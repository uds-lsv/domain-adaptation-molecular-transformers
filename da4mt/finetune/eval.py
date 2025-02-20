#!/usr/bin/env python3
"""Train models on embeddings stored in HDF5 files with multiple train/val/test splits.

This script implements a flexible training pipeline that:
1. Loads embeddings from HDF5 groups
2. Supports multiple train/val/test splits via JSON files
3. Trains models using the specified splits and target data
"""

import argparse
import json
import logging
import pathlib
import shutil
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple, Literal, List

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

from da4mt.cli import add_eval_args
from da4mt.utils import (
    ScaledLinearRegression,
    ClassificationMetrics,
    RegressionMetrics,
    Metrics,
)


def get_logger():
    logger = logging.getLogger("eamt.finetune")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


class ModelMetadata(NamedTuple):
    device: Literal["cpu", "cuda"]
    embedding_dim: int
    model_path: pathlib.Path
    num_samples: int
    domain_adaptation: Literal["cbert", "sbert", "mtr", "mlm"] = None
    pretraining: Literal["mlm", "mtr"] = None
    pretraining_size: int = None


def load_embeddings(
    hdf5_file: Path,
) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, ModelMetadata]]:
    """Load embeddings from all groups in HDF5 file.

    :param hdf5_file: Path to HDF5 file containing embedding datasets
    :return: Dictionary mapping group names to embedding arrays
    """
    embeddings = {}
    metadata = {}
    with h5py.File(hdf5_file, "r") as file:
        for name, group in file.items():
            dataset = group["embeddings"]
            embeddings[name] = dataset[:]  # Copy as numpy array
            metadata[name] = ModelMetadata(**dict(dataset.attrs))

    return embeddings, metadata


def train_and_eval_model(
    model: BaseEstimator, embeddings, targets, train_idx, test_idx, metrics_fn: Metrics
) -> Tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float]]:
    """Train model on provided data bundle.

    :param model: Model to train
    :param embeddings: Embeddings of this dataset
    :param targets: Target values of this dataset
    :param train_idx: Index of train split
    :return: Predictions on the training and test set
    """

    model.fit(embeddings[train_idx], targets[train_idx].ravel())

    train_metrics, train_preds = metrics_fn(
        embeddings[train_idx],
        targets[train_idx],
        targets.shape[1],
        return_predictions=True,
    )
    test_metrics, test_preds = metrics_fn(
        embeddings[test_idx],
        targets[test_idx],
        targets.shape[1],
        return_predictions=True,
    )

    return train_metrics, train_preds, test_metrics, test_preds


def store_predictions(
    h5_file: Path,
    group_name: str,
    split_name: str,
    train_smiles: List[str],
    test_smiles: List[str],
    train_preds: NDArray[np.float32],
    test_preds: NDArray[np.float32],
    train_metrics: Optional[dict] = None,
    test_metrics: Optional[dict] = None,
) -> None:
    """Store predictions in HDF5 file using nested group structure.

    Structure:
    group_name/
        ├── embeddings
        └── predictions/
            └── split_name/
                ├── train_smiles
                ├── train
                ├── test_smiles
                ├── test
                └── metrics (attributes)

    :param h5_file: Path to HDF5 file
    :param group_name: Name of the embedding group
    :param split_name: Name of the split (derived from split file)
    :param train_preds: Predictions on training set
    :param test_preds: Predictions on test set
    :param train_metrics: Optional dictionary of evaluation metrics on train set
    :param test_metrics: Optional dictionary of evaluation metrics on test set
    """
    with h5py.File(h5_file, "r+") as f:
        # Get or create the predictions group
        group = f[group_name]
        pred_group = group.require_group(
            "predictions"
        )  # Raises an error if the group already exists

        # Create or replace split group
        split_path = f"{split_name}"
        if split_path in pred_group:
            del pred_group[split_path]
        split_group = pred_group.create_group(split_path)

        # Store predictions
        split_group.create_dataset("train_smiles", data=train_smiles)
        split_group.create_dataset("train", data=train_preds)
        split_group.create_dataset("test_smiles", data=test_smiles)
        split_group.create_dataset("test", data=test_preds)

        # Store metrics as attributes if provided
        if train_metrics:
            for key, value in train_metrics.items():
                split_group.attrs[f"train_{key}"] = value
        if test_metrics:
            for key, value in test_metrics.items():
                split_group.attrs[f"test_{key}"] = value


def get_model(
    model_type: Literal["linear", "random-forest", "svm"],
    task: Literal["regression", "classification"],
) -> Tuple[BaseEstimator, Metrics]:
    if task == "classification":
        if model_type == "linear":
            model = LogisticRegression(random_state=42)
        elif model_type == "random-forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "svm":
            model = SVC(probability=True, C=5.0, kernel="rbf", random_state=42)

        metrics_cls = ClassificationMetrics(model)

    elif task == "regression":
        if model_type == "linear":
            model = ScaledLinearRegression()
        elif model_type == "random-forest":
            model = RandomForestRegressor(random_state=42)
        elif model_type == "svm":
            model = SVR(C=5.0, kernel="rbf")

        metrics_cls = RegressionMetrics(model)

    return model, metrics_cls


def eval(args) -> None:
    """Main entry point of the script."""

    # Setup logging
    logger = get_logger()

    # Load data
    embeddings_dict, metadata_dict = load_embeddings(args.hdf5_file)

    # Load targets and smiles
    df = pd.read_csv(args.targets)

    target_cols = [c for c in df.columns if c != "smiles"]
    logger.info(f"Target columns: {target_cols}")
    smiles = df["smiles"]
    targets = df[target_cols].values

    output_hdf5 = args.output_dir / args.hdf5_file.name

    # Copy clean embedding file to store the results
    # if the file does not exist otherwise require manual
    # flag to overwrite.
    if output_hdf5.exists() and args.overwrite_existing:
        shutil.copy2(args.hdf5_file, output_hdf5)
    elif not output_hdf5.exists():
        shutil.copy2(args.hdf5_file, output_hdf5)

    # Process each split
    for split_file in args.splits:
        logger.info(f"Processing split: {split_file}")

        with open(split_file, "r") as f:
            split = json.load(f)

        if not args.keep_val_separate:
            # Merge validation and train set
            split["train"] = split["train"] + split["val"]
            del split["val"]

        # Train on each embedding group
        for group_name, embeddings in embeddings_dict.items():
            logger.info(f"Training on group: {group_name}")

            for model_type in args.model:
                model, metrics_fn = get_model(model_type, args.task)

                train_metrics, train_preds, test_metrics, test_preds = (
                    train_and_eval_model(
                        model,
                        embeddings,
                        targets,
                        split["train"],
                        split["test"],
                        metrics_fn,
                    )
                )

                store_predictions(
                    output_hdf5,
                    group_name,
                    f"{model_type}/{split_file.name}",
                    smiles[split["train"]].to_list(),
                    smiles[split["test"]].to_list(),
                    train_preds,
                    test_preds,
                    train_metrics,
                    test_metrics,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models on HDF5 embeddings with multiple splits."
    )
    parser = add_eval_args(parser)
    eval(parser.parse_args())
