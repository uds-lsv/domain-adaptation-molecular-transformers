"""Cross-validated evaluation of pre-computed molecule embeddings using random forests."""
import logging
import pathlib
import sys
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Tuple

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor


class ModelMetadata(NamedTuple):
    device: Literal["cpu", "cuda"]
    embedding_dim: int
    model_path: pathlib.Path
    num_samples: int
    domain_adaptation: Literal["cbert", "sbert", "mtr", "mlm"] = None
    pretraining: Literal["mlm", "mtr"] = None
    pretraining_size: int = None


def get_logger():
    logger = logging.getLogger("eamt.finetune")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


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


class PrecomputedEmbeddingWrapper:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
        self.y_col = None

    def fit(self, train):
        assert self.y_col is not None
        self.model.fit(self.embeddings[train.index], train[self.y_col])

    def predict(self, test):
        pred = self.model.predict(self.embeddings[test.index])
        return pred

    def validate(self, train, test):
        self.fit(train)
        return self.predict(test)

    # cross_validate expects a callable that returns a model
    def __call__(self, y_col):
        self.y_col = y_col
        return self


def eval(args):
    # Import locally to avoid import exception, since not currently installed in
    # the docker image. Since __main__ imports from cli.py which imports eval from here
    # global imports will be tried, hence resulting in an exception no matter if we are
    # actually in the process of evaluation.
    from useful_rdkit_utils.split_utils import (
        cross_validate,
        get_butina_clusters,
        get_random_clusters,
    )

    logger = get_logger()
    logger.info(f"Evaluating {args.embedding_file}")

    # Load smiles for this dataset
    with h5py.File(args.embedding_file, "r") as file:
        ds_name = file.attrs["dataset_name"]

        # We need to remove the censored datapoints.
        # the CSV contains the correct indices so we don't
        # mess up the smiles <-> embedding mapping
        if ds_name.startswith("adme_microsom_stab"):
            ds_name += "_cleaned"
            index_col = [0]  # We need the indices
        else:
            index_col = False  # Use default indices

        ds_source = args.data_dir / f"{ds_name}.csv"
        assert ds_source.exists()

        df = pd.read_csv(ds_source, index_col=index_col)
        # Cross validate expects a SMILES column
        df = df.rename(columns={"smiles": "SMILES"})

    embeddings_for_model, metadata = load_embeddings(args.embedding_file)

    group_list = [
        ("random", get_random_clusters),
        ("butina", get_butina_clusters),
    ]
    model_list = [
        (name, PrecomputedEmbeddingWrapper(embeddings))
        for name, embeddings in embeddings_for_model.items()
    ]

    target_cols = [c for c in df.columns if c != "SMILES"]
    logger.info(f"Target columns: {target_cols}")
    for y_col in target_cols:
        results_df = cross_validate(df, model_list, y_col, group_list)

        results_df.to_csv(f"{args.output_dir}/{ds_name}_{y_col}.csv", index=False)
