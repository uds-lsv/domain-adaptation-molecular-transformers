import json
import logging
import math
import pathlib
from typing import Literal, List, Optional, Tuple

import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import BertConfig, TrainingArguments, BertTokenizerFast, Trainer

from da4mt.models.bert_for_regression import BertForRegression, BertForRegressionConfig
from da4mt.utils import PhysicoChemcialPropertyExtractor


def keep_subset(values, subset: Literal["all", "surface"], label_names: List[str]):
    """
    Keep only the values for which the label is in the subset of labels.
    :param values: List of arbitrary values
    :param subset: Subset of labels either 'all' or 'surface'.
    :param label_names: List of all labels in order.
    """
    if subset == "surface":
        keep = set(
            i
            for i, name in enumerate(label_names)
            if name in PhysicoChemcialPropertyExtractor.get_surface_descriptor_subset()
        )
    elif subset == "all":
        keep = range(len(label_names))
    else:
        raise ValueError(
            f"Invalid subset. Must be one of 'all', 'surface' not '{subset}.'"
        )

    return [v for i, v in enumerate(values) if i in keep]


def add_normalization_to_config(
    config: BertConfig, normalization_file: str, subset: Literal["all", "surface"]
):
    """
    :param config: Preconfigured BertConfig
    :param normalization_file: Path to JSON file with mean, std and label_name for each target
    :param subset: Which subset of labels to include, either 'all' or 'surface'
    :return: Config including normalization/std/id2label and list of all labels. List of all labels includes also
        those not in the subset.
    """
    with open(normalization_file) as f:
        normalization_values = json.load(f)

    full_labels = normalization_values["label_names"]

    # Remove mean/std/labels not in the subset
    config.norm_mean = keep_subset(normalization_values["mean"], subset, full_labels)
    config.norm_std = keep_subset(normalization_values["std"], subset, full_labels)
    config.id2label = {
        i: label
        for i, label in enumerate(
            keep_subset(normalization_values["label_names"], subset, full_labels)
        )
    }

    assert len(config.norm_mean) == len(
        config.norm_std
    ), "Size mismatch between mean and std arrays."

    config.num_labels = len(config.norm_mean)
    config.property_subset = subset
    return config, full_labels


class preprocess_function:
    def __init__(self, tokenizer, id2label, subset: Literal["all", "surface"]):
        self.tokenizer = tokenizer
        self.subset = subset
        self.label_names = id2label

    def __call__(self, examples, block_size=128):
        def _clean_property(x):
            return 0.0 if x == "" or "inf" in str(x) else float(x)

        batch_encoding = self.tokenizer(
            examples["smile"],
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
        # Examples is a list of molecules, each with K properties
        # Using label_ids instead of "label", lets HF automatically handle multiple labels
        # TODO: Consider normalizing data here, rather than inside the model class
        batch_encoding["label_ids"] = [
            keep_subset(
                [_clean_property(prop) for prop in labels],
                subset=self.subset,
                label_names=self.label_names,
            )
            for labels in examples["labels"]
        ]

        return batch_encoding


def train_mtr(
    model: BertForRegression,
    tokenizer: BertTokenizerFast,
    training_args: TrainingArguments,
    dataset: DatasetDict,
    property_subset: Literal["all", "surface"],
    model_dir: str,
    logger: logging.Logger,
    orig_labels: List[str],
    save_model: bool = True,
):
    """
    :param model: (Pretrained) model instance
    :param tokenizer: tokenizer of this model
    :param training_args: HuggingFace Training arguments
    :param dataset: DatasetDict with train and validation split
    :param property_subset: Subset of the physico chemical properties to use
    :param model_dir: Output directory
    :param logger: Logger instance
    :param orig_labels: List with all labels
    :param save_model: If true saved the model and tokenizer in the model_dir
    :return:
    """
    tokenized_dataset = dataset.map(
        preprocess_function(tokenizer, orig_labels, subset=property_subset),
        batched=True,
        remove_columns=["smile", "labels"],
    )

    # Some values are so large, that in float32 they become inf, we replace them by the maximum representable value
    def replace_inf_and_nan(samples):
        max_val = torch.finfo(samples["label_ids"].dtype).max
        min_val = torch.finfo(samples["label_ids"].dtype).min
        samples["label_ids"] = torch.nan_to_num(
            samples["label_ids"], 0.0, posinf=max_val, neginf=min_val
        )
        return samples

    tokenized_dataset.set_format(type="torch")
    tokenized_dataset = tokenized_dataset.map(replace_inf_and_nan, batched=True)

    assert not torch.any(torch.isinf(tokenized_dataset["train"]["label_ids"]))

    if tokenized_dataset.get("validation"):
        assert not torch.any(torch.isinf(tokenized_dataset["validation"]["label_ids"]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
    )
    trainer.train()

    if save_model:
        logger.info(f"Saving model and tokenizer to {model_dir}.")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

    if trainer.state.best_metric:
        assert not np.isnan(trainer.state.best_metric)
    return model, tokenizer, trainer.state.best_metric


def pretrain_mtr(
    model_dir: str,
    model_config: BertConfig,
    training_args: TrainingArguments,
    tokenizer_path: str,
    train_file: str,
    validation_file: str,
    normalization_file: str,
    train_size: str,
    logger: logging.Logger,
    property_subset: Literal["all", "surface"],
    selection: Literal["random", "cluster"] = "random",
    cluster_file: pathlib.Path = None,
):
    tokenizer = BertTokenizerFast(f"{tokenizer_path}-vocab.txt")

    model_config, orig_labels = add_normalization_to_config(
        model_config, normalization_file, property_subset
    )

    model = BertForRegression(model_config)

    logger.info(f"Training file: {train_file}.")
    logger.info(f"Validation file: {validation_file}.")
    logger.info(f"Normalization file: {normalization_file}.")

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
    )
    # Select given data range
    logger.info(f"Using {train_size * 100}% of the training data.")
    if train_size < 1.0:
        n_samples = math.ceil(raw_datasets["train"].num_rows * train_size)
        # We select the first N% of the dataset
        if selection == "random":
            indices = range(n_samples)
    else:
        if cluster_file is None:
            raise ValueError(
                "Selecting from cluster requires a 'custer_file' to be specified."
            )

        with open(cluster_file, "r") as f:
            indices = json.load(f)

    raw_datasets["train"] = raw_datasets["train"].select(indices)
    logger.info(f"Training data has {raw_datasets['train'].num_rows:_} entries.")
    train_mtr(
        model,
        tokenizer,
        training_args,
        raw_datasets,
        property_subset,
        model_dir,
        logger,
        orig_labels,
        save_model=True,
    )


def adapt_mtr(
    model_path: str,
    output_path: pathlib.Path,
    train_fp: pathlib.Path,
    normalization_fp: pathlib.Path,
    training_args: TrainingArguments,
    logger: logging.Logger,
    save_model: bool = True,
    splits_fp: Optional[pathlib.Path] = None,
) -> Tuple[BertForRegression, BertTokenizerFast, float]:
    """
    Adapts the pretrained model using Multitask Regression


    :param model_path: Path to the directory containing pretrained model
    :param output_path: Path where the model will be saved
    :param train_fp: Path to the training file. Should be in jsonl format with keys being the molecules in SMILES and
        the values the phyisico chemical properties
    :param normalization_fp: Filepath to the normalization values for each property
    :param training_args: Training Arguments for HF Trainer
    :param logger: Logger
    :param save_model: Where model should be saved, useful for cross validation
    :param splits_fp: Optional path to file specifying splits. Should be a json file with keys "train", "test", "val"
        with indices for the respective splits as values.
    :return:
    """
    logger.info(f"Running domain adaptation with mtr on {model_path}.")

    ds = load_dataset("json", data_files=str(train_fp))

    if splits_fp is not None:
        with open(splits_fp, "r") as f:
            split = json.load(f)
            ds = DatasetDict(
                train=ds["train"].select(split["train"]),
                validation=ds["train"].select(split["val"]),
            )
    logger.info(f"Normalization file: {normalization_fp}.")

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model_config = BertForRegressionConfig.from_pretrained(model_path)
    if model_config.norm_mean is None:
        # We use all descriptors if this model has not been pretrained
        # with MTR, otherwise we use the subset on that the model
        # has been trained.
        model_config, orig_labels = add_normalization_to_config(
            model_config, str(normalization_fp), "all"
        )
        assert model_config.num_labels != 0
        property_subset = "all"
    else:
        orig_labels = list(model_config.id2label.values())
        property_subset = model_config.property_subset

    model = BertForRegression.from_pretrained(model_path, config=model_config)

    return train_mtr(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        dataset=ds,
        property_subset=property_subset,
        model_dir=str(output_path),
        logger=logger,
        orig_labels=orig_labels,
        save_model=save_model,
    )
