import json
import logging
import math
import pathlib
from typing import Tuple, Union, Optional, Literal

from datasets import DatasetDict, load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BertTokenizer,
)


class tokenizer_function:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples, block_size=128, text_column_name="text"):
        # Remove empty lines
        examples[text_column_name] = [
            line
            for line in examples[text_column_name]
            if len(line) > 0 and not line.isspace()
        ]
        return self.tokenizer(
            examples[text_column_name],
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            padding="max_length",
            # We use this option because DataCollatorForLanguageModeling is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )


def pretrain_mlm(
    model_dir: str,
    model_config: BertConfig,
    training_args: TrainingArguments,
    tokenizer_path: str,
    train_file: str,
    validation_file: str,
    train_size: float,
    logger: logging.Logger,
    selection: Literal["random", "cluster"] = "random",
    cluster_file: Optional[pathlib.Path] = None,
):
    """
    :param model_dir: Output directory where model and tokenizer are saved
    :param tokenizer_path: Path of the trained Bert Tokenizer
    :param train_file: Path to the training file. Should contain one SMILES string per line
    :param validation_file: Path to the validation file. Should contain one SMILES string per line
    :param train_size: Proportion of the training set that is used
    """
    tokenizer = BertTokenizerFast(f"{tokenizer_path}-vocab.txt")
    model = BertForMaskedLM(model_config)

    logger.info(f"Training file: {train_file}.")
    logger.info(f"Validation file: {validation_file}.")

    raw_datasets = load_dataset(
        "text", data_files={"train": train_file, "validation": validation_file}
    )
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

    train_mlm(logger, model, model_dir, raw_datasets, tokenizer, training_args)


def adapt_mlm(
    model_path: str,
    output_path: pathlib.Path,
    train_fp: pathlib.Path,
    training_args: TrainingArguments,
    logger: logging.Logger,
    save_model: bool = True,
    splits_fp: Optional[pathlib.Path] = None,
) -> Tuple[BertForMaskedLM, BertTokenizerFast, float]:
    """
    Runs domain adaptation using Masked Language Modeling.
    If not splits are given, the full content of the train_file is used

    :param model_path: Path to the directory containing the pretrained model
    :param output_path: Directory where the adapted model will be saved
    :param train_fp: Path to the training file
    :param training_args: Training Arguments for HF Trainer
    :param logger: Logger
    :param save_model: If model should be saved, useful if cross validation needs to be done
    :param splits_fp: Path to the splits file. Should be a json file with "train", "val" and "test" keys
        that contain the indices of the respective datapoints as values. Test set is not used
    :return:
    """

    logger.info(f"Running domain adaptation with mlm on {model_path}.")

    # By default, returns a DatasetDict with a train key
    ds = load_dataset("csv", data_files=str(train_fp))
    ds = ds.select_columns(["smiles"])
    ds = ds.rename_column("smiles", "text")

    # If splits are specified we take the given subsets of the full dataset by index
    if splits_fp is not None:
        with open(splits_fp, "r") as f:
            split = json.load(f)

        ds = DatasetDict(
            train=ds["train"].select(split["train"]),
            validation=ds["train"].select(split["val"]),
        )

    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    return train_mlm(
        logger,
        model,
        output_path,
        ds,
        tokenizer,
        training_args,
        save_model=save_model,
    )


def train_mlm(
    logger: logging.Logger,
    model: BertForMaskedLM,
    model_dir: pathlib.Path,
    raw_datasets: DatasetDict,
    tokenizer: Union[BertTokenizer, BertTokenizerFast],
    training_args: TrainingArguments,
    save_model=True,
) -> Tuple[BertForMaskedLM, BertTokenizerFast, float]:
    """Train the given model using masked-language modeling

    :param logger: Logger instance
    :param model: BERTForMaskedML model instance
    :param model_dir: Output directory where the trained model will be stored
    :param raw_datasets: Untokenized `DatasetDict` with "train" and "validation" keys. Each value should be a dataset with a "text" column.
    :param tokenizer: BertTokenizer
    :param training_args: Additional arguments for training e.g. batch size
    :param save_model: If the model checkpoint should be saved to `model_dir`, defaults to True
    :return: Trained model, tokenizer and validation loss.
    """
    tokenized_dataset = raw_datasets.map(
        tokenizer_function(tokenizer),
        batched=True,
        # Remove the untokenized original text from the dataset, no longer required
        remove_columns=["text"],
        desc="Tokenize",
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
    )
    trainer.train()

    if save_model:
        logger.info(f"Saving model and tokenizer to {model_dir}.")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

    return model, tokenizer, trainer.state.best_metric
