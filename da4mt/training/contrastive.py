import json
import logging
import pathlib
import typing
from typing import List, Literal

import pandas as pd
from sentence_transformers import (
    models,
    SentenceTransformer,
    losses,
    InputExample,
    evaluation,
)
from sentence_transformers.evaluation import SimilarityFunction
from torch.utils.data import DataLoader
from transformers import TrainingArguments, BertTokenizerFast
from typing_extensions import NotRequired

from da4mt.utils import WandbLoggingLoss


class TrainingDataset(typing.TypedDict):
    train: List[InputExample]
    val: NotRequired[typing.Tuple[List[str], List[str], List[float]]]


def prepare_sbert_data(sample) -> InputExample:
    smiles_a, smiles_b, is_enumerated = sample
    return InputExample(texts=[smiles_a, smiles_b], label=float(is_enumerated))


def prepare_cbert_data(sample) -> InputExample:
    sent1, sent0, hard_neg = sample
    return InputExample(texts=[sent0, sent1, hard_neg])


def prepare_example(objective: Literal["sbert", "cbert"], example):
    if objective == "sbert":
        example_loader = prepare_sbert_data
    elif objective == "cbert":
        example_loader = prepare_cbert_data
    else:
        raise ValueError(f"Invalid contrastive objective {objective}.")

    return example_loader(example)


def train_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    data: TrainingDataset,
    logger: logging.Logger,
    objective: Literal["sbert", "cbert"],
    training_args: TrainingArguments,
):
    logger.info(f"Running domain adaptation with {objective}.")
    logger.info(f"Loading model from {model_path}.")
    word_embedding_model = models.Transformer(
        str(model_path),
        max_seq_length=128,
        tokenizer_args=dict(
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding="max_length",
        ),
    )
    word_embedding_model.tokenizer = BertTokenizerFast.from_pretrained(model_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # TODO: We should use MultipleNegativeRanking loss for both tasks
    #   But by default sentencetransformers then samples in batch negatives
    #   Make sure we want this.
    if objective == "sbert":
        loss = losses.CosineSimilarityLoss(model)
    elif objective == "cbert":
        loss = losses.MultipleNegativesRankingLoss(model)
    data_loader = DataLoader(
        data["train"], training_args.per_device_train_batch_size, shuffle=True
    )
    loss_wrapper = WandbLoggingLoss(loss, group=output_path.stem)

    # Calculates spearmanr of cosine similarities between sentences a and b
    # FIXME: For cbert, we always take the positive label, i.e. 1 as the similarity label
    #   In this case the compared to vector is a constant vector of ones, spearmanr
    #   does not work in this case and sentence-transformers returns -9999999
    evaluator = None
    if data.get("val") is not None:
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            *data["val"], write_csv=False, main_similarity=SimilarityFunction.COSINE
        )

    model.fit(
        train_objectives=[(data_loader, loss_wrapper)],
        epochs=training_args.num_train_epochs,
        warmup_steps=training_args.warmup_steps,
        optimizer_params={"lr": training_args.learning_rate},
        evaluator=evaluator,
        evaluation_steps=-1,  # Only at the end of an epoch
        callback=loss_wrapper.log_eval,
        save_best_model=False,  # We manually save the model at the end
    )

    return model, model.best_score


def load_data(train_fp, method: Literal["cbert", "sbert"], splits_fp=None):
    df = pd.read_csv(train_fp)
    samples = [prepare_example(method, e) for e in df.itertuples(index=False)]

    if splits_fp is not None:
        with open(splits_fp, "r") as f:
            split = json.load(f)

        # Sentence transformers validation expects a different format than just a List of InputExample.
        # Two lists of sentence and similarity scores for each pair
        # In our case we have anchor, positive pairs in the dataset
        # where the similarity is 1.0
        sentences_a = []
        sentences_b = []
        scores = []
        for sample in samples[split["val"]]:
            sentences_a.append(sample.texts[0])
            sentences_b.append(sample.texts[1])

            # For cbert the second text is always a positive sample
            # For sbert we have an explicit label
            if method == "sbert":
                scores.append(sample.label)
            else:
                scores.append(1.0)

        return {
            "train": samples[split["train"]],
            "val": (sentences_a, sentences_b, scores),
        }
    else:
        return {"train": samples}


def train_contrastive(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    train_fp: pathlib.Path,
    training_args: TrainingArguments,
    logger: logging.Logger,
    method: Literal["cbert", "sbert"],
    save_model: bool = True,
    splits_fp: typing.Optional[pathlib.Path] = None,
) -> typing.Tuple[SentenceTransformer, float]:
    """
    Runs domain adaptation using CBERT or SBERT contrastive learning.
    If no splits are given, the full content of the train_file is used

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
    logger.info(f"Running domain adaptation with {method} on {model_path}.")
    logger.info(f"Loading dataset: {train_fp}.")

    dataset = load_data(train_fp, method, splits_fp)

    model, score = train_model(
        model_path=model_path,
        output_path=output_path,
        logger=logger,
        training_args=training_args,
        data=dataset,
        objective=method,
    )

    if save_model:
        # When not calling this once before model.fit internal tokenizer saving
        # raises an error because of some serialization problems, although the exactly
        # same method is called. No idea what this changes.
        model.tokenizer.save_pretrained(output_path)
        model.save(str(output_path))

    return model, score
