import json
import logging
import os
import pathlib
import random
from typing import Literal, List, Generator, NamedTuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from transformers import BertTokenizerFast
import torch
import seaborn as sns

from da4mt.utils import get_task_head_and_metrics
from da4mt.filename_parser import parse_model_name


INTERMEDIATE_DIR = pathlib.Path("/data/users/mrdupont/da4mt/finetune_split_results")
INTERMEDIATE_DIR.mkdir(exist_ok=True, parents=True)


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch, and CUDA.

    :param seed: The random seed to use
    :param deterministic_cudnn: If True, configure CuDNN to use deterministic algorithms
    """
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For HuggingFace
    os.environ["HF_SEED"] = str(seed)


class CLSFeatureExtractionPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, sentencetransformer: SentenceTransformer, device):
        super().__init__()
        self.model = sentencetransformer
        self.device = device

    def transform(self, X, y=None):
        return self.model.encode(X, batch_size=256, device=self.device)

    def fit(self, X, y=None):
        return self


class Storage(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        self.input = X
        return X

    def fit(self, X, y=None):
        return self


class EvalResult(NamedTuple):
    split_name: Literal["train", "test", "val"]
    metric: str
    target_name: str
    value: float
    fold: str
    splitter: str
    model_path: str
    dataset_name: str
    domain_adaptation: Literal["mtr", "mlm", "cbert", "sbert", "none"]
    pretraining: Literal["mlm", "mtr"]
    train_size: float


def get_model(model_path: pathlib.Path, logger: logging.Logger) -> SentenceTransformer:
    """
    Load or create a SentenceTransformer model based on the model path.
    If the model is not a SentenceTransformer, i.e. was not trained using a
    contrastive fashion, the CLS token is used as the final embedding

    :param pathlib.Path model_path: Path to the model
    :param logging.Logger logger: Logger instance
    :return: Loaded SentenceTransformer model
    :rtype: SentenceTransformer
    """
    if any(
        contrastive_method in model_path.stem
        for contrastive_method in ("sbert", "cbert")
    ):
        logger.info("Loading the model as 'SentenceTransformer'")
        model = SentenceTransformer(model_path)
    else:
        logger.info("Creating new SentenceTransformer with '[CLS]' pooling")
        word_embedding_model = models.Transformer(
            model_path,
            max_seq_length=128,
            tokenizer_args=dict(
                add_special_tokens=True,
                truncation=True,
                max_length=128,
                padding="max_length",
            ),
        )
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        word_embedding_model.tokenizer = tokenizer
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls"
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def evaluate_model(
    X: List[str],
    y: np.ndarray,
    target_names: List[str],
    model_path: pathlib.Path,
    split_files: List[pathlib.Path],
    task: Literal["regression", "classification"],
    device: Literal["cpu", "cuda"],
    seed: int,
    logger: logging.Logger,
) -> Generator[EvalResult, None, None]:
    """
    Evaluate a model on the given dataset.

    :param List[str] X: List of SMILES strings
    :param np.ndarray y: Target values
    :param List[str] target_names: Names of the target variables
    :param pathlib.Path model_path: Path to the model
    :param List[pathlib.Path] split_files: List of paths to split files
    :param Literal["regression", "classification"] task: Type of task
    :param Literal["cpu", "cuda"] device: Device to use for computation
    :param logging.Logger logger: Logger instance
    :param str domain_adaptation: Type of domain adaptation
    :yield: Generator of evaluation results
    """
    model = get_model(model_path, logger)

    model_args = parse_model_name(model_path)

    seed_everything(seed)

    feature_extractor = CLSFeatureExtractionPipeline(
        sentencetransformer=model, device=device
    )
    estimator, metrics_cls = get_task_head_and_metrics(task)
    pipe = Pipeline(
        [
            ("embedding", feature_extractor),
            ("embedding_store", Storage()),
            ("classifier", estimator),
        ],
        verbose=True,
    )

    for split_file in split_files:
        # splite_file is {dataset}_{k}.{splitter}_splits.json
        splitter = split_file.suffixes[0].removeprefix(".").removesuffix("_splits")

        with open(split_file, "r") as f:
            split = json.load(f)

        # Train model
        train_indices = split["train"]
        if task == "regression":
            assert len(y.ravel()) == len(
                y
            ), "Regression dataset might have multiple targets"
            pipe.fit([X[i] for i in train_indices], y[train_indices].ravel())
        else:
            pipe.fit([X[i] for i in train_indices], y[train_indices])

        embeddings = pipe.named_steps["embedding_store"].input

        # Evaluate on all splits
        num_columns = len(target_names)
        for name, indices in split.items():
            logger.debug(f"{model_path}: {split_file} {name}")
            metrics_fn = metrics_cls(pipe)

            metrics, y_preds = metrics_fn(
                [X[i] for i in indices],
                y[indices],
                num_columns,
                return_predictions=True,
            )

            # Store embeddings and predictions
            with open(INTERMEDIATE_DIR / model_path.stem, "wb") as intermediate:
                np.savez(
                    intermediate,
                    embeddings=embeddings[indices],
                    y_pred=y_preds,
                    y_true=y[indices],
                )

            # Return metric values for this split
            for key, values in metrics.items():
                for target_name, value in zip(target_names, np.atleast_1d(values)):
                    result = EvalResult(
                        name,
                        key,
                        target_name,
                        value,
                        str(split_file),
                        splitter,
                        str(model_path),
                        model_args.dataset,
                        model_args.domain_adaptation,
                        model_args.pretraining,
                        model_args.train_size,
                    )

                    logger.debug(result)

                    yield result


def evaluate_models(
    X: List[str],
    y: np.ndarray,
    dataset_name: str,
    target_names: List[str],
    model_dir: pathlib.Path,
    split_files: List[pathlib.Path],
    pretrain_filter: Literal["mlm", "mtr"],
    task: Literal["regression", "classification"],
    device: Literal["cpu", "cuda"],
    seed: int,
    logger: logging.Logger,
    evaluation_type: Literal["domain_adapted", "pretrained", "untrained"],
) -> Generator[EvalResult, None, None]:
    """
    Evaluate models based on the specified evaluation type.

    :param List[str] X: List of SMILES strings
    :param np.ndarray y: Target values
    :param str dataset_name: Name of the dataset
    :param List[str] target_names: Names of the target variables
    :param pathlib.Path model_dir: Directory containing models
    :param List[pathlib.Path] split_files: List of paths to split files
    :param Literal["mlm", "mtr"] pretrain_filter: Pretraining filter
    :param Literal["regression", "classification"] task: Type of task
    :param Literal["cpu", "cuda"] device: Device to use for computation
    :param logging.Logger logger: Logger instance
    :param Literal["domain_adapted", "pretrained", "untrained"] evaluation_type: Type of evaluation to perform
    :yield: Generator of evaluation results
    """
    if evaluation_type == "domain_adapted":
        models = list(model_dir.glob(f"{dataset_name}_*_none-*/")) + list(
            model_dir.glob(f"{dataset_name}_*_{pretrain_filter}-*/")
        )
    elif evaluation_type == "pretrained":
        models = list(model_dir.glob(f"{pretrain_filter}-*/"))
    elif evaluation_type == "untrained":
        models = list(model_dir.glob("none-*/"))
    else:
        raise ValueError(f"Invalid evaluation_type: {evaluation_type}")

    logger.debug(f"Evaluating {evaluation_type} models: {models}")

    for model_path in models:
        yield from evaluate_model(
            X=X,
            y=y,
            target_names=target_names,
            model_path=model_path,
            split_files=split_files,
            task=task,
            device=device,
            seed=seed,
            logger=logger,
        )


def train_downstream_and_evaluate(
    train_file: pathlib.Path,
    splits_files: List[pathlib.Path],
    task: Literal["regression", "classification"],
    targets: Optional[List[str]],
    pretrain_dir: pathlib.Path,
    pretrain_filter: Literal["mlm", "mtr"],
    adapted_dir: pathlib.Path,
    output_dir: pathlib.Path,
    device: Literal["cpu", "cuda"],
    seed: int,
    logger: logging.Logger,
):
    """
    :param data_dir: Directory with preprocessed data files
    :param pretrain_dir: Directory with pretrained models
    :param adapted_dir: Directory with domain adapted models
    :param output_dir: Output directory for metrics
    :param dataset_name: Name of the dataset
    :param device: Torch device
    """
    dataset = pd.read_csv(train_file)
    dataset_name = train_file.stem

    logger.debug(f"Using split files: {splits_files}")

    # Make sure targets are correct
    if targets is None:
        targets = [c for c in dataset.columns if c != "smiles"]
    else:
        for target in targets:
            if target not in dataset.columns:
                raise ValueError(f"'{target}' is not a valid column in the dataset.")

    logger.debug(f"Using {targets} as targets.")

    X = dataset["smiles"].to_list()
    y = dataset[targets].values

    results = []
    for evaluation_type in ["untrained", "pretrained", "domain_adapted"]:
        model_dir = pretrain_dir if evaluation_type != "domain_adapted" else adapted_dir
        logger.debug(f"{model_dir=}")
        results.extend(
            evaluate_models(
                X=X,
                y=y,
                dataset_name=dataset_name,
                target_names=targets,
                model_dir=model_dir,
                split_files=splits_files,
                pretrain_filter=pretrain_filter,
                task=task,
                device=device,
                seed=seed,
                logger=logger,
                evaluation_type=evaluation_type,
            )
        )

    df = pd.DataFrame(
        results,
        columns=[
            "split",
            "metric",
            "task",
            "value",
            "fold",
            "splitter",
            "model",
            "dataset",
            "domain_adaptation",
            "pretraining",
            "train_size",
        ],
    )
    logger.info(f"Writing results to {output_dir}.")
    df["domain_adaptation"] = df["domain_adaptation"].fillna("none")
    df.to_csv(output_dir / f"{dataset_name}_{pretrain_filter}.csv")

    hue_order = ["none", "mlm", "mtr", "cbert", "sbert"]

    metric = "MAE" if task == "regression" else "AUROC"

    g = sns.catplot(
        df[(df["split"] == "test") & (df["metric"] == metric)],
        x="train_size",
        y="value",
        hue="domain_adaptation",
        errorbar="se",
        kind="point",
        capsize=0.03,
        linestyle="none",
        dodge=0.5,
        hue_order=hue_order,
    )
    g.set(ylabel=f"{metric} ± SE")
    g.figure.savefig(output_dir / f"{dataset_name}_{pretrain_filter}.png")
