import logging
import os
import pathlib
import random
import sys
from typing import List, Literal

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
import torch
from transformers import BertTokenizerFast
import numpy.typing as npt
import h5py

from da4mt.filename_parser import parse_model_name


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


def get_logger():
    logger = logging.getLogger("eamt.embed")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def load_model(model_path: pathlib.Path, logger: logging.Logger) -> SentenceTransformer:
    """
    Load or create a SentenceTransformer model based on the model path.
    If the model is not a SentenceTransformer, i.e. was not trained using a
    contrastive fashion, the CLS token is used as the final embedding

    :param pathlib.Path model_path: Path to the model
    """

    metadata = parse_model_name(model_path)
    if metadata.domain_adaptation == "sbert" or metadata.domain_adaptation == "cbert":
        logger.info("Loading the model as 'SentenceTransformer'")
        st = SentenceTransformer(model_path)
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
        st = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return st


def embed_smiles(
    model_path: pathlib.Path,
    smiles: List[str],
    device: Literal["cpu", "cuda"],
    logger: logging.Logger,
) -> npt.NDArray:
    model: SentenceTransformer = load_model(model_path, logger)
    embeddings: npt.NDArray = model.encode(
        smiles, batch_size=256, device=device, convert_to_numpy=True
    )

    return embeddings


def save_embeddings_to_hdf5(
    embeddings: npt.NDArray,
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_path: pathlib.Path,
    dataset_name: str,
    device: Literal["cpu", "cuda"],
    logger: logging.Logger,
) -> None:
    """
    Save embeddings to an HDF5 file with metadata.

    :param embeddings: NumPy array of embeddings
    :param model_path: Path to the model used for embedding
    :param output_dir: Directory to save the HDF5 file
    :param dataset_name: Name of the dataset used for embeddings
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create HDF5 filename based on dataset name
    hdf5_path = output_dir / f"{dataset_name}_embeddings.hdf5"

    model_metadata = parse_model_name(model_path)
    group_name = str(model_path.name)

    with h5py.File(hdf5_path, "a") as f:
        # Add file-level metadata if it doesn't exist
        if "dataset_path" not in f.attrs:
            logger.info("Initializing file-level metadata")
            f.attrs["dataset_path"] = str(dataset_path.absolute())
            f.attrs["dataset_name"] = dataset_name

        if group_name in f:
            logger.warning(f"A group with the name '{group_name}' already exists!")
            del f[group_name]  # Remove existing dataset if it exists

        # Create group and dataset
        group = f.create_group(group_name)
        dataset = group.create_dataset("embeddings", data=embeddings)

        # Store metadata as attributes
        dataset.attrs["model_path"] = str(model_path)
        dataset.attrs["embedding_dim"] = embeddings.shape[1]
        dataset.attrs["num_samples"] = embeddings.shape[0]
        dataset.attrs["domain_adaptation"] = (
            model_metadata.domain_adaptation if model_metadata.domain_adaptation else ""
        )  # Can be None
        dataset.attrs["pretraining"] = (
            model_metadata.pretraining if model_metadata.pretraining else ""
        )
        dataset.attrs["pretraining_size"] = model_metadata.train_size
        dataset.attrs["device"] = device


def embed(
    train_file: pathlib.Path,
    pretrain_dir: pathlib.Path,
    adapted_dir: pathlib.Path,
    output_dir: pathlib.Path,
    device: Literal["cpu", "cuda"],
    seed: int,
    logger: logging.Logger,
):
    dataset = pd.read_csv(train_file)
    smiles = dataset["smiles"].to_list()

    dataset_name = train_file.stem
    models = (
        # Domain adaptation of the untrained model
        list(adapted_dir.glob(f"{dataset_name}_*_none-*/"))
        # Domain adaptation of a pretrained model
        + list(adapted_dir.glob(f"{dataset_name}_*_mlm-*/"))
        + list(adapted_dir.glob(f"{dataset_name}_*_mtr-*/"))
        # Pretrained model
        + list(pretrain_dir.glob("mlm-*/"))
        + list(pretrain_dir.glob("mtr-*/"))
        # Untrained model
        + list(pretrain_dir.glob("none-*/"))
    )

    for model_path in models:
        logger.info(f"Creating embeddings for '{model_path}'")
        seed_everything(seed)
        embeddings = embed_smiles(model_path, smiles, device=device, logger=logger)

        save_embeddings_to_hdf5(
            embeddings=embeddings,
            model_path=model_path,
            output_dir=output_dir,
            dataset_path=train_file,
            dataset_name=dataset_name,
            device=device,
            logger=logger,
        )
        logger.info(f"Saved embeddings for '{model_path}'")


def embed_cli_wrapper(args):
    logger = get_logger()
    logger.debug(args)

    embed(
        train_file=args.train_file,
        pretrain_dir=args.pretraindir,
        adapted_dir=args.adapteddir,
        output_dir=args.outdir,
        device=args.device,
        seed=args.seed,
        logger=logger,
    )


if __name__ == "__main__":
    import argparse
    from da4mt.cli import add_embed_args

    parser = argparse.ArgumentParser()
    parser = add_embed_args(parser)

    embed_cli_wrapper(parser.parse_args())
