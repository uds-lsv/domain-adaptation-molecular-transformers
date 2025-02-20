#!/usr/bin/env python3
import json
import logging
import os
import pathlib
import random
import sys
from functools import lru_cache
from typing import List, Literal, NamedTuple, Optional, Tuple
import numpy.typing as npt

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.utils import gen_batches
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


DATASETS = [
    ("adme_microsom_stab_h", "regression", ("microsom_stab_h",)),
    ("adme_microsom_stab_r", "regression", ("microsom_stab_r",)),
    ("adme_permeability", "regression", ("permeability",)),
    ("adme_ppb_h", "regression", ("ppb_h",)),
    ("adme_ppb_r", "regression", ("ppb_r",)),
    ("adme_solubility", "regression", ("solubility",)),
]

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

print_handler = logging.StreamHandler(stream=sys.stderr)
print_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
print_handler.setFormatter(formatter)

logger.addHandler(print_handler)


class EmbeddingResult(NamedTuple):
    embeddings: npt.NDArray
    targets: np.ndarray
    target_cols: List[str]


class DatasetEmbedder:
    """
    A callable class for embedding datasets using specified models.

    This class loads datasets from a given directory, applies the specified
    embedding model, and caches the results for efficiency.

    :param dataset_dir: pathlib.Path
        The directory containing the datasets.
    """

    def __init__(self, dataset_dir: pathlib.Path):
        self.dataset_dir = dataset_dir
        self.embedding_methods = {
            "physchem": self.embed_physchem,
            "molbert": self.embed_molbert,
            "molformer": self.embed_molformer,
            "fingerprint": self.embed_fingerprints,
        }

    @lru_cache(maxsize=None)
    def __call__(
        self, dataset_name: str, method: str, targets: Optional[Tuple[str]] = None
    ) -> EmbeddingResult:
        """
        Load the dataset and apply the specified embedding method.

        :param dataset_name: Name of the dataset file (without full path)
        :param method: Name of the embedding method to use
        :return: NamedTuple containing embeddings and metadata
        :raises ValueError: If the method is not recognized
        """
        if method not in self.embedding_methods:
            raise ValueError(f"Unknown embedding method: {method}")

        targets, names = self._load_targets(dataset_name, targets)
        embed_func = self.embedding_methods[method]
        embeddings = embed_func(dataset_name)
        return EmbeddingResult(embeddings, targets, names)

    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Loads CSV dataset by name

        :param dataset_name: Name of the dataset
        :raises FileNotFoundError: Raises if not file with {dataset_name}.csv is found in
            self.dataset_dir
        :return: Pandas DataFrame of the CSV file
        """
        file_path = self.dataset_dir / f"{dataset_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_csv(file_path)
        return df

    def _load_smiles(self, dataset_name: str) -> List[str]:
        """Loads the SMILES string from the given dataset"""
        df = self._load_dataset(dataset_name)
        return list(df["smiles"])

    def _load_targets(
        self, dataset_name: str, targets: Optional[Tuple[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Loads the targets of the given dataset. If no targets are specified uses all columns except 'smiles'"""
        df = self._load_dataset(dataset_name)

        if targets is None:
            targets = [c for c in df.columns if c != "smiles"]
        else:
            targets = list(targets)

        logger.debug(f"Using target columns {targets}")

        return df[targets].values, targets

    def _load_mtr_normalization(
        self, dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MTR normalization values of the given dataset and
        return the means and std of each label as an numpy array
        """

        file_path = self.dataset_dir / f"{dataset_name}_normalization_values.json"
        with open(file_path, "r") as f:
            values = json.load(f)

        # Check values
        means = np.array(values["mean"])
        stds = np.array(values["std"])
        for i, (mean, std, name) in enumerate(
            zip(values["mean"], values["std"], values["label_names"])
        ):
            if std == 0.0:
                logger.warning(f"Encountered 0.0 std for {name}")
                stds[i] = 1
            elif np.isnan(std):
                logger.warning(f"Encountered NaN std for {name}")
            elif np.isinf(std):
                logger.warning(f"Encounterd inf std for {name}")
            if np.isnan(mean):
                logger.warning(f"Encountered NaN mean for {name}")
            elif np.isinf(mean):
                logger.warning(f"Encountered inf mean for {name}")

        return means, stds

    def _load_mtr(self, dataset_name: str) -> List[List[float]]:
        """
        Load the physchem properties of the dataset and normalize
        """

        file_path = self.dataset_dir / f"{dataset_name}_mtr.jsonl"
        data = []
        with open(file_path, "r") as file:
            for line in file:
                properties = json.loads(line.strip())

                # We don't need the SMILES string
                data.append(properties["labels"])

        data = np.array(data)
        means, std = self._load_mtr_normalization(dataset_name)

        normalized = (data - means) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized

    def embed_physchem(self, dataset: str) -> List[List[float]]:
        return self._load_mtr(dataset)

    def embed_molbert(self, dataset: str) -> List[List[float]]:
        from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

        molbert_ckpt = pathlib.Path(__file__).parent / "checkpoints" / "molbert.ckpt"
        featurizer = MolBertFeaturizer(str(molbert_ckpt), device="cpu")

        smiles = self._load_smiles(dataset)
        features = []
        # Molbert does no automatic batching
        for batch in gen_batches(len(smiles), 512):
            features.extend(featurizer.transform(smiles[batch])[0])

        return np.vstack(features)

    def embed_fingerprints(self, dataset: str):
        smiles = self._load_smiles(dataset)

        fps = []
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            fps.append(mfpgen.GetFingerprintAsNumPy(mol))

        return np.vstack(fps)


    def embed_molformer(self, dataset: str) -> List[List[float]]:
        from transformers import AutoModel, AutoTokenizer

        molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True,
        )
        molformer.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )

        smiles = self._load_smiles(dataset)
        outputs = []
        for batch in gen_batches(len(smiles), 512):
            model_inputs = tokenizer(
                smiles[batch],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            model_inputs = {k: v for k, v in model_inputs.items()}
            with torch.no_grad():
                model_outputs = molformer(**model_inputs)
            outputs.extend(model_outputs.pooler_output.cpu())

        return torch.vstack(outputs).numpy()


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


def save_embeddings_to_hdf5(
    embeddings: npt.NDArray,
    model_name: str,
    output_dir: pathlib.Path,
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
    hdf5_path = output_dir / f"{dataset_name}_comparison_embeddings.hdf5"

    with h5py.File(hdf5_path, "a") as f:
        # Add file-level metadata if it doesn't exist
        if "dataset_path" not in f.attrs:
            logger.info("Initializing file-level metadata")
            f.attrs["dataset_path"] = dataset_name
            f.attrs["dataset_name"] = dataset_name

        if model_name in f:
            logger.warning(f"A group with the name '{model_name}' already exists!")
            del f[model_name]  # Remove existing dataset if it exists

        # Create group and dataset
        group = f.create_group(model_name)
        dataset = group.create_dataset("embeddings", data=embeddings)

        # Store metadata as attributes
        dataset.attrs["model_path"] = str(model_name)
        dataset.attrs["embedding_dim"] = embeddings.shape[1]
        dataset.attrs["num_samples"] = embeddings.shape[0]
        dataset.attrs["device"] = device


def embed(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    device: Literal["cpu", "cuda"],
    seed: int,
):
    for dataset_name, task, targets in DATASETS:
        embedder = DatasetEmbedder(data_dir)
        for feature in embedder.embedding_methods.keys():
            seed_everything(seed)
            embeddings, y, target_cols = embedder(
                dataset_name, method=feature, targets=targets
            )

            save_embeddings_to_hdf5(
                embeddings=embeddings,
                model_name=feature,
                output_dir=output_dir,
                dataset_name=dataset_name,
                device=device,
                logger=logger,
            )
            logger.info(f"Saved embeddings for '{feature}'")


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    root = here.parent
    embed(data_dir=root / "data", output_dir=here / "embeddings", seed=0, device="cuda")
