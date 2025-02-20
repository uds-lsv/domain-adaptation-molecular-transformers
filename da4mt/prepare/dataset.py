import argparse
import logging
import pathlib
import sys
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from da4mt.cli import add_prepare_dataset_args
from da4mt.utils import extract_physicochemical_props, randomize_smiles


def get_logger():
    logger = logging.getLogger("eamt.perpare.dataset")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def validate_arguments(args):
    if not args.file.exists():
        raise ValueError(f"{args.file} does not seem to be a valid path")

    if args.file.suffix != ".csv":
        warnings.warn(f"Expecting a CSV file, got '{args.file.suffix}' instead.")

    if not args.output_dir.exists() or not args.output_dir.is_dir():
        raise ValueError(f"{args.output_dir} is not a directory.")


def make_descriptors(df: pd.DataFrame, output_dir: pathlib.Path, name: str):
    """Calculates and stores physicochemical properties

    :param df: Full dataframe containing SMILES and targets
    :param output_dir: Directory where the file will be saved
    :param name: Name of the dataset
    """
    logger = get_logger()

    extract_physicochemical_props(
        df["smiles"].tolist(),
        output_path=output_dir / f"{name}_mtr.jsonl",
        logger=logger,
        normalization_path=output_dir / f"{name}_normalization_values.json",
        subset="all",
    )


def create_pairs(
    smiles: List[str], p: float, seed: int = None
) -> List[Tuple[str, str, bool]]:
    """
    Creates pairs of SMILES strings in the SBERT fashion.
    Each pair contains the original SMILES string of the molecule and
    either (a) an enumeration of this molecules' strings string or (b)
    the SMILES string of a random molecule in the dataset
    :param smiles: List of SMILES strings
    :param p: Probability of being an enumeration of this molecule
    :param seed: Random seed
    :return: Pairs of SMILES strings
    """
    rng = np.random.default_rng(seed)
    pairs = []

    indices = np.arange(len(smiles))
    possible_partners = np.ones_like(indices, dtype=np.bool_)
    for idx, smi in enumerate(tqdm(smiles)):
        possible_partners[idx] = False
        if rng.random() < p:
            partner = randomize_smiles(smi, rng=rng)
            is_enumeration = True
        else:
            partner_idx = rng.choice(indices[possible_partners])
            partner = smiles[partner_idx]
            is_enumeration = False
            assert smi != partner, "Molecule is the same"

        pairs.append((smi, partner, is_enumeration))

        # Make sure this molecule is a possible partner for the next molecule
        possible_partners[idx] = True
    return pairs


def make_sbert(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    name: str,
    seed: int,
):
    logger = get_logger()
    pairs = create_pairs(df["smiles"].to_list(), p=0.5, seed=seed)
    enumerated_smiles_df = pd.DataFrame(
        pairs, columns=["smiles_a", "smiles_b", "is_enumerated"]
    )
    outfile = output_dir / f"{name}_sbert.csv"
    enumerated_smiles_df.to_csv(
        outfile,
        index=False,
        encoding="utf-8-sig",
    )
    logger.info(f"Saved enumerated smiles to {outfile} for SBERT domain adaptation")


def make_cbert(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    name: str,
    seed,
):
    logger = get_logger()
    outfile = output_dir / f"{name}_cbert.csv"
    triplets = create_triplets(df["smiles"].to_list(), seed=seed)
    results_df = pd.DataFrame(triplets, columns=["sent1", "sent0", "hard_neg"])
    results_df.to_csv(outfile, index=False, encoding="utf-8-sig")
    logger.info(f"Saved enumerated smiles to {outfile} for CBERT domain adaptation")


def create_triplets(smiles: List[str], seed: int = None) -> List[Tuple[str, str, str]]:
    """
    Creates triplets of SMILES strings in the simCSE fashion.
    Each pair contains the original SMILES string of the molecule and
     an enumeration of this molecules' strings string and
    the SMILES string of a random molecule in the dataset.
    :param smiles: List of SMILES strings
    :param seed: Random seed
    :return: Triplets of SMILES strings
    """
    rng = np.random.default_rng(seed)
    triplets = []

    indices = np.arange(len(smiles))
    possible_negative = np.ones_like(indices, dtype=np.bool_)
    for idx, smi in enumerate(tqdm(smiles)):
        possible_negative[idx] = (
            False  # Make sure this molecule can't be its own negative
        )
        positive = randomize_smiles(smi, rng=rng)  # Enumeration as positive sample
        negative_idx = rng.choice(
            indices[possible_negative]
        )  # SMILES of random other molecule as negative sample
        negative = smiles[negative_idx]
        assert smi != negative, "Molecule is the same"

        triplets.append((smi, positive, negative))

        # Make sure this molecule is a possible negative for the next molecule
        possible_negative[idx] = True
    return triplets


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_prepare_dataset_args(parser)

    args = parser.parse_args()
    validate_arguments(args)
    return args


def make_data(args):
    validate_arguments(args)
    logger = get_logger()

    filename = args.file.stem

    logger.info(f"Loading {args.file} as CSV.")
    df = pd.read_csv(args.file)

    if "smiles" not in df.columns:
        raise ValueError("CSV file must contain 'smiles' column.")

    # Make sure the dataset has a continous index for splitting
    df = df.reset_index(drop=True)
    # Some datasets in chembench have an extra explicit index column, we don't make us of it
    if "index" in df.columns:
        warnings.warn(
            "Dataset contains 'index' column. Assuming this is not a target and can be safely removed. Make sure this is intentional"
        )
        df = df.drop(columns=["index"])

    # Have everything in one folder even if redundant
    df.to_csv(args.output_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")

    # Create features and training data once
    make_sbert(df, args.output_dir, filename, args.seed)
    make_cbert(df, args.output_dir, filename, args.seed)
    make_descriptors(df, args.output_dir, filename)


if __name__ == "__main__":
    make_data(get_args())
