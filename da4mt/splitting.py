from itertools import chain
import math
from collections import defaultdict
from typing import Dict, List, TypedDict

import numpy as np


class DatasetSplit(TypedDict):
    train: List[int]
    val: List[int]
    test: List[int]


class SplittingError(Exception): ...


def assert_no_overlap(folds: List[DatasetSplit]) -> None:
    """
    Assert that no overlap between val and test sets exists in different folds.
    :param folds: List of DatasetSplit objects
    """
    # Turns this function off if pythons optimized mode is activated
    # which would deactivate the assert statements anyway
    if not __debug__:
        return

    for i, a in enumerate(folds):
        for j, b in enumerate(folds):
            if i == j:
                continue

            for key in a.keys():
                if key == "train":
                    continue

                overlap = set(a[key]) & set(b[key])
                assert (
                    len(overlap) == 0
                ), f"Overlap between {key} sets in fold {i} and {j}: {overlap}"
                assert len(a[key]) == len(
                    b[key]
                ), f"Different sized splits: {key}, {i}, {j}"


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Generate a Murcko scaffold from a SMILES string.

    :param str smiles: SMILES string of the molecule
    :param bool include_chirality: Whether to include chirality in the scaffold
    :return: Murcko scaffold SMILES string
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def generate_scaffold_ds(smiles: List[str]) -> Dict[str, List[int]]:
    """
    Generate a dictionary mapping scaffolds to molecule indices.

    :param smiles_list: List of SMILES strings
    :return: Dictionary mapping scaffolds to lists of molecule indices
    """
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles):
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(ind)

    return {key: sorted(value) for key, value in scaffolds.items()}


def k_fold_scaffold_split(smiles: List[str], n: int) -> List[DatasetSplit]:
    """
    Attempts to split the dataset in to k-folds of train/test/validation set where the splits are
    determined by the scaffolds present in the dataset.
    This may fail, if not enough scaffolds with sufficient number of molecules are available
    without overlap between the test/val in the different folds.

    The algorithms roughly works as follows:
        Given a mapping from scaffolds to molecules which contain this molecule

        a) Iterate over all scaffolds
        b) If scaffold has been *used* as validation or test set, pack into training set
        c) Otherwise:
            1) Attempt to pack all molecules with this scaffold into the validation set
            2) Otherwise: Attempt to pack into the test set
            3) Otherwise: Pack into the training set

            4) If packed into validation or test set, mark this scaffold as 'used'
        d) Repeat k times.

    :param smiles: List of SMILES strings
    :param n: Number of folds
    :return: List of DatasetSplit objects, one for each fold
    """
    test_frac = 0.1
    val_frac = 0.1

    test_cutoff = math.floor(test_frac * len(smiles))
    val_cutoff = math.floor(val_frac * len(smiles))
    scaffolds = generate_scaffold_ds(smiles)

    folds: List[DatasetSplit] = [{"test": [], "val": [], "train": []} for _ in range(n)]

    test_or_val = [[False] * len(scaffolds) for _ in range(n)]

    for k in range(n):
        for i, (scaffold, mol_indices) in enumerate(scaffolds.items()):
            # If used as validation/test set in any fold before, this has to go into the
            # training set of this fold
            has_been_test_or_val = any(test_or_val[j][i] for j in range(n))
            if has_been_test_or_val:
                folds[k]["train"].extend(mol_indices)
            else:
                if len(mol_indices) + len(folds[k]["val"]) <= val_cutoff:
                    folds[k]["val"].extend(mol_indices)
                    test_or_val[k][i] = True
                elif len(mol_indices) + len(folds[k]["test"]) <= test_cutoff:
                    folds[k]["test"].extend(mol_indices)
                    test_or_val[k][i] = True
                else:
                    # This scaffold has too many molecules for test/validation set
                    # and needs to be assigned to the training set.
                    folds[k]["train"].extend(mol_indices)

        if len(folds[k]["test"]) < test_cutoff or len(folds[k]["val"]) < val_cutoff:
            raise SplittingError(
                "Insufficient number of scaffolds to assign into val/test set without overlap."
            )

    assert_no_overlap(folds)
    return folds


def datasail_split(smiles: List[str], n: int) -> List[DatasetSplit]:
    """Splits the given list of smiles strings using datasail into n disjoint splits

    :param smiles: List of valid molecules as SMILES strings
    :param n: Number of splits to generate
    :return: List of Dictionaries with `train`, `test` and `val` keys, each containing the indices
        of the respective molecules for this split
    """
    from datasail.sail import datasail

    # This yields a dictionary with idx: split assignment of each molecule to
    # one of the 10 splits.
    idx_to_split, _, _ = datasail(
        techniques=["C1e"],
        splits=[1] * n,
        names=[str(split) for split in range(n)],
        runs=1,
        solver="SCIP",
        e_data=dict(enumerate(smiles)),
        e_type="M",
        epsilon=0.2,
    )

    if not idx_to_split:
        raise SplittingError(f"Splitting with {n=} failed.")

    idx_to_split: Dict[int, str] = idx_to_split["C1e"][0]  # Only 1 run
    # Splits is a dictionary of the form {idx: split | idx in range(len(smiles))}
    # Working with {split_1: [idx_1, idx_2, ...], split_2: [...] } is more intuitive
    split_to_idx: Dict[str, List[int]] = defaultdict(list)
    for idx, split in idx_to_split.items():
        assert split.isdigit()
        split_to_idx[int(split)].append(idx)

    folds = []
    for split in range(n):
        # We take the ith split as validation set and the i+1th split as test set
        val_split = split
        test_split = (split + 1) % n

        folds.append(
            {
                "train": list(
                    chain.from_iterable(
                        v
                        for k, v in split_to_idx.items()
                        if k != val_split and k != test_split
                    )
                ),
                "val": split_to_idx[val_split],
                "test": split_to_idx[test_split],
            }
        )
    return folds


def random_split(smiles: List[str], n: int, seed: int) -> List[DatasetSplit]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(smiles))
    rng.shuffle(indices)
    splits = np.array_split(indices, n)

    folds = []
    for split in range(n):
        val_split = split
        test_split = (split + 1) % n

        folds.append(
            {
                "train": np.concatenate(
                    [splits[k] for k in range(n) if k != val_split and k != test_split]
                ).tolist(),
                "val": splits[val_split].tolist(),
                "test": splits[test_split].tolist(),
            }
        )

    return folds
