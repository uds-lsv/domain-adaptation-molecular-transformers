#!/usr/bin/env python3

import os
from pathlib import Path
import sys
from typing import List, Tuple, Callable
from collections import namedtuple

import pandas as pd

# Add the da4mt module to the PATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR / "da4mt"))
sys.path.append(os.environ["PROJECT_ROOT"])
sys.path.append("/opt/conda/lib/python3.10/site-packages")  # Required to find datasail

from da4mt.splitting import datasail_split, k_fold_scaffold_split, SplittingError

NAMES = [
    'bace', 'bbbp', 'clintox', 'sider', 'toxcast', 'esol', 'lipop', 'freesolv',
    'hiv', 'adme_microsom_stab_h', 'adme_microsom_stab_r', 'adme_permeability',
    'adme_ppb_h', 'adme_ppb_r', 'adme_solubility'
]
DATA_DIR = ROOT_DIR / "data"
DATASETS = [f for f in DATA_DIR.glob("*.csv") if f.stem in NAMES]

Result = namedtuple('Result', ['dataset', 'splitter', 'n_splits', 'status'])

def process_dataset(dataset: Path, splitter: Callable, name: str, k: int) -> Result:
    """
    Process a single dataset with a given splitter and number of splits.

    :param Path dataset: Path to the dataset file
    :param Callable splitter: Function to split the data
    :param str name: Name of the splitter
    :param int k: Number of splits
    :return: A Result namedtuple containing the processing results
    :rtype: Result
    """
    df = pd.read_csv(dataset)
    try:
        splitter(df["smiles"].tolist(), k)
        status = "Success"
    except SplittingError:
        status = "Failure"
    
    return Result(dataset.stem, name, k, status)

def main() -> None:
    """
    Main function to process datasets and save results.
    """
    splitters: List[Tuple[Callable, str]] = [
        (k_fold_scaffold_split, "scaffold"),
        (datasail_split, "datasail"),
    ]
    k_values: List[int] = [3, 5]

    with open(ROOT_DIR / "splitting_results.csv", "w") as f:
        for splitter, name in splitters:
            for dataset in DATASETS:
                for k in k_values:
                    r = process_dataset(dataset, splitter, name, k)
                    print(r)
                    f.write(f"{r.dataset}, {r.splitter}, {r.n_splits}, {r.status}\n")

if __name__ == "__main__":
    main()