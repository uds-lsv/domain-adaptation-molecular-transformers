import json
import logging
import sys

import pandas as pd

from da4mt.splitting import datasail_split, k_fold_scaffold_split, random_split


def get_logger():
    logger = logging.getLogger("eamt.perpare.splits")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def make_splits(args):
    if len(args.splitter) != len(args.num_splits):
        raise ValueError(
            "Size mismatch between --splitter and --num-splits. Please provide the number of splits for each splitter."
        )

    logger = get_logger()
    filename = args.file.stem

    logger.info(f"Loading {args.file} as CSV.")
    df = pd.read_csv(args.file)

    if "smiles" not in df.columns:
        raise ValueError("CSV file must contain 'smiles' column.")

    # Splits dataset
    for splitter, n in zip(args.splitter, args.num_splits):
        if splitter == "datasail":
            splits = datasail_split(df["smiles"].to_list(), n)
        elif splitter == "scaffold":
            splits = k_fold_scaffold_split(df["smiles"].to_list(), n)
        elif splitter == "random":
            splits = random_split(df["smiles"].to_list(), n, args.seed)

        logger.info("Successfully created splits.")

        for i, split in enumerate(splits):
            with open(
                args.output_dir / f"{filename}_{i}.{splitter}_splits.json", "w"
            ) as f:
                json.dump(split, f, indent=2)
