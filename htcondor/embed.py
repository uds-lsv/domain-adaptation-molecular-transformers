#!/usr/bin/env python3
"""
Embed the given dataset using all untrained, pretrained models as well as all models that have been domain adapted with this dataset
"""

import os
import sys
import argparse

from setup import PATHS, DATASETS, run_command


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")
    parser.add_argument("dataset", help="Dataset name", choices=DATASETS)
    args = parser.parse_args()

    print(f"Arguments: {args}")

    train_file = PATHS.DATA_DIR / f"{args.dataset}.csv"

    # Check if files exist
    if not os.path.isfile(train_file):
        print(f"Error: Train file {train_file} does not exist.", file=sys.stderr)
        sys.exit(1)

    outdir = PATHS.EMBED_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "da4mt",
        "finetune",
        "embed",
        str(train_file),
        str(PATHS.ADAPT_DIR),
        str(PATHS.PRETRAIN_DIR),
        "--outdir",
        str(outdir),
    ]
    # Execute the command
    run_command(cmd)


if __name__ == "__main__":
    main()
