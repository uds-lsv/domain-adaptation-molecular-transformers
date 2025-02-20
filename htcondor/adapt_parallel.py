#!/usr/bin/env python3
"""
Adapt all pretrained models on the given dataset
"""

import argparse
from typing import List

from setup import PATHS, ADME_DATASETS, run_command

METHODS = ["mtr", "mlm", "sbert", "cbert"]


def get_mlm_command(model: str, dataset: str) -> List[str]:
    train_file = PATHS.DATA_DIR / f"{dataset}.csv"
    return [
        "python",
        "-m",
        "da4mt",
        "adapt",
        model,
        str(train_file),
        str(PATHS.ADAPT_DIR),
        "--method",
        "mlm",
    ]


def get_mtr_command(model: str, dataset: str) -> List[str]:
    train_file = PATHS.DATA_DIR / f"{dataset}_mtr.jsonl"
    normalization_file = PATHS.DATA_DIR / f"{dataset}_normalization_values.json"
    return [
        "python",
        "-m",
        "da4mt",
        "adapt",
        model,
        str(train_file),
        str(PATHS.ADAPT_DIR),
        "--normalization-file",
        str(normalization_file),
        "--method",
        "mtr",
    ]


def get_other_command(model: str, dataset: str, method: str) -> List[str]:
    train_file = PATHS.DATA_DIR / f"{dataset}_{method}.csv"
    return [
        "python",
        "-m",
        "da4mt",
        "adapt",
        model,
        str(train_file),
        str(PATHS.ADAPT_DIR),
        "--method",
        method,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        help=f"Directory of the pretrained model. Must be inside {PATHS.PRETRAIN_DIR}",
    )
    args = parser.parse_args()

    print(f"Running domain adaptation for {args.model_dir}")

    for dataset in ADME_DATASETS:
        for method in METHODS:

            # We adapt the mtr pretrained model only with mtr
            if args.model_dir == "mtr-bert-30" and method != "mtr":
                continue

            model = PATHS.PRETRAIN_DIR / args.model_dir
            assert model.exists()
            model = str(model)

            if method == "mlm":
                cmd = get_mlm_command(model, dataset)
            elif method == "mtr":
                cmd = get_mtr_command(model, dataset)
            else:
                # sbert/cbert
                cmd = get_other_command(model, dataset, method)

            run_command(cmd)


if __name__ == "__main__":
    main()
