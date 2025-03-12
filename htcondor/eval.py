#!/usr/bin/env python3
"""
Evaluate the given dataset on existing embeddings
"""

import argparse
import pathlib
import subprocess
import sys

from setup import PATHS, DATASETS, run_command


def run_eval(embedding_file, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "da4mt",
        "finetune",
        "eval",
        "--embedding-file",
        str(embedding_file),
        "--output-dir",
        str(outdir),
    ]

    # Execute the command
    run_command(cmd)


def run_eval_on_comparison_embeddings(dataset):
    root = pathlib.Path(__file__).parent.parent
    compare_dir = root / "comparison"
    compare_results_dir = (
        compare_dir
        / "results"
        / "results_with_clustered_pretraining_cleaned_adme_microsom"
    )
    compare_embeddings_dir = compare_dir / "embeddings"

    outdir = compare_results_dir / "cv"

    embedding_file = compare_embeddings_dir / f"{dataset}_comparison_embeddings.hdf5"

    run_eval(embedding_file, outdir=outdir)


def run_eval_end2end(
    train_file, task, pretrain_filter, splits_files, splitter, targets=None
):
    outdir = PATHS.RESULT_DIR / splitter
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "da4mt",
        "finetune",
        str(train_file),
        str(PATHS.ADAPT_DIR),
        str(PATHS.PRETRAIN_DIR),
        "--outdir",
        str(outdir),
        "--task",
        task,
        "--pretrain-filter",
        pretrain_filter,
        "--splits-files",
    ] + splits_files

    # Add targets to the command if they were provided
    if targets:
        cmd.extend(["--targets"] + targets)

    # Execute the command
    run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")
    parser.add_argument("dataset", help="Dataset name", choices=DATASETS)
    parser.add_argument(
        "task", help="Task name", choices=["regression", "classification"]
    )
    parser.add_argument("targets", nargs="*", help="Target column names")
    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Better than updating the docker image...
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--ignore-python-version",
            "useful_rdkit_utils",
        ]
    )

    embedding_file = PATHS.EMBED_DIR / f"{args.dataset}_embeddings.hdf5"
    outdir = PATHS.RESULT_DIR / "cv"
    outdir.mkdir(parents=True, exist_ok=True)
    run_eval(embedding_file, outdir=outdir)
    run_eval_on_comparison_embeddings(args.dataset)


if __name__ == "__main__":
    main()
