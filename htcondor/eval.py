#!/usr/bin/env python3
"""
Evaluate the given dataset on existing embeddings
"""

import os
import pathlib
import sys
import argparse
import warnings

from setup import PATHS, DATASETS, run_command



def run_eval(embedding_file, train_file, task, outdir, splits_files):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "da4mt",
        "finetune",
        "eval",
        "--hdf5-file",
        str(embedding_file),
        "--target",
        str(train_file),
        "--model",
        "linear",
        "random-forest",
        "svm",
        "--task",
        task,
        "--output-dir",
        str(outdir),
        "--splits",
    ] + splits_files

    # Execute the command
    run_command(cmd)



def run_eval_on_comparison_embeddings(dataset, train_file, task, splits_files, splitter):
    root = pathlib.Path(__file__).parent.parent
    compare_dir = root / "comparison"
    compare_results_dir = compare_dir / "results" / "results_with_clustered_pretraining_cleaned_adme_microsom"
    compare_embeddings_dir = compare_dir / "embeddings"

    outdir = compare_results_dir / splitter

    embedding_file = compare_embeddings_dir / f"{dataset}_comparison_embeddings.hdf5"

    run_eval(embedding_file, train_file, task, outdir=outdir, splits_files=splits_files)



def run_eval_on_our_models(embedding_file, train_file, task, splits_files, splitter):
    outdir = PATHS.RESULT_DIR / splitter
    run_eval(embedding_file, train_file, task, outdir=outdir, splits_files=splits_files)


def run_eval_end2end(train_file, task, pretrain_filter, splits_files, splitter, targets=None):
    outdir = PATHS.RESULT_DIR / splitter
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = (
        [
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
        ]
        + splits_files
    )

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

    train_file = PATHS.DATA_DIR / f"{args.dataset}.csv"

    # Check if files exist
    if not os.path.isfile(train_file):
        print(f"Error: Train file {train_file} does not exist.", file=sys.stderr)
        sys.exit(1)

    for splitter in ["datasail", "scaffold", "random"]:


        # ADME microsom contains censored value that we remove
        # See notebooks/adme_microsom_postprocess_splits.ipynb
        if args.dataset.startswith("adme_microsom_stab_"):
            splits_pattern = f"{args.dataset}_cleaned_*.{splitter}_splits.json"
        else:
            splits_pattern = f"{args.dataset}_*.{splitter}_splits.json"
        splits_files = list(map(str, PATHS.DATA_DIR.glob(splits_pattern)))

        if not splits_files:
            warnings.warn(f"Error: No split files found matching {splits_pattern}")
            continue

        run_eval_on_our_models(PATHS.EMBED_DIR / f"{args.dataset}_embeddings.hdf5", train_file, args.task, splits_files, splitter)
        # run_eval_on_comparison_embeddings(args.dataset, train_file, args.task, splits_files, splitter)
        # run_eval_end2end(train_file, args.task, splits_files, splitter, args.targets)


if __name__ == "__main__":
    main()
