import argparse
import logging
import os
import pathlib
import sys

from da4mt.training import train_contrastive, adapt_mlm, adapt_mtr
from da4mt.cli import add_adapt_args
from da4mt.utils import get_adapt_training_args


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_adapt_args(parser)
    args = parser.parse_args()

    # Add model specific directory
    # if args.output.is_dir():
    #     model_name = args.model.name
    #     data_set_name = args.dataset.stem.split("_")[0]
    #     args.output = args.output / f"{model_name}-{args.method}-{data_set_name}"
    #
    #     if not args.output.exists():
    #         args.output.mkdir()
    # else:
    #     raise argparse.ArgumentTypeError("'output' is supposed to be a directory")

    return args


def get_logger():
    logger = logging.getLogger("eamt.adapt")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def run_domain_adaptation(args):
    logger = get_logger()

    dataset_name = args.train_file.stem

    if args.method in dataset_name:
        # With mtr, cbert and sbert the dataset_name will be {name}_{method}
        output_path: pathlib.Path = args.output / f"{dataset_name}_{args.model.stem}"
    else:
        output_path: pathlib.Path = (
            args.output / f"{dataset_name}_{args.method}_{args.model.stem}"
        )
    if output_path.exists() and any(
        output_path.iterdir()
    ):  # Check if directory exists and is not empty
        logger.warning(
            f"Model directory '{output_path}' already exists assuming this model has already been adapted"
        )
        sys.exit(0)
    else:
        if not output_path.exists():
            output_path.mkdir()

    logger.info(f"Adapting model: {args.model}")
    os.environ["WANDB_RUN_GROUP"] = output_path.stem
    if args.method == "mlm":
        adapt_mlm(
            model_path=args.model,
            output_path=output_path,
            train_fp=args.train_file,
            training_args=get_adapt_training_args(str(output_path)),
            logger=logger,
            save_model=True,
        )
    elif args.method == "mtr":
        if args.normalization_file is None:
            raise ValueError("--normalization-file must be specified.")

        adapt_mtr(
            model_path=args.model,
            output_path=output_path,
            train_fp=args.train_file,
            normalization_fp=args.normalization_file,
            training_args=get_adapt_training_args(str(output_path)),
            logger=logger,
            save_model=True,
        )
    elif args.method in ("sbert", "cbert"):
        train_contrastive(
            model_path=args.model,
            output_path=output_path,
            train_fp=args.train_file,
            training_args=get_adapt_training_args(str(output_path)),
            method=args.method,
            logger=logger,
            save_model=True,
        )
    else:
        raise ValueError(f"Invalid method {args.method}")


if __name__ == "__main__":
    run_domain_adaptation(get_args())
