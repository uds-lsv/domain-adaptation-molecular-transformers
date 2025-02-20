import argparse
import logging
import pathlib
import sys

from transformers import BertConfig

from da4mt.cli import add_pretrain_args
from da4mt.utils import get_pretraining_args
from da4mt.training import pretrain_mlm, pretrain_mtr, train_tokenizer


def add_data_dir_to_file(args, data_dir: pathlib.Path, arg_name: str):
    """
    If `arg_name` is only a file, appends data_dir in front.
    """
    file = getattr(args, arg_name)
    if len(file.parts) == 1:
        setattr(args, arg_name, data_dir / file)


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_pretrain_args(parser)
    args = parser.parse_args()

    return args


def get_logger():
    logger = logging.getLogger("eamt.pretrain")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def run_pretraining(args):
    logger = get_logger()

    # If file is not a full path we assume that the file is
    # inside the data dir
    # default to guacamol dataset, see da4mt/cli.py:add_pretrain_args()
    for file in [
        "mlm_train_file",
        "mlm_val_file",
        "mtr_train_file",
        "mtr_val_file",
        "mtr_norm_file",
    ]:
        add_data_dir_to_file(args, args.data_dir, file)

    if args.train_tokenizer:
        train_tokenizer(
            dataset_path=str(args.mlm_train_file),
            output_path=str(args.model_dir),
            name="tokenizer",
            logger=logger,
        )

    model_config = BertConfig.from_json_file(args.model_config)
    model_config.train_size = args.train_size
    model_config.train_type = "mlm" if args.train_mlm else "mtr"

    if args.train_mlm:
        # e.g. mlm-bert-100 for 100%
        model_dir = str(args.model_dir / f"mlm-bert-{int(args.train_size * 100)}")

        pretrain_mlm(
            model_dir=model_dir,
            model_config=model_config,
            training_args=get_pretraining_args(model_dir),
            tokenizer_path=str(args.model_dir / "tokenizer"),
            train_file=str(args.mlm_train_file),
            validation_file=str(args.mlm_val_file),
            train_size=args.train_size,
            logger=logger,
            selection=args.selection,
            cluster_file=args.cluster_file,
        )

    if args.train_mtr:
        model_dir = str(
            args.model_dir / f"mtr_{args.mtr_subset}-bert-{int(args.train_size * 100)}"
        )
        pretrain_mtr(
            model_dir=model_dir,
            model_config=model_config,
            training_args=get_pretraining_args(model_dir),
            tokenizer_path=str(args.model_dir / "tokenizer"),
            train_file=str(args.mtr_train_file),
            validation_file=str(args.mtr_val_file),
            normalization_file=str(args.mtr_norm_file),
            train_size=args.train_size,
            property_subset=args.mtr_subset,
            logger=logger,
            selection=args.selection,
            cluster_file=args.cluster_file,
        )


if __name__ == "__main__":
    run_pretraining(get_args())
