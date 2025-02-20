import argparse
import pathlib

from da4mt.types import VALID_ADAPT_METHODS


def existing_directory(arg):
    path = pathlib.Path(arg)
    if not path.exists() or not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"{path} does not exist or is not a valid directory."
        )
    return path.absolute()


def add_prepare_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "file", type=pathlib.Path, help="CSV files with smiles column and targets"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Directory where the preprocessed data will be saved",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for stochastic data generation."
    )

    return parser


def add_prepare_pretraining_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Directory where the preprocessed data will be saved",
    )

    return parser


def add_prepare_splits_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "file", type=pathlib.Path, help="CSV files with smiles column and targets"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Directory where the preprocessed data will be saved",
    )

    parser.add_argument(
        "--splitter",
        nargs="+",
        choices=["scaffold", "random", "datasail"],
        required=True,
    )

    parser.add_argument(
        "--num-splits", "-n", nargs="+", type=int, help="Number of k-fold splits"
    )

    parser.add_argument(
        "--seed", default=0, type=int, help="Seed for random splitting."
    )

    return parser


def add_prepare_parser(parser: argparse.ArgumentParser):
    from da4mt.prepare.dataset import make_data
    from da4mt.prepare.pretraining import make_pretraining
    from da4mt.prepare.splits import make_splits

    subparsers = parser.add_subparsers(title="kind")
    dataset_parser = subparsers.add_parser("dataset", help="Run dataset preprocessing.")
    dataset_parser = add_prepare_dataset_args(dataset_parser)
    dataset_parser.set_defaults(func=make_data)

    splits_parser = subparsers.add_parser("splits", help="Split the dataset")
    splits_parser = add_prepare_splits_args(splits_parser)
    splits_parser.set_defaults(func=make_splits)

    pretraining_parser = subparsers.add_parser(
        "pretraining", help="Run pretraining preprocessing."
    )
    pretraining_parser = add_prepare_pretraining_args(pretraining_parser)
    pretraining_parser.set_defaults(func=make_pretraining)

    return parser


def add_pretrain_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "data_dir",
        help="Directory containing the output of the data preparation script.",
        type=existing_directory,
    )
    parser.add_argument(
        "model_dir",
        help="Directory where the trained models/tokenizers should be saved.",
        type=existing_directory,
    )
    parser.add_argument(
        "--model-config",
        help="Configuration file of the BERT model in JSON format.",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "models" / "bert_config.json",
    )
    parser.add_argument(
        "--train-size",
        default=1,
        help="Proportion of the training set to use. Defaults to 100%%",
        type=float,
    )
    # MLM specifics
    parser.add_argument(
        "--mlm-train-file",
        default="guacamol_v1_train.smiles",
        help="MLL/Tokenizer training file. By default the guacamol training set is used. If"
        "only a file is given, the full path is inferred as `data_dir/mlm_train_file`",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--mlm-val-file",
        default="guacamol_v1_valid.smiles",
        help="MLM validation file. See --mlm-train-file for more help",
        type=pathlib.Path,
    )

    # MTR specifics
    parser.add_argument(
        "--mtr-train-file",
        default="guacamol_v1_train_mtr.jsonl",
        help="MTR training file.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--mtr-val-file",
        default="guacamol_v1_valid_mtr.jsonl",
        help="MTR validation file.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--mtr-norm-file",
        default="guacamol_v1_normalization_values.json",
        help="MTR normalization file. Should contain the mean and std of all labels.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--mtr-subset",
        choices=["all", "surface"],
        default="all",
        help="Subset of properties to include as labels.",
    )

    parser.add_argument(
        "--selection",
        choices=["random", "cluster"],
        default="random",
        help="How to select the molecules in case --train-size is less then 100%.",
    )

    parser.add_argument(
        "--cluster-file",
        type=pathlib.Path,
        default=None,
        help="Path to a file containing the cluster assignment. Should be a JSON file containing"
        "just a 2D array with indices of the molecules for each cluster.",
    )

    parser.add_argument(
        "--train-tokenizer", action="store_true", default=False, help="Train tokenizer."
    )
    parser.add_argument(
        "--train-mlm",
        action="store_true",
        default=False,
        help="Pretrain with mlm objective.",
    )
    parser.add_argument(
        "--train-mtr",
        action="store_true",
        default=False,
        help="Pretrain with mtr objective.",
    )
    return parser


def add_adapt_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "model", help="Pre-trained model to be adapted.", type=pathlib.Path
    )

    parser.add_argument(
        "train_file", help="Path to the training file", type=pathlib.Path
    )

    parser.add_argument(
        "output", help="Output directory of the adapted model", type=pathlib.Path
    )
    parser.add_argument("--method", choices=VALID_ADAPT_METHODS, default="mlm")

    parser.add_argument(
        "--normalization-file",
        help="Path to the normalization values for MTR.",
        type=pathlib.Path,
    )

    return parser


def add_embed_args(parser: argparse.ArgumentParser):
    parser.add_argument("train_file", type=pathlib.Path)

    parser.add_argument(
        "adapteddir",
        help="Directory containing the domain adapted models.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "pretraindir",
        help="Directory containing the pretrained models.",
        type=pathlib.Path,
    )

    parser.add_argument(
        "--outdir", required=True, help="Output directory.", type=pathlib.Path
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument("--seed", type=int, default=0)
    return parser


def add_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hdf5-file",
        type=pathlib.Path,
        required=True,
        help="Path to HDF5 file containing embeddings",
    )
    parser.add_argument(
        "--splits",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="Paths to JSON files containing train/val/test splits",
    )
    parser.add_argument(
        "--targets",
        type=pathlib.Path,
        required=True,
        help="Path to CSV file containing target values",
    )

    parser.add_argument(
        "--model",
        choices=["linear", "random-forest", "svm"],
        nargs="+",
        help="Model to train. If --task=classification and --model=linear a logistic regression model is used otherwise"
        "a linear regression model. ",
    )

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Task of the dataset",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Output directory where the resulting HDF5 file will be stored",
    )

    parser.add_argument(
        "--keep-val-separate",
        required=False,
        action="store_true",
        default=False,
        help="Turns of merging validation set into training set.",
    )

    parser.add_argument(
        "--overwrite-existing",
        required=False,
        default=False,
        action="store_true",
        help="If set, overwrites existing hdf5 results file in the --output-dir by replacing it with the embedding file and running "
        "finetuning from scratch. If not set appends to the existing file, overwriting any results for models that already existed "
        "when the file was created.",
    )

    return parser


def add_finetune_args(parser: argparse.ArgumentParser):
    from da4mt.finetune.embed import embed_cli_wrapper
    from da4mt.finetune.eval import eval

    subparsers = parser.add_subparsers(title="step")
    embed_parser = subparsers.add_parser(
        "embed", help="Create and save embeddings for this dataset."
    )
    embed_parser = add_embed_args(embed_parser)
    embed_parser.set_defaults(func=embed_cli_wrapper)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate the embeddings of this dataset. Requires running 'finetune embed' first",
    )
    eval_parser = add_eval_args(eval_parser)
    eval_parser.set_defaults(func=eval)

    return parser
