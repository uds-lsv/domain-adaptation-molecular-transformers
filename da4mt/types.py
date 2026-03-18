"""Type aliases and constants for pre-training and adaptation methods and datasets."""
import typing

PretrainingMethod = typing.Literal["mlm", "mtr"]
AdaptMethod = typing.Literal["mlm", "mtr", "sbert", "cbert"]
DownstreamDataset = typing.Literal[
    "bace",
    "bbbp",
    "clintox",
    "hiv",
    "sider",
    "toxcast",
    "esol",
    "lipop",
    "freesolv",
]
VALID_PRETRAINING_METHOD = typing.get_args(PretrainingMethod)
VALID_ADAPT_METHODS = typing.get_args(AdaptMethod)
VALID_DATASET = typing.get_args(DownstreamDataset)
