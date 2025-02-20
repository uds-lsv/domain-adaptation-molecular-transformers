import argparse

from da4mt.cli import (
    add_prepare_parser,
    add_pretrain_args,
    add_adapt_args,
    add_finetune_args,
)
from da4mt.pretrain.__main__ import run_pretraining
from da4mt.adapt.__main__ import run_domain_adaptation

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title="Command")
prepare_parser = subparsers.add_parser("prepare", help="Run data preparation.")
prepare_parser = add_prepare_parser(prepare_parser)

pretrain_parser = subparsers.add_parser("pretrain", help="Pretrain models.")
pretrain_parser = add_pretrain_args(pretrain_parser)
pretrain_parser.set_defaults(func=run_pretraining)

adaptation_parser = subparsers.add_parser("adapt", help="Run domain adaptation.")
adaptation_parser = add_adapt_args(adaptation_parser)
adaptation_parser.set_defaults(func=run_domain_adaptation)

finetune_parser = subparsers.add_parser(
    "finetune", help="Embedding & evaluation pipeline"
)
finetune_parser = add_finetune_args(finetune_parser)

args = parser.parse_args()
args.func(args)
