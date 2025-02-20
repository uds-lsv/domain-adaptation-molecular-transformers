import argparse

from da4mt.cli import add_finetune_args

parser = argparse.ArgumentParser()
parser = add_finetune_args(parser)
args = parser.parse_args()
args.func(args)
