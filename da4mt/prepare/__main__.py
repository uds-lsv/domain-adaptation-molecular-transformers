import argparse
from da4mt.cli import add_prepare_parser

parser = argparse.ArgumentParser()
parser = add_prepare_parser(parser)

args = parser.parse_args()
args.func(args)
