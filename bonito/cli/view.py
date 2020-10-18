"""
Bonito model viewer - display a model architecture for a given config.
"""
import argparse
import toml
from bonito.model import Model


def main(args):
    model = Model(toml.load(args.config))
    print(model)
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("config")
    return parser
