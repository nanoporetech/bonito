"""
Bonito model viewer - display a model architecture for a given config.
"""

import os.path

import toml
import argparse
from bonito.util import load_symbol


def main(args):

    if os.path.isdir(args.config):
        config = toml.load(os.path.join(args.config, "config.toml"))
    else:
        config = toml.load(args.config)
    Model = load_symbol(config, "Model")
    model = Model(config)
    print(model)
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config")
    return parser
