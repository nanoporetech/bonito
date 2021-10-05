#!/usr/bin/env python3

"""
Bonito training.
"""

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

from bonito.util import __models__, default_config, default_data
from bonito.util import load_data, load_model, load_symbol, init, half_supported
from bonito.training import load_state, Trainer

import toml
import torch
import numpy as np


def main(args):

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    init(args.seed, args.device)
    device = torch.device(args.device)

    print("[loading data]")
    if args.directory is not None:
        train_loader, valid_loader = load_numpy(
            args.chunks, args.directory, args.batch
        )
    elif args.dataset_config is not None:
        train_loader, valid_loader = load_bonito_datasets(
            args.dataset_config, args.seed, args.batch, args.chunks, args.valid_chunks
        )
    else:
        raise ValueError(f"failed to load data")

    if args.pretrained:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        config_file = os.path.join(dirname, 'config.toml')
    else:
        config_file = args.config

    config = toml.load(config_file)

    argsdict = dict(training=vars(args))

    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    print("[loading model]")
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False)
    else:
        model = load_symbol(config, 'Model')(config)

    last_epoch = load_state(workdir, args.device, model)

    if args.multi_gpu:
        from torch.nn import DataParallel
        model = DataParallel(model)
        model.decode = model.module.decode
        model.alphabet = model.module.alphabet

    trainer = Trainer(model, device, train_loader, valid_loader, use_amp=half_supported() and not args.no_amp)
    trainer.fit(workdir, args.epochs, args.lr, last_epoch=last_epoch)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default="")

    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--directory", type=Path)
    data.add_argument("--dataset-config", type=Path)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid_chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--multi-gpu", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    return parser
