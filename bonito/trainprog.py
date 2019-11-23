#!/usr/bin/env python3

"""
Bonito training.
"""

import argparse
import os
from datetime import datetime

from bonito.model import Model
from bonito.util import load_data, init
from bonito.train import ReadDataSet, train, test

import toml
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

try: from apex import amp
except ImportError: pass


def main(args):

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("* error: %s exists." % workdir)
        exit(1)

    init(args.seed, args.device)
    device = torch.device(args.device)

    chunks, targets, target_lengths = load_data(limit=args.chunks, shuffle=True)

    split = np.floor(chunks.shape[0] * args.validation_split).astype(np.int32)
    train_dataset = ReadDataSet(chunks[:split], targets[:split], target_lengths[:split])
    test_dataset = ReadDataSet(chunks[split:], targets[split:], target_lengths[split:])
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=4, pin_memory=True)

    config = toml.load(args.config)
    argsdict = dict(training=vars(args))

    model = Model(config)

    weights = os.path.join(workdir, 'weights.tar')
    if os.path.exists(weights): model.load_state_dict(torch.load(weights))

    model.to(device)
    model.train()

    os.makedirs(workdir, exist_ok=True)
    torch.save(model, os.path.join(workdir, 'model.py'))

    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    optimizer = AdamW(model.parameters(), amsgrad=True, lr=args.lr)

    if args.amp:
        try:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        except NameError:
            print("* error: Cannot use AMP: Apex package needs to be installed manually, See https://github.com/NVIDIA/apex")
            exit(1)

    schedular = CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    log_interval = np.floor(len(train_dataset) / args.batch * 0.10)

    for epoch in range(1, args.epochs + 1):

        print("[Epoch %s]:" % epoch, workdir.split('/')[-1])
        train_loss, train_time = train(log_interval, model, device, train_loader, optimizer, epoch, use_amp=args.amp)
        test_loss, mean, median = test(model, device, test_loader)

        torch.save(model.state_dict(), os.path.join(workdir, "weights_%s.tar" % epoch))

        # TODO: make this a csv
        with open(os.path.join(workdir, 'training.log'), 'a') as logfile:
            now = datetime.today()
            logfile.write("%s Train Epoch %s: Loss %.2f - %.2f seconds\n" % (now, epoch, train_loss, train_time))
            logfile.write("%s Validation Loss %.2f\n" % (now, test_loss))
            logfile.write("%s Inference Accuracy %.2f %.3f \n\n" % (now, mean, median))

        if schedular: schedular.step()


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("training_directory")
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--chunks", default=1000000, type=int)
    parser.add_argument("--validation_split", default=0.99, type=float)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    return parser
