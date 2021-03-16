#!/usr/bin/env python3

"""
Bonito tuning.

  $ export CUDA_VISIBLE_DEVICES=0
  $ bonito tune /data/models/bonito-tune

"""

import os
import csv
from datetime import datetime
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

from bonito.training import ChunkDataSet, train, test
from bonito.training import func_scheduler, cosine_decay_schedule
from bonito.util import load_data, load_symbol, init, default_config

import toml
import torch
import numpy as np
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import SuccessiveHalvingPruner


def main(args):

    workdir = os.path.expanduser(args.tuning_directory)

    if os.path.exists(workdir) and not args.force:
        print("* error: %s exists." % workdir)
        exit(1)

    os.makedirs(workdir, exist_ok=True)

    init(args.seed, args.device)
    device = torch.device(args.device)

    print("[loading data]")
    train_data = load_data(limit=args.chunks, directory=args.directory)
    if os.path.exists(os.path.join(args.directory, 'validation')):
        valid_data = load_data(directory=os.path.join(args.directory, 'validation'), limit=10000)
    else:
        print("[validation set not found: splitting training set]")
        split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader = DataLoader(
        ChunkDataSet(*train_data), batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        ChunkDataSet(*valid_data), batch_size=args.batch, num_workers=4, pin_memory=True
    )

    def objective(trial):

        config = toml.load(args.config)

        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

        model = load_symbol(config, 'Model')(config)

        num_params = sum(p.numel() for p in model.parameters())

        print("[trial %s]" % trial.number)

        model.to(args.device)
        model.train()

        os.makedirs(workdir, exist_ok=True)

        scaler = GradScaler(enabled=True)
        optimizer = AdamW(model.parameters(), amsgrad=False, lr=lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

        if hasattr(model, 'seqdist'):
            criterion = model.seqdist.ctc_loss
        else:
            criterion = None

        lr_scheduler = func_scheduler(
            optimizer,
            cosine_decay_schedule(1.0, decay),
            args.epochs * len(train_loader),
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
        )

        for epoch in range(1, args.epochs + 1):

            try:
                train_loss, duration = train(
                    model, device, train_loader, optimizer, scaler=scaler, use_amp=True, criterion=criterion
                )
                val_loss, val_mean, val_median = test(model, device, test_loader, criterion=criterion)
                print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                    epoch, workdir, val_loss, val_mean, val_median
                ))
            except KeyboardInterrupt: exit()
            except Exception as e:
                print("[pruned] exception")
                raise optuna.exceptions.TrialPruned()

            if np.isnan(val_loss): val_loss = 9.9
            trial.report(val_loss, epoch)

            if trial.should_prune():
                print("[pruned] unpromising")
                raise optuna.exceptions.TrialPruned()

        trial.set_user_attr('val_loss', val_loss)
        trial.set_user_attr('val_mean', val_mean)
        trial.set_user_attr('val_median', val_median)
        trial.set_user_attr('train_loss', train_loss)
        trial.set_user_attr('model_params', num_params)

        torch.save(model.state_dict(), os.path.join(workdir, "weights_%s.tar" % trial.number))
        toml.dump(config, open(os.path.join(workdir, 'config_%s.toml' % trial.number), 'w'))

        print("[loss] %.4f" % val_loss)
        return val_loss

    print("[starting study]")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        storage='sqlite:///%s' % os.path.join(workdir, 'tune.db'),
        study_name='bonito-study',
        load_if_exists=True,
        pruner=SuccessiveHalvingPruner()
    )

    study.optimize(objective, n_trials=args.trials)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("tuning_directory")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--directory", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--trials", default=100, type=int)
    parser.add_argument("--chunks", default=250000, type=int)
    parser.add_argument("--max-params", default=7000000, type=int)
    parser.add_argument("--validation_split", default=0.90, type=float)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    return parser
