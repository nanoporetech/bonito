#!/usr/bin/env python3

"""
Bonito tuning.

  $ export CUDA_VISIBLE_DEVICES=0
  $ bonito tune /data/models/bonito-tune config/quartznet5x5.toml

"""

import os
import csv
from datetime import datetime
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

from bonito.model import Model
from bonito.util import load_data, init
from bonito.training import ChunkDataSet, train, test

import toml
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

try: from apex import amp
except ImportError: pass

import optuna

from optuna.pruners import HyperbandPruner


def main(args):

    workdir = os.path.expanduser(args.tuning_directory)

    if os.path.exists(workdir) and not args.force:
        print("* error: %s exists." % workdir)
        exit(1)

    os.makedirs(workdir, exist_ok=True)

    init(args.seed, args.device)
    device = torch.device(args.device)

    print("[loading data]")
    chunks, chunk_lengths, targets, target_lengths = load_data(limit=args.chunks, shuffle=True, directory=args.directory)
    split = np.floor(chunks.shape[0] * args.validation_split).astype(np.int32)
    train_dataset = ChunkDataSet(chunks[:split], chunk_lengths[:split], targets[:split], target_lengths[:split])
    test_dataset = ChunkDataSet(chunks[split:], chunk_lengths[split:], targets[split:], target_lengths[split:])
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=4, pin_memory=True)

    def objective(trial):

        config = toml.load(args.config)

        lr = 1e-3
        #lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        config['encoder']['activation'] = 'gelu'
        #config['block'][0]['stride'] = [trial.suggest_int('stride', 4, 6)]

        # C1
        config['block'][0]['kernel'] = [int(trial.suggest_discrete_uniform('c1_kernel', 1, 129, 2))]
        config['block'][0]['filters'] = trial.suggest_int('c1_filters', 1, 1024)

        # B1 - B5
        for i in range(1, 6):
            config['block'][i]['repeat'] = trial.suggest_int('b%s_repeat' % i, 1, 9)
            config['block'][i]['filters'] = trial.suggest_int('b%s_filters' % i, 1, 512)
            config['block'][i]['kernel'] = [int(trial.suggest_discrete_uniform('b%s_kernel' %i, 1, 129, 2))]

        # C2
        config['block'][-2]['kernel'] = [int(trial.suggest_discrete_uniform('c2_kernel', 1, 129, 2))]
        config['block'][-2]['filters'] = trial.suggest_int('c2_filters', 1, 1024)

        # C3
        config['block'][-1]['kernel'] = [int(trial.suggest_discrete_uniform('c3_kernel', 1, 129, 2))]
        config['block'][-1]['filters'] = trial.suggest_int('c3_filters', 1, 1024)

        model = Model(config)
        num_params = sum(p.numel() for p in model.parameters())

        print("[trial %s]" % trial.number)

        if num_params > args.max_params:
            print("[pruned] network too large")
            raise optuna.exceptions.TrialPruned()

        model.to(args.device)
        model.train()

        os.makedirs(workdir, exist_ok=True)

        optimizer = AdamW(model.parameters(), amsgrad=True, lr=lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        schedular = CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

        for epoch in range(1, args.epochs + 1):

            try:
                train_loss, duration = train(model, device, train_loader, optimizer, use_amp=True)
                val_loss, val_mean, val_median = test(model, device, test_loader)
                print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                    epoch, workdir, val_loss, val_mean, val_median
                ))
            except KeyboardInterrupt: exit()
            except:
                print("[pruned] exception")
                raise optuna.exceptions.TrialPruned()

            if np.isnan(val_loss): val_loss = 9.9
            trial.report(val_loss, epoch)

            if trial.should_prune():
                print("[pruned] unpromising")
                raise optuna.exceptions.TrialPruned()

        trial.set_user_attr('seed', args.seed)
        trial.set_user_attr('val_loss', val_loss)
        trial.set_user_attr('val_mean', val_mean)
        trial.set_user_attr('val_median', val_median)
        trial.set_user_attr('train_loss', train_loss)
        trial.set_user_attr('batchsize', args.batch)
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
        pruner=HyperbandPruner()
    )

    study.optimize(objective, n_trials=args.trials)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("tuning_directory")
    parser.add_argument("config")
    parser.add_argument("--directory", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--trials", default=100, type=int)
    parser.add_argument("--chunks", default=1000000, type=int)
    parser.add_argument("--max-params", default=7000000, type=int)
    parser.add_argument("--validation_split", default=0.90, type=float)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    return parser
