#!/usr/bin/env python3

"""
Bonito training.
"""

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from bonito.data import load_numpy, load_script
from bonito.util import __models_dir__, default_config
from bonito.util import load_model, load_symbol, init, half_supported
from bonito.training import Trainer

import toml
import torch
from torch.utils.data import DataLoader


def main(args):
    workdir = os.path.expanduser(args.training_directory)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)
    os.makedirs(workdir, exist_ok=True)

    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    if not args.pretrained:
        config = toml.load(args.config)
    else:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models_dir__, dirname)):
            dirname = os.path.join(__models_dir__, dirname)
        pretrain_file = os.path.join(dirname, 'config.toml')
        config = toml.load(pretrain_file)
        if 'lr_scheduler' in config:
            print(f"[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']

    argsdict = dict(training=vars(args))

    print("[loading model]")
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False)
    else:
        model = load_symbol(config, 'Model')(config)

    print("[loading data]")
    try:
        if (Path(args.directory) / "chunks.npy").exists():
            print(f"[loading data] - chunks from {args.directory}")
            train_loader_kwargs, valid_loader_kwargs = load_numpy(
                args.chunks,
                args.directory,
                valid_chunks=args.valid_chunks,
            )
        elif (Path(args.directory) / "dataset.py").exists():
            print(f"[loading data] - dynamically from {args.directory}/dataset.py")
            train_loader_kwargs, valid_loader_kwargs = load_script(
                args.directory,
                seed=args.seed,
                chunks=args.chunks,
                valid_chunks=args.valid_chunks,
                n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
                n_post_context_bases=getattr(model, "n_post_context_bases", 0),
                batch_size=args.batch,
                standardisation=config.get("standardisation", {}),
                log_dir=workdir,
                num_workers=args.num_workers,
            )
        else:
            raise FileNotFoundError(f"No suitable training data found at: {args.directory}")
    except Exception as e:
        raise IOError(f"Failed to load input data from {args.directory}") from e

    loader_kwargs = {
        "batch_size": args.batch, "num_workers": args.num_workers, "pin_memory": True
    }
    # Allow options from the train/valid_loader to override the loader_kwargs
    train_loader = DataLoader(**{**loader_kwargs, **train_loader_kwargs})
    valid_loader = DataLoader(**{**loader_kwargs, **valid_loader_kwargs})

    try:
        # Allow the train-loader to write meta-data fields to the config
        dataset_cfg = train_loader.dataset.dataset_config
    except AttributeError:
        dataset_cfg = {}
    toml.dump({**config, **argsdict, **dataset_cfg}, open(os.path.join(workdir, 'config.toml'), 'w'))

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        model, device, train_loader, valid_loader,
        use_amp=half_supported() and not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip,
        chunks_per_epoch=args.chunks,
        batch_size=args.batch,
    )

    if (',' in args.lr):
        lr = [float(x) for x in args.lr.split(',')]
    else:
        lr = float(args.lr)
    optim_kwargs = config.get("optim", {})
    trainer.fit(workdir, args.epochs, lr, **optim_kwargs)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=None, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    quantile_group = parser.add_mutually_exclusive_group()
    quantile_group.add_argument('--quantile-grad-clip', dest='quantile_grad_clip', action='store_true')
    quantile_group.add_argument('--no-quantile-grad-clip', dest='quantile_grad_clip', action='store_false')
    quantile_group.set_defaults(quantile_grad_clip=True)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser
