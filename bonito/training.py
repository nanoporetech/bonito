"""
Bonito train
"""

import os
import re
from glob import glob
from functools import partial
from time import perf_counter
from collections import OrderedDict
from datetime import datetime

from bonito.schedule import linear_warmup_cosine_decay
from bonito.util import accuracy, decode_ref, permute, concat, match_names
import bonito

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.cuda.amp as amp


def load_state(dirname, device, model, optim=None):
    """
    Load a model state dict from disk
    """
    model.to(device)
    if hasattr(model, "module"):
        model = model.module

    weight_no = optim_no = None

    optim_files = glob(os.path.join(dirname, "optim_*.tar"))
    optim_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in optim_files}

    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    weight_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files}

    if optim is not None:
        weight_no = optim_no = max(optim_nos & weight_nos, default=None)
    else:
        weight_no = max(weight_nos, default=None)

    to_load = []
    if weight_no:
        to_load.append(("weights", model))
    if optim_no:
        to_load.append(("optim", optim))

    if to_load:
        print("[picking up %s state from epoch %s]" % (', '.join([n for n, _ in to_load]), weight_no))
        for name, obj in to_load:
            state_dict = torch.load(
                os.path.join(dirname, '%s_%s.tar' % (name, weight_no)), map_location=device
            )
            if name == "weights":
                state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, obj).items()}
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict
            obj.load_state_dict(state_dict)
        epoch = weight_no
    else:
        epoch = 0

    return epoch


class Trainer:
    def __init__(
        self, model, device, train_loader, valid_loader, criterion=None,
        use_amp=True, lr_scheduler_fn=None, restore_optim=False,
        save_optim_every=10, grad_accum_split=1
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or (model.seqdist.ctc_loss if hasattr(model, 'seqdist') else model.ctc_label_smoothing_loss)
        self.use_amp = use_amp
        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.save_optim_every = save_optim_every
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.optimizer = None

    def train_one_step(self, batch):
        self.optimizer.zero_grad()

        losses = None
        with amp.autocast(enabled=self.use_amp):
            for data_, targets_, lengths_ in zip(*map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)):
                data_, targets_, lengths_ = data_.to(self.device), targets_.to(self.device), lengths_.to(self.device)
                scores_ = self.model(data_)
                losses_ = self.criterion(scores_.to(torch.float32), targets_, lengths_)

                if not isinstance(losses_, dict): losses_ = {'loss': losses_}

                losses_['loss'] = losses_['loss'] / self.grad_accum_split

                self.scaler.scale(losses_['loss']).backward()

                losses = {
                    k: (v.item() if losses is None else v.item() + losses[k])
                    for k, v in losses_.items()
                }

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return losses, grad_norm

    def train_one_epoch(self, loss_log, lr_scheduler):
        t0 = perf_counter()
        chunks = 0
        self.model.train()

        progress_bar = tqdm(
            total=len(self.train_loader), desc='[0/{}]'.format(len(self.train_loader.sampler)),
            ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
        )
        smoothed_loss = None

        with progress_bar:

            for batch in self.train_loader:

                chunks += batch[0].shape[0]

                losses, grad_norm = self.train_one_step(batch)

                smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)

                progress_bar.set_postfix(loss='%.4f' % smoothed_loss)
                progress_bar.set_description("[{}/{}]".format(chunks, len(self.train_loader.sampler)))
                progress_bar.update()

                if loss_log is not None:
                    lr = lr_scheduler.get_last_lr() if lr_scheduler is not None else [pg["lr"] for pg in optim.param_groups]
                    if len(lr) == 1: lr = lr[0]
                    loss_log.append({
                        'chunks': chunks,
                        'time': perf_counter() - t0,
                        'grad_norm': grad_norm,
                        'lr': lr,
                        **losses
                    })

                if lr_scheduler is not None: lr_scheduler.step()

        return smoothed_loss, perf_counter() - t0

    def validate_one_step(self, batch):
        data, targets, lengths = batch

        scores = self.model(data.to(self.device))
        losses = self.criterion(scores.to(torch.float32), targets.to(self.device), lengths.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()
        if hasattr(self.model, 'decode_batch'):
            seqs = self.model.decode_batch(scores)
        else:
            seqs = [self.model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
        refs = [decode_ref(target, self.model.alphabet) for target in targets]
        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]
        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            seqs, refs, accs, losses = zip(*(self.validate_one_step(batch) for batch in self.valid_loader))
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss = np.mean([(x['ctc_loss'] if isinstance(x, dict) else x) for x in losses])
        return loss, np.mean(accs), np.median(accs)

    def init_optimizer(self, lr, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)

    def get_lr_scheduler(self, epochs, last_epoch=0):
        return self.lr_scheduler_fn(self.optimizer, self.train_loader, epochs, last_epoch)

    def fit(self, workdir, epochs=1, lr=2e-3):
        if self.optimizer is None:
            self.init_optimizer(lr)

        last_epoch = load_state(workdir, self.device, self.model, self.optimizer if self.restore_optim else None)
        # override learning rate to new value
        for pg in self.optimizer.param_groups: pg["initial_lr"] = pg["lr"] = lr

        lr_scheduler = self.get_lr_scheduler(epochs, last_epoch=last_epoch)

        for epoch in range(1 + last_epoch, epochs + 1 + last_epoch):
            try:
                with bonito.io.CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                    train_loss, duration = self.train_one_epoch(loss_log, lr_scheduler)

                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))
                if epoch % self.save_optim_every == 0:
                    torch.save(self.optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))

                val_loss, val_mean, val_median = self.validate_one_epoch()
            except KeyboardInterrupt:
                break

            print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                epoch, workdir, val_loss, val_mean, val_median
            ))

            with bonito.io.CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                training_log.append({
                    'time': datetime.today(),
                    'duration': int(duration),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                    'validation_mean': val_mean,
                    'validation_median': val_median
                })
