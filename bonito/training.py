"""
Bonito train
"""

import os
import re
import time
from glob import glob
from itertools import starmap

from bonito.util import accuracy, decode_ref

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

try: from apex import amp
except ImportError: pass


criterion = nn.CTCLoss(reduction='mean')


class ChunkDataSet:
    def __init__(self, chunks, chunk_lengths, targets, target_lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.chunk_lengths = chunk_lengths
        self.targets = targets
        self.target_lengths = target_lengths

    def __getitem__(self, i):
        return (
            self.chunks[i],
            self.chunk_lengths[i].astype(np.int32),
            self.targets[i].astype(np.int32),
            self.target_lengths[i].astype(np.int32)
        )

    def __len__(self):
        return len(self.chunks)


def load_state(dirname, device, model, optim, use_amp=False):
    """
    Load a model and optimizer state dict from disk
    """
    model.to(device)

    if use_amp:
        try:
            model, optimizer = amp.initialize(model, optim, opt_level="O1", verbosity=0)
        except NameError:
            print("[error]: Cannot use AMP: Apex package needs to be installed manually, See https://github.com/NVIDIA/apex")
            exit(1)

    weight_no = optim_no = None

    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    if weight_files:
        weight_no = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])

    optim_files = glob(os.path.join(dirname, "optim_*.tar"))
    if optim_files:
        optim_no = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in optim_files])

    if weight_no and optim_no and weight_no == optim_no:
        print("[picking up from epoch %s]" % optim_no)
        model_dict = torch.load(
            os.path.join(dirname, 'weights_%s.tar' % weight_no), map_location=device
        )
        model.load_state_dict(model_dict)
        optim_dict = torch.load(
            os.path.join(dirname, 'optim_%s.tar' % optim_no), map_location=device
        )
        optim.load_state_dict(optim_dict)
        epoch = weight_no
    else:
        epoch = 0

    return epoch


def train(model, device, train_loader, optimizer, use_amp=False):

    chunks = 0
    model.train()
    t0 = time.perf_counter()

    progress_bar = tqdm(
        total=len(train_loader), desc='[0/{}]'.format(len(train_loader.dataset)),
        ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
    )

    with progress_bar:

        for data, out_lengths, target, lengths in train_loader:

            optimizer.zero_grad()

            chunks += data.shape[0]

            data = data.to(device)
            target = target.to(device)
            log_probs = model(data)

            loss = criterion(log_probs.transpose(0, 1), target, out_lengths / model.stride, lengths)

            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            progress_bar.set_postfix(loss='%.4f' % loss.item())
            progress_bar.set_description("[{}/{}]".format(chunks, len(train_loader.dataset)))
            progress_bar.update()

    return loss.item(), time.perf_counter() - t0


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    predictions = []
    prediction_lengths = []

    with torch.no_grad():
        for batch_idx, (data, out_lengths, target, lengths) in enumerate(test_loader, start=1):
            data, target = data.to(device), target.to(device)
            log_probs = model(data)
            test_loss += criterion(log_probs.transpose(1, 0), target, out_lengths / model.stride, lengths)
            predictions.append(torch.exp(log_probs).cpu())
            prediction_lengths.append(out_lengths / model.stride)

    predictions = np.concatenate(predictions)
    lengths = np.concatenate(prediction_lengths)

    references = [decode_ref(target, model.alphabet) for target in test_loader.dataset.targets]
    sequences = [model.decode(post[:n]) for post, n in zip(predictions, lengths)]

    if all(map(len, sequences)):
        accuracies = list(starmap(accuracy, zip(references, sequences)))
    else:
        accuracies = [0]

    mean = np.mean(accuracies)
    median = np.median(accuracies)
    return test_loss.item() / batch_idx, mean, median
