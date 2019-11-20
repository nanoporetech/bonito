import time
from itertools import starmap

from bonito.util import decode_ctc, stitch, identity, palign

import numpy as np
import torch
import torch.nn as nn

try: from apex import amp
except ImportError: pass


criterion = nn.CTCLoss()


class ReadDataSet:
    def __init__(self, reads, targets, lengths):
        self.reads = np.expand_dims(reads, axis=1)
        self.targets = targets
        self.lengths = lengths

    def __getitem__(self, i):
        return self.reads[i], self.targets[i].astype(np.int32), self.lengths[i].astype(np.int32)

    def __len__(self):
        return len(self.reads)


def train(log_interval, model, device, train_loader, optimizer, epoch, use_amp=False):

    t0 = time.perf_counter()
    chunks = 0

    model.train()
    for batch_idx, (data, target, lengths) in enumerate(train_loader, start=1):

        chunks += data.shape[0]

        data = data.to(device)
        target = target.to(device)

        # fixed sized output lengths
        out_lengths = torch.tensor(data.shape[-1]*len(lengths), dtype=torch.int32)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output.transpose(1, 0), target, out_lengths, lengths)

        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, chunks, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, chunks, len(train_loader.dataset),
        100. * batch_idx / len(train_loader),
        loss.item()))
    print('Train Epoch: %s [%.2f Seconds]' % (epoch, time.perf_counter() - t0))

    return loss.item(), time.perf_counter() - t0


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    predictions = []

    with torch.no_grad():
        for batch_idx, (data, target, lengths) in enumerate(test_loader, start=1):

            data, target = data.to(device), target.to(device)

            # fixed sized output lengths
            out_lengths = torch.tensor(data.shape[-1]*len(lengths), dtype=torch.int32)

            output = model(data)
            predictions.append(torch.exp(output).cpu())

            test_loss += criterion(output.transpose(1, 0), target, out_lengths, lengths)

    references = list(map(stitch, test_loader.dataset.targets))
    predictions = np.concatenate(predictions)
    sequences = list(map(decode_ctc, predictions))

    try:
        identities = list(starmap(identity, zip(references, sequences)))
        palign(references[0], sequences[0])
    except IndexError:
        # it might take a few epochs for sensible alignment on a small dataset
        identities = [0]

    mean = np.mean(identities)
    median = np.median(identities)

    print("* mean %.3f" % mean)
    print("* median %.3f" % median)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss / batch_idx))
    return test_loss.item(), mean, median
