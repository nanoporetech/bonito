"""
Bonito train
"""

import time
from itertools import starmap

from bonito.util import decode_ctc, decode_ref, accuracy

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

        optimizer.zero_grad()

        chunks += data.shape[0]
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        # fixed sized output lengths
        out_lengths = torch.tensor([output.shape[1]]*len(lengths), dtype=torch.int32)
        loss = criterion(output.transpose(0, 1), target, out_lengths, lengths)

        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                chunks, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )

    print('[{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
        chunks, len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item())
    )
    print('[%.2f Seconds]' % (time.perf_counter() - t0))

    return loss.item(), time.perf_counter() - t0


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    predictions = []

    with torch.no_grad():
        for batch_idx, (data, target, lengths) in enumerate(test_loader, start=1):

            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions.append(torch.exp(output).cpu())

            # fixed sized output lengths
            out_lengths = torch.tensor([output.shape[1]]*len(lengths), dtype=torch.int32)
            test_loss += criterion(output.transpose(1, 0), target, out_lengths, lengths)

    references = [decode_ref(target, model.alphabet) for target in test_loader.dataset.targets]
    sequences = [decode_ctc(post, model.alphabet) for post in np.concatenate(predictions)]

    if all(map(len, sequences)):
        accuracies = list(starmap(accuracy, zip(references, sequences)))
    else:
        accuracies = [0]

    mean = np.mean(accuracies)
    median = np.median(accuracies)

    print()
    print('Validation Loss:              %.4f' % (test_loss / batch_idx))
    print("Validation Accuracy (mean):   %.3f%%" % max(0, mean))
    print("Validation Accuracy (median): %.3f%%" % max(0, median))
    print()
    return test_loss.item() / batch_idx, mean, median
