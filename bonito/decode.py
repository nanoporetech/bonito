"""
Bonito Decoding functions
"""

from itertools import groupby

import numpy as np
from fast_ctc_decode import beam_search


def phred(prob, scale=0.7, bias=2.0):
    """
    Converts `prob` into a ascii encoded phred quality score between 0 and 40.
    """
    p = max(1 - prob, 1e-4)
    q = -10 * np.log10(p) * scale + bias
    return chr(int(q + 33))


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def greedy_ctc_decode(predictions, labels, qscores=False):
    """
    Greedy argmax decoder with collapsing repeats
    """
    pos = 0
    path = []
    qstring = []
    sequence = []

    for label, run in groupby(np.argmax(predictions, axis=1)):
        duration = len(list(run))
        if label:
            avg_label_prob = np.mean(predictions[pos:pos + duration, label])
            path.append(pos)
            qstring.append(phred(avg_label_prob) if qscores else '!')
            sequence.append(labels[label])
        pos += duration

    return ''.join(sequence), ''.join(qstring), path


def decode(predictions, alphabet, beam_size=5, threshold=0.1, qscores=False):
    """
    Decode model posteriors to sequence
    """
    alphabet = ''.join(alphabet)
    if beam_size == 1:
        sequence, qstring, path = greedy_ctc_decode(predictions, alphabet, qscores=qscores)
    else:
        sequence, path = beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)
        qstring = '!' * len(sequence)
    return sequence, qstring, path
