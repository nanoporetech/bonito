"""
Bonito Decoding functions
"""

from itertools import groupby

import numpy as np
from fast_ctc_decode import beam_search


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def greedy_ctc_decode(predictions, labels):
    """
    Greedy argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def decode(predictions, alphabet, beam_size=5, threshold=0.1):
    """
    Decode model posteriors to sequence
    """
    alphabet = ''.join(alphabet)
    if beam_size == 1:
        sequence = greedy_ctc_decode(predictions, alphabet)
    else:
        sequence, _ = beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)
    return sequence
