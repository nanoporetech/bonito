"""
Bonito Decoding functions
"""

from itertools import groupby

import numpy as np
from fast_ctc_decode import beam_search, viterbi_search


def phred(prob, scale=1.0, bias=0.0):
    """
    Converts `prob` into a ascii encoded phred quality score between 0 and 40.
    """
    p = max(1 - prob, 1e-4)
    q = -10 * np.log10(p) * scale + bias
    return chr(int(np.round(q) + 33))


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded if e)


def greedy_ctc_decode(predictions, labels, qscores=False):
    """
    Greedy argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def decode(predictions, alphabet, beam_size=5, threshold=0.1, qscores=False):
    """
    Decode model posteriors to sequence
    """
    alphabet = ''.join(alphabet)
    predictions = predictions.astype(np.float32)
    if beam_size == 1 or qscores:
        sequence, path = viterbi_search(predictions, alphabet, qstring=qscores)
    else:
        sequence, path = beam_search(predictions, alphabet, beqam_size, threshold)
    return sequence, path
