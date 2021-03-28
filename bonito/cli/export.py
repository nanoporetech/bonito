"""
Bonito Export
"""

import sys
import json
import torch
import hashlib
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args):
    model = torch.load(args.model, map_location='cpu')
    assert len(model) == 28
    model = list(model.items())
    jsn = serial_json([
        conv_json(model[0], model[1], stride=1),
        conv_json(model[2], model[3], stride=1),
        conv_json(model[4], model[5], stride=args.stride),
        reverse_json(lstm_json(*list(model[6:10]))),
        lstm_json(*list(model[10:14])),
        reverse_json(lstm_json(*list(model[14:18]))),
        lstm_json(*list(model[18:22])),
        reverse_json(lstm_json(*list(model[22:26]))),
        global_norm_json(model[26], model[27], scale=args.scale)
    ])
    model_md5 = file_md5(args.model)
    jsn["md5sum"] = model_md5
    json.dump(jsn, sys.stdout, cls=JsonEncoder)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.nn.Parameter):
            return obj.data
        elif isinstance(obj, torch.Tensor):
            return obj.detach_().numpy()
        else:
            return super(JsonEncoder, self).default(obj)


def file_md5(filename, nblock=1024):
    """
    Get md5 string from file.
    """
    hasher = hashlib.md5()
    block_size = nblock * hasher.block_size
    with open(filename, "rb") as fh:
        for blk in iter((lambda: fh.read(block_size)), b""):
            hasher.update(blk)
    return hasher.hexdigest()


def reshape(x, shape):
    return x.detach_().numpy().reshape(shape)


def test_name(name, expt):
    ns = name.split('.')
    assert ns[-1] == expt, 'Got {} expected {}'.format(ns[-1], expt)


def conv_json(w, b, activation='swish', stride=1):
    test_name(w[0], 'weight')
    test_name(b[0], 'bias')
    size, insize, winlen = w[1].shape
    assert b[1].shape == (size,)
    padding = (winlen // 2, (winlen - 1) // 2)
    res = OrderedDict([
        ("type", "convolution"),
        ("insize", insize),
        ("size", size),
        ("bias", b is not None),
        ("winlen", winlen),
        ("stride", stride),
        ("padding", padding),
        ("activation", activation)
    ])
    res['params'] = OrderedDict(
        [("W", w[1])] + [("b", b[1])] if b is not None else []
    )
    return res


def global_norm_json(w, b, activation='tanh', scale=5.0):
    test_name(w[0], 'weight')
    test_name(b[0], 'bias')
    size, insize = w[1].shape
    assert b[1].shape == (size,)
    res = OrderedDict([
        ('type', 'GlobalNormTransducer'),
        ('size', size),
        ('insize', insize),
        ('bias', b is not None),
        ('scale', scale),
        ("activation", activation)
    ])
    res['params'] = OrderedDict(
        [('W', w[1])] +
        [('b', b[1])] if b is not None else []
    )
    return res


def lstm_json(wih, whh, bih, bhh):
    test_name(wih[0], 'weight_ih_l0')
    test_name(whh[0], 'weight_hh_l0')
    test_name(bih[0], 'bias_ih_l0')
    test_name(bhh[0], 'bias_hh_l0')
    four_size, insize = wih[1].shape
    size = four_size // 4
    assert whh[1].shape == (four_size, size)
    assert bih[1].shape == (four_size,)
    res = OrderedDict([
        ('type', "LSTM"),
        ('activation', "tanh"),
        ('gate', "sigmoid"),
        ('size', size),
        ('insize', insize),
        ('bias', True)
    ])
    res['params'] = OrderedDict([
        ('iW', reshape(wih[1], (4, size, insize))),
        ('sW', reshape(whh[1], (4, size, size))),
        ('b', reshape(bih[1], (4, size)))
    ])
    return res


def reverse_json(layer):
    return OrderedDict([
        ('type', "reverse"), ('sublayers', layer)
    ])


def serial_json(sublayers):
    return OrderedDict([
       ('type', "serial"), ('sublayers', [jsn for jsn in sublayers])
    ])


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument('--scale', default=5.0, type=float)
    parser.add_argument('--stride', default=5, type=int)
    parser.add_argument('model')
    return parser
