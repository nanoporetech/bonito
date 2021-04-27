"""
Bonito Export
"""

import os
import re
import sys
import json
import torch
import bonito
import hashlib
import numpy as np
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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
            return obj.detach().numpy()
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


def reformat_output_layer(layer_dict):
    n_base, state_len, blank_score = [layer_dict.pop(k) for k in ['n_base', 'state_len', 'blank_score']]
    layer_dict['size'] = (n_base + 1) * n_base**state_len
    layer_dict['type'] = 'GlobalNormTransducer'
    if blank_score is not None:
        assert layer_dict['activation'] == 'tanh'
        params = layer_dict['params']
        params['W'] = torch.nn.functional.pad(
            params['W'].reshape([n_base**state_len, n_base, -1]),
            (0, 0, 1, 0),
            value=0.
        ).reshape((n_base + 1) * n_base**state_len, -1)

        params['b'] = torch.nn.functional.pad(
            params['b'].reshape(n_base**state_len, n_base),
            (1, 0),
            value=np.arctanh(blank_score / layer_dict['scale'])
        ).reshape(-1)

    return layer_dict


def to_guppy_dict(model, include_weights=True):
    guppy_dict = bonito.nn.to_dict(model.encoder, include_weights=include_weights)
    guppy_dict['sublayers'] = [x for x in guppy_dict['sublayers'] if x['type'] != 'permute']
    guppy_dict['sublayers'] = [dict(x, type='LSTM', activation='tanh', gate='sigmoid') if x['type'] == 'lstm' else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'] = [dict(x, padding=(x['padding'], x['padding'])) if x['type'] == 'convolution' else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'] = [{'type': 'reverse', 'sublayers': x} if x.pop('reverse', False) else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'][-1] = reformat_output_layer(guppy_dict['sublayers'][-1])
    return guppy_dict


def main(args):

    if not os.path.isdir(args.model):
        print("[error] file given - please provide a model directory to export.", file=sys.stderr)
        return 1

    model = bonito.util.load_model(args.model, device='cpu')
    jsn = to_guppy_dict(model)
    weight_files = glob(os.path.join(args.model, "weights_*.tar"))
    weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
    jsn["md5sum"] = file_md5(os.path.join(args.model, 'weights_%s.tar' % weights))
    json.dump(jsn, sys.stdout, cls=JsonEncoder)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument('model')
    return parser
