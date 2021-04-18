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


def to_guppy_dict(model, include_weights=True):
    guppy_dict = bonito.nn.to_dict(model.encoder, include_weights=include_weights)
    guppy_dict['sublayers'] = [x for x in guppy_dict['sublayers'] if x['type'] != 'permute']
    guppy_dict['sublayers'] = [dict(x, type='LSTM', activation='tanh', gate='sigmoid') if x['type'] == 'lstm' else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'] = [dict(x, padding=(x['padding'], x['padding'])) if x['type'] == 'convolution' else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'] = [{'type': 'reverse', 'sublayers': x} if x.pop('reverse', False) else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'][-1]['type'] = 'GlobalNormTransducer' # we call it Linear
    return guppy_dict


def main(args):
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
