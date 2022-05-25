"""
Bonito Export
"""

import io
import os
import re
import sys
import json

import toml
import torch
import bonito
import hashlib
import numpy as np
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import _load_model


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
    if os.path.isdir(args.model):
        weight_files = glob(os.path.join(args.model, "weights_*.tar"))
        last_checkpoint = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
        model_file = os.path.join(args.model, 'weights_%s.tar' % last_checkpoint)
    else:
        model_file = args.model

    if args.config is None:
        args.config = os.path.join(os.path.dirname(model_file), "config.toml")

    config = toml.load(args.config)
    model = _load_model(model_file, config, device='cpu')

    if args.format == 'guppy':
        jsn = to_guppy_dict(model)
        jsn["md5sum"] = file_md5(model_file)
        json.dump(jsn, sys.stdout, cls=JsonEncoder)
    elif args.format == 'torchscript':
        tmp_tensor = torch.rand(10, 1, 1000)
        model = model.float()
        traced_script_module = torch.jit.trace(model, tmp_tensor)
        buffer = io.BytesIO()
        torch.jit.save(traced_script_module, buffer)
        buffer.seek(0)
        sys.stdout.buffer.write(buffer.getvalue())
        sys.stdout.flush()
    else:
        raise NotImplementedError("Export format not supported")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument('model')
    parser.add_argument('--format', help='guppy or torchscript', default='guppy')
    parser.add_argument('--config', default=None,
                        help='config file to read settings from')
    return parser
