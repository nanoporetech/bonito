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
import base64
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.nn import fuse_bn_
from bonito.util import _load_model, get_last_checkpoint, set_config_defaults


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
        elif isinstance(obj, bytes):
            return obj.decode('ascii')
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


def save_tensor(directory, name, tensor):
    """
    Save a tensor `x` to `fn.tensor` for use with libtorch.
    """
    module = torch.nn.Module()
    param = torch.nn.Parameter(tensor, requires_grad=False)
    module.register_parameter("0", param)
    tensors = torch.jit.script(module)
    tensors.save(f"{directory}/{name}.tensor")


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


def to_guppy_dict(model, include_weights=True, binary_weights=True):
    guppy_dict = bonito.nn.to_dict(model.encoder, include_weights=include_weights)
    guppy_dict['sublayers'] = [x for x in guppy_dict['sublayers'] if x['type'] != 'permute']
    guppy_dict['sublayers'] = [dict(x, type='LSTM', activation='tanh', gate='sigmoid') if x['type'] == 'lstm' else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'] = [dict(x, padding=(x['padding'], x['padding'])) if x['type'] == 'convolution' else x for x in guppy_dict['sublayers']]
    guppy_dict['sublayers'][-1] = reformat_output_layer(guppy_dict['sublayers'][-1])
    if binary_weights:
        for layer_dict in guppy_dict['sublayers']:
            if 'params' in layer_dict:
                layer_dict['params'] = {
                    f'{k}_binary': base64.b64encode(v.data.detach().numpy().astype(np.float32).tobytes()) for (k, v) in layer_dict['params'].items()
                }
    guppy_dict['sublayers'] = [{'type': 'reverse', 'sublayers': x} if x.pop('reverse', False) else x for x in guppy_dict['sublayers']]

    return guppy_dict


def main(args):

    model_file = get_last_checkpoint(args.model) if os.path.isdir(args.model) else args.model

    if args.config is None:
        args.config = os.path.join(os.path.dirname(model_file), "config.toml")

    config = toml.load(args.config)
    config = set_config_defaults(config)
    model = _load_model(model_file, config, device='cpu')

    if args.fuse_bn:
        # model weights might be saved in half when training and PyTorch's bn fusion
        # code uses an op (rsqrt) that currently (1.11) only has a float implementation
        model = model.to(torch.float32).apply(fuse_bn_)

    if args.format == 'guppy':
        jsn = to_guppy_dict(model)
        jsn["md5sum"] = file_md5(model_file)
        json.dump(jsn, sys.stdout, cls=JsonEncoder)
    elif args.format == 'dorado':
        for name, tensor in model.encoder.state_dict().items():
            save_tensor(args.model, name, tensor)
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
    parser.add_argument('--format', choices=['guppy', 'dorado', 'torchscript'], default='guppy')
    parser.add_argument('--config', default=None, help='config file to read settings from')
    parser.add_argument('--fuse-bn', default=True, help='fuse batchnorm layers')
    return parser
