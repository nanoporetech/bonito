"""
Bonito Export
"""
import logging
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import toml
import torch

from bonito.nn import fuse_bn_, Clamp
from bonito.util import _load_model, get_last_checkpoint

logger = logging.getLogger(__name__)


def save_tensor(directory, name, tensor):
    """
    Save a tensor `x` to `fn.tensor` for use with libtorch.
    """
    module = torch.nn.Module()
    param = torch.nn.Parameter(tensor, requires_grad=False)
    module.register_parameter("0", param)
    tensors = torch.jit.script(module)
    tensors.save(directory / f"{name}.tensor")


def clean_config(config):
    """
    Strip any non-inference time features out of the model
    """
    config.pop("decoder", None)
    config.pop("aux_CRF_losses", None)
    config.pop("training", None)
    config.pop("basecaller", None)
    config.pop("lr_scheduler", None)
    config.pop("optim", None)

    expected_fields = ["qscore", "run_info", "scaling", "standardisation", "training_dataset"]
    for field in expected_fields:
        if field not in config:
            logger.warning(f"INFO: metadata '{field}' is not set in config")
    return config


def get_layer_order_map(base_encoder):
    # For models with clamp layers we have to reorder the output layers
    # so that Dorado can parse them correctly
    clamp_count = 0
    layer_order_map = {}
    for i, layer in enumerate(base_encoder):
        if isinstance(layer, Clamp):
            clamp_count += 1
        layer_order_map[str(i)] = str(i - clamp_count)
    return layer_order_map


def export_to_dorado(model, config_dict, output):
    output.mkdir(exist_ok=True, parents=True)

    if hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        # v5-style transformer models
        encoder = model.base_model.encoder
        config_dict["model"] = config_dict["model"]["base_model"]
    elif hasattr(model, "encoder") and hasattr(model.encoder, "base_encoder"):
        # v4-style lstm-models
        encoder = model.encoder.base_encoder
    else:
        encoder = model.encoder

    config_dict = clean_config(config_dict)
    with (output / "config.toml").open("w") as f:
        toml.dump(config_dict, f)

    for name, tensor in encoder.state_dict().items():
        save_tensor(output, name, tensor)

    if any(isinstance(encoder[i], Clamp) for i in range(len(encoder) - 1)):
        reorder_layers_without_clamp(encoder, output)


def reorder_layers_without_clamp(encoder, output):
    # In v4.0-v4.2 we had clamp layers after the convs which need removing
    layer_order_map = get_layer_order_map(encoder)
    for name, tensor in encoder.state_dict().items():
        # Rename the layers to avoid counting Clamps
        # We have to do this _after_ saving the file to get an identical object
        # since tensor.save() encodes the filename in the file
        old_layer_id = name.split(".")[0]
        new_layer_id = layer_order_map.get(old_layer_id, old_layer_id)
        new_name = name.replace(old_layer_id, new_layer_id, 1)
        if name != new_name:
            shutil.move(output / f"{name}.tensor", output / f"{new_name}.tensor")


def main(args):
    export_model(args.model, args.output, args.config)


def export_model(model, output, config_file):
    if config_file is None:
        config_file = model / "config.toml"

    config_dict = toml.load(config_file)
    model_file = get_last_checkpoint(model) if model.is_dir() else model
    model = _load_model(model_file, config_dict, device='cpu')

    # fuse conv+batchnorm
    # model weights might be saved in half when training and PyTorch's bn fusion
    # code uses an op (rsqrt) that currently (1.11) only has a float implementation
    model = model.to(torch.float32).apply(fuse_bn_)

    export_to_dorado(model, config_dict, output)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=None,
                        help='config file to read settings from')
    return parser
