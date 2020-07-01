"""
Bonito Model template
"""

import torch
import torch.nn as nn
from torch import sigmoid
from torch.jit import script
from torch.autograd import Function
from torch.nn import ReLU, LeakyReLU
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout

from fast_ctc_decode import beam_search, viterbi_search


@script
def swish_jit_fwd(x):
    return x * sigmoid(x)


@script
def swish_jit_bwd(x, grad):
    x_s = sigmoid(x)
    return grad * (x_s * (1 + x * (1 - x_s)))


class SwishAutoFn(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad)


class Swish(Module):
    """
    Swish Activation function

    https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return SwishAutoFn.apply(x)


activations = {
    "relu": ReLU,
    "swish": Swish,
}


class Model(Module):
    """
    Model template for QuartzNet style architectures

    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = config['block'][-1]['filters']
        self.encoder = Encoder(config)
        self.decoder = Decoder(self.features, len(self.alphabet))

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq


class Encoder(Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        if config['input'].get('embed'):
            features = 64
            encoder_layers = [Embed(size=64)]
        else:
            features = self.config['input']['features']
            encoder_layers = []

        activation = activations[self.config['encoder']['activation']]()

        for layer in self.config['block']:
            encoder_layers.append(
                Block(
                    features, layer['filters'], activation,
                    repeat=layer['repeat'], kernel_size=layer['kernel'],
                    stride=layer['stride'], dilation=layer['dilation'],
                    dropout=layer['dropout'], residual=layer['residual'],
                    separable=layer['separable'],
                )
            )

            features = layer['filters']

        self.encoder = Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

@script
def gauss(x, mu, sig):
    return (1.0 / sig) * torch.exp(-(((x-mu[:, None])**2) / (2*sig**2)))

@script
def diff_gauss(x, mu, sig1, sig2):
    return 0.5 * (gauss(x, mu, sig1) - gauss(x, mu, sig2))


class Embed(Module):
    def __init__(self, sig1=0.2, sig2=None, size=64, range_min=-2.2, range_max=2.2):
        super().__init__()
        self.size=size
        self.sig1, self.sig2 = [nn.Parameter(torch.tensor(x), requires_grad=False) for x in (sig1, (sig2 or 2*sig1))]
        self.ys = nn.Parameter(torch.linspace(range_min, range_max, size), requires_grad=False)

    def forward(self, x):
        return diff_gauss(x, self.ys, self.sig1, self.sig2)


class PixelShuffle1d(Module):
    """
    Upscales sample length, downscales channel length

    https://arxiv.org/pdf/1609.05158.pdf
    """
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.factor
        long_width = self.factor * short_width

        x = x.contiguous().view([batch_size, self.factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(batch_size, long_channel_len, long_width)


class TCSConv1d(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, 1, stride=1,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x


class Block(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(Block, self).__init__()

        self.use_res = residual
        self.unet = residual and stride[0] > 1
        self.conv = ModuleList()

        if self.unet:
            out_channels *= stride[0]
            _in_channels = out_channels
            self.conv.extend([
                Conv1d(in_channels, out_channels, 5, padding=2, stride=stride, bias=False),
                BatchNorm1d(out_channels),
                *self.get_activation(activation, dropout),
            ])
        else:
            _in_channels = in_channels

        padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        # add the first n - 1 convolutions + activation
        for idx in range(repeat - 1):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride if not self.unet and idx == 0 else 1, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride if repeat == 1 else 1, dilation=dilation,
                padding=padding, separable=separable
            )
        )

        if self.unet:
            self.conv.extend([
                *self.get_activation(activation, dropout),
                PixelShuffle1d(stride[0])
            ])
            out_channels //= stride[0]

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation(activation, dropout))

    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x += self.residual(x)
        return self.activation(_x)


class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = Sequential(Conv1d(features, classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.layers(x)
        return nn.functional.log_softmax(x.transpose(1, 2), dim=2)
