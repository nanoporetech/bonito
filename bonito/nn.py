"""
Bonito nn modules.
"""

import math
import torch
from torch import nn
from torch.nn import Module
from torch.nn.init import orthogonal_
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

layers = {}


def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer
    return layer


register(torch.nn.ReLU)
register(torch.nn.Tanh)


@register
class Swish(torch.nn.SiLU):
    pass


@register
class Serial(torch.nn.Sequential):

    def __init__(self, sublayers):
        super().__init__(*sublayers)

    def to_dict(self, include_weights=False):
        return {
            'sublayers': [to_dict(layer, include_weights) for layer in self._modules.values()]
        }


@register
class Reverse(Module):

    def __init__(self, sublayers):
        super().__init__()
        self.layer = Serial(sublayers) if isinstance(sublayers, list) else sublayers

    def forward(self, x):
        return self.layer(x.flip(0)).flip(0)

    def to_dict(self, include_weights=False):
        if isinstance(self.layer, Serial):
            return self.layer.to_dict(include_weights)
        else:
            return {'sublayers': to_dict(self.layer, include_weights)}


@register
class Convolution(Module):

    def __init__(self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None):
        super().__init__()
        self.conv = torch.nn.Conv1d(insize, size, winlen, stride=stride, padding=padding, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.conv.in_channels,
            "size": self.conv.out_channels,
            "bias": self.conv.bias is not None,
            "winlen": self.conv.kernel_size[0],
            "stride": self.conv.stride[0],
            "padding": self.conv.padding[0],
            "activation": self.activation.name if self.activation else None,
        }
        if include_weights:
            res['params'] = {
                'W': self.conv.weight, 'b': self.conv.bias if self.conv.bias is not None else []
            }
        return res


@register
class LinearCRFEncoder(Module):

    def __init__(self, insize, n_base, state_len, bias=True, scale=None, activation=None, blank_score=None):
        super().__init__()
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()
        self.scale = scale

    def forward(self, x):
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None:
            T, N, C = scores.shape
            s = torch.tensor(self.blank_score, device=scores.device, dtype=scores.dtype)
            scores = torch.cat([s.expand(T, N, C//self.n_base, 1), scores.reshape(T, N, C//self.n_base, self.n_base)], axis=-1).reshape(T, N, -1)
        return scores

    def to_dict(self, include_weights=False):
        res = {
            'insize': self.linear.in_features,
            'n_base': self.n_base,
            'state_len': self.state_len,
            'bias': self.linear.bias is not None,
            'scale': self.scale,
            'activation': self.activation.name if self.activation else None,
            'blank_score': self.blank_score,
        }
        if include_weights:
            res['params'] = {
                'W': self.linear.weight, 'b': self.linear.bias
                if self.linear.bias is not None else []
            }
        return res


@register
class SHA(Module):

    def __init__(self, dim, dropout=0., sha_sandwich_norm=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_q = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)
        self.bottom_sandwich_norm = nn.LayerNorm(dim) if sha_sandwich_norm else nn.Identity()
        self.layerscale = LayerScale(dim)

    def forward(self, x, kv):
        x = x.transpose(0, 1)
        kv = kv.transpose(0, 1)

        q = self.to_q(x)
        sim = torch.matmul(q, kv.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, kv)
        out = out.transpose(0, 1)
        out = self.bottom_sandwich_norm(out)
        return self.layerscale(out)

@register
class MHA(Module):

    def __init__(self, dim, heads=4, dim_head=64, dropout=0., causal = False, norm_inputs=False, rel_pos_emb=None):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.rel_pos_emb = rel_pos_emb

        # proposed https://openreview.net/forum?id=GMYWzWztDx5
        self.head_scale = nn.Parameter(torch.ones(1, heads, 1, 1))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)
        self.layerscale = LayerScale(dim)

        self.norm = nn.LayerNorm(dim) if norm_inputs else nn.Identity()

    def forward(self, x, kv=None, rot_pos_emb=None):
        n, b, d, h, device = *x.shape, self.heads, x.device

        x = self.norm(x)
        kv = x if kv is None else kv

        x = x.transpose(0, 1)
        kv = kv.transpose(0, 1)

        q, k, v = self.to_q(x), self.to_k(kv), self.to_v(kv)

        q, k, v = map(lambda t: t.reshape(b, -1, h, self.dim_head).transpose(1, 2), (q, k, v))

        if rot_pos_emb is not None:
            rot_pos_emb = rot_pos_emb[:, None]
            q = apply_rotary_pos_emb(rot_pos_emb, q)
            k = apply_rotary_pos_emb(rot_pos_emb, k)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device=device).triu(j - i + 1).bool()
            mask_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask[None, None, :, :], mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) * self.head_scale

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)

        out = out.transpose(0, 1)
        return self.layerscale(out)

# sinusoidal positional embedding

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        sinusoid_inp = t[:, None] * self.inv_freq[None, :]
        emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb.reshape(1, seq_len, -1)

# rotary embedding helper functions

def rotate_half(x):
    preceding_dims = x.shape[:-1]
    x = x.reshape(*preceding_dims, -1, 2)
    x1, x2 = x.unbind(dim=-1)
    out = torch.stack((-x2, x1), dim=-1)
    return out.reshape(*preceding_dims, -1)

def apply_rotary_pos_emb(freqs, t):
    """
    Rotary embedding - parameter-less relative positional encoding in the context of attention
    https://arxiv.org/abs/2104.09864
    """
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t =  (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return t

class Decoder(Module):

    def __init__(self, dim, num_tokens=5, depth=2, heads=4, dim_head=64, max_seq_len=1024, loss_weight=0.25, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.loss_weight = loss_weight
        self.token_emb = nn.Embedding(num_tokens + 2, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.norm_context = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        self.rot_pos_emb = SinusoidalEmbedding(dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MHA(dim, heads=heads, causal=True, norm_inputs=True, dropout=attn_dropout),
                MHA(dim, heads=heads, norm_inputs=True, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens + 2)
        )

    def forward(self, x, encoded, return_loss=False):
        device = x.device
        encoded = self.norm_context(encoded)

        if return_loss:
            is_padding = (x == 0)

            # reserve 2 special tokens for SOS and EOS. 0 is reused as SOS for non-reversed seq
            x = x + 2
            x = x.masked_fill(is_padding, 0)

            # unbind and reverse the nucleotide sequences and pad with batch first
            reversed_x_list = list(map(lambda t: t[t.nonzero()], torch.flip(x, (1,)).unbind(dim=0)))
            reversed_x = pad_sequence(reversed_x_list, batch_first=True).squeeze(-1)

            # add SOS token for reversed nucleotide sequence as 1
            reversed_x = F.pad(reversed_x, (1, 0), value=1)

            # add SOS token for nucleotide sequence as 0 (it is fine, even though it is padding)
            x = F.pad(x, (1, 0), value=0)

            # make room for EOS token for both original and reversed sequence
            x = F.pad(x, (0, 1), value=0)
            reversed_x = F.pad(reversed_x, (0, 1), value=0)

            # make sure original sequence and reverse sequence batches are concatenatable
            x = x[:, :reversed_x.shape[-1]]

            # set EOS as 2
            eos_indices = (x != 0).sum(dim=-1) + 1
            eos_mask = eos_indices[:, None] == torch.arange(reversed_x.shape[-1], device=device)[None, :]
            x = x.masked_fill(eos_mask, 2)
            reversed_x = reversed_x.masked_fill(eos_mask, 2)

            x, labels = x[:, :-1], x[:, 1:]
            reversed_x, reversed_labels = reversed_x[:, :-1], reversed_x[:, 1:]

            # concatenate original sequence and reversed sequence, and duplicate encoded memories for cross attention
            x = torch.cat((x, reversed_x), dim=0)
            encoded = torch.cat((encoded, encoded), dim=1)

        # embed tokens and add absolute positions
        x = self.token_emb(x)

        # positional embeddings
        pos_emb = self.pos_emb(torch.arange(x.shape[-2], device=device))
        x = x + pos_emb[None, :, :]

        rot_pos_emb = self.rot_pos_emb(x)

        # transformer layers
        x = x.transpose(0, 1)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, rot_pos_emb=rot_pos_emb) + x
            x = cross_attn(x, encoded) + x
            x = ff(x) + x

        x = x.transpose(0, 1)
        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = logits.transpose(1, 2)
        forward_logits, reversed_logits = logits.chunk(2, dim=0)

        # calculate forward and backward cross-entropy losses
        forward_loss = F.cross_entropy(forward_logits, labels, ignore_index=0)
        backward_loss = F.cross_entropy(reversed_logits, reversed_labels, ignore_index=0)
        loss = (forward_loss + backward_loss) * 0.5

        # return loss, weighted
        return loss * self.loss_weight

@register
class LayerScale(Module):
    """ https://arxiv.org/abs/2103.17239 """

    def __init__(self, features, eps=1e-5):
        super().__init__()
        scale = torch.zeros(1, 1, features).fill_(eps)
        self.scale = nn.Parameter(scale)

    def forward(self, x):
        return self.scale * x

@register
class FeedForward(Module):

    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.LayerNorm(dim),
            LayerScale(dim)
        )

    def forward(self, x):
        return self.net(x)

@register
class SHABlock(Module):
    """ https://arxiv.org/abs/1911.11423 """

    def __init__(self, dim, attn_dropout=0., ff_dropout=0., num_attn_heads=1, sha_sandwich_norm=False, ff_mult=4):
        super().__init__()
        self.attn_query_norm = nn.LayerNorm(dim)
        self.attn_kv_norm = nn.LayerNorm(dim)

        is_multiheaded = num_attn_heads > 1

        if is_multiheaded:
            self.attn = MHA(dim=dim, dropout=attn_dropout, heads=num_attn_heads)
        else:
            self.attn = SHA(dim=dim, dropout=attn_dropout, sha_sandwich_norm=sha_sandwich_norm)

        self.ff = FeedForward(dim=dim, dropout=ff_dropout, mult=ff_mult)

    def forward(self, x):
        kv = self.attn_kv_norm(x)
        q = self.attn_query_norm(x)

        x = self.attn(q, kv) + x
        x = self.ff(x) + x
        return x

@register
class Permute(Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}


def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5):
    x = torch.empty(size + (num_resample,), dtype=torch.float32, device=device).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)


class RNNWrapper(Module):
    def __init__(
            self, rnn_type, *args, reverse=False, orthogonal_weight_init=True, disable_state_bias=True, bidirectional=False, **kwargs
    ):
        super().__init__()
        if reverse and bidirectional:
            raise Exception("'reverse' and 'bidirectional' should not both be set to True")
        self.reverse = reverse
        self.rnn = rnn_type(*args, bidirectional=bidirectional, **kwargs)
        self.init_orthogonal(orthogonal_weight_init)
        self.init_biases()
        if disable_state_bias: self.disable_state_bias()

    def forward(self, x):
        if self.reverse: x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse: y = y.flip(0)
        return y

    def init_biases(self, types=('bias_ih',)):
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types):
                with torch.no_grad():
                    param.set_(0.5*truncated_normal(param.shape, dtype=param.dtype, device=param.device))

    def init_orthogonal(self, types=True):
        if not types: return
        if types == True: types = ('weight_ih', 'weight_hh')
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types):
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i:i+self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if 'bias_hh' in name:
                x.requires_grad = False
                x.zero_()


@register
class LSTM(RNNWrapper):

    def __init__(self, size, insize, bias=True, reverse=False):
        super().__init__(torch.nn.LSTM, size, insize, bias=bias, reverse=reverse)

    def to_dict(self, include_weights=False):
        res = {
            'size': self.rnn.hidden_size,
            'insize': self.rnn.input_size,
            'bias': self.rnn.bias,
            'reverse': self.reverse,
        }
        if include_weights:
            res['params'] = {
                'iW': self.rnn.weight_ih_l0.reshape(4, self.rnn.hidden_size, self.rnn.input_size),
                'sW': self.rnn.weight_hh_l0.reshape(4, self.rnn.hidden_size, self.rnn.hidden_size),
                'b': self.rnn.bias_ih_l0.reshape(4, self.rnn.hidden_size)
            }
        return res


def to_dict(layer, include_weights=False):
    if hasattr(layer, 'to_dict'):
        return {'type': layer.name, **layer.to_dict(include_weights)}
    return {'type': layer.name}


def from_dict(model_dict, layer_types=None):
    model_dict = model_dict.copy()
    if layer_types is None:
        layer_types = layers
    type_name = model_dict.pop('type')
    typ = layer_types[type_name]
    if 'sublayers' in model_dict:
        sublayers = model_dict['sublayers']
        model_dict['sublayers'] = [
            from_dict(x, layer_types) for x in sublayers
        ] if isinstance(sublayers, list) else from_dict(sublayers, layer_types)
    try:
        layer = typ(**model_dict)
    except Exception as e:
        raise Exception(f'Failed to build layer of type {typ} with args {model_dict}') from e
    return layer
