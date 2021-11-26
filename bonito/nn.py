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
import torch.cuda.amp as amp

layers = {}


def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer
    return layer


def stable_softmax(x, dim=-1, alpha=128):
    # stable softmax technique from https://arxiv.org/abs/2105.13290
    x = x / alpha
    x = x - torch.amax(x, dim=dim, keepdim=True).detach()
    return (x * alpha).softmax(dim=dim)


register(torch.nn.ReLU)
register(torch.nn.Tanh)


@register
class Swish(torch.nn.SiLU):
    pass


@register
class StableLayerNorm(nn.Module):
    """ stable layernorm technique from https://arxiv.org/abs/2105.13290 """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x / torch.amax(x, dim=-1, keepdim=True).detach()
        return self.norm(x)


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
    """ single-head attention from https://arxiv.org/abs/1911.11423 """

    def __init__(self, dim, dropout=0.):
        """
        Parameters:
            dim (int): feature dimension
            dropout (float): attention dropout
        """

        super().__init__()
        self.scale = dim ** -0.5
        self.to_q = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)
        self.bottom_sandwich_norm = nn.LayerNorm(dim)

        # zero init final layer, which is the gamma and beta of the post-layernorm
        last_layer = self.bottom_sandwich_norm
        nn.init.constant_(last_layer.weight, 0.)
        nn.init.constant_(last_layer.bias, 0.)

    def forward(self, x, kv):
        """
        Parameters:
            x (tensor): <seq, batch, dimension> input tensor
            kv (tensor): <seq, batch, dimension> contextual tensor - pass in the input tensor here for it to be self attention, otherwise it is cross attention
        Returns:
            out (tensor): <seq, batch, dimension>
        """

        x = x.transpose(0, 1)
        kv = kv.transpose(0, 1)

        # derive queries
        q = self.to_q(x)
        q = q * self.scale

        # measure similarity of queries to keys
        sim = torch.matmul(q, kv.transpose(-1, -2))

        # attention
        attn = stable_softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # aggregate values
        out = torch.matmul(attn, kv)
        out = out.transpose(0, 1)

        # final post-layernorm, from sandwich norm
        out = self.bottom_sandwich_norm(out)
        return out

@register
class MHA(Module):
    """ classic multi-head attention """

    def __init__(self, dim, heads=4, dim_head=64, dropout=0., causal=False, norm_inputs=False, kv_input_dim=None, use_scaled_cosine_sim_attn=False, init_learned_scale=-5.):
        """
        Parameters:
            dim (int): feature dimension
            heads (int): number of attention heads
            dim_head (int): dimension per attention head
            dropout (float): attention dropout
            causal (bool): autoregressive or not, will add causal masking if true
            norm_inputs (bool): whether to layernorm the inputs, as in the pre-layernorm architecture
            kv_input_dim (int): separate dimension for the input of the key / values, in the case of cross attending to encoder feature maps that have a greater dimension than the query embedding dimension
            use_scaled_cosine_sim_attn (bool): whether to use scaled cosine similarity attention
            init_learned_scale (float): initial learned temperature for cosine similarity attention, in logspace
        """

        super().__init__()
        inner_dim = heads * dim_head
        kv_input_dim = dim if kv_input_dim is None else kv_input_dim

        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal

        # whether to use scaled cosine similarity attention
        self.use_scaled_cosine_sim_attn = use_scaled_cosine_sim_attn
        if use_scaled_cosine_sim_attn:
            self.learned_scale = nn.Parameter(torch.ones(1, heads, 1, 1) * init_learned_scale)
        else:
            self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_input_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_input_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim) if norm_inputs else nn.Identity()

        # zero init final layer, which is the gamma and beta of the post-layernorm
        last_layer = self.to_out[-1]
        nn.init.constant_(last_layer.weight, 0.)
        nn.init.constant_(last_layer.bias, 0.)

    def forward(self, x, kv=None, rot_pos_emb=None):
        """
        Parameters:
            x (tensor): <seq, batch, dimension> input tensor
            kv (tensor, optional): <seq, batch, dimension> contextual tensor - if not passed in, will be self-attention
            rot_pos_emb (tensor, optional): <batch, seq, freq_dimension> rotary positional embedding for self attention - only applied if passed in
        Returns:
            out (tensor): <seq, batch, dimension>
        """

        n, b, d, h, device = *x.shape, self.heads, x.device

        x = self.norm(x)
        kv = x if kv is None else kv

        x = x.transpose(0, 1)
        kv = kv.transpose(0, 1)

        # derive queries, keys, values
        q, k, v = self.to_q(x), self.to_k(kv), self.to_v(kv)

        # split heads
        q, k, v = map(lambda t: t.reshape(b, -1, h, self.dim_head).transpose(1, 2), (q, k, v))

        if self.use_scaled_cosine_sim_attn:
            # proposed by https://arxiv.org/abs/2010.04245
            # validated at scale by https://arxiv.org/abs/2111.09883v1
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            scale = 1 / (self.learned_scale.exp() + 1e-2)
        else:
            scale = self.scale

        q = q * scale

        # apply rotary embeddings, if the rotary positional frequencies are passed in
        if rot_pos_emb is not None:
            rot_pos_emb = rot_pos_emb[:, None]
            q = apply_rotary_pos_emb(rot_pos_emb, q)
            k = apply_rotary_pos_emb(rot_pos_emb, k)

        # derive similarity of queries to keys
        sim = torch.matmul(q, k.transpose(-1, -2))

        # apply causal masking, if designated on module init
        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device=device).triu(j - i + 1).bool()
            mask_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask[None, None, :, :], mask_value)

        # attention
        attn = stable_softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # aggregate values
        out = torch.matmul(attn, v)

        # merge heads and combine heads to output
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)

        out = out.transpose(0, 1)
        return out

# ISAB

@register
class ISAB(Module):
    """ induced-set attention block, from https://arxiv.org/abs/1810.00825 """

    def __init__(self, *, dim, num_latents, heads=8, dim_head=64, dropout=0.):
        """
        Parameters:
            dim (int): feature dimension
            num_latents (int): number of latents
            heads (int): number of attention heads
            dim_head (int): dimension per attention head
            dropout (float): attention dropout
        """

        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, 1, dim))
        self.norm = nn.LayerNorm(dim)

        self.attn1 = MHA(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.attn2 = MHA(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x):
        batch = x.shape[1]
        x = self.norm(x)

        latents = self.latents.expand(-1, batch, -1)
        induced = self.attn1(latents, x)
        return self.attn2(x, induced)

@register
class ISABBlock(Module):
    """ induced-set attention transformer block, from https://arxiv.org/abs/1810.00825 """

    def __init__(self, dim, attn_dropout=0., ff_dropout=0., num_attn_heads=4, dim_head=64, ff_mult=4, num_latents=6):
        """
        Parameters:
            dim (int): feature dimension
            num_latents (int): number of latents
            attn_dropout (float): attention dropout
            ff_dropout (float): feedforward dropout
            num_attn_heads (int): number of attention heads
            dim_head (int): dimension per attention head
            ff_mult (int): expansion ratio for inner dimension of feedforward, typically kept at 4
            num_latents (int): number of latents for ISAB
        """

        super().__init__()
        self.attn = ISAB(dim=dim, heads=num_attn_heads, dim_head=dim_head, num_latents=num_latents, dropout=attn_dropout)
        self.ff = FeedForward(dim=dim, dropout=ff_dropout, mult=ff_mult)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    """
    Rotary embedding - parameter-less relative positional encoding in the context of attention
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim, theta = 10000):
        """
        Parameters:
            dim (int): feature dimension (should be smaller than dimension of attention head)
            theta (int): determines the frequencies of the positional embeddings, and also determines the max timestep in the position
        """

        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """
        Parameters:
            x (tensor): <batch, seq, dimension> input tensor
        Returns:
            out (tensor): <batch, seq, freq_dimension> frequencies for all positions in the sequence
        """

        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = t[:, None] * self.inv_freq[None, :]
        freqs = torch.stack((freqs, freqs), dim=-1)
        return freqs.reshape(1, seq_len, -1)

def rotate_half(x):
    preceding_dims = x.shape[:-1]
    x = x.reshape(*preceding_dims, -1, 2)
    x1, x2 = x.unbind(dim=-1)
    out = torch.stack((-x2, x1), dim=-1)
    return out.reshape(*preceding_dims, -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t =  (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# convolutional layer for absolute positional embedding

class CausalDepthwiseConv(Module):
    """ depthwise convolution with causality and sandwich normalization """

    def __init__(self, dim, kernel_size):
        """
        Parameters:
            dim (int): feature dimension
            kernel_size (int): kernel size of the convolution
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(1, 2, 0)

        # causal padding, which is kernel size minus 1
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.)

        x = self.conv(x)
        x = x.permute(2, 0, 1)
        return self.norm_out(x)

# transformer decoder

class Decoder(Module):

    def __init__(self, dim, num_tokens=5, depth=2, heads=4, dim_head=64, loss_weight=0.25, attn_dropout=0., ff_dropout=0., conv_kernel_size=5, token_emb_grad_frac=0.2, use_self_attn=True, use_scaled_cosine_sim_attn=False, num_encoder_layers_attend=1):
                """
        Parameters:
            dim (int): feature dimension
            num_tokens (int): number of tokens
            depth (int): transformer depth
            heads (int): number of attention heads
            dim_head (int): dimension per attention head
            loss_weight (float): weight on the auxiliary forward and backward autoregressive loss
            attn_dropout (float): attention dropout
            ff_dropout (float): feedforward dropout
            conv_kernel_size (int): kernel size of the position generating causal convolution
            token_emb_grad_frac (float): what fraction (0 - 1) of the gradients to pass back to the token embeddings. Cogview paper found that transformers are more stable if the token embeddings have a fraction of the gradient of the overall transformer
            use_self_attn (bool): whether to include self attention
            use_scaled_cosine_sim_attn (bool): whether to use scaled cosine similarity attention
            num_encoder_layers_attend (int): number of penultimate encoder feature maps to include in the cross attention
        """

        super().__init__()
        self.loss_weight = loss_weight

        # 2 reserved special tokens, for <sos> and <eos>
        num_reserved_tokens = 2

        # token embedding
        self.token_emb_grad_frac = token_emb_grad_frac
        self.token_emb = nn.Embedding(num_tokens + num_reserved_tokens, dim)
        nn.init.kaiming_normal_(self.token_emb.weight)

        # number of encoder layers, as well as the layer norm
        self.num_encoder_layers_attend = num_encoder_layers_attend
        self.norm_encoder_layers = StableLayerNorm(dim)

        self.layers = nn.ModuleList([])
        self.rot_pos_emb = RotaryEmbedding(max(dim_head // 2, 32)) # partial rotary embedding needs a minimum of dimension 32 to work well

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalDepthwiseConv(dim, kernel_size=conv_kernel_size),
                MHA(dim, heads=heads, causal=True, norm_inputs=True, dropout=attn_dropout, use_scaled_cosine_sim_attn=use_scaled_cosine_sim_attn) if use_self_attn else None,
                MHA(dim, kv_input_dim=dim * num_encoder_layers_attend, heads=heads, norm_inputs=True, dropout=attn_dropout, use_scaled_cosine_sim_attn=use_scaled_cosine_sim_attn),
                FeedForward(dim, dropout=ff_dropout)
            ]))

        self.to_logits = nn.Sequential(
            StableLayerNorm(dim),
            nn.Linear(dim, num_tokens + num_reserved_tokens)
        )

    def forward(self, x, encoder_layers, return_loss=False):
        """
        Parameters:
            x (tensor): <seq, batch, dimension> input tensor
            encoder_layers (list[tensor]): <seq, batch, dimension> list of encoder feature maps to cross attend to for autoregressive auxiliary loss
            return_loss (bool): whether to return the forwards and backwards auxiliary loss, or simply return the forward logits

        Returns: (return_loss = true)
            loss (tensor): <scalar> weighted forwards and backwards autoregressive auxiliary loss

        Returns: (return_loss = false)
            logits (tensor): <batch, seq, num_tokens> decoder forward logits
        """

        device = x.device

        # prepare the encoder layers for attending
        assert len(encoder_layers) >= self.num_encoder_layers_attend, f'designated number of encoder layers to attend to (num_encoder_layers_attend) must be less than or equal to the actual number of layers'

        encoder_layers = encoder_layers[-self.num_encoder_layers_attend:]
        encoder_layers = torch.stack(encoder_layers, dim=-2)
        encoded = self.norm_encoder_layers(encoder_layers)
        encoded = encoded.flatten(2)

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

        # stability measure from https://arxiv.org/abs/2105.13290
        x = x * self.token_emb_grad_frac + x.detach() * (1 - self.token_emb_grad_frac)

        rot_pos_emb = self.rot_pos_emb(x)

        # transformer layers
        x = x.transpose(0, 1)

        for conv, self_attn, cross_attn, ff in self.layers:
            x = conv(x) + x

            if self_attn is not None:
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
        # extra insurance that it is done in float32
        with amp.autocast(enabled=False):
            forward_loss = F.cross_entropy(forward_logits, labels, ignore_index=0)
            backward_loss = F.cross_entropy(reversed_logits, reversed_labels, ignore_index=0)
            loss = (forward_loss + backward_loss) * 0.5

        # return loss, weighted
        return loss * self.loss_weight

@register
class GEGLU(Module):
    """ gating with GELU nonlinearity, proposed in https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

@register
class FeedForward(Module):
    """ feedforward with sandwich normalization, gated GELU, as well as post-activation layernorm https://openreview.net/forum?id=GMYWzWztDx5 """

    def __init__(self, dim, mult=4, dropout=0.):
        """
        Parameters:
            dim (int): feature dimension
            mult (int): expansion ratio of the inner dimension of the feedforward, as a multiplier on the input dimension (dim)
            dropout (float): feedforward dropout
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.LayerNorm(dim * mult),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.LayerNorm(dim)
        )

        last_layer = self.net[-1]
        nn.init.constant_(last_layer.weight, 0.)
        nn.init.constant_(last_layer.bias, 0.)

    def forward(self, x):
        return self.net(x)

@register
class SHABlock(Module):
    """ single head attention transformer block, from https://arxiv.org/abs/1911.11423 """

    def __init__(self, dim, attn_dropout=0., ff_dropout=0., num_attn_heads=1, ff_mult=4, dim_head=64):
        """
        Parameters:
            dim (int): feature dimension
            attn_dropout (float): attention dropout
            ff_dropout (float): feedforward dropout
            num_attn_heads (int): number of attention heads
            ff_mult (int): feedforward expansion ratio of inner dimension
        """

        super().__init__()
        self.attn_query_norm = nn.LayerNorm(dim)
        self.attn_kv_norm = nn.LayerNorm(dim)

        is_multiheaded = num_attn_heads > 1

        if is_multiheaded:
            self.attn = MHA(dim=dim, dropout=attn_dropout, heads=num_attn_heads, dim_head=dim_head)
        else:
            self.attn = SHA(dim=dim, dropout=attn_dropout)

        self.ff = FeedForward(dim=dim, dropout=ff_dropout, mult=ff_mult)

    def forward(self, x):
        kv = self.attn_kv_norm(x)    # single head attention uses separate layernorms for queries than they do key / values
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
