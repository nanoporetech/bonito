import logging
import types
from functools import lru_cache

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.modules.mlp import GatedMlp
    from flash_attn.ops.triton.layer_norm import RMSNorm
except ImportError:
    logger.warning(
        "please install flash-attn to use the transformer module: "
        "`pip install flash-attn --no-build-isolation`"
    )

from bonito.crf.model import SeqdistModel
from bonito.nn import from_dict, register, LinearCRFEncoder, MakeContiguous, Permute, Serial


def deepnorm_params(depth):
    """
    Returns the DeepNorm (https://arxiv.org/abs/2203.00555) alpha and beta parameters.
    """
    alpha = round((2*depth)**0.25, 7)
    beta = round((8*depth)**(-1/4), 7)
    return alpha, beta


@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, qkv_bias=False, out_bias=True, rotary_dim=None, attn_window=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)

    def attn_func(self, qkv):
        if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (torch.is_autocast_enabled() or qkv.dtype == torch.half):
            attn_output = flash_attn_qkvpacked_func(qkv, window_size=self.attn_window)
        else:
            q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
            mask = sliding_window_mask(qkv.shape[1], self.attn_window, q.device)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.permute(0, 1, 3, 2, 4)
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape

        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)

        qkv = self.rotary_emb(qkv)

        attn_output = self.attn_func(qkv).reshape(N, T, self.d_model)

        out = self.out_proj(attn_output)

        return out


@register
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window=None):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "attn_window": attn_window
        }

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            qkv_bias=False,
            out_bias=True,
            attn_window=attn_window
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        x = self.norm1(self.self_attn(x), self.deepnorm_alpha*x)
        x = self.norm2(self.ff(x), self.deepnorm_alpha*x)
        return x

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs


def use_koi(self, **kwargs):
    # koi needs modified LinearCRFLayer settings
    def _expand_blanks(m):
        if isinstance(m, LinearCRFEncoder):
            m.expand_blanks = False
    self.encoder.apply(_expand_blanks)
    self.encoder = Serial([
        self.encoder,
        Permute([1, 0, 2]),
        MakeContiguous(),
    ])


def Model(config):
    model_config = {k: v for k, v in config["model"].items() if k != "package"}
    model = from_dict(model_config)
    model.config = config
    model.use_koi = types.MethodType(use_koi, model)
    return model
