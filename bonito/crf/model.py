"""
Bonito CTC-CRF Model.
"""

import torch
import numpy as np
from bonito.nn import Module, Convolution, SHABlock, ISABBlock, LinearCRFEncoder, Serial, Permute, layers, Decoder, from_dict

import seqdist.sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring

from functools import partial, wraps
from collections import Counter

def cache_on_first_run(fn):
    """ decorator for a function, whereby the function will only compute once on first execute, and the return value is cached for all subsequent executions """

    cached = None
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal cached
        if cached is not None:
            return cached
        output = fn(*args, **kwargs)
        cached = output
        return output
    return inner

def get_stride(m):
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1


class CTC_CRF(SequenceDist):

    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S:semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        #convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


class Encoder(Module):
    """
    An encoder wrapper class that accepts the convolutional encoder as well as the list of modules to execute sequentially (backbone)
    It will pipe the tensor through the convolutional encoder and backbone, and return the final output as well as the intermediate feature maps
    """

    def __init__(self, conv_encoder, backbone):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.backbone = backbone

    def forward(self, x):
        x = self.conv_encoder(x)

        fmaps = [x]
        for layer in self.backbone:
            x = layer(x)

            # only save feature maps for outputs from LSTMs
            if isinstance(x, (SHABlock, ISABBlock)):
                continue

            fmaps.append(x)

        return x, fmaps

def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation)


def rnn_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, attn_layers=[], num_attn_heads=1, dim_attn_head=64, attn_dropout=0., ff_dropout=0., use_isab_attn=False, isab_num_latents=6, weight_tie_attn_blocks=False):
    rnn = layers[rnn_type]

    rnns = [
        rnn(features, features, reverse=True), rnn(features, features),
        rnn(features, features, reverse=True), rnn(features, features),
        rnn(features, features, reverse=True)
    ]

    backbone = nn.ModuleList([])
    attn_layers_count = Counter(attn_layers) # allows for multiple attention blocks per layer

    attn_klass = SHABlock if not use_isab_attn else partial(ISABBlock, num_latents=isab_num_latents)

    # weight tie attention block parameters across all layers
    if weight_tie_attn_blocks:
        attn_klass = cache_on_first_run(attn_klass)

    for layer, rnn in enumerate(rnns):
        layer_num = layer + 1
        backbone.append(rnn)

        # add attention block(s) if the layer number is in attn_layers
        if layer_num in attn_layers_count:
            backbone.extend([attn_klass(features, attn_dropout=attn_dropout, ff_dropout=ff_dropout, num_attn_heads=num_attn_heads, dim_head=dim_attn_head) for _ in range(attn_layers_count[layer_num])])

    conv_encoder = Serial([
        conv(insize, 4, ks=5, bias=True, activation=activation),
        conv(4, 16, ks=5, bias=True, activation=activation),
        conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
        Permute([2, 0, 1])
    ])

    encoder = Encoder(conv_encoder, backbone)
    linear_crf = LinearCRFEncoder(features, n_base, state_len, bias=True, activation='tanh', scale=scale, blank_score=blank_score)

    return encoder, linear_crf

class SeqdistModel(Module):
    def __init__(self, encoder, linear_crf, decoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.decoder = decoder
        self.linear_crf = linear_crf
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x, targets=None, no_aux_loss=False):
        encoded, layer_fmaps = self.encoder(x)
        scores = self.linear_crf(encoded)
        scores = scores.to(torch.float32)

        if targets is None:
            return scores

        if self.decoder is None or no_aux_loss:
            return scores, torch.tensor([0], device=x.device)

        aux_loss = self.decoder(targets, layer_fmaps, return_loss=True)
        return scores, aux_loss

    def decode_batch(self, x):
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]


class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        if 'type' in config['encoder']: #new-style config
            encoder = from_dict(config['encoder'])
        else: #old-style
            encoder, linear_crf = rnn_encoder(seqdist.n_base, seqdist.state_len, insize=config['input']['features'], **config['encoder'])
            decoder = Decoder(config['encoder']['features'], **config['aux_decoder']) if config['aux_decoder']['loss_weight'] > 0 else None
        super().__init__(encoder, linear_crf, decoder, seqdist)
        self.config = config
