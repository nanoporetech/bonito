"""
Bonito CTC-CRF Model.
"""

import torch
import numpy as np

import koi.lstm
from koi.ctc import SequenceDist, Max, Log, semiring
from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse

from bonito.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, to_dict, from_dict, register


def get_stride(m, stride=1):
    if hasattr(m, "output_stride"):
        stride = m.output_stride(stride)
    elif hasattr(m, "stride"):
        s = m.stride
        if isinstance(s, tuple):
            assert len(s) == 1
            s = s[0]
        stride = stride * s
    else:
        for child in m.children():
            stride = get_stride(child, stride)
    return stride


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
        return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

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
        a_traceback = traceback.argmax(2)
        moves = (a_traceback % len(self.alphabet)) != 0
        paths = 1 + (torch.div(a_traceback, len(self.alphabet), rounding_mode="floor") % self.n_base)
        return torch.where(moves, paths, 0)

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
        logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
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


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None, norm=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation, norm=norm)


def rnn_encoder(n_base, state_len, insize=1, first_conv_size=4, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5, norm=None):
    rnn = layers[rnn_type]
    return Serial([
        conv(insize, first_conv_size, ks=5, bias=True, activation=activation, norm=norm),
        conv(first_conv_size, 16, ks=5, bias=True, activation=activation, norm=norm),
        conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation, norm=norm),
        Permute([2, 0, 1]),
        *(rnn(features, features, reverse=(num_layers - i) % 2) for i in range(num_layers)),
        LinearCRFEncoder(
            features, n_base, state_len, activation='tanh', scale=scale,
            blank_score=blank_score, expand_blanks=expand_blanks
        )
    ])

@register
class SeqdistModel(Module):
    def __init__(self, encoder, seqdist, n_pre_post_context_bases=None, target_projection=None):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

        if n_pre_post_context_bases is None:
            self.n_pre_context_bases = self.seqdist.state_len - 1
            self.n_post_context_bases = 1
        else:
            self.n_pre_context_bases, self.n_post_context_bases = n_pre_post_context_bases

        if target_projection is None:
            self.target_projection = None
        else:
            self.register_buffer('target_projection', torch.tensor([0] + target_projection), persistent=False)

    @classmethod
    def from_dict(cls, model_dict, layer_types=None):
        kwargs = dict(
            model_dict,
            encoder=from_dict(model_dict["encoder"], layer_types),
            seqdist=CTC_CRF(**model_dict["seqdist"])
        )
        return cls(**kwargs)

    def forward(self, x, *args):
        return self.encoder(x)

    def decode_batch(self, x):
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]

    def loss(self, scores, targets, target_lengths, **kwargs):
        if self.target_projection is not None:
            targets = self.target_projection[targets]
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)

    def use_koi(self, **kwargs):
        pass

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        res = {
            "encoder": to_dict(self.encoder),
            "seqdist": {"state_len": self.seqdist.state_len, "alphabet": self.seqdist.alphabet},
            "n_pre_post_context_bases": (self.n_pre_context_bases, self.n_post_context_bases),
        }
        if self.target_projection is not None:
            res["target_projection"] = self.target_projection.tolist()[1:]
        return res


class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        if 'type' in config['encoder']: #new-style config
            encoder = from_dict(config['encoder'])
        else: #old-style
            encoder = rnn_encoder(seqdist.n_base, seqdist.state_len, insize=config['input']['features'], **config['encoder'])

        super().__init__(encoder, seqdist, n_pre_post_context_bases=config['input'].get('n_pre_post_context_bases'))
        self.config = config

    def use_koi(self, **kwargs):
        self.encoder = koi.lstm.update_graph(
            self.encoder,
            batchsize=kwargs["batchsize"],
            chunksize=kwargs["chunksize"] // self.stride,
            quantize=kwargs["quantize"],
        )
