"""
Bonito CTC-CRF Model.
"""

import torch
import numpy as np
from bonito.nn import Permute, Scale, activations, rnns
from torch.nn import Sequential, Module, Linear, Tanh, Conv1d

import seqdist.sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring


def get_stride(m):
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Sequential):
        return int(np.prod([get_stride(x) for x in m]))
    return 1


class SeqdistModel(Module):
    def __init__(self, encoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.global_norm = GlobalNorm(seqdist)
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x):
        return self.global_norm(self.encoder(x))

    def decode_batch(self, x):
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]


def rnn_encoder(outsize, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0):
    activation = activations[activation]()
    rnn = rnns[rnn_type]

    return Sequential(
            conv(insize, 4, ks=5, bias=True), activation,
            conv(4, 16, ks=5, bias=True), activation,
            conv(16, features, ks=winlen, stride=stride, bias=True), activation,
            Permute(2, 0, 1),
            rnn(features, features, reverse=True), rnn(features, features),
            rnn(features, features, reverse=True), rnn(features, features),
            rnn(features, features, reverse=True),
            Linear(features, outsize, bias=True),
            Tanh(),
            Scale(scale),
        )

class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        encoder = rnn_encoder(
            outsize=seqdist.n_score(),
            insize=config['input']['features'],
            **config['encoder']
        )
        super().__init__(encoder, seqdist)
        self.config = config


def conv(c_in, c_out, ks, stride=1, bias=False, dilation=1, groups=1):
    if stride > 1 and dilation > 1:
        raise ValueError("Dilation and stride can not both be greater than 1")
    return Conv1d(
        c_in, c_out, ks, stride=stride, padding=(ks // 2) * dilation,
        bias=bias, dilation=dilation, groups=groups
    )


class GlobalNorm(Module):

    def __init__(self, seq_dist):
        super().__init__()
        self.seq_dist = seq_dist

    def forward(self, x):
        scores = x.to(torch.float32)
        return (scores - self.seq_dist.logZ(scores)[:, None] / len(scores)).to(x.dtype)


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

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean'):
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