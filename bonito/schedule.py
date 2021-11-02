import math

import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def linear_warmup_cosine_decay(end_ratio=0.1, **kwargs):
    """
    Linear warmup, cosine decay scheduler
    """
    return lambda optimizer, train_loader, epochs, last_epoch: func_scheduler(
        optimizer=optimizer,
        func=cosine_decay_schedule(1.0, end_ratio),
        total_steps=epochs * len(train_loader),
        warmup_steps=500,
        start_step=last_epoch*len(train_loader)
    )


def linear_warmup_const_inverse_sqrt_decay(warmup_steps, decay_start_step, scale, **kwargs):
    """
    Linear warmup, hold constant, inverse sqrt decay scheduler
    """
    def gen_sched(optimizer, train_loader, epochs, last_epoch):
        def sched(step):
            step = step + last_epoch*len(train_loader)
            if step < decay_start_step:
                return min(1.0, 0.1 + 0.9*step/warmup_steps)
            return 1.0 / math.sqrt(1 + ((step - decay_start_step) / scale))
        return LambdaLR(optimizer, sched)
    return gen_sched


def linear_cooldown(end_ratio, **kwargs):
    """
    Linear Cooldown Scheduler
    """
    return lambda optimizer, train_loader, epochs, last_epoch: func_scheduler(
        optimizer=optimizer,
        func=linear_schedule(1.0, end_ratio),
        total_steps=epochs * len(train_loader),
        start_step=0,
    )


def cosine_cooldown(end_ratio, **kwargs):
    """
    Cosine Cooldown Scheduler
    """
    return lambda optimizer, train_loader, epochs, last_epoch: func_scheduler(
        optimizer=optimizer,
        func=cosine_decay_schedule(1.0, end_ratio),
        total_steps=epochs * len(train_loader),
        warmup_steps=None,
        start_step=0
    )


def const_cooldown(**kwargs):
    return lambda optimizer, train_loader, epochs, last_epoch: func_scheduler(
        optimizer=optimizer,
        func=const_schedule(1.0),
        total_steps=epochs * len(train_loader),
        warmup_steps=None,
        start_step=0
    )



#-------------------------------------------------------------------------------


def const_schedule(y):
    """
    Constant Scheduler
    """
    return lambda t: y


def linear_schedule(y0, y1):
    """
    Linear Scheduler
    """
    return lambda t: y0 + (y1 - y0) * t


def cosine_decay_schedule(y0, y1):
    """
    Cosine Decay Scheduler
    """
    return lambda t: y1 + 0.5 * (y0 - y1) * (np.cos(t * np.pi) + 1.0)


def piecewise_schedule(knots, funcs):
    """
    Piecewise Scheduler
    """
    def f(t):
        i = np.searchsorted(knots, t)
        t0 = 0.0 if i == 0 else knots[i - 1]
        t1 = 1.0 if i == len(knots) else knots[i]
        return funcs[i]((t - t0) / (t1 - t0))
    return f


def func_scheduler(optimizer, func, total_steps, warmup_steps=None, warmup_ratio=0.1, start_step=0):
    """
    Learning Rate Scheduler
    """
    if warmup_steps:
        y0 = func(0.0)
        func = piecewise_schedule(
            [warmup_steps / total_steps],
            [linear_schedule(warmup_ratio * y0, y0), func]
        )
    return LambdaLR(optimizer, (lambda step: func((step + start_step) / total_steps)))
