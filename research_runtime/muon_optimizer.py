"""Muon for the 2D attention-LoRA matrices, with FP32 arithmetic on T4.

Algorithm follows PyTorch 2.11 Muon / Keller Jordan: Nesterov momentum,
five quintic Newton-Schulz steps and match_rms_adamw shape scaling.
The scaling name is an RMS convention, not an Adam optimizer or state.
Unlike torch.optim.Muon, NS runs in FP32: T4 has no native BF16 support.
No optimizer is selected implicitly for an unsupported parameter.
"""
import math
import torch


def orthogonalize(matrix, steps=5, eps=1e-7):
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError('Muon requires a finite matrix gradient')
    x = matrix.float().clone()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445*x + (-4.7750*gram + 2.0315*(gram @ gram)) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=5e-5, momentum=.95, weight_decay=0.):
        if not (0 < lr <= .01 and 0 <= momentum < 1 and weight_decay == 0):
            raise ValueError('Unregistered Muon settings')
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=0.,
                                     nesterov=True, ns_steps=5, eps=1e-7,
                                     adjust_lr_fn='match_rms_adamw', ns_dtype='float32'))
        if any(p.ndim != 2 or p.dtype != torch.float32 for g in self.param_groups for p in g['params']):
            raise ValueError('Every trained LoRA parameter must be a FP32 matrix')

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not used by this registered trainer')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    raise RuntimeError('Missing Muon gradient')
                if p.grad.is_sparse or not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Invalid Muon gradient')
                state = self.state[p]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['step'] = 0
                buf = state['momentum_buffer']
                buf.lerp_(p.grad, 1-group['momentum'])
                direction = p.grad.lerp(buf, group['momentum'])
                update = orthogonalize(direction, group['ns_steps'], group['eps'])
                if not torch.isfinite(update).all():
                    raise FloatingPointError('Nonfinite Muon direction')
                scale = .2 * math.sqrt(max(p.shape))
                p.add_(update, alpha=-group['lr']*scale)
                state['step'] += 1
