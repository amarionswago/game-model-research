"""Causal learned probability mixture and exact recomputed chain-rule gradients.

The independent control trains its adapters on separate NLLs and trains the
same gate on the mixture NLL, with no gate-to-adapter gradient. The joint arm
includes that gradient. No launch, model loading, or data selection here.
"""
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MixingGate(torch.nn.Module):
    def __init__(self, hidden_size, *, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, 2*hidden_size+1, device=device, dtype=torch.float32))

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[0] != 2:
            raise ValueError('Gate expects [two members, target positions, hidden size]')
        x = hidden.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        x = torch.cat([x[0], x[1], torch.ones_like(x[0, :, :1])], -1)
        return F.linear(x, self.weight).squeeze(-1)


def mixture(logps, hidden, gate=None):
    if logps.ndim != 2 or logps.shape[0] != 2 or hidden.shape[:2] != logps.shape:
        raise ValueError('Aligned [member, target] probabilities and hidden states required')
    if gate is None:
        weights = torch.full_like(logps, -math.log(2))
    else:
        a = gate(hidden)
        weights = torch.stack([F.logsigmoid(a), F.logsigmoid(-a)])
    weighted = logps + weights
    return weighted.logsumexp(0), weighted.softmax(0), weights.exp()


def selected(runtime, member, record):
    runtime.model.set_adapter('member'+str(member))
    encoded = record['encoded']
    device = runtime.input_device
    ids = torch.tensor([encoded['input_ids']], device=device, dtype=torch.long)
    labels = torch.tensor([encoded['labels']], device=device, dtype=torch.long)
    states = runtime.backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(ids.shape[1], device=device)[None], use_cache=False).last_hidden_state
    labels = labels.to(states.device)
    mask = labels[:, 1:] != -100
    hidden = states[:, :-1][mask]
    targets = labels[:, 1:][mask]
    def head_scores(h, y):
        logits = runtime.head(h).float()
        return logits.gather(-1, y[:, None]).squeeze(-1)-logits.logsumexp(-1)
    values = []
    for start in range(0, targets.numel(), 64):
        h, y = hidden[start:start+64], targets[start:start+64]
        values.append(checkpoint(head_scores, h, y, use_reentrant=False)
                      if torch.is_grad_enabled() else head_scores(h, y))
    logps = torch.cat(values)
    if logps.numel() != encoded['supervised_tokens']:
        raise ValueError('Complete target count changed')
    if not torch.isfinite(logps).all() or not torch.isfinite(hidden).all():
        raise FloatingPointError('Nonfinite target probability or gate input')
    return logps, hidden


def recomputed_backward(forward, *, mode, gate=None, denominator, single_member=0, replay_atol=1e-3):
    if mode not in ('joint_gate', 'independent_gate', 'joint_uniform', 'single'):
        raise ValueError('Unknown arm')
    if (gate is not None) != (mode in ('joint_gate', 'independent_gate')):
        raise ValueError('Gate does not match arm')
    if mode == 'single':
        logp, hidden = forward(single_member)
        if denominator < logp.numel():
            raise ValueError('Invalid denominator')
        (-logp.sum()/denominator).backward()
        return {'logp': logp.detach().cpu(), 'gate_mean': None, 'responsibility_mean': None}
    with torch.no_grad():
        captured = [forward(i) for i in (0, 1)]
        logs = torch.stack([x[0] for x in captured]).detach()
        states = torch.stack([x[1] for x in captured]).detach()
    if denominator < logs.shape[1]:
        raise ValueError('Invalid denominator')
    # Small leaf graph computes all covectors, including the gate's path through
    # member hidden states. Recompute each backbone separately to bound memory.
    lp = logs.clone().requires_grad_(True)
    hs = states.clone().requires_grad_(True)
    combined, responsibilities, weights = mixture(lp, hs, gate)
    loss = -combined.sum()/denominator
    variables = [lp, hs] + ([] if gate is None else list(gate.parameters()))
    gradients = torch.autograd.grad(loss, variables, allow_unused=True)
    for parameter, gradient in zip([] if gate is None else gate.parameters(), gradients[2:]):
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError('Invalid gate gradient')
        parameter.grad = gradient.detach().clone() if parameter.grad is None else parameter.grad + gradient.detach()
    for i in (0, 1):
        actual_lp, actual_h = forward(i)
        if not torch.allclose(actual_lp.detach(), logs[i], rtol=0, atol=replay_atol):
            raise RuntimeError('Recomputed target probabilities changed')
        if not torch.allclose(actual_h.detach(), states[i], rtol=0, atol=replay_atol):
            raise RuntimeError('Recomputed causal gate inputs changed')
        if mode == 'independent_gate':
            (-actual_lp.sum()/(2*denominator)).backward()
        elif gradients[1] is None:
            actual_lp.backward(gradients[0][i])
        else:
            torch.autograd.backward((actual_lp, actual_h), (gradients[0][i], gradients[1][i]))
    return {'logp': combined.detach().cpu(), 'gate_mean': weights.detach().mean(1).cpu().tolist(),
            'responsibility_mean': responsibilities.detach().mean(1).cpu().tolist()}


@torch.no_grad()
def record_logp(runtime, record, *, mode, gate, single_member=0):
    if mode == 'single':
        return selected(runtime, single_member, record)[0]
    values = [selected(runtime, i, record) for i in (0, 1)]
    return mixture(torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values]), gate)[0]


class StepRejected(RuntimeError):
    pass


def update(runtime, optimizers, records, *, mode, gate, single_member=0, policy_probe=None, progress=None):
    started = time.monotonic()
    runtime.model.train()
    members = [single_member] if mode == 'single' else [0, 1]
    parameters = [p for i in members for p in runtime.groups[i].values()]
    if gate is not None:
        parameters += list(gate.parameters())
    if any(p.grad is not None for g in runtime.groups for p in g.values()) or any(p.grad is not None for p in parameters):
        raise ValueError('Uncleared gradients')
    denominator = sum(r['encoded']['supervised_tokens'] for r in records)
    before, diagnostics = [], []
    distribution_before = None
    if policy_probe is not None:
        with torch.no_grad():
            distribution_before = policy_probe().detach().cpu()
        runtime.model.train()
    try:
        for n, record in enumerate(records):
            result = recomputed_backward(lambda i:selected(runtime, i, record), mode=mode, gate=gate,
                                         denominator=denominator, single_member=single_member)
            before.append(result.pop('logp')); diagnostics.append(result)
            if progress:
                progress(completed_records=n+1, records=len(records), target_tokens=denominator)
        for p in parameters:
            if p.grad is None or p.grad.dtype != torch.float32 or not torch.isfinite(p.grad).all():
                raise FloatingPointError('Missing/non-FP32/nonfinite intended gradient')
        member_norms = [float(torch.linalg.vector_norm(torch.stack([p.grad.norm().to(parameters[0].device)
                       for p in runtime.groups[i].values()]))) for i in members]
        if min(member_norms) <= 0:
            raise FloatingPointError('Missing member learning signal')
        gate_norm = None if gate is None else float(gate.weight.grad.norm())
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        for optimizer in optimizers:
            optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Nonfinite post-update parameter')
        runtime.model.eval()
        after = [record_logp(runtime, r, mode=mode, gate=gate, single_member=single_member).cpu() for r in records]
        delta = torch.cat(after)-torch.cat(before)
        drift = {'max_absolute_target_logp_change':float(delta.abs().max()),
                 'mean_absolute_target_logp_change':float(delta.abs().mean())}
        change = None
        if policy_probe is not None:
            with torch.no_grad():
                distribution_after = policy_probe().detach().cpu()
            if not torch.isfinite(distribution_after).all():
                raise FloatingPointError('Nonfinite policy probe')
            change = {'max_full_vocabulary_KL':float((distribution_before.exp()*(distribution_before-distribution_after)).sum(-1).max()),
                      'max_probability_difference':float((distribution_before.exp()-distribution_after.exp()).abs().max())}
        if drift['max_absolute_target_logp_change'] > 1. or drift['mean_absolute_target_logp_change'] > .25:
            raise StepRejected('Target likelihood trust limit exceeded: '+str(drift))
        if change is not None and (change['max_full_vocabulary_KL'] > .01 or change['max_probability_difference'] > .05):
            raise StepRejected('Policy distribution trust limit exceeded: '+str(change))
        return {'target_tokens':denominator, 'records':len(records), 'complete_EOS_targets':len(records),
                'loss_before':float(-torch.cat(before).mean()), 'loss_after':float(-torch.cat(after).mean()),
                'member_gradient_norms':member_norms, 'gate_gradient_norm':gate_norm,
                'gradient_norm_before_clip':norm, 'learning_rates':[o.param_groups[0]['lr'] for o in optimizers],
                'target_likelihood_change':drift, 'policy_probe_change':change,
                'gate_diagnostics':diagnostics, 'seconds':time.monotonic()-started}
    finally:
        for group in runtime.groups:
            for p in group.values():
                p.grad = None
        if gate is not None:
            for p in gate.parameters():
                p.grad = None
