"""Causal learned probability mixture and exact recomputed chain-rule gradients.

The independent control trains its adapters on separate NLLs and trains the
same gate on the mixture NLL, with no gate-to-adapter gradient. The joint arm
includes that gradient. No launch, model loading, or data selection here.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def validate_members(members):
    members = tuple(members)
    if len(members) not in (2, 4) or len(set(members)) != len(members) or any(
            type(m) is not int or m < 0 for m in members):
        raise ValueError('Two or four distinct nonnegative member IDs required')
    return members


class MixingGate(torch.nn.Module):
    def __init__(self, hidden_size, *, members=(0, 1), device=None):
        super().__init__()
        self.members = validate_members(members)
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.zeros(len(self.members)-1,
            len(self.members)*hidden_size+1, device=device, dtype=torch.float32))

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[0] != len(self.members) or hidden.shape[2] != self.hidden_size:
            raise ValueError('Gate expects aligned [members, target positions, hidden size]')
        x = hidden.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        x = torch.cat([*x.unbind(0), torch.ones_like(x[0, :, :1])], -1)
        logits = F.linear(x, self.weight)
        return logits.squeeze(-1) if len(self.members) == 2 else logits

    def log_weights(self, hidden):
        logits = self(hidden)
        if len(self.members) == 2:
            return torch.stack([F.logsigmoid(logits), F.logsigmoid(-logits)])
        return torch.cat([logits, torch.zeros_like(logits[:, :1])], -1).log_softmax(-1).T


def mixture(logps, hidden, gate=None):
    if logps.ndim != 2 or logps.shape[0] not in (2, 4) or hidden.shape[:2] != logps.shape:
        raise ValueError('Aligned [member, target] probabilities and hidden states required')
    if not torch.isfinite(logps).all() or not torch.isfinite(hidden).all():
        raise FloatingPointError('Nonfinite probability or gate input')
    weights = (torch.full_like(logps, -math.log(logps.shape[0])) if gate is None
               else gate.log_weights(hidden))
    if not torch.isfinite(weights).all():
        raise FloatingPointError('Nonfinite gate weights')
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


def recomputed_backward(forward, *, mode, gate=None, denominator, members=(0, 1), single_member=0, replay_atol=1e-3):
    """Accumulate gradients; the caller must discard ALL gradients on any error."""
    members = validate_members(members)
    if gate is not None and gate.members != members:
        raise ValueError("Gate/member order mismatch")
    if not math.isfinite(denominator) or denominator <= 0 or replay_atol < 0 or not math.isfinite(replay_atol):
        raise ValueError("Invalid denominator or replay tolerance")
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
        captured = [forward(i) for i in members]
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
    for i, member in enumerate(members):
        actual_lp, actual_h = forward(member)
        if not torch.allclose(actual_lp.detach(), logs[i], rtol=0, atol=replay_atol):
            raise RuntimeError('Recomputed target probabilities changed')
        if not torch.allclose(actual_h.detach(), states[i], rtol=0, atol=replay_atol):
            raise RuntimeError('Recomputed causal gate inputs changed')
        if mode == 'independent_gate':
            (-actual_lp.sum()/(len(members)*denominator)).backward()
        elif gradients[1] is None:
            actual_lp.backward(gradients[0][i])
        else:
            torch.autograd.backward((actual_lp, actual_h), (gradients[0][i], gradients[1][i]))
    return {'logp': combined.detach().cpu(), 'gate_mean': weights.detach().mean(1).cpu().tolist(),
            'responsibility_mean': responsibilities.detach().mean(1).cpu().tolist()}


@torch.no_grad()
def record_logp(runtime, record, *, mode, gate, members=(0, 1), single_member=0):
    if mode == 'single':
        return selected(runtime, single_member, record)[0]
    values = [selected(runtime, i, record) for i in validate_members(members)]
    return mixture(torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values]), gate)[0]
