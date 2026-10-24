"""Memory-bounded token-mixture objectives. AI assistance: authored by Codex.

These primitives do not select data, step optimizers, or launch research jobs.
Members must be deterministic between collection and recomputation. In particular,
do not step any member or switch its adapter during checkpointed backward.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint


def target_log_probabilities(hidden, head, labels, *, block_size=128, checkpoint_head=True):
    """Return one FP32 log probability per supervised causal target, in row order.

    hidden: [batch, sequence, hidden]; labels: [batch, sequence], -100 for
    unsupervised positions. Position t-1 predicts label t. No full-sequence
    vocabulary tensor is retained. Padding must also be masked in the backbone.
    """
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2] or block_size < 1:
        raise ValueError("Invalid hidden/labels shape or block size")
    if labels.dtype != torch.long:
        raise ValueError("Labels must be int64")
    labels = labels.to(hidden.device)
    selected = labels[:, 1:] != -100
    if not selected.any():
        raise ValueError("No supervised causal targets")
    states = hidden[:, :-1][selected]
    targets = labels[:, 1:][selected]

    def score(h, y):
        logits = head(h).float()
        return logits.gather(-1, y[:, None]).squeeze(-1) - logits.logsumexp(-1)

    pieces = []
    for start in range(0, len(targets), block_size):
        h, y = states[start:start + block_size], targets[start:start + block_size]
        if checkpoint_head and torch.is_grad_enabled():
            value = checkpoint(score, h, y, use_reentrant=False)
        else:
            value = score(h, y)
        pieces.append(value)
    return torch.cat(pieces)


def mixture_statistics(member_logp):
    """Uniform-mixture NLL and detached token responsibilities [member, token]."""
    if member_logp.ndim != 2 or min(member_logp.shape) == 0:
        raise ValueError("Expected nonempty [member, supervised-token] log probabilities")
    if not torch.isfinite(member_logp).all():
        raise FloatingPointError("Nonfinite member log probabilities")
    values = member_logp.float()
    loss = -(values.logsumexp(0) - math.log(values.shape[0])).mean()
    return loss, values.detach().softmax(0)


def backward_recomputed(forward_member, members, *, objective="team", denominator=None,
                        scale_loss=None, replay_atol=1e-5):
    """Accumulate gradients for a deterministic, unchanged population snapshot.

    forward_member(i) returns a 1-D tensor of supervised target log probabilities.
    denominator is the total target-token batch (for microbatch accumulation),
    not K times that total. For independent loss the coefficient is 1/K; for
    team loss it is the detached responsibility. No optimizer is stepped here.
    Collection/recomputation are both charged to the team; independent uses one
    pass per member. Caller must discard all gradients if this function raises.
    """
    if members < 1 or objective not in {"team", "independent"}:
        raise ValueError("Invalid population or objective")
    if replay_atol < 0:
        raise ValueError("Negative replay tolerance")
    collected = None
    if objective == "team":
        with torch.no_grad():
            collected = torch.stack([forward_member(i).detach().float() for i in range(members)])
            loss, responsibilities = mixture_statistics(collected)
        count = collected.shape[1]
    else:
        count, loss = None, 0.0
    backward = scale_loss if scale_loss is not None else lambda x: x
    observed = []
    for i in range(members):
        current = forward_member(i)
        if current.ndim != 1 or current.numel() == 0:
            raise ValueError("Member must return a nonempty target vector")
        if not torch.isfinite(current).all():
            raise FloatingPointError("Nonfinite recomputed log probabilities")
        observed.append(current.detach().float())
        if count is None:
            count = current.numel()
        if current.numel() != count:
            raise ValueError("Member target alignment mismatch")
        total = count if denominator is None else denominator
        if total < count:
            raise ValueError("Denominator is smaller than the microbatch")
        if objective == "team":
            if not torch.allclose(current.detach().float(), collected[i], rtol=0, atol=replay_atol):
                raise RuntimeError("Collection/recomputation mismatch; discard this update")
            surrogate = -(responsibilities[i].to(current.device) * current.float()).sum() / total
        else:
            surrogate = -current.float().sum() / (members * total)
            loss += float(-current.detach().float().mean()) / members
        backward(surrogate).backward()
    # Diagnostic values reuse the actual forwards. They add neither an objective
    # term nor extra member passes to the cheaper independent control.
    observed = torch.stack(observed)
    if collected is None:
        _, responsibilities = mixture_statistics(observed)
    return {"loss": float(loss), "supervised_tokens": count,
            "forward_calls": members * (2 if objective == "team" else 1),
            "mean_responsibility": responsibilities.mean(1).tolist(),
            "member_mean_nll": (-observed.mean(1)).tolist()}
