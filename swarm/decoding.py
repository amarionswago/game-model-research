"""Uniform probability combination and declared sampling order. AI-authored.

Each member must score the same observed prefix with its own KV cache. These
functions do not perform member forwards, select answers, or read grading data.
"""
import hashlib
import math

import torch


def item_seed(training_seed, question_id):
    value = f'cs-k2-001|{training_seed}|{question_id}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') % (2**63 - 1)


def combined_log_probabilities(member_logits):
    """[member, vocabulary] logits -> normalized FP32 mixture log probabilities."""
    if member_logits.ndim != 2 or min(member_logits.shape) == 0:
        raise ValueError('Expected a nonempty population with a shared vocabulary')
    if not torch.isfinite(member_logits).all():
        raise FloatingPointError('Nonfinite member logits')
    logp = member_logits.float().log_softmax(-1)
    return torch.logsumexp(logp, dim=0) - math.log(member_logits.shape[0])


def sample_combined(member_logits, *, temperature, top_p, generator):
    """Mix probabilities first, then temperature/top-p, then one shared token.

    The generator must match the logits' device. No member-specific temperature,
    top-p filtering, repetition penalty or answer-dependent selection occurs.
    """
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('Invalid temperature or nucleus mass')
    combined = combined_log_probabilities(member_logits)
    probabilities = (combined / temperature).softmax(-1)
    sorted_p, indices = torch.sort(probabilities, descending=True, stable=True)
    # Keep the token that crosses the threshold and at least the highest token.
    remove = sorted_p.cumsum(-1) - sorted_p >= top_p
    sorted_p = sorted_p.masked_fill(remove, 0)
    sorted_p /= sorted_p.sum()
    position = torch.multinomial(sorted_p, 1, generator=generator)
    return int(indices[position].item())
