"""Exact target-token batches and atomic population updates. AI-authored.

Dataset admission and the model loader belong to the run manifest/entry point.
These routines never download data, launch jobs, or choose experimental arms.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch

from swarm.cooperative import backward_recomputed


@dataclass(frozen=True)
class LossSegment:
    record: dict
    positions: tuple

    def batch(self, device):
        """Keep the entire causal input; select only this update's loss positions."""
        ids = torch.tensor([self.record['input_ids']], device=device, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        selected = list(self.positions)
        labels[0, selected] = torch.tensor([self.record['labels'][i] for i in selected], device=device)
        attention = torch.tensor([self.record.get('attention_mask', [1] * ids.shape[1])], device=device)
        return {'input_ids': ids, 'labels': labels, 'attention_mask': attention}


class TargetStream:
    """Consume an immutable, already-admitted record sequence without overshoot.

    A record can contribute to consecutive updates. Each update keeps the full
    causal prefix and masks positions charged to another update. The cursor is
    checkpointable and contains no shuffled or hidden iterator state.
    """
    def __init__(self, records, *, cursor=None):
        self.records = records
        self.cursor = deepcopy(cursor or {'record': 0, 'target_offset': 0, 'supervised_tokens': 0})
        if set(self.cursor) != {'record', 'target_offset', 'supervised_tokens'}:
            raise ValueError('Invalid target stream cursor')
        if any(type(v) is not int or v < 0 for v in self.cursor.values()):
            raise ValueError('Negative or noninteger cursor')
        if self.cursor['record'] > len(records):
            raise ValueError('Cursor past the immutable stream')
        if self.cursor['record'] == len(records) and self.cursor['target_offset']:
            raise ValueError('Offset after the end of the stream')

    @staticmethod
    def positions(record):
        ids, labels = record['input_ids'], record['labels']
        mask = record.get('attention_mask', [1] * len(ids))
        if len(ids) != len(labels) or len(ids) != len(mask) or len(ids) < 2:
            raise ValueError('Malformed admitted record')
        if labels[0] != -100:
            raise ValueError('Position zero has no preceding causal prediction')
        positions = [i for i in range(1, len(ids)) if labels[i] != -100]
        if not positions or any(labels[i] != ids[i] or mask[i] != 1 for i in positions):
            raise ValueError('Invalid causal targets or padding supervision')
        return positions

    def take(self, count):
        """Return exactly count supervised positions; exhaustion leaves state intact."""
        if type(count) is not int or count <= 0:
            raise ValueError('Positive integer target budget required')
        cursor = deepcopy(self.cursor)
        segments, remaining = [], count
        while remaining:
            if cursor['record'] == len(self.records):
                raise EOFError('Insufficient frozen targets; no implicit replay permitted')
            record = self.records[cursor['record']]
            positions = self.positions(record)
            offset = cursor['target_offset']
            if offset >= len(positions):
                raise ValueError('Invalid offset within admitted record')
            chosen = positions[offset:offset + remaining]
            segments.append(LossSegment(record, tuple(chosen)))
            remaining -= len(chosen)
            cursor['target_offset'] += len(chosen)
            if cursor['target_offset'] == len(positions):
                cursor['record'] += 1
                cursor['target_offset'] = 0
        cursor['supervised_tokens'] += count
        self.cursor = cursor
        return segments


def learning_rate_multiplier(update, total_updates, *, warmup_fraction=.05, floor=.1):
    """Multiplier for a zero-based update, including warm-up and a final floor."""
    if not (0 <= update < total_updates) or not 0 < warmup_fraction < 1 or not 0 <= floor <= 1:
        raise ValueError('Invalid schedule')
    warmup = max(1, math.ceil(total_updates * warmup_fraction))
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup + 1) / max(1, total_updates - warmup)
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def population_update(forward_member, segments, parameter_groups, optimizers, *,
                      objective, scaler=None, clip_norm=1., require_cuda=True):
    """Accumulate every member before changing any weights; then update together.

    forward_member(member_index, segment) returns target log probabilities in
    segment.positions order. Production requires CUDA parameters. Explicitly
    disabling require_cuda is reserved for software fixtures, not CPU training.
    Nonfinite/missing gradients abort all optimizers before any step. An optimizer
    or scaler exception during stepping invalidates the process: reload the last
    complete checkpoint, never continue with a partly updated population.
    """
    if not parameter_groups or len(parameter_groups) != len(optimizers) or not segments:
        raise ValueError('Incomplete update')
    parameters = [p for group in parameter_groups for p in group]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Adapters must not share trainable parameter objects')
    if any(not group for group in parameter_groups):
        raise ValueError('Empty adapter parameter group')
    for group, optimizer in zip(parameter_groups, optimizers):
        owned = [p for opt_group in optimizer.param_groups for p in opt_group['params']]
        if len(owned) != len(group) or {id(p) for p in owned} != {id(p) for p in group}:
            raise ValueError('Optimizer ownership differs from its member')
    if any(p.grad is not None for p in parameters):
        raise ValueError('Update must begin with cleared gradients')
    if require_cuda and any(p.device.type != 'cuda' for p in parameters):
        raise RuntimeError('The registered model training requires GPU parameters')
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('Expected FP32 adapter/gradient state')
    tokens = sum(len(segment.positions) for segment in segments)
    if tokens <= 0 or clip_norm <= 0:
        raise ValueError('Invalid target budget or clipping threshold')
    total_loss, forward_calls = 0., 0
    mean_responsibility = [0.] * len(parameter_groups)
    member_mean_nll = [0.] * len(parameter_groups)
    try:
        for segment in segments:
            result = backward_recomputed(
                lambda member: forward_member(member, segment), len(parameter_groups),
                objective=objective, denominator=tokens,
                scale_loss=scaler.scale if scaler is not None else None,
                replay_atol=1e-3 if require_cuda else 1e-5,
            )
            if result['supervised_tokens'] != len(segment.positions):
                raise ValueError('Forward target alignment differs from frozen loss mask')
            total_loss += result['loss'] * len(segment.positions) / tokens
            forward_calls += result['forward_calls']
            for member in range(len(parameter_groups)):
                weight = len(segment.positions) / tokens
                mean_responsibility[member] += result['mean_responsibility'][member] * weight
                member_mean_nll[member] += result['member_mean_nll'][member] * weight
        if scaler is not None:
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Missing/nonfinite population gradient; no optimizer stepped')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True))
        if scaler is None:
            for optimizer in optimizers:
                optimizer.step()
        else:
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Nonfinite updated parameter; restore last full checkpoint')
        return {'loss': total_loss, 'supervised_tokens_per_member': tokens,
                'gradient_bearing_member_targets': tokens * len(parameter_groups),
                'batch_input_tokens': sum(len(s.record['input_ids']) for s in segments),
                'member_forward_input_tokens_excluding_checkpoint_recompute':
                    sum(len(s.record['input_ids']) for s in segments) * len(parameter_groups) * (2 if objective == 'team' else 1),
                'forward_calls': forward_calls, 'population_gradient_norm_before_clip': norm,
                'optimizer_steps_per_member': 1, 'objective': objective,
                'mean_target_responsibility': mean_responsibility,
                'member_mean_nll': member_mean_nll}
    finally:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
