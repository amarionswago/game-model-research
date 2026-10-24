"""Checkpoint ownership and balanced paired-domain evaluation scheduling."""
from dataclasses import dataclass
import copy
import time

import torch
from gpu_policy import Population, synchronize


@dataclass(frozen=True)
class Condition:
    id: str
    checkpoint: str
    members: tuple[int, ...]


CONDITIONS = {
    'member0': Condition('member0', 'joint_gate', (0,)),
    'member1': Condition('member1', 'joint_gate', (1,)),
    'independent_gate': Condition('independent_gate', 'independent_gate', (0, 1)),
    'joint_gate': Condition('joint_gate', 'joint_gate', (0, 1)),
}


def balanced_orders(condition_ids):
    ids = tuple(condition_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= CONDITIONS.keys():
        raise ValueError('Conditions must be a nonempty unique subset of the registry')
    n = len(ids)
    first = [0]
    for step in range(1, n):
        first.append((step+1)//2 if step % 2 else n-step//2)
    rows = [tuple(ids[(value+offset) % n] for value in first) for offset in range(n)]
    if n % 2 and n > 1:
        rows += [tuple(reversed(row)) for row in rows]
    return rows


def paired_blocks(prompts, condition_ids, *, completed_task_ids=()):
    prompts = list(prompts)
    ids = [p['id'] for p in prompts]
    completed = set(completed_task_ids)
    if len(prompts) % 2 or len(set(ids)) != len(ids) or not completed <= set(ids):
        raise ValueError('Paired tasks require an even unique prompt list and known completed IDs')
    orders = balanced_orders(condition_ids)
    domains = sorted({p['dataset'] for p in prompts})
    if len(domains) != 2:
        raise ValueError('Exactly two prompt datasets are required')
    offsets = {domain: i for i, domain in enumerate(domains)}
    domain_counts = {domain: 0 for domain in domains}
    blocks = []
    for pair_index in range(len(prompts)//2):
        pair = prompts[pair_index*2:pair_index*2+2]
        if {p['dataset'] for p in pair} != set(domains):
            raise ValueError('Every admitted pair must contain one task from each dataset')
        statuses = [p['id'] in completed for p in pair]
        if any(statuses) and not all(statuses):
            raise ValueError('Completed task IDs must describe complete domain pairs')
        tasks = []
        for prompt in pair:
            domain = prompt['dataset']
            order = orders[(domain_counts[domain]+offsets[domain]) % len(orders)]
            domain_counts[domain] += 1
            tasks.append({'prompt': prompt, 'condition_order': list(order)})
        if not all(statuses):
            blocks.append({'pair_index': pair_index, 'tasks': tasks})
    return blocks


def block_budget_seconds(block, *, response_seconds_cap, bookkeeping_seconds=60, export_reserve_seconds=180):
    if response_seconds_cap <= 0 or bookkeeping_seconds < 0 or export_reserve_seconds < 0:
        raise ValueError('Positive response cap and nonnegative reserves are required')
    return sum(len(task['condition_order']) for task in block['tasks'])*response_seconds_cap+bookkeeping_seconds+export_reserve_seconds


def admits_block(block, *, now, deadline, response_seconds_cap, bookkeeping_seconds=60, export_reserve_seconds=180):
    required = block_budget_seconds(block, response_seconds_cap=response_seconds_cap,
                                   bookkeeping_seconds=bookkeeping_seconds,
                                   export_reserve_seconds=export_reserve_seconds)
    return deadline-now >= required


class CheckpointBank:
    def __init__(self, runtime):
        self.runtime = runtime
        self.states = {}
        self.active = None
        self.runtime.checkpoint_generation = getattr(runtime, 'checkpoint_generation', 0)

    def capture(self, checkpoint, gate, *, sha256):
        if checkpoint not in ('joint_gate', 'independent_gate') or checkpoint in self.states:
            raise ValueError('Checkpoint must be a new joint or independent state')
        if gate is None or self.runtime.restored_identity['stage'] != checkpoint:
            raise ValueError('Checkpoint identity and learned gate must match')
        if self.runtime.restored_cursor['completed_updates'] != 7:
            raise ValueError('Only the latest seven-update control checkpoints are admitted')
        if len(sha256) != 64:
            raise ValueError('An exact source SHA256 is required')
        values = {name: parameter.detach().cpu().clone()
                  for group in self.runtime.groups for name, parameter in group.items()}
        if any(not torch.isfinite(value).all() for value in values.values()) or not torch.isfinite(gate.weight).all():
            raise ValueError('Checkpoint tensors must be finite')
        self.states[checkpoint] = dict(adapters=values, gate=copy.deepcopy(gate).eval().requires_grad_(False),
                                       sha256=sha256, identity=copy.deepcopy(self.runtime.restored_identity),
                                       cursor=copy.deepcopy(self.runtime.restored_cursor))
        self.active = checkpoint
        self.runtime.checkpoint_generation += 1

    @torch.no_grad()
    def activate(self, condition_id):
        condition = CONDITIONS[condition_id]
        state = self.states[condition.checkpoint]
        synchronize()
        started = time.monotonic()
        changed = self.active != condition.checkpoint
        if changed:
            for group in self.runtime.groups:
                for name, parameter in group.items():
                    parameter.copy_(state['adapters'][name])
            self.runtime.restored_identity = copy.deepcopy(state['identity'])
            self.runtime.restored_cursor = copy.deepcopy(state['cursor'])
            self.runtime.checkpoint_generation += 1
            self.active = condition.checkpoint
        synchronize()
        gate = state['gate'] if len(condition.members) == 2 else None
        population = Population(self.runtime, gate, members=condition.members)
        return population, dict(checkpoint_sha256=state['sha256'], checkpoint=condition.checkpoint,
                                checkpoint_switched=changed, checkpoint_switch_seconds=time.monotonic()-started,
                                condition=condition.id, member_ids=list(condition.members))
