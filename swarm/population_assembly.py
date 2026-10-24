"""Import source-bound formed adapters; no formation optimizer state is carried.

Only tiny synthetic artifacts may be opened locally. Production import belongs
inside the admitted hardware worker, after external source hashes are checked.
"""
import json
import re
from pathlib import Path

import torch

from swarm.population_fork import file_sha256, tensor_sha256
from swarm.population_objective import MixingGate


SCHEMA = 'hive-formed-adapter-v1'
PROVENANCE_FIELDS = {
    'parent_checkpoint_sha256', 'parent_member_index', 'base_manifest_sha256',
    'tokenizer_sha256', 'code_sha256', 'recipe_sha256', 'data_manifest_sha256',
    'family_manifest_sha256', 'batch_manifest_sha256', 'retained_updates',
    'physical_optimizer_calls', 'replay_optimizer_calls', 'supervised_tokens',
    'recovery_verified', 'artifact_kind',
}


def validate_provenance(value):
    if not isinstance(value, dict) or set(value) != PROVENANCE_FIELDS:
        raise ValueError('Incomplete/extra formed-adapter provenance')
    for key in PROVENANCE_FIELDS:
        if key.endswith('_sha256') and not (isinstance(value[key], str) and
                re.fullmatch('[0-9a-f]{64}', value[key])):
            raise ValueError('Invalid provenance digest: '+key)
    for key in ('parent_member_index', 'retained_updates', 'physical_optimizer_calls',
                'replay_optimizer_calls', 'supervised_tokens'):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('Invalid provenance counter: '+key)
    if value['parent_member_index'] != 0 or value['recovery_verified'] is not True:
        raise ValueError('Common Member0 lineage and verified recovery required')
    if value['physical_optimizer_calls'] < value['retained_updates']+value['replay_optimizer_calls']:
        raise ValueError('Physical optimizer calls cannot omit retained/replay calls')
    if value['artifact_kind'] not in ('formed', 'unchanged_parent'):
        raise ValueError('Unknown adapter artifact kind')
    if (value['artifact_kind'] == 'formed') != (value['retained_updates'] > 0):
        raise ValueError('Unchanged parents must remain explicit')
    if value['retained_updates'] > 0 and value['supervised_tokens'] == 0:
        raise ValueError('Formed adapter requires a declared target dose')
    json.dumps(value, allow_nan=False)


def canonical_name(name, member):
    marker = f'.member{member}.'
    if not isinstance(name, str) or name.count(marker) != 1 or '.lora_' not in name:
        raise ValueError('Invalid adapter matrix name')
    return name.replace(marker, '.member0.')


def assemble(sources, groups, *, hidden_size, purpose='formed_population'):
    """Validate all four inputs, then return independent CPU tensors and zero gate.

    sources is an ordered list of {member, path, expected_file_sha256,
    expected_provenance}. groups describes actual destination tensors. It is
    inspected for names/shapes only and is never mutated here.
    """
    if len(sources) != 4 or len(groups) != 4 or purpose not in (
            'formed_population', 'numerical_duplicate_reference'):
        raise ValueError('Four explicit sources/groups and a known purpose required')
    hashes = [s['expected_file_sha256'] for s in sources]
    if purpose == 'formed_population' and len(set(hashes)) != 4:
        raise ValueError('Duplicate artifact requires explicit numerical-reference purpose')
    values, mapping = {}, []
    common = None
    for member, (source, group) in enumerate(zip(sources, groups)):
        if set(source) != {'member', 'path', 'expected_file_sha256', 'expected_provenance'} or type(source['member']) is not int or source['member'] != member:
            raise ValueError('Source slot/order mismatch')
        expected = source['expected_provenance']
        validate_provenance(expected)
        lineage = {key: expected[key] for key in ('parent_checkpoint_sha256',
            'parent_member_index', 'base_manifest_sha256', 'tokenizer_sha256')}
        if common is not None and common != lineage:
            raise ValueError('Population parents disagree on common source/base/tokenizer')
        common = lineage
        path = Path(source['path'])
        if file_sha256(path) != source['expected_file_sha256']:
            raise ValueError('Formed artifact file identity mismatch')
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(artifact, dict) or set(artifact) != {
                'schema', 'adapters', 'tensor_sha256', 'provenance'} or artifact['schema'] != SCHEMA:
            raise ValueError('Invalid formed-adapter schema')
        validate_provenance(artifact['provenance'])
        if artifact['provenance'] != expected:
            raise ValueError('Formed artifact receipt mismatch')
        adapters = artifact['adapters']
        names = {canonical_name(name, member): name for name in group}
        if not names or not isinstance(adapters, dict) or set(adapters) != set(names):
            raise ValueError('Incomplete/extra adapter matrices')
        for canonical, name in names.items():
            value, destination = adapters[canonical], group[name]
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.dtype != torch.float32 or destination.dtype != torch.float32 or value.shape != destination.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid adapter shape/dtype/value: '+canonical)
        if tensor_sha256(adapters) != artifact['tensor_sha256']:
            raise ValueError('Adapter tensor digest mismatch')
        values.update({name: adapters[canonical].detach().clone() for canonical, name in names.items()})
        mapping.append({'member': member, 'path': str(path.resolve()),
            'file_sha256': source['expected_file_sha256'], 'tensor_sha256': artifact['tensor_sha256'],
            'provenance': dict(expected)})
    gate = MixingGate(hidden_size, members=(0, 1, 2, 3))
    values['mix_gate.weight'] = gate.weight.detach().clone()
    return {'schema': 'hive-four-member-assembly-v1', 'purpose': purpose,
        'members': mapping, 'adapters': values, 'tensor_sha256': tensor_sha256(values),
        'gate_initialization': 'uniform_reference_logits_zero',
        'fresh_optimizer_scheduler_scaler_required': True}


def activate(runtime, assembly, gate):
    """Validate before copying; caller owns the full recovery transaction.

    A copy/device failure may leave partial state and MUST trigger rollback.
    Successful activation advances the generation, invalidating old caches.
    """
    parameters = {name: p for group in runtime.groups for name, p in group.items()}
    if len(runtime.groups) != 4 or gate.members != (0, 1, 2, 3):
        raise ValueError('Four-member activation required')
    parameters['mix_gate.weight'] = gate.weight
    values = assembly['adapters']
    if set(parameters) != set(values) or tensor_sha256(values) != assembly['tensor_sha256']:
        raise ValueError('Assembly tensor identity mismatch')
    for name, p in parameters.items():
        v = values[name]
        if v.shape != p.shape or v.dtype != p.dtype or not torch.isfinite(v).all():
            raise ValueError('Invalid activation tensor: '+name)
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(values[name])
            if not torch.equal(p.detach().cpu(), values[name]):
                raise RuntimeError('Activation copy mismatch')
    runtime.checkpoint_generation = getattr(runtime, 'checkpoint_generation', 0)+1
    return runtime.checkpoint_generation


def duplicate_binary_gate(binary_gate):
    """Engineering reference only: order (parent0, parent1, parent0, parent1).

    Four logits [a,0,a,0] split each binary parent's mass equally. This is not
    the uniform initialization used for actual formed specialist comparisons.
    """
    if binary_gate.members != (0, 1):
        raise ValueError('Canonical binary reference required')
    result = MixingGate(binary_gate.hidden_size, members=(0, 1, 2, 3),
        device=binary_gate.weight.device)
    with torch.no_grad():
        for row in (0, 2):
            result.weight[row, :2*binary_gate.hidden_size].copy_(binary_gate.weight[0, :-1])
            result.weight[row, -1].copy_(binary_gate.weight[0, -1])
    return result
