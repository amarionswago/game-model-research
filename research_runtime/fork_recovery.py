"""Exact revision-fork initialization and guarded reuse of checkpoint primitives."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import torch

from stream_hash import update_tensor_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def tensor_identity(tensor):
    require(tensor.layout == torch.strided and tensor.device.type != 'meta', 'Unhashable tensor layout/device')
    result = hashlib.sha256()
    update_tensor_hash(result, tensor)
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': result.hexdigest()}


def named_parameters(model):
    rows = list(model.named_parameters(remove_duplicate=False))
    require(len({name for name, _ in rows}) == len(rows), 'Duplicate parameter name')
    return dict(rows)


def frozen_identities(model, excluded_names=(), *, frozen_modules=None):
    excluded = frozenset(excluded_names)
    modules = {'model': model, **{str(name): module for name, module in (frozen_modules or {}).items()}}
    require(len(modules) == 1 + len(frozen_modules or {}), 'Frozen module name collides with model')
    output = {}
    for prefix, module in modules.items():
        for name, value in module.named_parameters(remove_duplicate=False):
            if prefix == 'model' and name in excluded:
                continue
            require(not value.requires_grad and value.grad is None, 'Original/base/gate parameters must be frozen and clear')
            output[prefix + '/parameter/' + name] = tensor_identity(value)
        for name, value in module.named_buffers(remove_duplicate=False):
            require(not value.requires_grad, 'Frozen buffers cannot require gradients')
            output[prefix + '/buffer/' + name] = tensor_identity(value)
    return output


@dataclass(frozen=True)
class ForkBinding:
    fork_member: int
    parameter_names: tuple[str, ...]
    parameters: object
    frozen_fingerprints: object

    def __post_init__(self):
        object.__setattr__(self, 'parameters', MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, 'frozen_fingerprints', MappingProxyType(deepcopy(dict(self.frozen_fingerprints))))


def _fork_parameters(model, fork_member):
    require(type(fork_member) is int and fork_member >= 2, 'Revision fork must be member2 or later')
    destination = 'member' + str(fork_member)
    require('member0' in model.peft_config and destination in model.peft_config, 'Missing source or installed fork adapter')
    require(model.peft_config['member0'].to_dict() == model.peft_config[destination].to_dict(), 'Fork/source adapter configuration mismatch')
    parameters = named_parameters(model)
    source = {name: p for name, p in parameters.items() if '.member0.' in name and '.lora_' in name}
    fork = {name: p for name, p in parameters.items() if '.' + destination + '.' in name and '.lora_' in name}
    require(source and fork, 'Missing source or fork LoRA parameters')
    expected = {name.replace('.member0.', '.' + destination + '.'): p for name, p in source.items()}
    require(set(expected) == set(fork), 'Missing or extra fork LoRA keys')
    require(all('.lora_' in name for name in parameters if '.' + destination + '.' in name), 'Unsupported non-LoRA fork parameters')
    require(len({id(p) for p in fork.values()}) == len(fork), 'Duplicate fork parameter object')
    fork_storage = set()
    original_storage = {(str(p.device), p.untyped_storage().data_ptr()) for name, p in parameters.items() if name not in fork}
    for name, parameter in fork.items():
        original = expected[name]
        require(parameter.shape == original.shape, 'Fork/source shape mismatch')
        require(parameter.dtype == original.dtype, 'Fork/source dtype mismatch')
        require(parameter.dtype == torch.float32 and original.dtype == torch.float32
                and parameter.ndim == original.ndim == 2, 'Fork/source must be FP32 2D LoRA matrices')
        require(parameter.device == original.device, 'Fork/source device mismatch')
        require(torch.isfinite(original).all(), 'Nonfinite source adapter')
        storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
        require(storage not in original_storage and storage not in fork_storage, 'Aliased fork storage')
        fork_storage.add(storage)
    return parameters, expected, fork


def copy_into_installed_fork(model, *, fork_member, frozen_modules=None):
    parameters, source, fork = _fork_parameters(model, fork_member)
    frozen = frozen_identities(model, fork, frozen_modules=frozen_modules)
    outside = list(model.buffers())
    for module in (frozen_modules or {}).values():
        outside.extend(module.parameters())
        outside.extend(module.buffers())
    outside_storage = {(str(p.device), p.untyped_storage().data_ptr()) for p in outside}
    require(all((str(p.device), p.untyped_storage().data_ptr()) not in outside_storage for p in fork.values()),
            'Fork storage aliases a frozen buffer/external module')
    require(all(p.grad is None for p in fork.values()), 'Fork creation requires cleared gradients')
    with torch.no_grad():
        for name, parameter in fork.items():
            parameter.copy_(source[name])
    model.set_adapter('member' + str(fork_member))
    binding = ForkBinding(fork_member, tuple(sorted(fork)), fork, frozen)
    assert_fork_state(model, binding, frozen_modules=frozen_modules)
    require(all(torch.equal(p, source[name]) for name, p in fork.items()), 'Fork copy is not exact')
    return binding


def install_revision_fork(model, *, fork_member=2, frozen_modules=None):
    require(type(fork_member) is int and fork_member >= 2, 'Revision fork must be member2 or later')
    require('member0' in model.peft_config, 'Missing source adapter')
    destination = 'member' + str(fork_member)
    require(destination not in model.peft_config, 'Destination adapter already installed')
    before = frozen_identities(model, frozen_modules=frozen_modules)
    model.add_adapter(destination, deepcopy(model.peft_config['member0']))
    binding = copy_into_installed_fork(model, fork_member=fork_member, frozen_modules=frozen_modules)
    require(dict(binding.frozen_fingerprints) == before, 'Installing fork changed original frozen tensors')
    return binding


def assert_fork_state(model, binding, *, frozen_modules=None):
    parameters, _, actual_fork = _fork_parameters(model, binding.fork_member)
    require(set(binding.parameter_names) == set(actual_fork) == set(binding.parameters), 'Fork binding names changed')
    require(all(actual_fork[name] is binding.parameters[name] for name in actual_fork), 'Fork parameter object changed')
    require({name for name, p in parameters.items() if p.requires_grad} == set(binding.parameter_names),
            'Only declared fork parameters may be trainable')
    require(frozen_identities(model, binding.parameter_names, frozen_modules=frozen_modules)
            == dict(binding.frozen_fingerprints), 'Original frozen tensor identity changed')
    return dict(actual_fork)


def _checkpoint_identity(identity, binding):
    require(bool(identity), 'Explicit checkpoint identity required')
    frozen_hash = hashlib.sha256(json.dumps(dict(binding.frozen_fingerprints), sort_keys=True).encode()).hexdigest()
    return {'caller': deepcopy(identity), 'fork_member': binding.fork_member,
            'fork_parameter_names': list(binding.parameter_names), 'frozen_state_sha256': frozen_hash}


def _optimizer_membership(binding, optimizers, schedulers):
    require(len(optimizers) == len(schedulers) > 0, 'Optimizer/scheduler counts differ')
    owned = [id(p) for optimizer in optimizers for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in binding.parameters.values()}
    require(len(owned) == len(set(owned)) and set(owned) == expected, 'Optimizers must own exactly the fork once')
    require(all(scheduler.optimizer is optimizer for optimizer, scheduler in zip(optimizers, schedulers)),
            'Scheduler is attached to a different optimizer')


def save_fork(path, model, binding, optimizers, schedulers, *, checkpoint_io, identity, cursor,
              frozen_modules=None, scaler=None):
    assert_fork_state(model, binding, frozen_modules=frozen_modules)
    require(all(torch.isfinite(p).all() for p in binding.parameters.values()), 'Cannot save nonfinite fork tensors')
    _optimizer_membership(binding, optimizers, schedulers)
    checkpoint_io.save_population(path, dict(binding.parameters), optimizers, schedulers,
        identity=_checkpoint_identity(identity, binding), cursor=cursor, scaler=scaler)
    assert_fork_state(model, binding, frozen_modules=frozen_modules)
    return file_sha256(path)


def restore_fork(path, expected_sha256, model, binding, optimizers, schedulers, *, checkpoint_io,
                 identity, frozen_modules=None, scaler=None):
    require(file_sha256(path) == expected_sha256, 'Checkpoint file identity changed')
    assert_fork_state(model, binding, frozen_modules=frozen_modules)
    _optimizer_membership(binding, optimizers, schedulers)
    cursor = checkpoint_io.load_population(path, dict(binding.parameters), optimizers, schedulers,
        identity=_checkpoint_identity(identity, binding), scaler=scaler)
    assert_fork_state(model, binding, frozen_modules=frozen_modules)
    return cursor
