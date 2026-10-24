"""Four-adapter/gate binding around the unchanged guarded transaction engine."""
import time

import torch

import fork_recovery as recovery
from effective_state_guard import assert_effective_state
from guarded_transactions import GuardedForkTransactions
from swarm.population_runtime import Population


def bind(runtime, gate, *, frozen_modules):
    if 'mix_gate' in runtime.model._modules:
        recovery.require(runtime.model.mix_gate is gate, 'Different registered gate')
    else:
        runtime.model.add_module('mix_gate', gate)
    parameters = {n:p for g in runtime.groups for n,p in g.items()}
    parameters['mix_gate.weight'] = gate.weight
    recovery.require(len(runtime.groups) == 4 and gate.members == (0, 1, 2, 3), 'Four-member binding required')
    all_parameters = recovery.named_parameters(runtime.model)
    recovery.require(all(all_parameters.get(n) is p for n,p in parameters.items()), 'Unregistered mutable tensors')
    runtime.model.requires_grad_(False)
    for p in runtime.model.parameters(): p.grad = None
    for p in parameters.values(): p.requires_grad_(True)
    frozen = recovery.frozen_identities(runtime.model, parameters, frozen_modules=frozen_modules)
    binding = recovery.ForkBinding(0, tuple(sorted(parameters)), parameters, frozen)
    assert_binding(runtime, gate, binding, frozen_modules=frozen_modules)
    return binding


def assert_binding(runtime, gate, binding, *, frozen_modules):
    actual = recovery.named_parameters(runtime.model)
    expected = {n:p for g in runtime.groups for n,p in g.items()}
    expected['mix_gate.weight'] = gate.weight
    recovery.require(tuple(sorted(expected)) == binding.parameter_names and set(expected) == set(binding.parameters), 'Mutable inventory changed')
    recovery.require(all(actual.get(n) is p is binding.parameters[n] for n,p in expected.items()), 'Mutable object changed')
    recovery.require({n for n,p in actual.items() if p.requires_grad} == set(expected), 'Trainable membership changed')
    outside = [p for n,p in actual.items() if n not in expected]+list(runtime.model.buffers())
    for module in frozen_modules.values(): outside += list(module.parameters())+list(module.buffers())
    stores = {(str(p.device), p.untyped_storage().data_ptr()) for p in outside}
    for p in expected.values():
        key = (str(p.device), p.untyped_storage().data_ptr())
        recovery.require(p.dtype == torch.float32 and p.ndim == 2 and key not in stores, 'Invalid or aliased mutable matrix')
        stores.add(key)
    recovery.require(recovery.frozen_identities(runtime.model, expected, frozen_modules=frozen_modules)
        == dict(binding.frozen_fingerprints), 'Immutable population state changed')


class CollectiveTransactions(GuardedForkTransactions):
    """Only the single-fork boundary assertion changes; serialization is inherited.

    The gate is registered on the model so inherited gradient clearing, device
    synchronization and complete-state checks include it. No unchecked path.
    """
    def __init__(self, runtime, gate, binding, optimizers, schedulers, *, effective_snapshot,
                 identity, frozen_modules, event):
        self.gate = gate
        super().__init__(runtime, binding, optimizers, schedulers, effective_snapshot=effective_snapshot,
            identity=identity, frozen_modules=frozen_modules,
            population_factory=lambda:Population(runtime, gate, members=(0, 1, 2, 3)), event=event)

    def _boundary(self, label):
        self._sync(); started = time.monotonic()
        assert_binding(self.runtime, self.gate, self.binding, frozen_modules=self.frozen_modules)
        named_seconds = time.monotonic()-started
        supplemental = assert_effective_state(self.runtime, self.effective_snapshot)
        recovery._optimizer_membership(self.binding, self.optimizers, self.schedulers)
        recovery.require(all(p.grad is None for p in self.binding.parameters.values()), 'Uncleared collective gradients')
        if not label.startswith('before_restore_'):
            recovery.require(all(torch.isfinite(p).all() for p in self.binding.parameters.values()), 'Nonfinite collective state')
        self._sync()
        self.event('frozen_state_checked', boundary=label, seconds=time.monotonic()-started,
            named_seconds=named_seconds, supplemental=supplemental)
