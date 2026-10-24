"""Guarded checkpoint transactions; experimental, with no measured-cost admission.

The complete named and supplemental guards run exactly once on each side of
each successful transaction. There is no public unchecked checkpoint method,
skip flag, or reusable boundary receipt. The caller serializes model mutation.
"""
from copy import deepcopy
from enum import Enum, auto
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import tempfile
import time

import torch

import adapter_checkpoint as checkpoint_io
import fork_recovery as recovery
from effective_state_guard import assert_effective_state


class _State(Enum):
    READY = auto()
    ACTIVE = auto()
    INVALID = auto()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected):
    """Bit-exact recursive state comparison, including tensor dtype and shape."""
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
                and recovery.tensor_identity(actual) == recovery.tensor_identity(expected))
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_same(actual[k], expected[k]) for k in actual)
    if isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _hash_open(source):
    source.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(block)
    source.seek(0)
    return digest.hexdigest()


class GuardedForkTransactions:
    """One serialized process-local owner for guarded save and full restoration.

    A failed operation permanently invalidates this owner and retires its prior
    activation. Pending files are evidence only: only a successfully returned
    receipt may enter the caller's checkpoint manifest. No CUDA throughput or
    whole-process memory bound is implied by the streaming hash helper.
    """

    def __init__(self, runtime, binding, optimizers, schedulers, *, effective_snapshot,
                 identity, frozen_modules, population_factory, event, scaler=None):
        _require(callable(population_factory) and callable(event), 'Explicit population factory/event required')
        _require(bool(identity), 'Explicit checkpoint identity required')
        self.runtime = runtime
        self.binding = binding
        self.optimizers = tuple(optimizers)
        self.schedulers = tuple(schedulers)
        self.effective_snapshot = effective_snapshot
        self.identity = deepcopy(identity)
        self.frozen_modules = dict(frozen_modules)
        self.population_factory = population_factory
        self.event = event
        self.scaler = scaler
        self._state = _State.READY
        self._generation = self._read_generation()
        self.last_pending = None
        self.invalidation_event_error = None
        recovery._optimizer_membership(binding, self.optimizers, self.schedulers)

    @property
    def valid(self):
        return self._state is not _State.INVALID

    def _read_generation(self):
        generation = getattr(self.runtime, 'checkpoint_generation', None)
        _require(type(generation) is int and generation >= 0, 'Invalid activation generation')
        return generation

    def _begin(self, label):
        _require(self._state is _State.READY, 'Checkpoint transaction owner unavailable or invalid')
        self._state = _State.ACTIVE
        try:
            _require(isinstance(label, str) and label and '/' not in label and '\\' not in label,
                     'Explicit checkpoint label required')
            generation = self._read_generation()
            _require(generation >= self._generation, 'Activation generation moved backwards')
            self._generation = generation
            self._clear_gradients()
        except BaseException:
            self._invalidate()
            raise

    def _invalidate(self):
        self._state = _State.INVALID
        current = getattr(self.runtime, 'checkpoint_generation', None)
        # Even a partly loaded state cannot be used through an older population.
        current = current if type(current) is int and current >= 0 else self._generation
        previous = max(current, self._generation)
        self.runtime.checkpoint_generation = previous + 1
        self._generation = self.runtime.checkpoint_generation
        try:
            self.event('activation_invalidated', previous_generation=previous,
                       new_generation=self._generation, owner_valid=False)
        except BaseException as error:
            # The caller's original transaction failure remains primary. A
            # failed event sink cannot make this owner valid or authorize more
            # model operations. Its missing causal evidence is not a pass.
            self.invalidation_event_error = repr(error)

    def _clear_gradients(self):
        for parameter in self.runtime.model.parameters():
            parameter.grad = None
        for module in self.frozen_modules.values():
            for parameter in module.parameters():
                parameter.grad = None

    def _sync(self):
        devices = {p.device for p in self.runtime.model.parameters() if p.device.type == 'cuda'}
        devices.update(p.device for m in self.frozen_modules.values() for p in m.parameters()
                       if p.device.type == 'cuda')
        for device in sorted(devices, key=str):
            torch.cuda.synchronize(device)

    def _boundary(self, label):
        self._sync()
        started = time.monotonic()
        recovery.assert_fork_state(self.runtime.model, self.binding, frozen_modules=self.frozen_modules)
        named_seconds = time.monotonic() - started
        supplemental = assert_effective_state(self.runtime, self.effective_snapshot)
        recovery._optimizer_membership(self.binding, self.optimizers, self.schedulers)
        _require(all(p.grad is None for p in self.binding.parameters.values()),
                 'Fork requires cleared gradients at checkpoint boundary')
        # Original restore semantics recover a damaged mutable fork. Frozen
        # tensors/metadata are always checked; the incoming payload was already
        # checked finite, and the restored fork must be finite at the postcheck.
        # This private phase distinction is not an optional caller bypass.
        if not label.startswith('before_restore_'):
            _require(all(bool(torch.isfinite(p).all()) for p in self.binding.parameters.values()),
                     'Fork must be finite with cleared gradients at checkpoint boundary')
        self._sync()
        elapsed = time.monotonic() - started
        self.event('frozen_state_checked', boundary=label, seconds=elapsed,
                   named_seconds=named_seconds, supplemental=supplemental)

    def _identity(self):
        return recovery._checkpoint_identity(self.identity, self.binding)

    def _validate_payload(self, state, cursor):
        _require(type(state) is dict and set(state) == {
            'format', 'identity', 'cursor', 'adapters', 'optimizers', 'schedulers', 'scaler', 'rng'},
            'Checkpoint state inventory mismatch')
        _require(type(state['format']) is int and state['format'] == 1 and _same(state['identity'], self._identity()),
                 'Checkpoint identity mismatch')
        _require(bool(cursor) and _same(state['cursor'], cursor), 'Checkpoint cursor mismatch')
        _require(type(state['adapters']) is dict and set(state['adapters']) == set(self.binding.parameters),
                 'Adapter keys mismatch')
        for name, parameter in self.binding.parameters.items():
            value = state['adapters'][name]
            _require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                     and value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                     'Adapter shape/dtype/finiteness mismatch')
        _require(type(state['optimizers']) is list and len(state['optimizers']) == len(self.optimizers)
                 and type(state['schedulers']) is list and len(state['schedulers']) == len(self.schedulers),
                 'Population optimizer/scheduler count mismatch')
        _require((state['scaler'] is None) == (self.scaler is None), 'Cannot discard or invent precision-scaler state')
        rng = state['rng']
        _require(type(rng) is dict and set(rng) == {'python', 'torch', 'cuda'}, 'RNG state inventory mismatch')
        random.Random().setstate(rng['python'])
        torch.Generator(device='cpu').set_state(rng['torch'])
        _require(type(rng['cuda']) is list and len(rng['cuda']) ==
                 (torch.cuda.device_count() if torch.cuda.is_available() else 0), 'CUDA RNG device-count mismatch')
        for value in rng['cuda']:
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                     and value.device.type == 'cpu' and value.ndim == 1, 'Invalid CUDA RNG state')

    def _assert_restored(self, saved, cursor):
        actual = {'format': 1, 'identity': self._identity(), 'cursor': cursor,
                  'adapters': dict(self.binding.parameters),
                  'optimizers': [opt.state_dict() for opt in self.optimizers],
                  'schedulers': [scheduler.state_dict() for scheduler in self.schedulers],
                  'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                  'rng': {'python': random.getstate(), 'torch': torch.get_rng_state(),
                          'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        _require(_same(actual, saved), 'Checkpoint did not restore the complete state exactly')

    def save(self, path, *, cursor, label):
        self._begin(label)
        path_text = str(path)
        path = Path(path)
        try:
            _require(not path.exists() and not path.is_symlink(), 'Checkpoint publication already exists')
            _require(bool(cursor), 'Explicit progress cursor required')
            self._boundary('before_save_' + label)
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = Path(tempfile.mkdtemp(prefix='.' + path.name + '.pending-', dir=path.parent))
            self.last_pending = str(pending)
            (pending / 'INELIGIBLE.json').write_text(json.dumps({
                'eligible': False, 'reason': 'Pending evidence is never a manifest checkpoint',
                'requested_publication': path_text, 'label': label}, sort_keys=True) + '\n')
            artifact = pending / 'state.pt'
            checkpoint_io.save_population(artifact, dict(self.binding.parameters), self.optimizers,
                self.schedulers, identity=self._identity(), cursor=cursor, scaler=self.scaler)
            self._boundary('after_save_' + label)
            receipt = {'path': path_text, 'sha256': recovery.file_sha256(artifact),
                       'bytes': artifact.stat().st_size, 'cursor': deepcopy(cursor)}
            _require(self._read_generation() == self._generation, 'Activation changed during save')
            # Atomic no-replace publication, on the same filesystem. No earlier
            # error path produces a final artifact, and all pending data remains.
            os.link(artifact, path)
            self._state = _State.READY
            return receipt
        except BaseException:
            self._invalidate()
            raise

    def restore(self, path, expected_sha256, *, expected_cursor, label):
        self._begin(label)
        path_text = str(path)
        path = Path(path)
        try:
            _require(not (path.parent / 'INELIGIBLE.json').exists(), 'Pending checkpoint is ineligible')
            _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
                     and all(c in '0123456789abcdef' for c in expected_sha256), 'Invalid checkpoint digest')
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, 'rb') as source:
                info = os.fstat(source.fileno())
                _require(stat.S_ISREG(info.st_mode), 'Checkpoint must be a regular file')
                _require(_hash_open(source) == expected_sha256, 'Checkpoint file identity changed')
                saved = torch.load(source, map_location='cpu', weights_only=True)
                self._validate_payload(saved, expected_cursor)
                self._boundary('before_restore_' + label)
                stale = self.population_factory()
                _require(stale.runtime is self.runtime and stale.checkpoint_generation == self._generation,
                         'Population factory returned a stale or foreign activation')
                source.seek(0)
                cursor = checkpoint_io.load_population(source, dict(self.binding.parameters), self.optimizers,
                    self.schedulers, identity=self._identity(), scaler=self.scaler)
                _require(_hash_open(source) == expected_sha256, 'Checkpoint changed during restore')
                _require(_same(cursor, expected_cursor), 'Restored cursor changed')
                self._assert_restored(saved, cursor)
                _require(self._read_generation() == self._generation, 'Activation changed during restore')
                previous = self._generation
                self.runtime.checkpoint_generation = previous + 1
                self._generation = previous + 1
                try:
                    stale._forward(self.binding.fork_member, [1], 0, None)
                except RuntimeError as error:
                    _require('earlier checkpoint activation' in str(error), 'Wrong stale-activation rejection')
                else:
                    raise RuntimeError('Stale population was not invalidated')
                self.event('activation_advanced', reason='restore_' + label, previous_generation=previous,
                           stale_activation_rejected=True)
                self._boundary('after_restore_' + label)
                _require(self._read_generation() == self._generation, 'Activation changed after restore')
                current = path.stat(follow_symlinks=False)
                _require((current.st_dev, current.st_ino, current.st_size) ==
                         (info.st_dev, info.st_ino, info.st_size), 'Checkpoint path changed during restore')
            receipt = {'path': path_text, 'sha256': expected_sha256, 'bytes': info.st_size,
                       'cursor': deepcopy(cursor)}
            self._state = _State.READY
            return receipt
        except BaseException:
            self._invalidate()
            raise
