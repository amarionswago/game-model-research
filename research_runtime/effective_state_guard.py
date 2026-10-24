"""Supplement complete named-tensor hashes with effective NF4 state and alias checks.

This is a process-local boundary guard, not an initial decoded-weight admission.
Pair every assertion with fork_recovery.assert_fork_state. Packed/dense named
tensor contents are deliberately not hashed here a second time.
"""
from dataclasses import dataclass, is_dataclass
from enum import Enum
import functools
import hashlib
import json
import marshal
from pathlib import Path
import struct
import sys
import time
import types

import torch

from stream_hash import update_tensor_hash


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage._cdata), int(storage.data_ptr()), int(storage.nbytes()))


def geometry(tensor):
    require(tensor.layout == torch.strided and tensor.device.type != 'meta', 'Unsupported tensor layout/device')
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'storage_offset': tensor.storage_offset(), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'storage': list(storage_key(tensor)),
            'requires_grad': tensor.requires_grad, 'is_conj': tensor.is_conj(), 'is_neg': tensor.is_neg()}


def same_slice(actual, expected, label):
    require(geometry(actual) == geometry(expected), 'Sparse parent-slice geometry differs: ' + label)


def first_difference(left, right, path='state'):
    if type(left) is not type(right):
        return path + ' (type)'
    if isinstance(left, dict):
        if set(left) != set(right):
            return path + ' (keys)'
        for key in sorted(left):
            if left[key] != right[key]:
                return first_difference(left[key], right[key], path + '/' + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ' (length)'
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return first_difference(a, b, path + '/' + str(index))
    return path


class _Manifest:
    def __init__(self, runtime, fingerprints, source_files, arithmetic_modules, active_policy):
        from bitsandbytes.functional import QuantState
        self.QuantState = QuantState
        self.runtime = runtime
        self.fingerprints = fingerprints
        self.source_files = source_files
        self.arithmetic_modules = arithmetic_modules
        self.active_policy = active_policy
        self.storages, self.states, self.coverage, self.mutable_storage = {}, {}, {}, set()
        self.hash_bytes = 0
        self.call_stack = set()
        self.functions = {}
        self.config_stack = set()
        self.quant_paths = {}
        for kind, iterator in [('parameter', runtime.model.named_parameters(remove_duplicate=False)),
                               ('buffer', runtime.model.named_buffers(remove_duplicate=False))]:
            for name, tensor in iterator:
                key = 'model/' + kind + '/' + name
                skey = storage_key(tensor)
                if key not in fingerprints:
                    if kind == 'parameter':
                        self.mutable_storage.add(skey)
                    continue
                expected = fingerprints[key]
                require(list(tensor.shape) == expected['shape'] and str(tensor.dtype) == expected['dtype'],
                        'Named frozen tensor coverage geometry differs: ' + key)
                if tensor.is_contiguous() and tensor.storage_offset() == 0 and tensor.numel() * tensor.element_size() == skey[-1]:
                    self.coverage.setdefault(skey, []).append(key)

    def tensor(self, tensor):
        require(not tensor.requires_grad and tensor.grad is None, 'Supplemental frozen tensor is trainable or has a gradient')
        skey = storage_key(tensor)
        require(skey not in self.mutable_storage, 'Quantization metadata aliases an excluded/trainable parameter')
        key = '|'.join(map(str, skey))
        if key not in self.storages:
            if skey in self.coverage:
                # Entire storage is already covered by fresh named-tensor hashing.
                entry = {'named_tensor_coverage': sorted(self.coverage[skey]), 'bytes_hashed_here': 0}
            else:
                require(tensor.layout == torch.strided and tensor.device.type != 'meta', 'Unsupported metadata storage')
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    tensor.untyped_storage(), 0, (skey[-1],), (1,))
                hasher = hashlib.sha256()
                update_tensor_hash(hasher, raw)
                entry = {'sha256_raw_storage': hasher.hexdigest(), 'bytes_hashed_here': skey[-1]}
                self.hash_bytes += skey[-1]
            self.storages[key] = entry
        return {'tensor_object_id': id(tensor), **geometry(tensor), 'storage_record': key}

    def function(self, function):
        owner = None
        if isinstance(function, (staticmethod, classmethod)):
            function = function.__func__
        if isinstance(function, types.MethodType):
            owner, function = id(function.__self__), function.__func__
        key = str(id(function)) + '/' + str(owner)
        if key in self.functions:
            return {'function_ref': key}
        identity = {'object_id': id(function), 'owner_id': owner,
                    'module': getattr(function, '__module__', None),
                    'qualname': getattr(function, '__qualname__', getattr(function, '__name__', None)),
                    'type': type(function).__module__ + '.' + type(function).__qualname__}
        self.functions[key] = identity
        if id(function) in self.call_stack:
            return {'function_ref': key}
        self.call_stack.add(id(function))
        try:
            if hasattr(function, '__code__'):
                identity.update(code_sha256=digest(marshal.dumps(function.__code__)),
                                defaults=self.value(function.__defaults__), kwdefaults=self.value(function.__kwdefaults__))
                if function.__closure__:
                    identity['closure'] = [self.value(cell.cell_contents) for cell in function.__closure__]
            if isinstance(function, type):
                identity['class_functions'] = {name: self.function(value) for name, value in vars(function).items()
                    if name in {'forward', 'backward', 'apply', 'as_dict', 'from_dict', '__call__', '__torch_function__'}
                    and isinstance(value, (types.FunctionType, staticmethod, classmethod))}
            if isinstance(function, functools.partial):
                identity.update(partial_function=self.function(function.func),
                                partial_args=self.value(function.args), partial_keywords=self.value(function.keywords))
            return {'function_ref': key}
        finally:
            self.call_stack.remove(id(function))

    def value(self, value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            return {'float64_bytes': struct.pack('!d', value).hex()}
        if isinstance(value, Enum):
            return {'enum_class': type(value).__module__ + '.' + type(value).__qualname__, 'name': value.name, 'value': self.value(value.value)}
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, self.QuantState):
            return self.state(value)
        if isinstance(value, (torch.dtype, torch.device)):
            return {'type': type(value).__name__, 'value': str(value)}
        if isinstance(value, (list, tuple, torch.Size)):
            return {'sequence_type': type(value).__name__, 'items': [self.value(x) for x in value]}
        if isinstance(value, (set, frozenset)):
            return {'set_type': type(value).__name__, 'items': sorted((self.value(x) for x in value), key=canonical)}
        if isinstance(value, dict):
            require(all(type(key) in (str, int) for key in value), 'Unsupported metadata map key')
            return {'mapping': [[self.value(k), self.value(v)] for k, v in sorted(value.items(), key=lambda item: str(item[0]))]}
        if isinstance(value, types.ModuleType):
            return {'module_object_id': id(value), 'module': value.__name__}
        if isinstance(value, torch.nn.Module):
            return {'module_object_id': id(value), 'class': type(value).__module__ + '.' + type(value).__qualname__}
        if isinstance(value, (staticmethod, classmethod)):
            return self.function(value)
        if callable(value):
            return self.function(value)
        if is_dataclass(value):
            return {'dataclass': type(value).__module__ + '.' + type(value).__qualname__,
                    'attributes': self.value(vars(value))}
        if hasattr(value, 'to_dict') and callable(value.to_dict):
            require(id(value) not in self.config_stack, 'Cyclic configuration object')
            self.config_stack.add(id(value))
            try:
                return {'configuration_object_id': id(value), 'class': type(value).__module__ + '.' + type(value).__qualname__,
                        'attributes': self.value(vars(value)), 'serialized': self.value(value.to_dict())}
            finally:
                self.config_stack.remove(id(value))
        raise ValueError('Unsupported effective-state metadata type: ' + str(type(value)))

    def state(self, state):
        require(type(state) is self.QuantState, 'Unexpected QuantState implementation')
        key = str(id(state))
        if key in self.states:
            return {'quant_state_id': key}
        self.states[key] = None
        require(state.nested is (state.state2 is not None), 'QuantState nesting flag inconsistent')
        require(getattr(state, 'packing_format_for_cpu', False) is False, 'CPU-repacked NF4 is outside canonical contract')
        require(state.dtype in (torch.float16, torch.bfloat16, torch.float32), 'Unsupported canonical decode dtype')
        self.states[key] = {'class_identity': self.function(type(state)),
                            'attributes': {key: self.value(value) for key, value in sorted(vars(state).items())},
                            'packing_format_for_cpu_effective': False}
        return {'quant_state_id': key}

    def packed(self, tensor, name):
        key = 'model/parameter/' + name
        require(key in self.fingerprints and storage_key(tensor) in self.coverage,
                'Packed parent is not fully covered by named frozen hashes: ' + name)
        require(tensor.dtype == torch.uint8 and tensor.is_contiguous()
                and not tensor.requires_grad and tensor.grad is None, 'Unexpected packed parent contract')
        return {'name': key, 'geometry': geometry(tensor), 'tensor_object_id': id(tensor),
                'content_identity_owned_by_fork_guard': self.fingerprints[key]}

    def sparse(self, module, name):
        require(set(module._nf4_views) == {'gate_up_proj', 'down_proj'}, 'Sparse projection inventory changed')
        projections = {}
        for projection, views in sorted(module._nf4_views.items()):
            chain = module.parametrizations[projection]
            require(len(chain) == 1, 'Unexpected expert parametrization chain')
            packed, state = chain.original, chain[0].quant_state
            parent_name = name + '.parametrizations.' + projection + '.original'
            parent = self.packed(packed, parent_name)
            require(len(state.shape) == 3 and state.quant_type == 'nf4', 'Unexpected expert parent quantization')
            count, rows, cols = state.shape
            size, blocks = rows * cols, rows * cols // state.blocksize
            require(size % state.blocksize == 0 and size % 2 == 0 and packed.numel() * 2 == count * size,
                    'Expert parent packing/block alignment differs')
            require(len(views) == count, 'Sparse expert count changed')
            self.quant_paths[parent_name] = self.state(state)
            logical = []
            for index, pair in enumerate(views):
                require(isinstance(pair, tuple) and len(pair) == 2, 'Sparse entry schema changed')
                data, view = pair
                label = name + '/' + projection + '/' + str(index)
                begin, end = index * blocks, (index + 1) * blocks
                expected = packed.reshape(-1)[index * size // 2:(index + 1) * size // 2].view(-1, 1)
                same_slice(data, expected, label + '/packed')
                require(tuple(view.shape) == (rows, cols) and view.dtype == state.dtype
                        and view.blocksize == state.blocksize and view.quant_type == state.quant_type
                        and view.nested == state.nested and view.code is state.code and view.offset is state.offset,
                        'Sparse quant-state parent relationship differs: ' + label)
                same_slice(view.absmax, state.absmax[begin:end], label + '/absmax')
                if state.nested:
                    nested, target = state.state2, view.state2
                    require(blocks % nested.blocksize == 0 and not nested.nested and target is not None,
                            'Sparse nested block alignment differs')
                    require(tuple(target.shape) == (blocks,) and target.blocksize == nested.blocksize
                            and target.dtype == nested.dtype and target.code is nested.code
                            and target.quant_type is None and target.offset is None
                            and target.state2 is None and target.nested is False, 'Sparse nested-state relationship differs')
                    same_slice(target.absmax, nested.absmax[begin // nested.blocksize:end // nested.blocksize], label + '/nested_absmax')
                else:
                    require(view.state2 is None, 'Unexpected sparse nested state')
                logical.append({'index': index, 'packed_geometry': geometry(data), 'packed_tensor_object_id': id(data),
                                'state': self.state(view), 'parent_block_interval': [begin, end]})
            projections[projection] = {'parent': parent, 'state': self.state(state), 'views': logical}
        return projections

    def build(self):
        from bitsandbytes.nn import Linear4bit
        from bitsandbytes import functional as bnb
        import torch.nn.utils.parametrize as parametrize
        require(parametrize._cache_enabled == 0 and not parametrize._cache, 'Dense parametrization cache persists at boundary')
        require(self.source_files and self.arithmetic_modules, 'Explicit source and arithmetic identities required')
        sources = {str(path): file_digest(path) for path in self.source_files}
        require(sources == dict(self.source_files), 'Pinned effective-state source file changed')
        modules, linears, experts = {}, {}, {}
        for name, module in self.runtime.model.named_modules(remove_duplicate=False):
            settings = {}
            for key, value in sorted(vars(module).items()):
                if key.startswith('_') or key in {'training', 'active_adapter', 'active_adapters'}:
                    continue
                settings[key] = self.value(value)
            modules[name] = {'class': self.function(type(module)), 'forward': self.function(module.forward),
                'call': self.function(module.__call__), 'call_impl': self.function(module._call_impl),
                'execution_overrides': {key: self.value(getattr(module, key)) for key in (
                    '_compiled_call_impl', '_gradient_checkpointing_func', '_disable_adapters') if hasattr(module, key)},
                'public_settings': settings, 'forward_hooks': self.value(module._forward_hooks),
                'forward_pre_hooks': self.value(module._forward_pre_hooks),
                'backward_hooks': self.value(module._backward_hooks),
                'backward_pre_hooks': self.value(module._backward_pre_hooks),
                'full_backward_hook_flag': self.value(module._is_full_backward_hook),
                'always_call_hook_ids': self.value(module._forward_hooks_always_called),
                'hook_kwargs_ids': self.value(module._forward_hooks_with_kwargs),
                'pre_hook_kwargs_ids': self.value(module._forward_pre_hooks_with_kwargs)}
            if isinstance(module, Linear4bit):
                packed, state = module.weight, module.weight.quant_state
                require(state is not None and state.quant_type == 'nf4', 'Missing/non-NF4 linear state')
                require(getattr(module, 'quant_state', state) is state and packed.module is module,
                        'Linear/module quant-state link differs')
                self.quant_paths[name + '.weight'] = self.state(state)
                attrs = {key: self.value(value) for key, value in sorted(vars(packed).items()) if key != 'module'}
                linears[name] = {'packed': self.packed(packed, name + '.weight'), 'weight_attributes': attrs}
                if self.active_policy:
                    canonical_forward = getattr(self.arithmetic_modules.get('nf4_linear'), 'canonical_forward', None)
                    require(module.compute_dtype == torch.float32 and module.compute_type_is_set is True
                            and canonical_forward is not None
                            and getattr(module.forward, '__func__', None) is canonical_forward,
                            'Active numerical policy is not installed on every NF4 linear')
            if hasattr(module, '_nf4_views'):
                experts[name] = self.sparse(module, name)
        # Discover quant-state attributes independently of the linear/view traversal.
        for name, module in self.runtime.model.named_modules(remove_duplicate=False):
            for key, value in vars(module).items():
                if isinstance(value, self.QuantState):
                    require(str(id(value)) in self.states, 'Uncovered module QuantState: ' + name + '/' + key)
            for key, parameter in module._parameters.items():
                if parameter is not None and getattr(parameter, 'quant_state', None) is not None:
                    require(str(id(parameter.quant_state)) in self.states, 'Uncovered parameter QuantState')
        require(linears and experts, 'No qualified linear or sparse expert state found')
        arithmetic = {}
        for name, module in sorted(self.arithmetic_modules.items()):
            arithmetic[name] = {key: self.value(value) for key, value in sorted(vars(module).items())
                                if not key.startswith('__')}
        surfaces = {'torch_linear': self.function(torch.nn.functional.linear),
                    'dequantize_4bit': self.function(bnb.dequantize_4bit),
                    'dequantize_blockwise': self.function(bnb.dequantize_blockwise),
                    'QuantState': self.function(bnb.QuantState)}
        glm = sys.modules.get('transformers.models.glm4_moe_lite.modeling_glm4_moe_lite')
        if glm is not None:
            surfaces['glm'] = {key: self.function(getattr(glm, key)) for key in (
                'eager_attention_forward', 'Glm4MoeLiteTopkRouter', 'Glm4MoeLiteRMSNorm', 'Glm4MoeLiteRotaryEmbedding')}
        cuda_ops = sys.modules.get('bitsandbytes.backends.cuda.ops')
        if cuda_ops is not None:
            surfaces['cuda_gemm_dispatch'] = self.function(cuda_ops._gemm_4bit_use_custom_fn)
        rotary = self.runtime.backbone.rotary_emb
        rotary_state = {key: self.value(value) for key, value in sorted(vars(rotary).items())
                        if key not in {'_parameters', '_modules', '_buffers'} and not key.startswith('_forward')
                        and key != 'training'}
        rotary_state['buffers'] = {key: self.tensor(value) for key, value in rotary._buffers.items() if value is not None}
        contract = getattr(self.runtime, 'nf4_linear_contract', None)
        if self.active_policy:
            require(isinstance(contract, dict) and contract.get('linears') == len(linears)
                    and contract.get('forward_backward_same_weight') is True
                    and contract.get('parameter_values_changed') is False, 'Missing active NF4 numerical contract')
            require(not torch.is_autocast_enabled(torch.device(self.runtime.input_device).type),
                    'Autocast is enabled inside the canonical FP32 policy')
        torch_arithmetic = {
            'default_dtype': str(torch.get_default_dtype()),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cuda_matmul_allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'cuda_matmul_allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'autocast': {device: {'enabled': torch.is_autocast_enabled(device),
                                  'dtype': str(torch.get_autocast_dtype(device))} for device in ('cpu', 'cuda')},
        }
        module_api = torch.nn.modules.module
        global_hooks = {key: self.value(value) for key, value in vars(module_api).items()
                        if key.startswith('_global_') and 'hook' in key}
        return {'schema': 'hive-effective-frozen-state-v1', 'active_policy': self.active_policy,
                'decode_convention': 'dequantize_4bit in QuantState.dtype; canonical linears then promote to FP32',
                'frozen_fingerprints_sha256': digest(canonical(self.fingerprints)),
                'source_files': sources, 'model_modules': modules, 'linear_parents': linears,
                'sparse_experts': experts, 'quant_paths': self.quant_paths, 'quant_states': self.states,
                'metadata_storages': self.storages, 'rotary': rotary_state, 'arithmetic_modules': arithmetic,
                'torch_arithmetic': torch_arithmetic,
                'global_module_hooks': global_hooks,
                'numerical_surfaces': surfaces, 'nf4_linear_contract': self.value(contract), 'function_identities': self.functions,
                'counts': {'nf4_linears': len(linears), 'nf4_expert_tensors': sum(len(x) for x in experts.values()),
                           'sparse_views': sum(len(p['views']) for x in experts.values() for p in x.values()),
                           'quant_state_objects': len(self.states), 'metadata_storages': len(self.storages)},
                'composition': 'Requires fresh fork_recovery.assert_fork_state at this boundary; named tensor bytes not rehashed here'}


@dataclass(frozen=True)
class EffectiveStateSnapshot:
    manifest_bytes: bytes
    frozen_fingerprints_bytes: bytes
    source_files: tuple
    arithmetic_modules: tuple
    active_policy: bool

    @property
    def sha256(self):
        return digest(self.manifest_bytes)

    def manifest(self):
        return json.loads(self.manifest_bytes)


def capture_effective_state(runtime, *, frozen_fingerprints, source_files, arithmetic_modules,
                            expected_counts=None, active_policy=True):
    """Capture after decoded admission/fork setup, paired with fresh named hashes."""
    before = time.monotonic()
    fingerprints = json.loads(canonical(dict(frozen_fingerprints)))
    builder = _Manifest(runtime, fingerprints, dict(source_files), dict(arithmetic_modules), active_policy)
    manifest = builder.build()
    for key, value in (expected_counts or {}).items():
        require(manifest['counts'].get(key) == value, 'Effective-state inventory count differs: ' + key)
    snapshot = EffectiveStateSnapshot(canonical(manifest), canonical(fingerprints), tuple(sorted(source_files.items())),
                                      tuple(sorted(arithmetic_modules.items())), active_policy)
    return snapshot, {'passed': True, 'sha256': snapshot.sha256, 'counts': manifest['counts'],
                      'supplemental_bytes_hashed': builder.hash_bytes, 'seconds': time.monotonic() - before,
                      'named_tensor_content_bytes_hashed': 0}


def assert_effective_state(runtime, snapshot):
    """Assert same process-local topology and complete metadata at a matching boundary."""
    before = time.monotonic()
    builder = _Manifest(runtime, json.loads(snapshot.frozen_fingerprints_bytes), dict(snapshot.source_files),
                        dict(snapshot.arithmetic_modules), snapshot.active_policy)
    current = builder.build()
    expected = snapshot.manifest()
    require(current == expected, 'Effective frozen state changed at ' + first_difference(expected, current))
    return {'passed': True, 'sha256': snapshot.sha256, 'counts': current['counts'],
            'supplemental_bytes_hashed': builder.hash_bytes, 'seconds': time.monotonic() - before,
            'named_tensor_content_bytes_hashed': 0}
