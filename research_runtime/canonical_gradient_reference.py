"""Streamed derivative reference under the SHARED canonical forward convention.

No optimizer, trainer, admission, PEFT forward, custom NF4 backward or checkpoint
is used by the reference. Production uses the caller's reviewed scorer. Reference
layers use native HF eager experts and explicit dense frozen weights / LoRA A,B.
The canonical linear/router/norm/attention FORWARD operations are shared and are
NOT independently certified here. Routes are recomputed, never supplied.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType

import torch
import torch.nn.functional as F
from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite as glm
from stable_arithmetic import ROW_BLOCK, KEY_SPAN, attention_piece, precise_attention, stable_linear


THRESHOLDS = MappingProxyType(dict(primal_atol=1e-6, primal_rtol=1e-6,
    target_logp_atol=1e-3, objective_atol=1e-4, gradient_atol=2e-6,
    gradient_rtol=1e-3, gradient_relative_l2=1e-3, gradient_cosine_min=.999999,
    head_fp64_atol=1e-12, head_fp64_rtol=1e-10))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Canonical gradient reference deadline reached')


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, tuple):
        return tuple(_cpu(x) for x in value)
    return value


def _device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_device(x, device) for x in value)
    return value


def _primal(actual, expected, label):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    _require(actual.shape == expected.shape and actual.dtype == expected.dtype,
             label + ': shape/dtype changed')
    _require(bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
             label + ': nonfinite values')
    error = float((actual - expected).abs().max()) if actual.numel() else 0.
    _require(torch.allclose(actual, expected, atol=THRESHOLDS['primal_atol'],
                            rtol=THRESHOLDS['primal_rtol']), label + ': primal mismatch')
    return error


def _route(actual, expected, label):
    _require(len(actual) == len(expected) == 3, label + ': router tuple changed')
    _require(torch.equal(actual[2].detach().cpu(), expected[2]),
             label + ': route indices/order mismatch')
    return dict(logits_max_abs=_primal(actual[0], expected[0], label + ' logits'),
                weights_max_abs=_primal(actual[1], expected[1], label + ' weights'),
                indices_and_order_exact=True)


class BoundaryCapture:
    """Initial-forward observations only; remove hooks BEFORE backward/recompute."""
    def __init__(self, runtime, record):
        self.runtime = runtime
        self.encoded = record.serial_record()['encoded']
        self.record_sha256 = hashlib.sha256(json.dumps(self.encoded, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        self.inputs, self.outputs, self.kwargs, self.routes = {}, {}, {}, {}
        self.norm_output = None
        self.input_ids_checked = False
        self.input_requires_grad = {}
        self.handles = []

    def __enter__(self):
        def backbone_pre(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            _require(isinstance(ids, torch.Tensor) and ids.detach().cpu().tolist() ==
                     [self.encoded['input_ids']], 'Captured input IDs differ from record')
            _require(kwargs.get('use_cache') is False, 'Cache must be explicitly disabled')
            self.input_ids_checked = True
        self.handles.append(self.runtime.backbone.register_forward_pre_hook(backbone_pre, with_kwargs=True))
        for index, layer in enumerate(self.runtime.backbone.layers):
            def before(module, args, kwargs, index=index):
                _require(index not in self.inputs, 'Repeated initial layer forward')
                x = args[0] if args else kwargs['hidden_states']
                value = _cpu(x)
                if index and torch.equal(value, self.outputs[index-1]):
                    value = self.outputs[index-1]  # Share adjacent boundary storage.
                self.inputs[index] = value
                self.input_requires_grad[index] = bool(x.requires_grad)
                _require(kwargs.get('past_key_values') is None and not kwargs.get('use_cache', False),
                         'Reference does not admit cached layers')
                self.kwargs[index] = {name: _cpu(kwargs[name]) for name in
                    ('attention_mask', 'position_ids', 'position_embeddings') if name in kwargs}
                _require('position_embeddings' in self.kwargs[index], 'Missing native rotary boundary')
            def after(module, args, kwargs, output, index=index):
                _require(isinstance(output, torch.Tensor), 'Unexpected decoder output')
                self.outputs[index] = _cpu(output)
            self.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(after, with_kwargs=True))
            if isinstance(layer.mlp, glm.Glm4MoeLiteMoE):
                def router_after(module, args, output, index=index):
                    _require(index not in self.routes, 'Repeated initial router forward')
                    self.routes[index] = _cpu(output)
                self.handles.append(layer.mlp.gate.register_forward_hook(router_after))
        def norm_after(module, args, output):
            _require(self.norm_output is None, 'Repeated initial final norm')
            self.norm_output = _cpu(output)
        self.handles.append(self.runtime.backbone.norm.register_forward_hook(norm_after))
        return self

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if exc[0] is None:
            count = len(self.runtime.backbone.layers)
            _require(self.input_ids_checked and set(self.inputs) == set(range(count)) and
                     set(self.outputs) == set(range(count)) and self.norm_output is not None,
                     'Incomplete production boundary capture')
            _require(all(self.input_requires_grad.values()), 'Production layer input lacks gradient path')

    def storage_bytes(self):
        seen = set()
        def size(value):
            if isinstance(value, torch.Tensor):
                key = (value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())
                if key in seen:
                    return 0
                seen.add(key)
                return value.untyped_storage().nbytes()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, tuple):
                return sum(size(v) for v in value)
            return 0
        return sum(size(v) for v in (self.inputs, self.outputs, self.kwargs, self.routes, self.norm_output))


def _attention_without_checkpoint(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
    _require(not dropout, 'Reference requires zero attention dropout')
    key = glm.repeat_kv(key, module.num_key_value_groups)
    value = glm.repeat_kv(value, module.num_key_value_groups)
    qn, kn = query.shape[-2], key.shape[-2]
    pieces = []
    for start in range(0, qn, ROW_BLOCK):
        end = min(start + ROW_BLOCK, qn)
        mask = attention_mask
        if mask is not None and mask.shape[-2] == qn:
            mask = mask[..., start:end, :]
        pieces.append(attention_piece(query[:, :, start:end], key, value, mask,
                                      scaling, kn-qn+start))
    return torch.cat(pieces, dim=-2).transpose(1, 2).contiguous(), None


@contextmanager
def reference_forward_context():
    """Nest after production backward inside the still-active policy_numerics."""
    _require(F.linear is stable_linear and glm.eager_attention_forward is precise_attention,
             'Reference requires the active canonical production policy')
    _require(ROW_BLOCK == 64 and KEY_SPAN == 1600, 'Canonical arithmetic dimensions changed')
    current = glm.eager_attention_forward
    try:
        glm.eager_attention_forward = _attention_without_checkpoint
        yield
    finally:
        glm.eager_attention_forward = current


def _frozen_value(module, field):
    """Decode full native base tensors; never reuse sparse expert forward/views."""
    if field == 'weight' and hasattr(module, 'base_layer'):
        module = module.base_layer
    parametrizations = getattr(module, 'parametrizations', {})
    if field in parametrizations:
        chain = parametrizations[field]
        _require(len(chain) == 1 and hasattr(chain[0], 'quant_state'), 'Unknown frozen parametrization')
        from bitsandbytes.functional import dequantize_4bit
        value = dequantize_4bit(chain.original, chain[0].quant_state)
    else:
        value = getattr(module, field)
        state = getattr(value, 'quant_state', None)
        if state is not None:
            from bitsandbytes.functional import dequantize_4bit
            _require(state.quant_type == 'nf4' and not getattr(state, 'packing_format_for_cpu', False),
                     'Unknown canonical NF4 storage')
            value = dequantize_4bit(value.reshape(-1, 1), state)
    _require(not value.requires_grad, 'Reference base must be frozen')
    return value.detach().to(dtype=torch.float32, copy=True)


class ExplicitLoRALinear(torch.nn.Module):
    def __init__(self, weight, source, member):
        super().__init__()
        _require(not source.use_dora.get(member, False) and not source.merged and
                 not source.fan_in_fan_out and not source.disable_adapters,
                 'Unsupported reference LoRA convention')
        _require(list(source.active_adapters) == [member], 'Active adapter boundary changed')
        _require(float(getattr(source.lora_dropout[member], 'p', 0.)) == 0., 'LoRA dropout must be zero')
        _require(member not in getattr(source, 'lora_variant', {}), 'LoRA variants are not admitted')
        _require(source.base_layer.bias is None and source.lora_A[member].bias is None and
                 source.lora_B[member].bias is None, 'Reference only admits bias-free LoRA')
        a, b = source.lora_A[member].weight, source.lora_B[member].weight
        _require(a.ndim == b.ndim == 2 and a.dtype == b.dtype == torch.float32 and
                 a.requires_grad and b.requires_grad, 'Fork matrices must be trainable FP32 matrices')
        self.register_buffer('weight', weight.detach())
        self.A = torch.nn.Parameter(a.detach().clone())
        self.B = torch.nn.Parameter(b.detach().clone())
        self.scale = float(source.scaling[member])
        _require(math.isfinite(self.scale), 'Nonfinite LoRA scaling')

    def forward(self, inputs):
        # Unmerged operation order matches PEFT; gradients use stock autograd.
        return F.linear(inputs, self.weight) + F.linear(F.linear(inputs, self.A), self.B) * self.scale


def independent_layer(runtime, index, *, fork_member, trainable_names):
    """A fresh dense frozen HF layer with explicit independent A/B leaves."""
    original = runtime.backbone.layers[index]
    config = deepcopy(runtime.model.config)
    config._attn_implementation = config._experts_implementation = 'eager'
    with torch.device('meta'):
        layer = glm.Glm4MoeLiteDecoderLayer(config, index)
    values = {}
    for name in layer.state_dict():
        path, field = name.rsplit('.', 1)
        values[name] = _frozen_value(original.get_submodule(path), field)
    layer.load_state_dict(values, strict=True, assign=True)
    layer.requires_grad_(False)
    layer.eval()
    layer.gradient_checkpointing = False
    model_names = {id(p): name for name, p in runtime.model.named_parameters()}
    mapping = {}
    member = 'member' + str(fork_member)
    for path, source in original.named_modules():
        if not hasattr(source, 'lora_A'):
            continue
        _require(member in source.lora_A and member in source.lora_B, 'Missing fork in layer')
        dense = layer.get_submodule(path)
        replacement = ExplicitLoRALinear(dense.weight, source, member)
        parent, field = path.rsplit('.', 1)
        setattr(layer.get_submodule(parent), field, replacement)
        for which in ('A', 'B'):
            source_p = getattr(source, 'lora_' + which)[member].weight
            name = model_names[id(source_p)]
            _require(name in trainable_names and name not in mapping, 'Reference fork membership differs')
            mapping[name] = getattr(replacement, which)
    _require(bool(mapping), 'No reference fork matrices in layer')
    return layer, mapping


def _head_cotangent(runtime, capture, denominator, *, deadline):
    device = runtime.backbone.norm.weight.device
    x = capture.outputs[len(runtime.backbone.layers)-1].to(device).requires_grad_()
    norm = glm.Glm4MoeLiteRMSNorm(runtime.model.config.hidden_size,
                                 runtime.model.config.rms_norm_eps).to(device)
    with torch.no_grad():
        norm.weight.copy_(runtime.backbone.norm.weight.float())
    norm.requires_grad_(False)
    hidden = norm(x)
    norm_error = _primal(hidden, capture.norm_output, 'Final norm')
    labels = torch.tensor(capture.encoded['labels'], device=device)
    positions = (labels[1:] != -100).nonzero().flatten()
    targets = labels[1:][positions]
    _require(len(targets) == capture.encoded['supervised_tokens'], 'Head target alignment changed')
    # encoded_record checks EOS/prefix/padding; independently select shifted labels.
    weight = _frozen_value(runtime.head, 'weight')
    _require(weight.device == device, 'Head/norm device split is outside this reference')
    cotangent = torch.zeros_like(hidden)
    logs, losses, checks = [], [], []
    for start in range(0, len(targets), ROW_BLOCK):
        _check_time(deadline)
        pos, y = positions[start:start+ROW_BLOCK], targets[start:start+ROW_BLOCK]
        h = hidden.detach()[0, pos].clone().requires_grad_()
        logits = F.linear(h, weight).double()
        loss = F.cross_entropy(logits, y, reduction='sum') / denominator
        gh, gz = torch.autograd.grad(loss, (h, logits))
        with torch.no_grad():
            maximum = logits.max(-1, keepdim=True).values
            exponentials = (logits - maximum).exp()
            normalizer = exponentials.sum(-1, keepdim=True)
            logps = logits.gather(-1, y[:, None]).squeeze(-1) - (maximum + normalizer.log()).squeeze(-1)
            explicit = exponentials / normalizer
            explicit[torch.arange(len(y), device=device), y] -= 1.
            explicit /= denominator
            _require(torch.allclose(gz, explicit, atol=THRESHOLDS['head_fp64_atol'],
                                    rtol=THRESHOLDS['head_fp64_rtol']), 'FP64 head cotangent mismatch')
            direct_loss = -logps.sum() / denominator
            _require(torch.allclose(loss, direct_loss, atol=THRESHOLDS['head_fp64_atol'],
                                    rtol=THRESHOLDS['head_fp64_rtol']), 'FP64 head objective mismatch')
            cotangent[0, pos] = gh
            logs.append(logps.cpu())
            losses.append(float(loss))
            checks.append(dict(tokens=len(y), objective_abs=float((loss-direct_loss).abs()),
                               cotangent_max_abs=float((gz-explicit).abs().max())))
        del h, logits, loss, gh, gz, exponentials, normalizer, explicit, maximum
    initial, = torch.autograd.grad(hidden, x, cotangent)
    return initial.detach().cpu(), torch.cat(logs), math.fsum(losses), dict(
        norm_max_abs=norm_error, chunks=checks, target_positions=positions.cpu().tolist(),
        target_count=len(targets), denominator=denominator,
        masked_final_hidden_cotangent_zero=bool(torch.equal(cotangent[0, labels[1:].ne(-100).logical_not().nonzero().flatten()],
            torch.zeros_like(cotangent[0, labels[1:].ne(-100).logical_not().nonzero().flatten()]))),
        last_input_position_cotangent_zero=bool(torch.count_nonzero(cotangent[:, -1]) == 0))


def reference_record(runtime, capture, *, fork_member, trainable_names, denominator,
                     deadline=None, progress=None):
    """Consume captured boundaries after its production graph was consumed."""
    _require(not capture.handles, 'Production capture hooks must be removed before reference')
    with reference_forward_context():
        cotangent, logps, objective, head = _head_cotangent(runtime, capture, denominator, deadline=deadline)
        gradients, layers = {}, []
        for index in reversed(range(len(runtime.backbone.layers))):
            _check_time(deadline)
            layer, mapping = independent_layer(runtime, index, fork_member=fork_member,
                                               trainable_names=trainable_names)
            device = layer.input_layernorm.weight.device
            x = capture.inputs[index].to(device).requires_grad_()
            routes = []
            hook = None
            if isinstance(layer.mlp, glm.Glm4MoeLiteMoE):
                hook = layer.mlp.gate.register_forward_hook(lambda module, args, output: routes.append(output))
            try:
                output = layer(x, **{k: _device(v, device) for k, v in capture.kwargs[index].items()}, use_cache=False)
                error = _primal(output, capture.outputs[index], 'Layer ' + str(index))
                route = None
                if index in capture.routes:
                    _require(len(routes) == 1, 'Missing independently recomputed route')
                    route = _route(routes[0], capture.routes[index], 'Layer ' + str(index))
                else:
                    _require(not routes, 'Unexpected reference route')
                values = torch.autograd.grad(output, (x, *mapping.values()),
                    grad_outputs=cotangent.to(device), allow_unused=False)
                cotangent = values[0].detach().cpu()
                for (name, parameter), value in zip(mapping.items(), values[1:]):
                    _require(name not in gradients and value.shape == parameter.shape and
                             bool(torch.isfinite(value).all()), 'Invalid reference matrix gradient')
                    gradients[name] = value.detach().cpu()
                prefix = len(capture.encoded['input_ids']) - capture.encoded['supervised_tokens']
                layers.append(dict(index=index, primal_max_abs=error, route=route,
                    matrices=len(mapping), input_cotangent_norm=float(cotangent.double().norm()),
                    conditioning_input_cotangent_norm=float(cotangent[:, :prefix].double().norm())))
            finally:
                if hook is not None:
                    hook.remove()
            del layer, mapping, x, output, values, routes
            if progress is not None:
                progress(phase='gradient_reference_layer', layer=index, record_sha256=capture.record_sha256)
        _require(set(gradients) == set(trainable_names), 'Reference does not cover every fork matrix')
    return gradients, logps, objective, dict(head=head, layers=layers,
        record_sha256=capture.record_sha256, captured_cpu_storage_bytes=capture.storage_bytes())


def compare_gradients(actual, expected):
    _require(bool(expected) and set(actual) == set(expected), 'Gradient key sets differ')
    squared_error = squared_actual = squared_expected = dot = 0.
    matrices = []
    for name in sorted(expected):
        a, b = actual[name].detach().cpu(), expected[name].detach().cpu()
        _require(a.shape == b.shape and a.dtype == b.dtype == torch.float32 and a.ndim == 2,
                 'Gradient shape/dtype mismatch: ' + name)
        _require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'Nonfinite gradient: ' + name)
        difference = (a-b).abs()
        allowance = THRESHOLDS['gradient_atol'] + THRESHOLDS['gradient_rtol'] * b.abs()
        _require(bool((difference <= allowance).all()), 'Gradient element tolerance exceeded: ' + name)
        ad, bd = a.double(), b.double()
        squared_error += float((ad-bd).square().sum())
        squared_actual += float(ad.square().sum())
        squared_expected += float(bd.square().sum())
        dot += float((ad*bd).sum())
        matrices.append(dict(name=name, elements=a.numel(), max_abs=float(difference.max()),
                             max_tolerance_fraction=float((difference/allowance).max())))
    _require(squared_expected > 0 and squared_actual > 0, 'Entire fork gradient is zero')
    relative = math.sqrt(squared_error/squared_expected)
    cosine = dot/math.sqrt(squared_actual*squared_expected)
    _require(relative <= THRESHOLDS['gradient_relative_l2'] and
             cosine >= THRESHOLDS['gradient_cosine_min'], 'Aggregate gradient tolerance exceeded')
    return dict(passed=True, matrices=matrices, matrix_count=len(matrices),
                elements=sum(x['elements'] for x in matrices), relative_l2=relative,
                cosine=cosine, production_norm=math.sqrt(squared_actual),
                reference_norm=math.sqrt(squared_expected))


@dataclass
class GradientReferenceResult:
    diagnostics: dict
    production_gradients: dict
    reference_gradients: dict


def run_gradient_reference(runtime, records, *, fork_member, trainable_names, denominator,
                           selected_revision, historical_selected, frozen_modules=(),
                           deadline=None, progress=None):
    """Zero-optimizer full-batch preflight; clears model grads on entry AND exit.

    Invoke inside policy_numerics after training/checkpoint setup. Caller must
    restore its complete C0/RNG afterward. Deadline is time.monotonic seconds;
    it is checked between bounded operations, not a preemptive cancellation.
    """
    from revision_objective import target_nll, require_trainable_membership
    records, names = tuple(records), frozenset(trainable_names)
    _require(type(fork_member) is int and fork_member >= 2 and bool(records), 'Invalid fork/batch')
    _require(type(denominator) is int and denominator == sum(len(r.target_ids) for r in records),
             'Batch must use its exact total target denominator')
    _require(runtime.model.training and all(getattr(l, 'gradient_checkpointing', False)
             for l in runtime.backbone.layers), 'Production training/checkpoint setup is missing')
    _require(F.linear is stable_linear and glm.eager_attention_forward is precise_attention,
             'Call inside the active canonical numerical policy')
    parameters = require_trainable_membership(runtime.model, names, frozen_modules=frozen_modules)
    _require(all('.member'+str(fork_member)+'.' in name and '.lora_' in name for name in names),
             'Only the declared fork may be differentiated')
    devices = sorted({str(p.device) for p in runtime.model.parameters() if p.device.type == 'cuda'})
    def sync():
        for device in devices:
            torch.cuda.synchronize(device)
    total_started = time.monotonic()
    production_loss = reference_loss = 0.
    reference_sum, details = {}, []
    runtime.model.zero_grad(set_to_none=True)
    try:
        for record_index, record in enumerate(records):
            _check_time(deadline)
            sync(); started = time.monotonic()
            with BoundaryCapture(runtime, record) as capture:
                logs, hidden = selected_revision(runtime, record, fork_member=fork_member,
                    trainable_names=names, historical_selected=historical_selected,
                    frozen_modules=frozen_modules)
            sync(); forward_seconds = time.monotonic()-started
            loss = target_nll(logs, denominator=denominator)
            production_loss += float(loss.detach())
            saved_logs = logs.detach().cpu()
            sync(); started = time.monotonic()
            loss.backward()
            sync(); backward_seconds = time.monotonic()-started
            del logs, hidden, loss
            require_trainable_membership(runtime.model, names, frozen_modules=frozen_modules)
            sync(); started = time.monotonic()
            gradients, ref_logs, ref_loss, detail = reference_record(runtime, capture,
                fork_member=fork_member, trainable_names=names, denominator=denominator,
                deadline=deadline, progress=progress)
            sync(); reference_seconds = time.monotonic()-started
            _require(saved_logs.shape == ref_logs.shape and bool(torch.isfinite(ref_logs).all()),
                     'Reference target shape/finiteness differs')
            log_error = float((saved_logs.double()-ref_logs).abs().max())
            _require(log_error <= THRESHOLDS['target_logp_atol'], 'Target log-probability mismatch')
            reference_loss += ref_loss
            for name, value in gradients.items():
                if name in reference_sum:
                    reference_sum[name].add_(value)
                else:
                    reference_sum[name] = value.clone()
            detail.update(record_index=record_index, target_logp_max_abs=log_error,
                production_forward_seconds=forward_seconds, production_backward_seconds=backward_seconds,
                reference_seconds=reference_seconds)
            details.append(detail)
            del capture, gradients, ref_logs, saved_logs
        _require(abs(production_loss-reference_loss) <= THRESHOLDS['objective_atol'],
                 'Full-batch objective mismatch')
        production = {}
        for name, parameter in parameters.items():
            _require(parameter.grad is not None, 'Missing production fork gradient: ' + name)
            production[name] = _cpu(parameter.grad)
        comparison = compare_gradients(production, reference_sum)
        require_trainable_membership(runtime.model, names, frozen_modules=frozen_modules)
        return GradientReferenceResult(dict(passed=True, thresholds=dict(THRESHOLDS),
            denominator=denominator, record_count=len(records), production_objective=production_loss,
            reference_objective=reference_loss, objective_abs=abs(production_loss-reference_loss),
            records=details, gradients=comparison, total_seconds=time.monotonic()-total_started,
            physical_optimizer_calls=0, reference_convention='shared canonical forward; independent dense-base/unmerged-LoRA VJP',
            shared_forward_not_independently_certified=['stable_linear','padded_row_call/router/norm',
                'attention_piece','native HF decoder/router/expert forward'],
            recomputed_routes_never_injected=True), production, reference_sum)
    finally:
        runtime.model.zero_grad(set_to_none=True)
