"""Atomic recovery of a frozen-base population's trainable state. AI-authored."""
from copy import deepcopy
from pathlib import Path
import random

import torch


def compare_population_parameters(parameters, expected, *, rtol=1e-5, atol=1e-6):
    """Return durable replay diagnostics without relaxing the acceptance test."""
    if set(parameters) != set(expected) or not parameters:
        raise ValueError('Replay parameter keys mismatch')
    differences, failures = {}, {}
    for name, parameter in parameters.items():
        actual = parameter.detach().cpu()
        reference = expected[name].detach().cpu()
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise ValueError('Replay parameter shape/dtype mismatch')
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
        differences[name] = float((actual-reference).abs().max()) if finite else None
        if not torch.allclose(actual, reference, rtol=rtol, atol=atol):
            failures[name] = {'finite': finite, 'elements_outside_tolerance':
                             int((~torch.isclose(actual, reference, rtol=rtol, atol=atol)).sum())}
    return {'passed': not failures, 'rtol': rtol, 'atol': atol,
            'max_abs_difference': max((v for v in differences.values() if v is not None), default=None),
            'parameter_max_abs_differences': differences, 'failed_parameters': failures}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def save_population(path, parameters, optimizers, schedulers, *, identity, cursor, scaler=None):
    """Save at an update boundary; cursor includes sampler position/loss masks.

    identity must identify the immutable base, code, recipe, data and tokenizer.
    Parameters must contain adapters only, including currently inactive members.
    The caller owns NumPy or other sampler RNGs and includes them in cursor using
    primitive/tensor types if it uses them; global Python/Torch RNGs are stored.
    """
    if not parameters or len(optimizers) != len(schedulers):
        raise ValueError("Missing parameters or mismatched optimizer/scheduler counts")
    if any(p.grad is not None for p in parameters.values()):
        raise ValueError("Checkpoint requires cleared gradients at an update boundary")
    if not identity or not cursor:
        raise ValueError("Checkpoint needs explicit identity and progress")
    ids = [id(p) for opt in optimizers for group in opt.param_groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in parameters.values()}:
        raise ValueError("Optimizers must own every adapter exactly once")
    state = {'format': 1, 'identity': deepcopy(identity), 'cursor': _cpu(cursor),
             'adapters': _cpu(parameters), 'optimizers': [_cpu(opt.state_dict()) for opt in optimizers],
             'schedulers': [deepcopy(s.state_dict()) for s in schedulers],
             'scaler': deepcopy(scaler.state_dict()) if scaler is not None else None,
             'rng': {'python': random.getstate(), 'torch': torch.get_rng_state(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_population(path, parameters, optimizers, schedulers, *, identity, scaler=None):
    """Load a checkpoint produced by this runner after validating its identity.

    Instantiate the same parameter groups/scheduler/scaler configuration first.
    Incompatible restore failures invalidate the process; never continue from a
    partly restored state. Does not load a new base or change active adapters.
    """
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 1 or state.get('identity') != identity:
        raise ValueError("Checkpoint identity mismatch")
    if set(state['adapters']) != set(parameters):
        raise ValueError("Adapter keys mismatch")
    if len(state['optimizers']) != len(optimizers) or len(state['schedulers']) != len(schedulers):
        raise ValueError("Population optimizer/scheduler count mismatch")
    if (state['scaler'] is None) != (scaler is None):
        raise ValueError("Cannot discard or invent precision-scaler state")
    if len(state['rng']['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA RNG device-count mismatch")
    for name, parameter in parameters.items():
        saved = state['adapters'][name]
        if saved.shape != parameter.shape or saved.dtype != parameter.dtype or not torch.isfinite(saved).all():
            raise ValueError("Adapter shape/dtype/finiteness mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state['adapters'][name])
            parameter.grad = None
    for opt, saved in zip(optimizers, state['optimizers']):
        opt.load_state_dict(saved)
    for scheduler, saved in zip(schedulers, state['schedulers']):
        scheduler.load_state_dict(saved)
    if scaler is not None:
        scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng']['python'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['cursor']
