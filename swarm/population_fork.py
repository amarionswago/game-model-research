"""Assemble separately formed specialists into one matched continuation fork.

AI-authored. Reads adapter checkpoints only. Does not train, launch workers or
carry formation optimizer state into a fresh I/C continuation stage.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import random

import torch


def file_sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024**2),b''):value.update(chunk)
    return value.hexdigest()


def tensor_sha256(parameters):
    """Same canonical parameter digest emitted by each formation worker."""
    value=hashlib.sha256()
    for name,tensor in sorted(parameters.items()):
        if not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32:
            raise ValueError('Expected FP32 adapter tensors')
        value.update(name.encode());value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def assemble_specialists(initial_path, checkpoints, *, initial_sha256,
                         initial_tensor_sha256, expected_recipe,
                         expected_identities, expected_sha256,
                         formation_updates=128):
    """Return a joint adapter/RNG fork after validating both complete parents.

    Caller supplies separately verified provider output hashes and expected
    identities. Partial formation is rejected, not relabelled a completed S.
    I/C use the same saved common RNG and fresh Adam/scheduler/scaler state.
    """
    if len(checkpoints)!=2 or len(expected_identities)!=2 or len(expected_sha256)!=2:
        raise ValueError('Exactly two specialist parents required')
    if formation_updates!=128:
        raise ValueError('Declared full formation dose is 128 updates')
    if file_sha256(initial_path)!=initial_sha256:
        raise ValueError('Common initialization file hash mismatch')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
    if initial.get('recipe_identity')!=expected_recipe:
        raise ValueError('Common initialization recipe mismatch')
    if tensor_sha256(initial['adapters'])!=initial_tensor_sha256:
        raise ValueError('Common initialization tensor digest mismatch')
    if set(initial['rng'])!={'python','torch','cuda'}:
        raise ValueError('Common initialization RNG state incomplete')
    common=('base','tokenizer','code','recipe','seed')
    if any(expected_identities[0].get(k)!=expected_identities[1].get(k) for k in common):
        raise ValueError('Specialists do not share base/code/recipe/seed identity')
    if expected_identities[0].get('recipe')!=expected_recipe:
        raise ValueError('Parent and initialization recipes differ')
    adapters={};parents=[]
    for member,path in enumerate(checkpoints):
        if file_sha256(path)!=expected_sha256[member]:
            raise ValueError('Specialist checkpoint file hash mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        identity=expected_identities[member]
        if state.get('format')!=1 or state.get('identity')!=identity:
            raise ValueError('Specialist checkpoint identity mismatch')
        if identity.get('stage')!=f'S_formation_member{member}' or identity.get('seed')!=17:
            raise ValueError('Wrong specialist role or seed')
        cursor=state['cursor']
        if (cursor.get('completed_updates')!=formation_updates or not cursor.get('recovery_verified')
                or cursor.get('stream',{}).get('supervised_tokens')!=8192*formation_updates):
            raise ValueError('Specialist formation is incomplete or recovery unverified')
        expected={n:v for n,v in initial['adapters'].items() if f'.member{member}.' in n}
        if not expected or set(state['adapters'])!=set(expected):
            raise ValueError('Specialist parameter ownership mismatch')
        for name,tensor in state['adapters'].items():
            reference=expected[name]
            if (tensor.shape!=reference.shape or tensor.dtype!=torch.float32
                    or not torch.isfinite(tensor).all()):
                raise ValueError('Invalid specialist adapter tensor')
            adapters[name]=tensor.clone()
        parents.append({'member':member,'sha256':expected_sha256[member],
                        'identity':deepcopy(identity),'completed_updates':formation_updates})
    if set(adapters)!=set(initial['adapters']):
        raise ValueError('Assembled population has missing or foreign parameters')
    return {'format':1,'stage':'S_assembled','adapters':adapters,
            'rng':deepcopy(initial['rng']),'parents':parents,
            'initial_population_sha256':initial_sha256,
            'initial_tensor_sha256':initial_tensor_sha256,
            'assembled_tensor_sha256':tensor_sha256(adapters),
            'continuation_rng_source':'common saved post-initialization RNG, restored identically in I/C',
            'fresh_optimizers_schedulers_scaler_required':True,
            'formation_optimizer_states_carried_forward':False}


def restore_specialist_fork(fork, named_groups):
    """Restore identical adapters/RNG before each fresh continuation arm."""
    if (fork.get('format')!=1 or fork.get('stage')!='S_assembled'
            or not fork.get('fresh_optimizers_schedulers_scaler_required')):
        raise ValueError('Invalid assembled specialist fork')
    if len(named_groups)!=2:
        raise ValueError('Exactly two runtime members required')
    parameters={n:p for group in named_groups for n,p in group.items()}
    if (len(parameters)!=sum(len(g) for g in named_groups)
            or set(parameters)!=set(fork['adapters'])):
        raise ValueError('Runtime population parameter mismatch')
    if tensor_sha256(fork['adapters'])!=fork['assembled_tensor_sha256']:
        raise ValueError('Assembled adapter digest changed')
    cuda_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(fork['rng']['cuda'])!=cuda_count:
        raise ValueError('Continuation CUDA RNG device-count mismatch')
    for name,parameter in parameters.items():
        saved=fork['adapters'][name]
        if parameter.shape!=saved.shape or parameter.dtype!=saved.dtype or not torch.isfinite(saved).all():
            raise ValueError('Runtime adapter shape/dtype/finiteness mismatch')
    with torch.no_grad():
        for name,parameter in parameters.items():
            parameter.copy_(fork['adapters'][name]);parameter.grad=None
    random.setstate(fork['rng']['python']);torch.set_rng_state(fork['rng']['torch'])
    if fork['rng']['cuda']:torch.cuda.set_rng_state_all(fork['rng']['cuda'])
    return fork['assembled_tensor_sha256']
