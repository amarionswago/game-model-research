"""Restore exact exported NF4 bytes; never requantize or construct an optimizer."""
from pathlib import Path
from types import SimpleNamespace
import gc, hashlib, json, re
import torch

def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_base(folder, *, placement, progress=lambda **kw: None):
    from transformers import AutoModelForCausalLM, Glm4MoeLiteConfig
    from bitsandbytes.nn import Linear4bit, Params4bit
    from bitsandbytes.nn.parametrize import replace_parameter_4bit_prequantized
    from bitsandbytes.functional import dequantize_4bit
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    config_dict = dict(manifest['config'])
    config_dict.pop('quantization_config', None)
    config = Glm4MoeLiteConfig(**config_dict)
    config._experts_implementation = 'eager'
    with torch.device('meta'):
        base = AutoModelForCausalLM.from_config(config, dtype=torch.float16, attn_implementation='eager')
    records = manifest['records']
    expected = set(dict(base.named_parameters())) | {
        name+'.'+key for name, module in base.named_modules()
        for key, value in module._buffers.items()
        if value is not None and key not in module._non_persistent_buffers_set}
    assert {r['name'] for r in records} == expected, 'Incomplete/extra exported model state'
    for index, record in enumerate(records):
        name = record['name']; path = folder/record['file']
        assert path.stat().st_size == record['bytes'] and file_sha(path) == record['sha256']
        package = torch.load(path, map_location='cpu', weights_only=True)
        parent_name, key = name.rsplit('.', 1)
        module = base.get_submodule(parent_name)
        device = torch.device(placement(name))
        if record['kind'] == 'nf4_linear':
            assert isinstance(module, torch.nn.Linear) and module.bias is None and key == 'weight'
            with torch.device('meta'):
                replacement = Linear4bit(module.in_features, module.out_features, bias=False,
                                         compute_dtype=torch.float16, compress_statistics=True, quant_type='nf4')
            replacement.weight = Params4bit.from_prequantized(package['packed'], dict(package['quant_state']),
                                                             device=device, requires_grad=False, module=replacement)
            replacement.compute_type_is_set = True
            outer, attr = parent_name.rsplit('.', 1)
            setattr(base.get_submodule(outer), attr, replacement)
            actual = dequantize_4bit(replacement.weight, replacement.weight.quant_state)
        elif record['kind'] == 'nf4_expert':
            setattr(module, key, torch.nn.Parameter(package['packed'].to(device), requires_grad=False))
            replace_parameter_4bit_prequantized(module, key, dict(package['quant_state']), device)
            actual = getattr(module, key)
        else:
            value = package['value'].to(device)
            if record['kind'] == 'buffer': setattr(module, key, value)
            else: setattr(module, key, torch.nn.Parameter(value, requires_grad=False))
            actual = value
        expected_digest = record.get('decoded_sha256', record.get('value_sha256'))
        assert list(actual.shape) == record['shape'] and digest(actual) == expected_digest, name
        del actual, package
        if index % 20 == 0 or index+1 == len(records):
            progress(stage='load_exact_nf4', loaded_tensors=index+1, total_tensors=len(records))
        gc.collect()
    rotary = base.model.rotary_emb
    device = torch.device(placement('model.embed_tokens.weight'))
    rotary.inv_freq = torch.tensor(manifest['rotary_inv_freq'], device=device, dtype=torch.float32)
    rotary.original_inv_freq = rotary.inv_freq.clone()
    rotary.attention_scaling = manifest['rotary_attention_scaling']
    assert not any(p.is_meta for p in base.parameters())
    assert not any(b.is_meta for b in base.buffers())
    base.is_loaded_in_4bit = True
    base.requires_grad_(False)
    return base, manifest

def wrap_population(base, *, rank=16, placement):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(task_type='CAUSAL_LM', target_modules=['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'],
                        r=rank, lora_alpha=2*rank, lora_dropout=0, bias='none')
    model = get_peft_model(base, config, adapter_name='member0')
    model.add_adapter('member1', config)
    groups = [{n: p for n, p in model.named_parameters() if '.lora_' in n and f'.member{i}.' in n} for i in (0,1)]
    assert all(len(g) == base.config.num_hidden_layers*10 for g in groups)
    assert all(p.dtype == torch.float32 for g in groups for p in g.values())
    def move(value, device):
        if isinstance(value, torch.Tensor): return value.to(device)
        if isinstance(value, tuple): return tuple(move(v,device) for v in value)
        if isinstance(value, list): return [move(v,device) for v in value]
        if isinstance(value, dict): return {k:move(v,device) for k,v in value.items()}
        return value
    def hook(device):
        def pre(module, args, kwargs): return move(args,device), move(kwargs,device)
        return pre
    for i, layer in enumerate(base.model.layers):
        layer.register_forward_pre_hook(hook(placement(f'model.layers.{i}.input_layernorm.weight')), with_kwargs=True)
    base.model.norm.register_forward_pre_hook(hook(placement('model.norm.weight')), with_kwargs=True)
    model.eval()
    return SimpleNamespace(model=model, groups=groups, backbone=base.model, head=base.lm_head,
                           input_device=placement('model.embed_tokens.weight'))

def restore(runtime, path, expected_sha):
    from gated_training import MixingGate
    assert file_sha(path) == expected_sha
    state = torch.load(path, map_location='cpu', weights_only=True)
    runtime.restored_cursor = dict(state['cursor'])
    runtime.restored_identity = dict(state['identity'])
    values = state['adapters']
    parameters = {n:p for group in runtime.groups for n,p in group.items()}
    assert set(values) == set(parameters) | ({'mix_gate.weight'} if 'mix_gate.weight' in values else set())
    with torch.no_grad():
        for name, param in parameters.items():
            value = values[name]
            assert value.shape == param.shape and value.dtype == param.dtype and torch.isfinite(value).all()
            param.copy_(value)
            assert torch.equal(param.cpu(), value)
    gate = None
    if 'mix_gate.weight' in values:
        gate = MixingGate(runtime.head.weight.shape[1], device=runtime.head.weight.device)
        with torch.no_grad(): gate.weight.copy_(values['mix_gate.weight'])
    return gate
