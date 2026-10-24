"""Read only the selected experts from existing block-aligned NF4 storage.

No requantization and no new parameter values. Nested scales are sliced at
their own block boundaries. An independent full-dequantization comparison
is mandatory on every mounted expert tensor before this path is enabled.
"""
import math
from types import MethodType
import torch
import torch.nn.functional as F


def expert_views(packed, state):
    from bitsandbytes.functional import QuantState
    if len(state.shape) != 3 or state.quant_type != 'nf4' or packed.dtype != torch.uint8:
        raise ValueError('Expected the pinned packed 3D NF4 expert tensor')
    count, rows, cols = state.shape
    size = rows*cols
    if size % state.blocksize or size % 2:
        raise ValueError('Expert does not align to quantization blocks')
    blocks = size // state.blocksize
    flat = packed.reshape(-1)
    if flat.numel()*2 != count*size:
        raise ValueError('Unexpected expert packing size')
    views = []
    for i in range(count):
        begin, end = i*blocks, (i+1)*blocks
        nested = None
        if state.nested:
            s2 = state.state2
            if blocks % s2.blocksize or s2.nested:
                raise ValueError('Nested scale boundary is not independent')
            nested = QuantState(absmax=s2.absmax[begin//s2.blocksize:end//s2.blocksize],
                                shape=(blocks,), code=s2.code, blocksize=s2.blocksize,
                                dtype=s2.dtype)
        view = QuantState(absmax=state.absmax[begin:end], shape=(rows,cols),
                          code=state.code, blocksize=state.blocksize, quant_type=state.quant_type,
                          dtype=state.dtype, offset=state.offset, state2=nested)
        views.append((flat[i*size//2:(i+1)*size//2].view(-1,1), view))
    return views


def sparse_forward(self, hidden_states, top_k_index, top_k_weights):
    from bitsandbytes.functional import dequantize_4bit
    output = torch.zeros_like(hidden_states)
    # Same expert order and token accumulation order as the pinned HF eager path.
    mask = F.one_hot(top_k_index, num_classes=self.num_experts+1).permute(2,1,0)
    for idx_tensor in (mask.sum(dim=(-1,-2)) > 0).nonzero():
        idx = int(idx_tensor[0])
        if idx == self.num_experts:
            continue
        slots, tokens = torch.where(mask[idx])
        gate_weight = dequantize_4bit(*self._nf4_views['gate_up_proj'][idx])
        down_weight = dequantize_4bit(*self._nf4_views['down_proj'][idx])
        gate, up = F.linear(hidden_states[tokens], gate_weight).chunk(2, dim=-1)
        current = F.linear(self.act_fn(gate)*up, down_weight)
        current = current * top_k_weights[tokens,slots,None]
        output.index_add_(0, tokens, current.to(output.dtype))
    return output


def install_sparse_experts(runtime):
    from bitsandbytes.functional import dequantize_4bit
    checked = []
    for name, module in runtime.model.named_modules():
        if not name.endswith('.mlp.experts'):
            continue
        views = {}
        for pname in ('gate_up_proj','down_proj'):
            plist = module.parametrizations[pname]
            if len(plist) != 1:
                raise ValueError('Unexpected expert parametrization stack')
            packed, state = plist.original, plist[0].quant_state
            slices = expert_views(packed,state)
            # One full tensor live at a time; all 64 expert slices are checked.
            full = dequantize_4bit(packed,state)
            for idx, (data, qs) in enumerate(slices):
                if not torch.equal(full[idx],dequantize_4bit(data,qs)):
                    raise RuntimeError('Sliced NF4 weights differ from original dequantization')
            checked.append({'tensor':name+'.'+pname,'shape':list(state.shape),
                            'all_expert_slices_bit_equal':True})
            del full
            views[pname] = slices
        module._nf4_views = views
        module._original_experts_forward = module.forward
        module.forward = MethodType(sparse_forward,module)
    if len(checked) != 92:
        raise ValueError('Incomplete expert tensor verification')
    return {'passed':True,'tensors':checked,'parameter_values_changed':False}
