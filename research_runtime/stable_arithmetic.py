"""Fixed arithmetic shapes for full-sequence and incremental execution.

Zero padding changes neither parameters nor the mathematical causal policy.
The 1,600-key workspace covers the registered 1,536-token cache guard
plus its appended target in the full-sequence comparison.
This candidate still requires actual-model numerical admission.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROW_BLOCK = 64
KEY_SPAN = 1600
STOCK_LINEAR = F.linear


def stable_linear(inputs, weights, bias=None):
    shape = inputs.shape
    rows = inputs.reshape(-1, shape[-1])
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        weight = weights.float()
        bias = None if bias is None else bias.float()
        outputs = []
        for start in range(0, len(rows), ROW_BLOCK):
            part = rows[start:start+ROW_BLOCK].float()
            count = len(part)
            part = F.pad(part, (0, 0, 0, ROW_BLOCK-count)).contiguous()
            outputs.append(STOCK_LINEAR(part, weight, bias)[:count])
        if not outputs:
            return STOCK_LINEAR(rows.float(), weight, bias).reshape(*shape[:-1], weight.shape[0]).to(inputs.dtype)
        return torch.cat(outputs).reshape(*shape[:-1], weight.shape[0]).to(inputs.dtype)


def padded_row_call(function, module, inputs):
    """Keep router selection and normalization reduction shapes fixed too."""
    shape = inputs.shape
    rows = inputs.reshape(-1, shape[-1])
    results = []
    for start in range(0, len(rows), ROW_BLOCK):
        part = rows[start:start+ROW_BLOCK]
        count = len(part)
        output = function(module, F.pad(part, (0, 0, 0, ROW_BLOCK-count)))
        results.append(tuple(x[:count] for x in output) if isinstance(output, tuple) else output[:count])
    if isinstance(results[0], tuple):
        return tuple(torch.cat([result[i] for result in results]) for i in range(len(results[0])))
    return torch.cat(results).reshape(shape)


def attention_piece(q, k, v, mask, scaling, offset):
    qn, kn = q.shape[-2], k.shape[-2]
    if qn > ROW_BLOCK or kn > KEY_SPAN:
        raise ValueError('Attention exceeds the registered fixed arithmetic span')
    with torch.autocast(device_type=q.device.type, enabled=False):
        query = F.pad(q.float(), (0, 0, 0, ROW_BLOCK-qn)).contiguous()
        key = F.pad(k.float(), (0, 0, 0, KEY_SPAN-kn)).contiguous()
        value = F.pad(v.float(), (0, 0, 0, KEY_SPAN-kn)).contiguous()
        scores = query @ key.transpose(-1, -2) * scaling
        if mask is None:
            qp = torch.arange(offset, offset+ROW_BLOCK, device=q.device)[:, None]
            kp = torch.arange(KEY_SPAN, device=q.device)[None, :]
            blocked = (kp > qp) | (kp >= kn)
            scores = scores.masked_fill(blocked, float('-inf'))
        else:
            # Expand a broadcast query dimension before padding, so padding
            # never converts an all-query mask into a first-query-only mask.
            if mask.shape[-2] == 1 and qn != 1:
                mask = mask.expand(*mask.shape[:-2], qn, mask.shape[-1])
            if mask.shape[-2:] != (qn, kn):
                raise ValueError('Unexpected attention mask shape')
            if mask.dtype == torch.bool:
                mask = F.pad(mask, (0, KEY_SPAN-kn, 0, ROW_BLOCK-qn), value=False)
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                mask = F.pad(mask.float(), (0, KEY_SPAN-kn, 0, ROW_BLOCK-qn), value=float('-inf'))
                scores = scores + mask
        empty = scores.isneginf().all(-1, keepdim=True)
        probabilities = scores.masked_fill(empty, 0.).softmax(-1).masked_fill(empty, 0.)
        return (probabilities @ value)[..., :qn, :].to(q.dtype)


def precise_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
    if dropout:
        raise ValueError('Deterministic arithmetic requires zero dropout')
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import repeat_kv
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    qn, kn = query.shape[-2], key.shape[-2]
    pieces = []
    for start in range(0, qn, ROW_BLOCK):
        end = min(start+ROW_BLOCK, qn)
        mask = attention_mask
        if mask is not None and mask.shape[-2] == qn:
            mask = mask[..., start:end, :]
        args = (query[:, :, start:end], key, value, mask, scaling, kn-qn+start)
        part = checkpoint(attention_piece, *args, use_reentrant=False) if torch.is_grad_enabled() else attention_piece(*args)
        pieces.append(part)
    return torch.cat(pieces, dim=-2).transpose(1, 2).contiguous(), None
