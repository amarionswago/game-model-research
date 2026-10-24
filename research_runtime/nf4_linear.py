"""Frozen NF4 linear with the same decoded weights in forward and backward.

The canonical weights are dequantized in quant_state.dtype, then promoted to
FP32. This matches the independent reference and existing expert tensors.
No requantization, tolerance relaxation, or persistent dense weight copy.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F


def decoded_weight(packed, state):
    from bitsandbytes.functional import dequantize_4bit
    if state.quant_type != 'nf4' or len(state.shape) != 2:
        raise ValueError('Expected a 2D NF4 frozen linear')
    if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Unsupported canonical dequantization dtype')
    return dequantize_4bit(packed.reshape(-1, 1), state).to(torch.float32)


class FrozenNF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, packed, state):
        if inputs.dtype != torch.float32 or packed.requires_grad:
            raise ValueError('Canonical policy requires FP32 inputs and frozen NF4 weights')
        ctx.save_for_backward(packed)
        ctx.state = state
        ctx.state_dtype = state.dtype
        weight = decoded_weight(packed, state)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.state.dtype != ctx.state_dtype:
            raise RuntimeError('Quantization convention changed between forward and backward')
        if not ctx.needs_input_grad[0]:
            return None, None, None
        packed, = ctx.saved_tensors
        weight = decoded_weight(packed, ctx.state)
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            # The forward uses fixed 64-row products. Match that shape in its
            # derivative, including the unchanged explicit-gradient guard.
            from stable_arithmetic import stable_linear
            grad_input = stable_linear(grad_output.float(), weight.t())
        return grad_input, None, None


def canonical_forward(module, inputs):
    if module.bias is not None:
        raise ValueError('Pinned GLM canonical linears must have no bias')
    if module.weight.quant_state is None:
        raise ValueError('Missing NF4 quantization state')
    return FrozenNF4Linear.apply(inputs, module.weight, module.weight.quant_state)


def verify_nf4_linears(model):
    """Check every installed forward and input gradient against explicit weights."""
    from bitsandbytes.nn import Linear4bit
    from bitsandbytes.functional import dequantize_4bit
    checks=[]
    for name,module in model.named_modules():
        if not isinstance(module,Linear4bit):
            continue
        state=module.weight.quant_state
        if getattr(module.forward,'__func__',None) is not canonical_forward:
            raise ValueError('Canonical forward is not installed: '+name)
        n,k=state.shape
        x=torch.linspace(-.2,.2,2*k,device=module.weight.device,dtype=torch.float32).reshape(2,k).requires_grad_()
        cotangent=torch.linspace(-.1,.1,2*n,device=x.device,dtype=torch.float32).reshape(2,n)
        # Independent explicit-matrix autograd is the reference for this check.
        weight=dequantize_4bit(module.weight.reshape(-1,1),state).float()
        actual=module(x)
        expected=F.linear(x,weight)
        grad_actual,=torch.autograd.grad(actual,x,cotangent)
        grad_expected,=torch.autograd.grad(expected,x,cotangent)
        output_error=float((actual.detach()-expected.detach()).abs().max())
        gradient_error=float((grad_actual-grad_expected).abs().max())
        passed=bool(torch.isfinite(actual).all() and torch.isfinite(grad_actual).all()
                    and torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
                    and torch.allclose(grad_actual,grad_expected,atol=1e-6,rtol=1e-6))
        checks.append({'name':name,'weight_shape':[n,k],'stored_dtype':str(state.dtype),
                       'max_output_error':output_error,'max_input_gradient_error':gradient_error,'passed':passed})
        if not passed:
            raise RuntimeError('NF4 forward/gradient comparison failed: '+str(checks[-1]))
        del x,cotangent,weight,actual,expected,grad_actual,grad_expected
    if not checks:
        raise ValueError('No NF4 linears checked')
    return {'passed':True,'linears':checks,'optimizer_updates':0}


@contextmanager
def canonical_nf4_linears(model):
    """Temporarily replace only frozen bnb Linear4bit forwards; PEFT stays intact."""
    from bitsandbytes.nn import Linear4bit
    changed = []
    try:
        for name, module in model.named_modules():
            if not isinstance(module, Linear4bit):
                continue
            state = module.weight.quant_state
            if module.weight.requires_grad or state is None or state.quant_type != 'nf4' or module.bias is not None:
                raise ValueError('Unsupported frozen NF4 module: '+name)
            if getattr(state, 'packing_format_for_cpu', False):
                raise ValueError('CPU repacked NF4 weights are outside this contract')
            existed = 'forward' in vars(module)
            previous = vars(module).get('forward')
            changed.append((module, existed, previous))
            module.forward = MethodType(canonical_forward, module)
        if not changed:
            raise ValueError('No frozen NF4 linears were found')
        yield {'linears': len(changed), 'weight_definition': 'dequantize in stored dtype then promote to FP32',
               'forward_backward_same_weight': True, 'parameter_values_changed': False}
    finally:
        for module, existed, previous in reversed(changed):
            if existed:
                module.forward = previous
            else:
                del module.forward
