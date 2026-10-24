"""Shared differentiable FP32-state arithmetic for rollout and policy updates.

Derived from the validated inference precision policy, with checkpointed
attention chunks to bound backward storage. This is a NEW gradient policy;
the worker must establish cache/full/recompute agreement before updating.
All patches are serialized and restored, including on exceptions.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from nf4_linear import canonical_nf4_linears


from stable_arithmetic import attention_piece, precise_attention, stable_linear, padded_row_call


@contextmanager
def policy_numerics(runtime):
    bnb_ops=None
    if torch.device(runtime.input_device).type=='cuda':
        import bitsandbytes.backends.cuda.ops as bnb_ops
    from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite as glm
    router=glm.Glm4MoeLiteTopkRouter.forward
    norm=glm.Glm4MoeLiteRMSNorm.forward
    dispatch=None if bnb_ops is None else bnb_ops._gemm_4bit_use_custom_fn
    linear=F.linear; eager=glm.eager_attention_forward
    backend=runtime.model.config._attn_implementation
    module_dtypes=[];handles=[]
    def router_fp32(self,inputs):
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            return padded_row_call(router,self,inputs)
    def norm_stable(self,inputs):
        return padded_row_call(norm,self,inputs)
    try:
        glm.Glm4MoeLiteTopkRouter.forward=router_fp32
        glm.Glm4MoeLiteRMSNorm.forward=norm_stable
        if bnb_ops is not None:bnb_ops._gemm_4bit_use_custom_fn=lambda *args:False
        F.linear=stable_linear
        runtime.model.config._attn_implementation='eager'
        glm.eager_attention_forward=precise_attention
        for module in runtime.model.modules():
            if 'compute_dtype' in vars(module) and 'compute_type_is_set' in vars(module):
                if module.bias is not None:raise ValueError('Refuse frozen bias mutation')
                module_dtypes.append((module,module.compute_dtype,module.compute_type_is_set))
                module.compute_dtype=torch.float32;module.compute_type_is_set=True
        handles.append(runtime.backbone.embed_tokens.register_forward_hook(lambda module,args,out:out.float()))
        with canonical_nf4_linears(runtime.model) as contract:
            runtime.nf4_linear_contract=contract
            with torch.autocast(device_type=torch.device(runtime.input_device).type,enabled=False):yield
    finally:
        for handle in handles:handle.remove()
        for module,dtype,flag in module_dtypes:module.compute_dtype=dtype;module.compute_type_is_set=flag
        F.linear=linear
        if bnb_ops is not None:bnb_ops._gemm_4bit_use_custom_fn=dispatch
        glm.Glm4MoeLiteTopkRouter.forward=router;glm.eager_attention_forward=eager
        glm.Glm4MoeLiteRMSNorm.forward=norm
        runtime.model.config._attn_implementation=backend
