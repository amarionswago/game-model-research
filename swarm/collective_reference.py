"""Independent layer VJPs with the collective gate's causal head covectors.

The layer reconstruction/reverse sweep is copied exactly from the qualified
canonical reference except its explicit head-covector call. No optimizer here.
"""
import torch
import torch.nn.functional as F
from canonical_gradient_reference import (reference_forward_context, independent_layer,
    _check_time, _primal, _route, _device, _frozen_value, _require, glm)


def collective_head(runtime, capture, logp_covector, hidden_covector, *, deadline):
    _check_time(deadline)
    device = runtime.backbone.norm.weight.device
    x = capture.outputs[len(runtime.backbone.layers)-1].to(device).requires_grad_()
    norm = glm.Glm4MoeLiteRMSNorm(runtime.model.config.hidden_size, runtime.model.config.rms_norm_eps).to(device)
    with torch.no_grad(): norm.weight.copy_(runtime.backbone.norm.weight.float())
    norm.requires_grad_(False)
    hidden = norm(x)
    error = _primal(hidden, capture.norm_output, 'Final norm')
    labels = torch.tensor(capture.encoded['labels'],device=device)
    positions = (labels[1:]!=-100).nonzero().flatten()
    target = labels[1:][positions]
    weight = _frozen_value(runtime.head,'weight')
    cotangent = torch.zeros_like(hidden); logs=[]
    for start in range(0,len(target),64):
        _check_time(deadline); pos=positions[start:start+64]; y=target[start:start+64]
        h=hidden.detach()[0,pos].clone().requires_grad_()
        logits=F.linear(h,weight).double()
        lp=logits.log_softmax(-1).gather(-1,y[:,None]).squeeze(-1)
        objective=(lp*logp_covector[start:start+len(pos)].to(device)).sum()
        gh,=torch.autograd.grad(objective,h)
        cotangent[0,pos]=gh+hidden_covector[start:start+len(pos)].to(device)
        logs.append(lp.detach().cpu())
    initial,=torch.autograd.grad(hidden,x,cotangent)
    return initial.detach().cpu(),torch.cat(logs),None,dict(norm_max_abs=error,target_positions=positions.cpu().tolist())


def reference_record(runtime, capture, *, fork_member, trainable_names, logp_covector, hidden_covector,
                     deadline=None, progress=None):
    """Consume captured boundaries after its production graph was consumed."""
    _require(not capture.handles, 'Production capture hooks must be removed before reference')
    with reference_forward_context():
        cotangent, logps, objective, head = collective_head(runtime, capture, logp_covector, hidden_covector, deadline=deadline)
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
