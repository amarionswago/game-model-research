"""Nonmutating K4 checks and independent dense-layer gradient reference."""
from types import SimpleNamespace
import time

import numpy as np
import torch

from canonical_gradient_reference import BoundaryCapture, compare_gradients, THRESHOLDS
from nf4_linear import verify_nf4_linears
from swarm.collective_contract import ARMS, atomic, require
from swarm.collective_reference import reference_record
from swarm.population_objective import MixingGate, mixture, recomputed_backward, selected
from swarm.population_assembly import duplicate_binary_gate
from swarm.population_runtime import Population


def reference_covectors(logs, hidden, gate, denominator, mode):
    lp=logs.detach().clone().requires_grad_(); hs=hidden.detach().clone().requires_grad_()
    weight=gate.weight.detach().clone().requires_grad_()
    k,t,h=hs.shape
    normalized=hs/torch.sqrt((hs**2).mean(-1,keepdim=True)+1e-6)
    features=torch.cat([normalized.permute(1,0,2).reshape(t,k*h),torch.ones(t,1,device=hs.device)],1)
    logits=features@weight.T
    logits=torch.cat([logits,torch.zeros(t,1,device=hs.device)],1)
    combined=torch.logsumexp(logits.log_softmax(-1).T+lp,dim=0)
    gl,gh,gg=torch.autograd.grad(-combined.sum()/denominator,(lp,hs,weight))
    if mode=='independent_gate': gl=torch.full_like(gl,-1/(k*denominator));gh=torch.zeros_like(gh)
    return gl.detach(),gh.detach(),gg.detach(),combined.detach()


def gradient_reference(controller, mode, deadline):
    r,g,c=controller.runtime,controller.gate,controller.config
    denominator=c['target_tokens_per_arm_update']; controller.clear(); diagnostics=[]
    for row in controller.rows:
        recomputed_backward(lambda i:selected(r,i,row), members=(0,1,2,3), mode=mode,gate=g,denominator=denominator)
    production={n:p.grad.detach().cpu().clone() for n,p in controller.binding.parameters.items()}
    controller.clear(); reference={}
    for row in controller.rows:
        require(time.monotonic()<deadline,'Gradient reference deadline')
        with torch.no_grad(): values=[selected(r,i,row) for i in range(4)]
        logs=torch.stack([v[0] for v in values]);hidden=torch.stack([v[1] for v in values])
        gl,gh,gg,combined=reference_covectors(logs,hidden,g,denominator,mode)
        actual=mixture(logs,hidden,g)[0]
        objective_error=float((torch.abs(actual.sum()-combined.sum())/denominator).detach())
        require(objective_error<=THRESHOLDS['objective_atol'],'Independent collective objective differs')
        reference['mix_gate.weight']=reference.get('mix_gate.weight',torch.zeros_like(gg.cpu()))+gg.cpu()
        for member in range(4):
            # Capture hooks are removed before independent dense-layer replay.
            record=SimpleNamespace(serial_record=lambda row=row:row)
            with BoundaryCapture(r,record) as capture:
                actual_lp,actual_h=selected(r,member,row)
            require(torch.allclose(actual_lp.detach(),logs[member],atol=1e-3,rtol=0),'Reference capture logp replay changed')
            del actual_lp,actual_h
            gradients,ref_logs,_,detail=reference_record(r,capture,fork_member=member,
                trainable_names=set(r.groups[member]),logp_covector=gl[member],hidden_covector=gh[member],
                deadline=deadline,progress=lambda **fields:controller.event('independent_gradient_reference',arm=mode,member=member,**fields))
            require(float((ref_logs-logs[member].double().cpu()).abs().max())<=THRESHOLDS['target_logp_atol'],'Independent target probability differs')
            for name,value in gradients.items(): reference[name]=reference.get(name,torch.zeros_like(value))+value
            diagnostics.append(dict(record_id=row['record_id'],member=member,detail=detail,objective_error=objective_error))
            del capture,gradients
    comparison=compare_gradients(production,reference)
    controller.clear()
    return dict(arm=mode,gradients=comparison,records=diagnostics,physical_optimizer_calls=0,
        convention='shared canonical forward; independent dense frozen layer/unmerged LoRA VJP; explicit collective head and gate covectors')


def numerical_check(controller):
    r,g,c=controller.runtime,controller.gate,controller.config
    deadline=min(controller.deadline-c['budget']['evaluation']-c['budget']['recovery'],
        time.monotonic()+c['budget']['numerical'])
    report=dict(schema='hive-collective-admission-v1',passed=False,physical_optimizer_calls=0,checks=[])
    atomic(controller.root/'collective-admission.json',report)
    require(torch.count_nonzero(g.weight)==0,'Candidate initial gate must be uniform')
    report['nf4_linears']=verify_nf4_linears(r.model)
    # A declared nonzero diagnostic gate exercises hidden-to-gate derivatives;
    # the owner restores the complete initial state before any optimizer call.
    with torch.no_grad():
        g.weight.copy_(torch.linspace(-.02,.03,g.weight.numel(),device=g.weight.device).reshape_as(g.weight))
    r.model.eval()
    seed=controller.prompts[0]['input_ids']
    # A small independent K2 reduction is a diagnostic only. Preserve every
    # formed adapter and retire activations on both sides of temporary copies.
    original={n:p.detach().clone() for group in r.groups[2:] for n,p in group.items()}
    try:
        with torch.no_grad():
            for member in (2,3):
                for name,p in r.groups[member].items():p.copy_(r.groups[member-2][name.replace(f'.member{member}.',f'.member{member-2}.')])
        r.checkpoint_generation+=1
        binary=MixingGate(g.hidden_size,members=(0,1),device=g.weight.device)
        with torch.no_grad():binary.weight.copy_(torch.linspace(-.01,.02,binary.weight.numel(),device=g.weight.device).reshape_as(binary.weight))
        doubled=duplicate_binary_gate(binary)
        pair=Population(r,binary,members=(0,1)).forced(seed,[len(seed)-1],deadline=deadline)
        four=Population(r,doubled,members=(0,1,2,3)).forced(seed,[len(seed)-1],deadline=deadline)
        require(float(np.max(np.abs(pair[2]-four[2])))<=.001 and
            float(np.max(np.abs(pair[3]-four[3][:,[0,2]].sum(-1))))<=.0001,'Duplicate K2 reduction differs')
        report['checks'].append(dict(id='duplicate_K2_reference_only',passed=True))
    finally:
        with torch.no_grad():
            for group in r.groups[2:]:
                for name,p in group.items():p.copy_(original[name])
        r.checkpoint_generation+=1
        del original
    sequence=(seed*((1536+len(seed)-1)//len(seed)))[:1536]
    population=Population(r,g,members=(0,1,2,3))
    positions=[63,64,767,768,1534,1535]
    a=population.forced(sequence,positions,chunk_size=64,deadline=deadline)
    b=population.forced(sequence,positions,chunk_size=63,deadline=deadline)
    error=float(np.max(np.abs(a[0]-b[0])));gate_error=float(np.max(np.abs(a[3]-b[3])))
    probability_error=float(np.max(np.abs(np.exp(a[2])-np.exp(b[2]))))
    kl=float(np.max(np.sum(np.exp(a[2])*(a[2]-b[2]),axis=-1)))
    require(error<=.001 and gate_error<=.0001 and probability_error<=.001 and kl<=.0001
        and np.array_equal(a[2].argmax(-1),b[2].argmax(-1)),'K4 context partition differs')
    require(np.isfinite(a[2]).all() and np.max(np.abs(np.exp(a[2]).sum(-1)-1))<=.0001,'Collective probabilities are not normalized')
    report['checks'].append(dict(id='context1536',logp_error=error,gate_error=gate_error,probability_error=probability_error,kl=kl))
    for member in range(4):
        solo=Population(r,None,members=(member,)).forced(sequence,positions,chunk_size=64,deadline=deadline)
        require(float(np.max(np.abs(solo[0][0]-a[0][member])))<=.001,'Solo/collective member differs')
    # Full-prefix/recomputed causal targets versus the cached member values.
    for row in controller.rows:
        encoded=row['encoded'];pos=list(range(encoded['prompt_tokens']-1,len(encoded['input_ids'])-1))
        cached=population.forced(encoded['input_ids'][:-1],pos,deadline=deadline)
        for member in range(4):
            with torch.no_grad():full,_=selected(r,member,row)
            target=encoded['input_ids'][encoded['prompt_tokens']:]
            require(float(np.max(np.abs(full.cpu().numpy()-cached[0][member,np.arange(len(target)),target])))<=.001,'Full/recomputed/cached targets differ')
    controller.memory()
    sample=population.generate(seed,eos_ids=set(),max_new_tokens=32,deadline=deadline)
    controller.memory()
    require(len(sample['tokens'])==32,'Incomplete numerical generation')
    require(set(sample['cache'])=={'0','1','2','3'} and all(v['dtypes']==['torch.float32'] for v in sample['cache'].values()),'Four FP32 caches required')
    sequence=seed+sample['tokens'][:-1]; positions=list(range(len(seed)-1,len(sequence)))
    replay=population.forced(sequence,positions,deadline=deadline)
    require(replay[2].argmax(-1).tolist()==sample['tokens'],'Greedy generation replay differs')
    selected_logs=np.stack([replay[0][i,np.arange(32),sample['tokens']] for i in range(4)],-1)
    require(float(np.max(np.abs(selected_logs-np.asarray(sample['member_logps']))))<=.001 and
        float(np.max(np.abs(replay[3]-np.asarray(sample['member_weights']))))<=.0001,'Generated policy replay differs')
    report['generation']=sample
    r.model.train()
    report['gradient_references']=[gradient_reference(controller,arm,deadline) for arm in ARMS]
    report['passed']=True;atomic(controller.root/'collective-admission.json',report)
    return report
