"""Reuse exact accepted K2 admission, then install source-bound K4 parents."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from runtime_admission import admit as admit_parent
from swarm.collective_contract import require, sha
from swarm.population_assembly import assemble, activate
from swarm.population_objective import MixingGate


def admit(config, parents, deadline, root):
    require(config['fixture'] is False,'Synthetic package cannot enter pretrained admission')
    require(torch.cuda.is_available(),'Pretrained CPU work forbidden')
    sources=[]
    for parent in parents:
        provenance=parent['expected_provenance']
        require(provenance['parent_checkpoint_sha256']==config['checkpoint_sha256'] and
            provenance['parent_member_index']==0 and provenance['base_manifest_sha256']==config['export_manifest_sha256']
            and provenance['tokenizer_sha256']==config['tokenizer_sha256'],'Formed lineage differs from admitted runtime')
        matches=[p for p in Path('/kaggle/input').rglob(Path(parent['path']).name)
            if p.is_file() and sha(p)==parent['expected_file_sha256']]
        require(len(matches)==1,'Missing/ambiguous source-bound formed artifact')
        sources.append(dict(parent,path=str(matches[0])))
    loaded=admit_parent(config,deadline,root)
    runtime=loaded.population.runtime
    for member in (2,3): runtime.model.add_adapter('member'+str(member),deepcopy(runtime.model.peft_config['member0']))
    runtime.groups=[{n:p for n,p in runtime.model.named_parameters() if '.lora_' in n and f'.member{i}.' in n} for i in range(4)]
    require(all(len(g)==470 and sum(p.numel() for p in g.values())==21031936 for g in runtime.groups),'Four-member matrix inventory differs')
    assembly=assemble(sources,runtime.groups,hidden_size=runtime.head.weight.shape[1])
    gate=MixingGate(runtime.head.weight.shape[1],members=(0,1,2,3),device=runtime.head.weight.device)
    activate(runtime,assembly,gate)
    # The unchanged baseline admission's host bank has no reference in the
    # returned ParentRuntime. No mutable checkpoint bank is retained here.
    return SimpleNamespace(runtime=runtime,gate=gate,tokenizer=loaded.tokenizer,
        assembly=assembly,frozen_modules={})
