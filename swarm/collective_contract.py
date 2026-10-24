"""Frozen matched-exposure recipe, input identities and complete-panel slots."""
import json
import math
from pathlib import Path

from formation_contract import atomic, append, canonical, digest, encoded, read, require, reserve, sha

ARMS = ('joint_gate', 'independent_gate')


def recipe_digest(config):
    return digest({k:v for k,v in config.items() if k not in ('recipe_sha256', 'guard_source_files')})


def validate(config, rows, prompts):
    require(config['schema'] == 'hive-collective-config-v1', 'Unknown collective schema')
    require(config['members'] == [0, 1, 2, 3] and config['arms'] == list(ARMS), 'Explicit paired four-member order required')
    require(config['max_sequence'] == 1536 and config['gate_initialization'] == 'uniform_reference_logits_zero', 'Context/gate contract differs')
    require(config['normalization'] == 'joint_sum_over_batch_targets;independent_each_sum_over_K_batch_targets', 'Objective normalization differs')
    require(config['evaluation_fallback'] == 'last_complete_common_round_including_zero', 'Undeclared evaluation fallback')
    require(config['optimizer'] == dict(name='Muon', lr=5e-5, momentum=.95, weight_decay=0., grad_clip=1., scheduler='constant', precision='FP32', scaler=None), 'Unregistered optimizer')
    require(type(config['max_rounds']) is int and 1 <= config['max_rounds'] <= 2, 'Bounded complete rounds required')
    require(config['physical_optimizer_call_cap'] == 5*(2*config['max_rounds']+2), 'Both first-step replays must be charged')
    require(config['seed'] == 17 and config['cash_ceiling'] == 0, 'Seed/cash contract differs')
    parts = {'installation','admission','numerical','training','evaluation','recovery','supervisor','export'}
    b = config['budget']
    require(set(b) == parts and all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in b.values()), 'Missing finite budget')
    require(sum(b.values()) == config['provider_seconds'], 'Budget sum differs')
    require(config['worker_boundary_seconds'] == config['provider_seconds']-b['supervisor']-b['export'], 'Worker/export deadline differs')
    require(config['operation_seconds'] > 0 and config['restore_seconds'] > 0 and b['recovery'] >= 3*config['restore_seconds'], 'Recovery reserve missing')
    require(b['training'] >= 4*config['operation_seconds'], 'First complete pair and both replays need reserve')
    require(b['evaluation'] >= 2*len(prompts)*config['response_seconds_cap'], 'Complete paired panel needs reserve')
    require(config['fixture'] is True or (config['max_new_tokens'] == 512 and config['response_seconds_cap'] == 480), 'Production scoring bounds differ')
    require(rows and len({r['record_id'] for r in rows}) == len(rows), 'Empty/duplicate training records')
    require(len({r['family_id'] for r in rows}) == len(rows), 'Duplicate training families')
    for row in rows: encoded(row, config['eos_token_ids'], 1536)
    require(config['record_order'] == [r['record_id'] for r in rows], 'Whole-record order differs')
    require(config['target_tokens_per_arm_update'] == sum(r['encoded']['supervised_tokens'] for r in rows), 'Exact target denominator differs')
    require(config['target_tokens_per_arm_update'] <= 1024, 'Bounded complete-batch dose exceeded')
    require(prompts and len({p['record_id'] for p in prompts}) == len(prompts), 'Empty/duplicate panel')
    families = {a for r in rows for a in [r['family_id'], *r.get('family_aliases', [])]}
    for p in prompts:
        require(p['record_id'] not in config['record_order'] and p['family_id'] not in families, 'Training/evaluation collision')
        require(p['input_ids'] and all(type(t) is int and t>=0 for t in p['input_ids']) and len(p['input_ids'])+config['max_new_tokens']<=1536, 'Invalid prospective prompt')
        require(not {'target','solution','tests','labels','reference','answer'} & p.keys(), 'Gold/tests in worker prompt')
    return rows


def slots(prompts):
    return [dict(record_id=p['record_id'], condition=arm, state='unstarted', answer=None, error=None)
            for arm in ARMS for p in prompts]


def inputs(root):
    root = Path(root); config = read(root/'collective-config.json'); hashes = read(root/'source-hashes.json')
    for name, expected in hashes.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and not (root/p).is_symlink(), 'Unsafe payload path')
        require(sha(root/p) == expected, 'Payload changed: '+name)
    require(sha(root/'training-records.jsonl') == config['training_records_sha256'] and sha(root/'prompts.jsonl') == config['prompt_sha256'], 'Native input bytes differ')
    rows = [json.loads(x) for x in (root/'training-records.jsonl').read_bytes().splitlines()]
    prompts = [json.loads(x) for x in (root/'prompts.jsonl').read_bytes().splitlines()]
    validate(config, rows, prompts)
    require(recipe_digest(config) == config['recipe_sha256'], 'Recipe identity differs')
    admission = read(root/'training-admission.json')
    require(admission['schema'] == 'hive-collective-source-admission-v1' and admission['admitted'] is True,
        'Missing scoped collective source admission')
    require(admission['fixture'] is config['fixture'] and admission['record_ids'] == config['record_order'] and
        admission['training_records_sha256'] == config['training_records_sha256'] and
        admission['recipe_sha256'] == config['recipe_sha256'], 'Admission binding differs')
    parents = read(root/'formed-parents.json')
    require(len(parents) == 4 and [p['member'] for p in parents] == config['members'], 'Missing formed parent')
    require(len({p['expected_file_sha256'] for p in parents}) == 4 and all(
        p['expected_provenance']['artifact_kind'] == 'formed' and p['expected_provenance']['retained_updates']>0 for p in parents), 'Four actual formed artifacts required')
    require(digest(parents) == admission['formed_parents_sha256'] == config['formed_parents_sha256'], 'Formed parent bindings differ')
    return config, rows, prompts, parents, hashes
