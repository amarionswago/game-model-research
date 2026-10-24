"""Source-bound complete-answer formation contract. No model/provider imports."""
import hashlib
import json
import math
import os
from pathlib import Path

PARENT = '3fdd016f85e3840e553acd88e80fc0cab798fbeee40472ad805a317576823fe0'


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def recipe_digest(config):
    return digest({k:v for k,v in config.items() if k not in ('recipe_sha256','guard_source_files')})


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def atomic(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.partial')
    with pending.open('wb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


def append(path, value):
    with Path(path).open('ab') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def encoded(row, eos, context):
    e = row['encoded']
    ids, labels = e['input_ids'], e['labels']
    prefix, count = e['prompt_tokens'], e['supervised_tokens']
    require(type(prefix) is int and prefix > 0 and type(count) is int and count >= 2, 'Invalid complete target')
    require(type(ids) is list and all(type(x) is int and x >= 0 for x in ids), 'Invalid token IDs')
    require(len(ids) == prefix + count <= context, 'Truncation/context mismatch')
    require(labels == [-100] * prefix + ids[prefix:], 'Prompt/target causal mask differs')
    require(e['attention_mask'] == [1] * len(ids) and e['thinking_target'] is False, 'Attention/output convention differs')
    require(ids[-1] in eos and not any(x in eos for x in ids[prefix:-1]), 'Complete single terminal EOS required')
    require(row['supervised_tokens'] == count and row['input_tokens'] == len(ids), 'Declared native counts differ')
    require(row['split'] == 'train' and row['role'] in ('possible_training', 'rehearsal'), 'Evaluation/unsupported role in training')
    require(row['record_id'] and row['family_id'] and row['source_ref'] and row['qualification_ref'], 'Source identity missing')
    for key in ('prompt_sha256', 'target_sha256'):
        require(isinstance(row[key], str) and len(row[key]) == 64, 'Source text hash missing')
    return e


def validate_recipe(config, rows, prompts):
    require(config['schema'] == 'hive-formation-config-v1', 'Unknown formation schema')
    require(config['checkpoint_sha256'] == PARENT and config['parent_member_index'] == 0, 'Common Member0 parent required')
    require(config['max_sequence'] == 1536 and config['max_new_tokens'] == 512
            and config['response_seconds_cap'] == 480, 'Frozen evaluation bounds differ')
    require(config['optimizer'] == dict(name='Muon', lr=5e-5, momentum=.95, weight_decay=0.,
            grad_clip=1., scheduler='constant', precision='FP32', scaler=None), 'Unregistered optimizer settings')
    require(type(config['retained_updates']) is int and 1 <= config['retained_updates'] <= 4, 'Retained-update bound')
    require(config['physical_call_cap'] == config['retained_updates'] + 1, 'First replay must be charged')
    require(config['seed'] == 17 and config['cash_ceiling'] == 0, 'Seed/cash contract differs')
    budget = config['budget']
    parts = ('installation', 'admission', 'training_admission', 'updates', 'evaluation', 'recovery', 'supervisor', 'export')
    require(set(budget) == set(parts) and all(type(v) is int and v > 0 for v in budget.values()), 'Incomplete bounded budget')
    require(sum(budget.values()) == config['provider_seconds'], 'Budget does not sum to provider cap')
    require(budget['updates'] == config['physical_call_cap'] * config['operation_seconds'], 'Physical-operation envelope differs')
    require(config['evaluation_conditions'] in (['parent','formed'],['formed']), 'Unknown evaluation selection')
    if config['evaluation_conditions'] == ['formed']:
        parent=config['parent_reuse']
        require(parent['checkpoint_sha256']==PARENT and parent['member_ids']==[0]
                and parent['prompt_sha256']==config['prompt_sha256'] and len(parent['prepared_sha256'])==64,
                'Exact parent-reference binding required')
    else:
        require(config.get('parent_reuse') is None,'Unused parent reference')
    require(budget['evaluation'] == len(config['evaluation_conditions']) * len(prompts) * config['response_seconds_cap'], 'Selected answer slots need reserve')
    require(config['worker_boundary_seconds'] == config['provider_seconds'] - budget['supervisor'] - budget['export'], 'Worker/export boundaries differ')
    require(config['operation_seconds'] > 0 and config['restore_seconds'] > 0, 'Missing operation caps')
    require(budget['recovery'] >= 3 * config['restore_seconds'], 'Final and failed-operation recovery reserves missing')
    by_id = {r['record_id']: r for r in rows}
    require(rows and len(by_id) == len(rows), 'Missing/duplicate training records')
    for row in rows:
        encoded(row, config['eos_token_ids'], config['max_sequence'])
    batch = config['batch']
    require(batch and len(batch) == len({r['record_id'] for r in batch}), 'Duplicate record in fixed complete batch')
    require({r['record_id'] for r in batch} == set(by_id), 'Package must contain exactly its selected complete batch')
    for descriptor in batch:
        require(descriptor['target_start'] == 0 and descriptor['target_stop'] == by_id[descriptor['record_id']]['supervised_tokens'], 'Target slicing forbidden')
    require(sum(r['supervised_tokens'] for r in rows) == config['targets_per_update'], 'Exact denominator differs')
    require(config['targets_per_update'] <= config['max_targets_per_update'] <= 1024, 'Selected dose exceeds bounded screen')
    require(len({r['family_id'] for r in rows}) == len(rows), 'Duplicate training families')
    require(prompts and len({r['record_id'] for r in prompts}) == len(prompts), 'Missing/duplicate evaluation slots')
    train_families = {a for r in rows for a in [r['family_id'], *r.get('family_aliases', [])]}
    for p in prompts:
        require(p['record_id'] not in by_id and p['family_id'] not in train_families, 'Training/evaluation collision')
        require(p['input_ids'] and all(type(t) is int and t >= 0 for t in p['input_ids'])
                and len(p['input_ids']) + config['max_new_tokens'] <= config['max_sequence'], 'Evaluation context differs')
        require(not {'target', 'solution', 'tests', 'labels', 'reference'} & p.keys(), 'Grading asset in inference inputs')
    return [by_id[d['record_id']] for d in batch]


def inputs(root):
    root = Path(root)
    config = read(root/'formation-config.json')
    hashes = read(root/'source-hashes.json')
    for name, expected in hashes.items():
        require(Path(name).name == name and sha(root/name) == expected, 'Payload changed: ' + name)
    require(sha(root/'training-records.jsonl') == config['training_records_sha256'], 'Training bytes differ')
    require(sha(root/'prompts.jsonl') == config['prompt_sha256'], 'Evaluation bytes differ')
    rows = [json.loads(x) for x in (root/'training-records.jsonl').read_bytes().splitlines()]
    prompts = [json.loads(x) for x in (root/'prompts.jsonl').read_bytes().splitlines()]
    rows = validate_recipe(config, rows, prompts)
    admission = read(root/'training-admission.json')
    require(admission['schema'] == 'hive-formation-admission-v1'
            and admission['training_records_sha256'] == config['training_records_sha256']
            and admission['recipe_sha256'] == config['recipe_sha256'] == recipe_digest(config)
            and admission['data_manifest_sha256'] == config['data_manifest_sha256']
            and admission['record_ids'] == [r['record_id'] for r in rows], 'Scoped source-SFT admission differs')
    require(admission['admitted'] is True and admission['objective'] == 'original_question_complete_answer_sft', 'No scoped supervised admission')
    return config, rows, prompts, hashes


def reserve(now, deadline, required, label):
    require(all(math.isfinite(x) for x in (now, deadline, required)) and required >= 0, 'Invalid clock/reserve')
    require(deadline - now >= required, 'Insufficient reserve: ' + label)


def initial_slots(prompts, conditions=('parent','formed')):
    return [dict(record_id=p['record_id'], condition=c, state='unstarted' if c in conditions else 'external_pending', answer=None, error=None)
            for p in prompts for c in ('parent', 'formed')]
