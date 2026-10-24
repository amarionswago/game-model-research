"""Conditional revision data admission and serial single-member NLL. No trainer."""
from dataclasses import dataclass, asdict
import hashlib
import json
from types import MappingProxyType
from pathlib import PurePath
from typing import Callable, Mapping

import torch

IGNORE = -100
SOURCE_ROLES = frozenset({
    'protocol', 'training_manifest', 'prompts', 'raw_stages', 'raw_task_results',
    'generation_source', 'tokenizer_manifest', 'checkpoint_provenance', 'grading_evidence',
    'grading_protocol', 'grading_input_bindings',
    'adaptation_split',
})


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def object_digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(blobs: Mapping[str, bytes], digest: str) -> bytes:
    require(isinstance(digest, str) and len(digest) == 64, 'Malformed source hash')
    require(digest in blobs, 'Missing bound artifact: ' + digest)
    raw = blobs[digest]
    require(sha256(raw) == digest, 'Source bytes changed: ' + digest)
    return raw


def json_lines(raw):
    require(raw.endswith(b'\n'), 'Incomplete JSONL artifact')
    return [json.loads(line) for line in raw.splitlines()]


@dataclass(frozen=True)
class AdmissionPolicy:
    approved_admission_hashes: frozenset[str]
    trusted_sources: Mapping[str, str]
    checkpoint_sha256: str
    eos_ids: frozenset[int]
    max_sequence: int
    purpose: str = 'actual_training'

    def __post_init__(self):
        require(self.purpose in {'actual_training', 'verification_fixture'}, 'Unknown purpose')
        require(bool(self.eos_ids) and all(type(t) is int and t >= 0 for t in self.eos_ids), 'Invalid EOS set')
        require(type(self.max_sequence) is int and self.max_sequence > 1, 'Invalid context limit')
        required = {'protocol', 'training_manifest', 'prompts', 'generation_source',
                    'tokenizer_manifest', 'checkpoint_provenance', 'grading_protocol', 'grading_input_bindings'}
        required.add('adaptation_split')
        require(set(self.trusted_sources) == required, 'Incomplete externally trusted source set')
        object.__setattr__(self, 'trusted_sources', MappingProxyType(dict(self.trusted_sources)))


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    target_ids: tuple[int, ...]
    eos_ids: frozenset[int]

    def serial_record(self):
        n = len(self.input_ids)
        require(n == len(self.labels) == len(self.attention_mask), 'Unaligned encoded arrays')
        require(all(type(t) is int and t >= 0 for t in self.input_ids), 'Invalid input token')
        require(all(type(a) is int and a in (0, 1) for a in self.attention_mask), 'Invalid attention mask')
        length = sum(self.attention_mask)
        require(length >= 2 and self.attention_mask == (1,) * length + (0,) * (n - length),
                'Only contiguous right padding is supported')
        require(all(y == IGNORE for y in self.labels[length:]), 'Padding has supervised labels')
        targets = self.target_ids
        require(len(targets) >= 2 and targets[-1] in self.eos_ids
                and not any(t in self.eos_ids for t in targets[:-1]), 'Complete visible target and final EOS required')
        prefix = length - len(targets)
        require(prefix >= 1, 'At least one conditioning token is required')
        require(self.input_ids[prefix:length] == targets, 'Target input IDs changed')
        require(self.labels[:length] == (IGNORE,) * prefix + targets, 'Only contiguous revision and EOS may be supervised')
        return {'encoded': {'input_ids': list(self.input_ids[:length]),
                            'labels': list(self.labels[:length]), 'supervised_tokens': len(targets)}}


@dataclass(frozen=True)
class AdmittedPair:
    task_id: str
    family_id: str
    arm: str
    bundle_sha256: str
    admission_sha256: str
    source_bindings: tuple[tuple[str, str], ...]
    revision: EncodedRecord
    answer_only: EncodedRecord
    training_eligible: bool


def encoded_record(prompt_ids, target_ids, *, eos_ids, max_sequence):
    prompt, target = tuple(prompt_ids), tuple(target_ids)
    require(len(prompt) + len(target) <= max_sequence, 'Complete target exceeds context; truncation forbidden')
    record = EncodedRecord(prompt + target, (IGNORE,) * len(prompt) + target,
                           (1,) * (len(prompt) + len(target)), target, frozenset(eos_ids))
    record.serial_record()
    return record


def validate_pair(bundle_sha256, admission_sha256, blobs, policy, tokenizer, loop):
    """Read only externally pinned admissions. A passing grade is insufficient."""
    require(admission_sha256 in policy.approved_admission_hashes, 'No explicit semantic admission')
    admission = json.loads(read_bound(blobs, admission_sha256))
    bundle = json.loads(read_bound(blobs, bundle_sha256))
    require(bundle.get('schema') == 'revision-trajectory-v1', 'Unknown trajectory schema')
    require(admission.get('decision') == 'admitted_revision_sft', 'Candidate or rejected admission')
    require(admission.get('bundle_sha256') == bundle_sha256, 'Admission targets different trajectory bytes')
    origin = 'actual_saved_generation' if policy.purpose == 'actual_training' else 'verification_fixture'
    require(bundle.get('origin') == admission.get('origin') == origin, 'Synthetic/source-replacement origin forbidden')
    require(admission.get('reviewer_id') and admission.get('independent_review') is True, 'Independent reviewer required')
    required_findings = ('whole_solution_reviewed', 'standalone_final_valid', 'actual_complete_trajectory',
                         'training_family_verified', 'no_reference_replacement', 'no_fabricated_wrong_draft')
    require(all(admission.get(name) is True for name in required_findings), 'Incomplete semantic admission')
    review = read_bound(blobs, admission['semantic_review_sha256'])
    require(bool(review.strip()), 'Empty semantic review evidence')
    sources = bundle['source_bindings']
    require(set(sources) == SOURCE_ROLES, 'Missing or unknown provenance role')
    payloads = {name: read_bound(blobs, digest) for name, digest in sources.items()}
    require(all(sources[name] == digest for name, digest in policy.trusted_sources.items()), 'Untrusted upstream source')
    protocol = json.loads(payloads['protocol'])
    require(protocol['checkpoint_sha256'] == policy.checkpoint_sha256, 'Protocol checkpoint mismatch')
    require(protocol['caps']['context_tokens'] == policy.max_sequence == loop.MAX_SEQUENCE, 'Context limit differs from frozen protocol')
    checkpoint = json.loads(payloads['checkpoint_provenance'])
    require(checkpoint['checkpoint_sha256'] == policy.checkpoint_sha256, 'Checkpoint provenance mismatch')
    token_manifest = json.loads(payloads['tokenizer_manifest'])
    require(bool(token_manifest['files']), 'Empty tokenizer binding')
    require(frozenset(token_manifest['eos_ids']) == policy.eos_ids
            and token_manifest['max_sequence'] == policy.max_sequence, 'Tokenizer EOS/context policy mismatch')
    for digest in token_manifest['files'].values():
        read_bound(blobs, digest)
    require(tokenizer.source_manifest_sha256 == sources['tokenizer_manifest'], 'Tokenizer loader identity mismatch')
    require(sha256(loop.source_bytes) == sources['generation_source'], 'Renderer source mismatch')
    tasks = json_lines(payloads['prompts'])
    require(len({t['id'] for t in tasks}) == len(tasks), 'Duplicate task IDs')
    task = next((t for t in tasks if t['id'] == bundle['task_id']), None)
    require(task is not None, 'Task absent from frozen prompts')
    require(task['group_id'] == bundle['family_id'], 'Wrong family ID')
    manifest = json.loads(payloads['training_manifest'])
    require(manifest['split'] == 'training' and manifest['development_excluded'] is True
            and manifest['sealed_excluded'] is True, 'Unqualified training manifest')
    family_rows = manifest['records']
    require(len({r['id'] for r in family_rows}) == len(family_rows), 'Duplicate provenance IDs')
    family = next((r for r in family_rows if r['id'] == task['id']), None)
    require(family is not None and family['family_id'] == task['group_id']
            and family['domain'] == task['domain']
            and family['problem_sha256'] == object_digest(task['messages']), 'Training-family provenance mismatch')
    require(task['id'] in protocol['ids']
            and protocol['family_ids'][protocol['ids'].index(task['id'])] == task['group_id'], 'Task outside pilot protocol')
    split = json.loads(payloads['adaptation_split'])
    require(split['schema'] == 'hive-correction-adaptation-split-v1'
            and split['pilot_protocol_sha256'] == sources['protocol'], 'Adaptation split protocol mismatch')
    split_rows = split['records']
    require(len({r['id'] for r in split_rows}) == len(split_rows)
            and len({r['family_id'] for r in split_rows}) == len(split_rows), 'Duplicate adaptation task/family')
    role = next((r for r in split_rows if r['id'] == task['id']), None)
    require(role is not None and role['family_id'] == task['group_id'] and role['domain'] == task['domain'],
            'Task absent from bound adaptation split')
    require(role['future_role'] == 'possible_correction_training', 'Correction-development family cannot enter training')
    require(loop.encode(tokenizer, task['messages']) == task['input_ids'], 'Original prompt replay differs')
    arm = bundle['arm']
    require(arm in {'peer', 'self', 'direct'}, 'Unknown revision arm')
    rows = json_lines(payloads['raw_stages'])
    branch = [s for s in rows if s['id'] == task['id'] and s['arm'] in {'shared', arm}]
    expected = [('draft', 'shared', 0)]
    if arm != 'direct':
        expected.append(('critic', arm, 1 if arm == 'peer' else 0))
    expected.append(('revision', arm, 0))
    require([(s['stage'], s['arm'], s['member']) for s in branch] == expected, 'Missing, duplicated or reordered trajectory stages')
    draft, critic = branch[0], None if arm == 'direct' else branch[1]
    for row in branch:
        require(row['group_id'] == task['group_id'] and row['domain'] == task['domain'], 'Stage family mismatch')
        require(row['checkpoint_sha256'] == policy.checkpoint_sha256 and row['checkpoint'] == 'joint_gate', 'Stage checkpoint mismatch')
        require(row['member_ids'] == [row['member']] and row['condition'] == 'member' + str(row['member']), 'Stage member mismatch')
        spec = loop.DRAFT if row['stage'] == 'draft' else loop.REVISION if row['stage'] == 'revision' else loop.CRITICS[arm]
        require(row['budget'] == asdict(spec), 'Stage budget differs from frozen controller')
        require(protocol['caps'][row['stage'] + '_tokens'] == spec.max_tokens
                and protocol['caps'][row['stage'] + '_seconds'] == spec.seconds, 'Protocol stage allowance differs')
        tokens = row['tokens']
        require(isinstance(tokens, list) and all(type(t) is int and 0 <= t < len(tokenizer) for t in tokens), 'Invalid stage tokens')
        require(row['stop'] == 'eos' and len(tokens) >= 2 and tokens[-1] in policy.eos_ids
                and not any(t in policy.eos_ids for t in tokens[:-1]), 'Every required stage must finish with EOS')
        require(len(tokens) <= spec.max_tokens and len(row['input_ids']) + spec.max_tokens <= policy.max_sequence,
                'Stage exceeds complete output/context bounds')
        require(row['text'] == tokenizer.decode(tokens, skip_special_tokens=True) and row['text'].strip(), 'Changed or empty stage text')
        require(sum(row['forward_calls'].values()) > 0, 'Stage has no recorded model forward')
        use_draft = None if row['stage'] == 'draft' else draft
        use_critic = critic if row['stage'] == 'revision' else None
        messages = loop.dialogue(task, row['stage'], use_draft, use_critic)
        require(row['messages'] == messages, 'Rendered dialogue changed')
        require(row['input_ids'] == loop.encode(tokenizer, messages)
                and row['prompt_sha256'] == object_digest(row['input_ids']), 'Stage prompt replay differs')
        require(row['parent_draft_sha256'] == (None if use_draft is None else object_digest(draft['tokens']))
                and row['parent_critic_sha256'] == (None if use_critic is None else object_digest(critic['tokens'])), 'Stage parent binding differs')
    results = json_lines(payloads['raw_task_results'])
    require(len({r['id'] for r in results}) == len(results), 'Duplicate saved task result')
    result = next((r for r in results if r['id'] == task['id']), None)
    require(result is not None and result['group_id'] == task['group_id'] and result['optimizer_updates'] == 0, 'Missing frozen task result')
    final, revision = result['finals'][arm], branch[-1]
    require(final['fallback'] is False and final['answer_stage'] == 'revision' and final['answer_arm'] == arm,
            'Fallback is not a complete revision target')
    require(final['tokens'] == revision['tokens'] and final['text'] == revision['text'], 'Final answer substituted')
    require(admission['target_tokens_sha256'] == object_digest(revision['tokens'])
            and admission['final_text_sha256'] == sha256(revision['text'].encode()), 'Reviewed target changed')
    grade_artifacts = json.loads(payloads['grading_evidence'])
    require({'summary.json', 'per-task.json'} <= set(grade_artifacts), 'Missing frozen grade artifacts')
    grade_bytes = {name: read_bound(blobs, digest) for name, digest in grade_artifacts.items()}
    summary = json.loads(grade_bytes['summary.json'])
    require(summary['audit_passed'] is True and summary['optimizer_updates'] == 0, 'Failed frozen grading audit')
    require(summary['grading_protocol_sha256'] == sources['grading_protocol']
            and summary['source_bindings_sha256'] == sources['grading_input_bindings'], 'Grading protocol/source identity mismatch')
    raw_outputs = summary['raw_outputs']
    require(len(raw_outputs) == 2, 'Unexpected grading raw output set')
    expected_raw = {'trajectories.jsonl': sources['raw_stages'], 'task-results.jsonl': sources['raw_task_results']}
    require({PurePath(path).name: entry['sha256'] for path, entry in raw_outputs.items()} == expected_raw
            and all(entry['exists'] is True for entry in raw_outputs.values()), 'Grade refers to different raw outputs')
    grades = json.loads(grade_bytes['per-task.json'])
    require(len({r['id'] for r in grades}) == len(grades), 'Duplicate graded task ID')
    graded = next((r for r in grades if r['id'] == task['id']), None)
    require(graded is not None and graded['family_id'] == task['group_id'] and graded['domain'] == task['domain']
            and graded['saved_task_result'] is True and graded['trace_complete'] is True, 'Incomplete graded trajectory identity')
    outcome = graded['outcomes'][arm]
    require(outcome['grade']['passed'] is True, 'Frozen grader did not pass final revision')
    require(outcome['grade']['raw_text_sha256'] == sha256(revision['text'].encode())
            and outcome['answer_tokens_sha256'] == object_digest(revision['tokens']), 'Frozen grade target mismatch')
    require(outcome['stages_eos'] is True and outcome['answer_eos'] is True and outcome['fallback'] is False
            and outcome['answer_stage'] == [arm, 'revision'], 'Frozen grade completion differs')
    common = {'eos_ids': policy.eos_ids, 'max_sequence': policy.max_sequence}
    correction = encoded_record(revision['input_ids'], revision['tokens'], **common)
    control = encoded_record(task['input_ids'], revision['tokens'], **common)
    require(correction.target_ids == control.target_ids, 'Control target IDs differ')
    all_bindings = tuple(sorted({**sources, 'semantic_review': admission['semantic_review_sha256']}.items()))
    return AdmittedPair(task['id'], task['group_id'], arm, bundle_sha256, admission_sha256,
                        all_bindings, correction, control, policy.purpose == 'actual_training')


def require_trainable_membership(model, names, *, frozen_modules=()):
    names = frozenset(names)
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    require(bool(names) and actual == names, 'Trainable parameter membership changed')
    for module in frozen_modules:
        require(all(not p.requires_grad and p.grad is None for p in module.parameters()), 'Frozen policy or gate has gradients enabled')
    for name, parameter in model.named_parameters():
        require(name in names or parameter.grad is None, 'Frozen parameter has a gradient')
    return {name: parameter for name, parameter in model.named_parameters() if name in names}


def selected_revision(runtime, record, *, fork_member, trainable_names, historical_selected: Callable,
                      frozen_modules=()):
    require(type(fork_member) is int and fork_member >= 2, 'Original draft/critic adapters may not be trained')
    require(all('.member' + str(fork_member) + '.' in n and '.lora_' in n for n in trainable_names),
            'Trainables must belong only to the forked LoRA adapter')
    require_trainable_membership(runtime.model, trainable_names, frozen_modules=frozen_modules)
    logps, hidden = historical_selected(runtime, fork_member, record.serial_record())
    require_trainable_membership(runtime.model, trainable_names, frozen_modules=frozen_modules)
    require(logps.numel() == len(record.target_ids), 'Supervised target count changed')
    require(torch.isfinite(logps).all() and torch.isfinite(hidden).all(), 'Nonfinite selected output')
    return logps, hidden


def target_nll(logps, *, denominator):
    require(type(denominator) is int and denominator >= logps.numel() > 0, 'Invalid target-token denominator')
    require(logps.ndim == 1 and torch.isfinite(logps).all(), 'Invalid selected log probabilities')
    return -logps.sum() / denominator
