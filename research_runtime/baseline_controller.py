"""Fixed seven-case parent inference controller and bounded entry. AI-authored.

Importing this module does not import a model, install packages or call a provider.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zlib


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def config_and_prompts(root):
    config = read_json(root / 'baseline-config.json')
    hashes = read_json(root / 'source-hashes.json')
    for name, expected in hashes.items():
        require(Path(name).name == name, 'Non-flat payload path')
        require(digest((root / name).read_bytes()) == expected, 'Payload changed: ' + name)
    raw = (root / 'prompts.jsonl').read_bytes()
    require(digest(raw) == config['prompt_sha256'], 'Prompt bytes changed')
    prompts = [json.loads(line) for line in raw.splitlines()]
    by_id = {p['record_id']: p for p in prompts}
    require(len(prompts) == len(by_id) == len(config['ids']) == 7, 'Expected exactly seven unique prompts')
    require(set(by_id) == set(config['ids']), 'Panel identity mismatch')
    for p in prompts:
        require(digest(p['messages'][0]['content'].encode()) == p['prompt_sha256'], 'Prompt text changed')
        require(p['input_ids'] and all(type(t) is int and t >= 0 for t in p['input_ids']), 'Invalid prefix')
        require(len(p['input_ids']) + config['max_new_tokens'] <= config['max_sequence'], 'Context exceeded')
    return config, [by_id[i] for i in config['ids']], hashes


def initial_report(config, prompts, hashes):
    return dict(schema='hive-parent-baseline-slots-v1', status='unstarted',
                source_hashes=hashes, prompt_sha256=config['prompt_sha256'],
                checkpoint_sha256=config['checkpoint_sha256'], member_ids=[0], optimizer_updates=0,
                slots=[dict(record_id=p['record_id'], prompt_sha256=p['prompt_sha256'],
                            input_ids_sha256=digest(canonical(p['input_ids'])), state='unstarted',
                            started_at_seconds=None, finished_at_seconds=None,
                            operation_elapsed_seconds=None, answer=None, error=None) for p in prompts])


def run_worker(root, entry_started, *, runtime_factory=None, clock=time.monotonic):
    root = Path(root)
    config, prompts, hashes = config_and_prompts(root)
    report = initial_report(config, prompts, hashes)
    worker_deadline = entry_started + config['worker_boundary_seconds']
    save = lambda: atomic(root / 'slots.json', {**report, 'entry_elapsed_seconds': clock()-entry_started})
    save()  # Every slot exists before importing the numerical runtime.
    try:
        require(worker_deadline-clock() >= config['admission_seconds'] + config['bookkeeping_seconds'],
                'Insufficient reserve for numerical admission')
        report['status'] = 'admitting'
        save()
        if runtime_factory is None:
            from runtime_admission import admit
            runtime_factory = admit
        admission_deadline = min(clock()+config['admission_seconds'], worker_deadline-config['bookkeeping_seconds'])
        runtime = runtime_factory(config, admission_deadline, root)
        require(clock() <= admission_deadline, 'Numerical admission overshot its deadline')
        require(runtime.identity == {'checkpoint_sha256': config['checkpoint_sha256'], 'member_ids': [0]},
                'Wrong parent/member identity')
        report['admission_passed'] = True
        report['status'] = 'generating'
        save()
        for prompt, slot in zip(prompts, report['slots']):
            now = clock()
            if worker_deadline-now < config['response_seconds_cap'] + config['bookkeeping_seconds']:
                report['budget_stop_before'] = slot['record_id']
                break
            slot.update(state='started', started_at_seconds=now-entry_started)
            save()
            try:
                answer = runtime.generate(prompt['input_ids'], eos_ids=config['eos_token_ids'],
                                          max_new_tokens=config['max_new_tokens'],
                                          deadline=now+config['response_seconds_cap'])
                # Save the complete native return before decoding, which may itself fail.
                slot['answer'] = answer
                save()
                require(answer['member_ids'] == [0], 'Non-parent answer')
                require(answer['stop'] in ('eos', 'resource_deadline', 'max_new_tokens'), 'Unknown native stop')
                require(type(answer['seconds']) in (int, float) and math.isfinite(answer['seconds'])
                        and answer['seconds'] >= 0, 'Invalid native timing')
                require(isinstance(answer['tokens'], list) and len(answer['tokens']) <= config['max_new_tokens']
                        and all(type(t) is int and t >= 0 for t in answer['tokens']), 'Invalid generated tokens')
                answer['raw_text'] = runtime.decode(answer['tokens'], skip_special_tokens=False)
                # Remove only one terminal stop ID. Other special tokens remain visible.
                scoring_tokens = answer['tokens'][:-1] if answer['tokens'] and answer['tokens'][-1] in config['eos_token_ids'] else answer['tokens']
                answer['text'] = runtime.decode(scoring_tokens, skip_special_tokens=False)
                answer['decoding'] = config['decoding']
                slot['state'] = 'captured'
            except Exception as error:
                partial = getattr(error, 'baseline_partial_answer', None)
                if partial is not None and slot['answer'] is None:
                    slot['answer'] = partial
                    try:
                        partial['raw_text'] = runtime.decode(partial['tokens'], skip_special_tokens=False)
                        tokens = partial['tokens']
                        scoring_tokens = tokens[:-1] if tokens and tokens[-1] in config['eos_token_ids'] else tokens
                        partial['text'] = runtime.decode(scoring_tokens, skip_special_tokens=False)
                        partial['decoding'] = config['decoding']
                    except Exception as decode_error:
                        slot['partial_decode_error'] = {'type': type(decode_error).__name__, 'message': str(decode_error)}
                slot.update(state='failed', error={'type': type(error).__name__, 'message': str(error),
                                                  'traceback': traceback.format_exc()})
                report['status'] = 'failed'
            finally:
                slot['finished_at_seconds'] = clock()-entry_started
                slot['operation_elapsed_seconds'] = clock()-now
                slot['deadline_overshoot_seconds'] = max(0, clock()-now-config['response_seconds_cap'])
                save()
            if slot['state'] == 'failed':
                break  # No retry; preserve later slots as unstarted.
        if report['status'] != 'failed':
            report['status'] = 'complete' if all(s['state'] == 'captured' for s in report['slots']) else 'partial'
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()})
    finally:
        save()
    return report


def install(config, root, timeout):
    with (root / 'pip-install.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--disable-pip-version-check',
                        *config['packages']], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def launch(root, started, timeout):
    child = subprocess.Popen([sys.executable, '-u', 'baseline_controller.py', 'worker', str(started)],
                             cwd=root, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)


def entry(payload, hashes, root, *, installer=install, launcher=launch, clock=time.monotonic):
    started = clock()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(not any(root.iterdir()), 'Preserve existing entry outputs')
    require(payload.keys() == hashes.keys(), 'Payload membership differs')
    for name, value in payload.items():
        require(Path(name).name == name, 'Unsafe payload name')
        raw = zlib.decompress(base64.b64decode(value, validate=True))
        require(digest(raw) == hashes[name], 'Payload hash mismatch')
        if name.endswith('.py'):
            compile(raw, name, 'exec')
        (root / name).write_bytes(raw)
    atomic(root / 'source-hashes.json', hashes)
    config, prompts, _ = config_and_prompts(root)
    atomic(root / 'slots.json', initial_report(config, prompts, hashes))
    report = dict(schema='hive-parent-baseline-entry-v1', status='installing', source_hashes=hashes,
                  provider_seconds=config['provider_seconds'], worker_boundary_seconds=config['worker_boundary_seconds'],
                  planned_questions=7, optimizer_updates=0, paid_scaling=False)
    save = lambda: atomic(root / 'entry-report.json', {**report, 'elapsed_seconds': clock()-started})
    save()
    previous = Path.cwd()
    try:
        os.chdir(root)
        remaining_install = config['installation_seconds']-(clock()-started)
        require(remaining_install > 0, 'Entry unpack consumed installation reserve')
        installer(config, root, remaining_install)
        require(clock()-started <= config['installation_seconds'], 'Installation overshot reserve')
        os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='17')
        remaining = config['worker_boundary_seconds']-(clock()-started)
        require(remaining >= config['admission_seconds']+config['bookkeeping_seconds'], 'Worker reserve unavailable')
        report['status'] = 'worker_running'
        save()
        report['worker_returncode'] = launcher(root, started, remaining)
        report['status'] = read_json(root/'slots.json')['status'] if report['worker_returncode'] == 0 else 'failed'
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()})
    finally:
        os.chdir(previous)
        save()
        files = {p.name: digest(p.read_bytes()) for p in root.iterdir()
                 if p.name in ('entry-report.json', 'slots.json', 'probe-report.json', 'pip-install.log')}
        atomic(root/'output-manifest.json', {'schema': 'hive-parent-baseline-export-v1', 'files': files,
                                            'source_hashes': hashes, 'entry_elapsed_seconds': clock()-started})
    return report


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] == 'worker', 'Use the prepared entry')
    result = run_worker(Path.cwd(), float(sys.argv[2]))
    sys.exit(0 if result['status'] in ('complete', 'partial') else 1)
