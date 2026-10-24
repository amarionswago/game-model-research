"""Checkpointed GPU training stage over frozen admitted records. AI-authored.

The entry point must supply the registered GLM loader, admitted data manifest and
arm-specific frozen stream. This module neither chooses data nor launches jobs.
Early valid updates remain in the stage; recovery replay work is charged.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import time
import traceback
from uuid import uuid4

import torch

from swarm.adapter_checkpoint import compare_population_parameters, load_population, save_population
from swarm.adapter_training import TargetStream, population_update, learning_rate_multiplier
from swarm.run_reports import write_report


def update_time_reserve(measured_seconds, *, updates=1, extra_seconds=30.):
    """Conservative admission estimate, not a guarantee about an unseen batch.

    A later batch can cost more than prior ones. The outer hard timer remains
    necessary, but known update timings must not be ignored when starting work.
    """
    if updates < 1 or extra_seconds < 0 or not math.isfinite(extra_seconds):
        raise ValueError('Invalid update time reserve')
    values = [float(value) for value in measured_seconds]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Invalid measured update duration')
    return 1.25 * max(values, default=0.) * updates + extra_seconds


def run_stage(records, forward_member, named_groups, *, identity, output_directory,
              total_updates, learning_rate, objective, max_seconds, resume=False,
              target_tokens=8192, charged_prior_seconds=None, on_update=None):
    """Train one declared stage, checking actual updates and checkpoint recovery.

    Each named_groups entry maps one member's parameter names to FP32 tensors.
    Formation calls use one member and its own stream; continuation uses the
    common stream and the selected independent/team objective. The caller must
    preserve initializers, formation forks, and all control/replication manifests.
    """
    required = {'base', 'tokenizer', 'data', 'code', 'recipe', 'stage', 'seed'}
    if not required.issubset(identity) or any(not identity[key] and identity[key] != 0 for key in required):
        raise ValueError('Missing immutable stage identity')
    if total_updates < 3 or target_tokens != 8192 or learning_rate not in (5e-5, 2e-5) or max_seconds <= 0:
        raise ValueError('Stage differs from the registered initial recipe')
    if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('This stage requires the two registered T4 GPUs')
    if not named_groups or any(not group for group in named_groups):
        raise ValueError('No member parameters')
    parameters = {name: parameter for group in named_groups for name, parameter in group.items()}
    if len(parameters) != sum(len(group) for group in named_groups):
        raise ValueError('Parameter names overlap across members')
    groups = [list(group.values()) for group in named_groups]
    if any(p.device.type != 'cuda' or p.dtype != torch.float32 for p in parameters.values()):
        raise ValueError('FP32 adapters must already be loaded on the GPUs')
    optimizers = [torch.optim.AdamW(group, lr=learning_rate, betas=(.9,.95), eps=1e-8, weight_decay=.01)
                  for group in groups]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lambda step: learning_rate_multiplier(min(step,total_updates-1), total_updates))
                  for opt in optimizers]
    scaler = torch.amp.GradScaler('cuda', init_scale=32.)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'population.pt'
    report_path = output / 'stage.json'
    if checkpoint.exists() and not resume:
        raise FileExistsError('Existing stage checkpoint requires explicit resume')
    previous_attempts = sorted(output.glob('attempt-*.json'))
    if previous_attempts and not resume:
        raise FileExistsError('Preserve the previous attempt; start with explicit recovery')
    if resume and charged_prior_seconds is None:
        raise ValueError('Resume requires measured prior elapsed charges, including failed work')
    elapsed_charged = float(charged_prior_seconds or 0.)
    observed_previous = sum(json.loads(p.read_text()).get('attempt_elapsed_seconds', 0.) for p in previous_attempts)
    if elapsed_charged < observed_previous:
        raise ValueError('Prior time charge omits recorded attempt work')
    attempt_path = output / ('attempt-' + uuid4().hex + '.json')
    started, last_save = time.monotonic(), time.monotonic()
    cursor = {'completed_updates': 0, 'stream': {'record':0,'target_offset':0,'supervised_tokens':0},
              'recovery_verified': False, 'physical_gradient_bearing_member_targets': 0,
              'elapsed_training_seconds': 0.}
    if resume:
        cursor = load_population(checkpoint, parameters, optimizers, schedulers, identity=identity, scaler=scaler)
    stream = TargetStream(records, cursor=cursor['stream'])
    elapsed_before = max(cursor['elapsed_training_seconds'], elapsed_charged)
    report = {'started_at':datetime.now(timezone.utc).isoformat(), 'identity':identity,
              'status':'running', 'objective':objective, 'completed_updates':cursor['completed_updates'],
              'recovery_verified':cursor['recovery_verified'], 'measurements':[]}
    measured_step_seconds = [measurement['seconds'] for path in previous_attempts
                             for measurement in json.loads(path.read_text()).get('measurements', [])
                             if 'seconds' in measurement]

    def persist_report():
        report['attempt_elapsed_seconds'] = time.monotonic() - started
        report['charged_stage_elapsed_seconds'] = elapsed_before + report['attempt_elapsed_seconds']
        report['attempt_file'] = attempt_path.name
        write_report(attempt_path, report)
        write_report(report_path, report)

    def save():
        nonlocal last_save
        cursor['stream'] = dict(stream.cursor)
        cursor['elapsed_training_seconds'] = elapsed_before + time.monotonic() - started
        save_population(checkpoint, parameters, optimizers, schedulers, identity=identity, cursor=cursor, scaler=scaler)
        write_report(output/'checkpoint-metadata.json', {'identity':identity,'cursor':cursor,
                     'saved_at':datetime.now(timezone.utc).isoformat()})
        last_save = time.monotonic()
        report.update(cursor=cursor, completed_updates=cursor['completed_updates'],
                      elapsed_seconds=cursor['elapsed_training_seconds'], recovery_verified=cursor['recovery_verified'])
        persist_report()

    def step(segments):
        for gpu in range(2):
            torch.cuda.reset_peak_memory_stats(gpu)
        beginning = time.monotonic()
        result = population_update(forward_member, segments, groups, optimizers, objective=objective, scaler=scaler)
        for schedule in schedulers:
            schedule.step()
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        result.update(seconds=time.monotonic()-beginning,
                      peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
                      peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)],
                      host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        measured_step_seconds.append(result['seconds'])
        cursor['physical_gradient_bearing_member_targets'] += result['gradient_bearing_member_targets']
        return result

    def time_available_for(updates=1, *, extra_seconds=30.):
        remaining = max_seconds - elapsed_before - (time.monotonic() - started)
        needed = update_time_reserve(measured_step_seconds, updates=updates, extra_seconds=extra_seconds)
        if remaining < needed:
            report.update(status='time_limit_checkpointed_partial_stage',
                          stop_reason='Insufficient time for measured update cost and checkpoint reserve',
                          time_admission={'remaining_seconds':remaining, 'estimated_required_seconds':needed,
                                          'updates':updates, 'measured_updates':len(measured_step_seconds)})
            return False
        return True

    def check_step_limits(result):
        if elapsed_before + time.monotonic() - started >= max_seconds:
            raise TimeoutError('Stage time allowance reached during recovery; restore saved checkpoint')
        if any(result['peak_reserved_bytes'][i] > torch.cuda.get_device_properties(i).total_memory-int(.75*1024**3) for i in range(2)):
            raise MemoryError('Recovery update lacks registered device headroom')
        if result['host_peak_rss_bytes'] > 28*1024**3:
            raise MemoryError('Recovery update exceeds registered host memory')

    def verify_recovery(probe_segment):
        with torch.no_grad():
            expected_logp = [forward_member(i, probe_segment).detach().float().cpu() for i in range(len(groups))]
        loaded = load_population(checkpoint, parameters, optimizers, schedulers, identity=identity, scaler=scaler)
        if loaded['stream'] != cursor['stream']:
            raise RuntimeError('Saved sampler cursor differs from the training stream')
        with torch.no_grad():
            differences = [float((forward_member(i, probe_segment).detach().float().cpu()-expected_logp[i]).abs().max()) for i in range(len(groups))]
        report['recovery'] = {'status': 'checking', 'logp_max_abs_differences': differences,
                              'parameter_rtol': 1e-5, 'parameter_atol': 1e-6}
        persist_report()
        if max(differences) > 1e-3:
            raise RuntimeError('Post-reload target log probabilities differ by more than 1e-3')
        # Replay the next update from the saved state; retain exactly one
        # effective update and separately charge the discarded computation.
        load_population(checkpoint, parameters, optimizers, schedulers, identity=identity, scaler=scaler)
        next_segments = stream.take(target_tokens)
        first = step(next_segments)
        report['measurements'].append(dict(first, completed_update=2, discarded_recovery_replay=True))
        persist_report()
        check_step_limits(first)
        expected = {name:p.detach().cpu().clone() for name,p in parameters.items()}
        load_population(checkpoint, parameters, optimizers, schedulers, identity=identity, scaler=scaler)
        repeated = step(next_segments)
        replay_measurement = dict(repeated, completed_update=2, recovery_replay_candidate=True,
                                  retained_recovery_replay=False)
        report['measurements'].append(replay_measurement)
        persist_report()
        check_step_limits(repeated)
        comparison = compare_population_parameters(parameters, expected, rtol=1e-5, atol=1e-6)
        report['recovery'].update(status='comparison_passed' if comparison['passed'] else 'comparison_failed',
                                  parameter_comparison=comparison)
        persist_report()  # Preserve the numerical evidence even when the guard fails.
        if not comparison['passed']:
            raise RuntimeError('Resumed next update differs from uninterrupted update')
        cursor['completed_updates'] += 1
        cursor['recovery_verified'] = True
        report['recovery'].update({'next_update_parameter_max_abs_difference':comparison['max_abs_difference'],
                              'additional_replay_targets':first['gradient_bearing_member_targets'],
                              'extra_forward_calls_for_reload_comparison':2*len(groups)})
        save()
        replay_measurement['retained_recovery_replay'] = True
        report['recovery']['status'] = 'verified_checkpoint_saved'
        persist_report()
        print(json.dumps({'event':'recovery_verified','stage':identity['stage'],
                          'retained_updates':cursor['completed_updates'],
                          'discarded_replay_updates':1}),flush=True)

    try:
        if not resume:
            save()
        if resume and not cursor['recovery_verified']:
            if cursor['completed_updates'] > 1:
                raise RuntimeError('Unverified recovery after multiple updates is not a valid resume state')
            if cursor['completed_updates'] == 1:
                if not time_available_for(2, extra_seconds=60.):
                    save()
                    return report
                probe_segment = TargetStream(records).take(target_tokens)[0]
                verify_recovery(probe_segment)
        while cursor['completed_updates'] < total_updates:
            if not time_available_for():
                break
            segments = stream.take(target_tokens)
            result = step(segments)
            cursor['completed_updates'] += 1
            result['completed_update'] = cursor['completed_updates']
            report['measurements'].append(result)
            report['completed_updates'] = cursor['completed_updates']
            persist_report()
            print(json.dumps({'event':'optimizer_update_finished','stage':identity['stage'],
                              'retained_updates':cursor['completed_updates'],
                              'loss':result['loss'],'seconds':result['seconds'],
                              'supervised_targets_per_member':target_tokens,
                              'peak_reserved_bytes':result['peak_reserved_bytes']}),flush=True)
            enough_memory = all(result['peak_reserved_bytes'][i] <= torch.cuda.get_device_properties(i).total_memory-int(.75*1024**3) for i in range(2))
            if not enough_memory:
                report['status'] = 'headroom_limit_checkpointed_partial_stage'
                save()
                break
            if result['host_peak_rss_bytes'] > 28*1024**3:
                report['status'] = 'host_memory_limit_checkpointed_partial_stage'
                save()
                break
            if cursor['completed_updates'] == 1 and not cursor['recovery_verified']:
                save()
                if not time_available_for(2, extra_seconds=60.):
                    break
                verify_recovery(segments[0])
            if on_update is not None:
                if cursor['completed_updates'] <= 10:
                    save()  # Preserve early valid updates before callback checks.
                stop_reason = on_update(cursor, report)
                if stop_reason:
                    report.update(status='resource_review_checkpointed_partial_stage', stop_reason=str(stop_reason))
                    save()
                    break
            if cursor['completed_updates'] % 16 == 0 or time.monotonic() - last_save >= 900:
                save()
            print(f"stage={identity['stage']} updates={cursor['completed_updates']}/{total_updates} loss={result['loss']:.6f}", flush=True)
        if cursor['completed_updates'] == total_updates:
            report['status'] = 'stage_complete'
        save()
        return report
    except BaseException as error:
        # The previous checkpoint is the sole resumable state. A failed step can
        # have partially mutated an optimizer; never overwrite that checkpoint.
        traceback.print_exc()
        report.update(status='failed_restore_last_checkpoint', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc(),
                      failed_attempt_elapsed_seconds=time.monotonic()-started)
        persist_report()
        raise
