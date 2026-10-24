"""Bounded original-question SFT from Member0; guarded recovery and counted replay.

The historical revision helpers below supply only masks, single-member likelihood,
and state recovery. No critique, rollout replacement, or correction objective runs.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import signal
import time

import torch
import nf4_linear
import sparse_experts
import stable_arithmetic
import rl_numerics
from canonical_gradient_reference import run_gradient_reference
from effective_state_guard import capture_effective_state, assert_effective_state
from fork_recovery import install_revision_fork, assert_fork_state, file_sha256
from fork_inference import RevisionPopulation
from gated_training import selected
from guarded_transactions import GuardedForkTransactions, _same
from muon_optimizer import Muon
from revision_objective import EncodedRecord, selected_revision, target_nll
from formation_contract import atomic, append, digest, require, reserve


def tensor_sha256(parameters):
    # Exact swarm.population_fork.tensor_sha256 convention, without its old recipe.
    value = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        require(tensor.dtype == torch.float32, 'FP32 adapter export required')
        value.update(name.encode()); value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class Formation:
    def __init__(self, runtime, bank, tokenizer, config, rows, prompts, root, deadline, *, fixture=False):
        self.runtime, self.bank, self.tokenizer = runtime, bank, tokenizer
        self.config, self.rows, self.prompts = config, rows, prompts
        self.root, self.deadline, self.fixture = Path(root), deadline, fixture
        require(fixture or all(p.device.type == 'cuda' for p in runtime.groups[0].values()), 'Pretrained CPU training forbidden')
        self.started = time.monotonic()
        self.physical_calls = self.replay_calls = self.retained = 0
        self.backward_targets = self.reference_targets = 0
        self.checkpoints, self.timings = {}, dict(save=[], restore=[], update=[], frozen=[])
        self.binding = self.guard = self.transactions = None
        self.valid = True
        self.replay_verified = False
        self.slots = [dict(record_id=p['record_id'], condition=c, state='unstarted' if c in config['evaluation_conditions'] else 'external_pending', answer=None, error=None)
                      for p in prompts for c in ('parent', 'formed')]
        self.peaks = {}
        admission_path=self.root/'probe-report.json'
        if admission_path.exists():
            admission=json.loads(admission_path.read_bytes())
            for i,(allocated,reserved) in enumerate(zip(admission.get('whole_admission_peak_allocated_bytes',[]),
                                                       admission.get('whole_admission_peak_reserved_bytes',[]))):
                self.peaks[str(i)]=dict(allocated=allocated,reserved=reserved)
        self.records = []
        for row in rows:
            e = row['encoded']; n = e['prompt_tokens']
            record = EncodedRecord(tuple(e['input_ids']), tuple(e['labels']), tuple(e['attention_mask']),
                                   tuple(e['input_ids'][n:]), frozenset(config['eos_token_ids']))
            record.serial_record(); self.records.append(record)
        self.frozen_modules = {name: state['gate'] for name, state in bank.states.items()}
        # The host checkpoint bank is immutable state too, including its adapters.
        self.bank_metadata = self.bank_signature()
        for name, state in bank.states.items():
            holder = torch.nn.Module()
            for index, (_, tensor) in enumerate(sorted(state.get('adapters', {}).items())):
                holder.register_buffer('tensor_' + str(index), tensor)
            self.frozen_modules['bank_' + name] = holder
        self.save_status('unstarted')

    def bank_signature(self):
        return digest({name: dict(metadata={k:v for k,v in state.items() if k not in ('gate','adapters')},
            gate_object=id(state['gate']),adapters={k:id(v) for k,v in state.get('adapters',{}).items()})
            for name,state in self.bank.states.items()})

    def event(self, stage, **fields):
        if stage == 'frozen_state_checked':
            require(self.bank_metadata == self.bank_signature(), 'Frozen checkpoint bank structure/metadata changed')
        append(self.root/'events.jsonl', dict(stage=stage, elapsed_seconds=time.monotonic()-self.started,
            physical_optimizer_calls=self.physical_calls, replay_optimizer_calls=self.replay_calls,
            retained_updates=self.retained, activation_generation=self.runtime.checkpoint_generation, **fields))

    def save_status(self, status, **fields):
        atomic(self.root/'slots.json', dict(schema='hive-formation-slots-v1', status=status, slots=self.slots,
            physical_optimizer_calls=self.physical_calls, replay_optimizer_calls=self.replay_calls,
            retained_updates=self.retained, **fields))

    def memory(self):
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
            now = dict(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))
            old = self.peaks.setdefault(str(device), dict(allocated=0, reserved=0))
            for key in now: old[key] = max(old[key], now[key])
        return dict(gpu=self.peaks, host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

    @contextmanager
    def phase(self, label, seconds, *, reserve_seconds=0):
        start = time.monotonic()
        stop = min(start+seconds, self.deadline-reserve_seconds)
        reserve(start, stop, .01, label)
        old_handler = signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError(label)))
        signal.setitimer(signal.ITIMER_REAL, stop-start)
        self.event(label+'_started', cap_seconds=stop-start)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            elapsed = time.monotonic()-start
            self.event(label+'_finished', seconds=elapsed, overshoot_seconds=max(0.,time.monotonic()-stop), memory=self.memory())

    def clear(self):
        for p in self.runtime.model.parameters(): p.grad = None
        for module in self.frozen_modules.values():
            for p in module.parameters(): p.grad = None

    def frozen(self, label):
        start = time.monotonic()
        try:
            assert_fork_state(self.runtime.model, self.binding, frozen_modules=self.frozen_modules)
            report = assert_effective_state(self.runtime, self.guard)
            require(self.bank_metadata == self.bank_signature(), 'Checkpoint bank metadata changed')
        except BaseException:
            self.valid = False
            raise
        self.timings['frozen'].append(time.monotonic()-start)
        self.event('frozen_state_checked', boundary=label, supplemental=report, seconds=self.timings['frozen'][-1])

    def population(self):
        require(self.valid, 'Formation runtime invalid')
        return RevisionPopulation(self.runtime, fork_member=2, expected_parameter_names=self.binding.parameter_names,
            expected_checkpoint_generation=self.runtime.checkpoint_generation, frozen_modules=tuple(self.frozen_modules.values()))

    def advance(self, reason, stale):
        previous = self.runtime.checkpoint_generation
        self.runtime.checkpoint_generation = previous+1
        if stale is not None:
            try: stale._forward(2, [1], 0, None)
            except RuntimeError as error: require('earlier checkpoint activation' in str(error), 'Wrong activation rejection')
            else: raise RuntimeError('Stale activation survived')
        self.event('activation_advanced', reason=reason, previous_generation=previous, stale_activation_rejected=stale is not None)

    def save(self, name, cursor):
        start = time.monotonic(); self.clear()
        require(self.valid and self.transactions.valid, 'Transaction owner invalid')
        try:
            receipt = self.transactions.save(self.root/(name+'.pt'), cursor=cursor, label=name)
        except BaseException:
            self.valid = False
            raise
        self.checkpoints[name] = receipt
        atomic(self.root/'checkpoint-manifest.json', self.checkpoints)
        self.timings['save'].append(time.monotonic()-start)
        return receipt

    def restore(self, name):
        start = time.monotonic(); self.clear()
        require(self.valid and self.transactions.valid, 'Transaction owner invalid')
        value = self.checkpoints[name]
        try:
            result = self.transactions.restore(value['path'], value['sha256'], expected_cursor=value['cursor'], label=name)
        except BaseException:
            self.valid = False
            raise
        self.timings['restore'].append(time.monotonic()-start)
        return result['cursor']

    def cursor(self, count):
        return dict(completed_updates=count, completed_epochs=count,
            next_record=0, batch_sha256=self.config['batch_manifest_sha256'],
            records=count*len(self.records), supervised_tokens=count*self.config['targets_per_update'])

    def setup(self):
        require(self.bank.active == 'joint_gate' and self.bank.states['joint_gate']['sha256'] == self.config['checkpoint_sha256'], 'Parent identity changed')
        model = self.runtime.model
        model.requires_grad_(False); self.clear()
        for module in self.frozen_modules.values(): module.requires_grad_(False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads(); model.train()
        self.binding = install_revision_fork(model, fork_member=2, frozen_modules=self.frozen_modules)
        if not self.fixture:
            require(len(self.binding.parameters) == 470 and sum(p.numel() for p in self.binding.parameters.values()) == 21031936, 'Production fork inventory differs')
        options = self.config['optimizer']
        self.optimizer = Muon([self.binding.parameters[n] for n in self.binding.parameter_names],
            lr=options['lr'], momentum=options['momentum'], weight_decay=options['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step:1.)
        sources = self.config['guard_source_files']
        self.guard, report = capture_effective_state(self.runtime, frozen_fingerprints=self.binding.frozen_fingerprints,
            source_files=sources, arithmetic_modules=dict(nf4_linear=nf4_linear, sparse_experts=sparse_experts,
                stable_arithmetic=stable_arithmetic, rl_numerics=rl_numerics), active_policy=True,
            expected_counts=None if self.fixture else dict(nf4_linears=376,nf4_expert_tensors=92,sparse_views=5888))
        atomic(self.root/'effective-state-capture.json', report)
        (self.root/'effective-state-manifest.json').write_bytes(self.guard.manifest_bytes+b'\n')
        random.seed(self.config['seed']); torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.config['seed'])
        self.advance('common_member0_fork_installed', None)
        self.identity = {key:self.config[key] for key in ('checkpoint_sha256','parent_member_index','recipe_sha256',
            'training_records_sha256','data_manifest_sha256','family_manifest_sha256','batch_manifest_sha256','code_sha256','tokenizer_sha256')}
        self.transactions = GuardedForkTransactions(self.runtime,self.binding,[self.optimizer],[self.scheduler],
            effective_snapshot=self.guard, identity=self.identity, frozen_modules=self.frozen_modules,
            population_factory=self.population, event=self.event)
        self.save('initial', self.cursor(0))
        self.restore('initial')
        require(not self.optimizer.state, 'Initial optimizer not fresh')

    def score(self, backward=False):
        logs, policies = [], []
        for record in self.records:
            self.event('training_forward_started', targets=len(record.target_ids), backward=backward)
            logps, hidden = selected_revision(self.runtime, record, fork_member=2,
                trainable_names=self.binding.parameter_names, historical_selected=selected,
                frozen_modules=tuple(self.frozen_modules.values()))
            logs.append(logps.detach().cpu())
            with torch.no_grad(): policies.append(self.runtime.head(hidden[[0,-1]].detach()).float().log_softmax(-1).cpu())
            if backward:
                self.backward_targets += len(record.target_ids)
                self.event('backward_started', target_contributions_attempted=self.backward_targets)
                target_nll(logps, denominator=self.config['targets_per_update']).backward()
            del logps, hidden
        return torch.cat(logs), torch.cat(policies)

    def preflight(self):
        budget = self.config['budget']
        reserve(time.monotonic(), self.deadline, budget['training_admission']+budget['updates']+budget['evaluation']+budget['recovery'], 'training admission plus complete endpoint')
        with self.phase('training_admission', budget['training_admission'], reserve_seconds=budget['updates']+budget['evaluation']+budget['recovery']):
            self.setup()
            self.event('gradient_reference_started', planned_targets=self.config['targets_per_update'], physical_optimizer_calls_in_reference=0)
            def counted_selected(runtime,record,**kwargs):
                self.reference_targets += len(record.target_ids)
                self.event('reference_production_forward_started',target_presentations_attempted=self.reference_targets)
                return selected_revision(runtime,record,**kwargs)
            reference = run_gradient_reference(self.runtime,self.records,fork_member=2,
                trainable_names=self.binding.parameter_names,denominator=self.config['targets_per_update'],
                selected_revision=counted_selected,historical_selected=selected,
                frozen_modules=tuple(self.frozen_modules.values()),
                deadline=min(self.deadline-budget['updates']-budget['evaluation']-budget['recovery'],
                    self.started+budget['training_admission']),progress=lambda **fields:self.event('gradient_reference_progress',**fields))
            atomic(self.root/'gradient-reference.json',reference.diagnostics)
            del reference
            self.clear(); self.frozen('after_gradient_reference'); self.restore('initial')
            require(max(self.timings['restore']) <= self.config['restore_seconds'], 'Measured full restore exceeds frozen recovery cap')
            self.event('training_admission_passed', physical_update_cost_measured=False, restore_seconds=self.timings['restore'])

    def update(self, count, *, replay=False):
        budget = self.config['budget']; cap = self.config['operation_seconds']
        remaining_calls = self.config['physical_call_cap']-self.physical_calls
        reserve(time.monotonic(), self.deadline, remaining_calls*cap+budget['evaluation']+budget['recovery'], 'remaining update/replay/evaluation')
        name = 'replay1' if replay else 'update'+str(count)
        start = time.monotonic()
        with self.phase(name, cap, reserve_seconds=budget['evaluation']+budget['recovery']):
            self.restore('initial' if count == 1 else 'update'+str(count-1))
            stale = self.population(); self.runtime.model.train(); self.clear()
            before, policies_before = self.score(backward=True)
            params = [self.binding.parameters[n] for n in self.binding.parameter_names]
            require(all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in params), 'Invalid intended gradients')
            norm = float(torch.nn.utils.clip_grad_norm_(params,self.config['optimizer']['grad_clip'],error_if_nonfinite=True))
            require(norm > 0, 'Zero fork gradient')
            self.frozen('before_'+name)
            require(self.physical_calls < self.config['physical_call_cap'], 'Physical call ceiling')
            self.physical_calls += 1; self.replay_calls += int(replay)
            append(self.root/'physical-calls.jsonl',dict(physical_call=self.physical_calls,operation=name,
                replay=replay,phase='invocation_started',targets=self.config['targets_per_update']))
            try:
                self.optimizer.step(); self.scheduler.step()
            finally:
                # Even a partly mutating/throwing optimizer invalidates every old cache.
                self.advance(name,stale)
            append(self.root/'physical-calls.jsonl',dict(physical_call=self.physical_calls,operation=name,phase='invocation_returned'))
            require(all(bool(torch.isfinite(p).all()) for p in params), 'Nonfinite updated fork')
            self.clear(); self.frozen('after_'+name); self.runtime.model.eval()
            with torch.no_grad(): after, policies_after = self.score()
            delta = (after-before).abs(); kl = (policies_before.exp()*(policies_before-policies_after)).sum(-1)
            probability = (policies_before.exp()-policies_after.exp()).abs()
            diagnostics = dict(loss_before=float(-before.mean()),loss_after=float(-after.mean()),
                max_target_logp_change=float(delta.max()),mean_target_logp_change=float(delta.mean()),
                max_policy_KL=float(kl.max()),max_policy_probability_change=float(probability.max()),gradient_norm=norm)
            atomic(self.root/(name+'-diagnostics.json'),diagnostics)
            require(float(delta.max())<=1. and float(delta.mean())<=.25 and float(kl.max())<=.01
                    and float(probability.max())<=.05,'Frozen trust boundary exceeded')
            saved = self.save(name,self.cursor(count))
            if replay:
                expected=torch.load(self.checkpoints['update1']['path'],map_location='cpu',weights_only=True)
                actual=torch.load(saved['path'],map_location='cpu',weights_only=True)
                require(_same(actual,expected),'Complete state/cursor/RNG replay differs')
                initial=torch.load(self.checkpoints['initial']['path'],map_location='cpu',weights_only=True)
                changed=sum(not torch.equal(expected['adapters'][n],initial['adapters'][n]) for n in self.binding.parameter_names)
                require(changed>0,'No changed adapter matrices')
                self.replay_verified=True
                atomic(self.root/'replay-verification.json',dict(passed=True,complete_state_exact=True,changed_matrices=changed,
                    physical_calls=self.physical_calls,first_sha256=self.checkpoints['update1']['sha256'],replay_sha256=saved['sha256']))
                del expected,actual,initial
                self.restore('update1')
            else:
                self.retained=count
            self.event('physical_update_completed',operation=name,**diagnostics)
        self.timings['update'].append(time.monotonic()-start)
        require(self.timings['update'][-1]<=cap,'Actual update/replay overshot cap')
        if replay:
            # Real first update/replay is already charged; later work uses measured cost.
            measured=2*max(self.timings['update'])
            atomic(self.root/'measured-update-cost.json',dict(actual_seconds=self.timings['update'],
                conservative_next_operation_seconds=measured,operation_cap_seconds=cap,
                within_cap=measured<=cap,physical_optimizer_calls=self.physical_calls))
            if self.config['retained_updates']>1: require(measured<=cap,'Measured cost blocks subsequent updates')

    def export_adapter(self):
        self.clear(); self.frozen('before_adapter_export')
        adapters={n.replace('.member2.','.member0.'):p.detach().cpu().clone() for n,p in self.binding.parameters.items()}
        provenance={key:self.config[key] for key in ('parent_member_index','tokenizer_sha256','code_sha256','recipe_sha256',
            'data_manifest_sha256','family_manifest_sha256','batch_manifest_sha256')}
        provenance.update(parent_checkpoint_sha256=self.config['checkpoint_sha256'],base_manifest_sha256=self.config['export_manifest_sha256'],
            retained_updates=self.retained,physical_optimizer_calls=self.physical_calls,replay_optimizer_calls=self.replay_calls,
            supervised_tokens=self.retained*self.config['targets_per_update'],recovery_verified=self.replay_verified,artifact_kind='formed')
        require(self.retained==self.config['retained_updates'] and self.replay_verified,'Incomplete formation cannot be exported as formed')
        state=dict(schema='hive-formed-adapter-v1',adapters=adapters,tensor_sha256=tensor_sha256(adapters),provenance=provenance)
        pending=self.root/'formed-adapter.pt.pending'
        require(not pending.exists() and not (self.root/'formed-adapter.pt').exists(),'Preserve export evidence')
        torch.save(state,pending)
        reread=torch.load(pending,map_location='cpu',weights_only=True)
        require(_same(state,reread),'Adapter export changed bytes/metadata')
        self.frozen('after_adapter_export')
        pending.rename(self.root/'formed-adapter.pt')
        atomic(self.root/'formed-adapter.json',dict(file='formed-adapter.pt',sha256=file_sha256(self.root/'formed-adapter.pt'),
            tensor_sha256=state['tensor_sha256'],provenance=provenance))

    def evaluate(self):
        budget=self.config['budget']
        reserve(time.monotonic(),self.deadline,budget['evaluation']+budget['recovery'],'complete paired evaluation')
        by_id={p['record_id']:p for p in self.prompts}
        # Fixed condition-block order avoids fourteen full-state restorations; disclosed in recipe.
        for condition,checkpoint in [('parent','initial'),('formed','update'+str(self.retained))]:
            if condition not in self.config['evaluation_conditions']: continue
            with self.phase('activate_'+condition,self.config['restore_seconds']): self.restore(checkpoint)
            self.runtime.model.eval()
            for slot in (s for s in self.slots if s['condition']==condition):
                prompt=by_id[slot['record_id']]; population=self.population()
                slot.update(state='started',checkpoint_sha256=self.checkpoints[checkpoint]['sha256'])
                self.save_status('evaluating'); start=time.monotonic()
                try:
                    self.memory()
                    answer=population.generate(prompt['input_ids'],eos_ids=self.config['eos_token_ids'],
                        max_new_tokens=self.config['max_new_tokens'],deadline=min(start+self.config['response_seconds_cap'],
                            self.deadline-self.config['restore_seconds']))
                    slot['answer']=answer; self.save_status('evaluating')
                    tokens=answer['tokens']; clean=tokens[:-1] if tokens and tokens[-1] in self.config['eos_token_ids'] else tokens
                    answer['raw_text']=self.tokenizer.decode(tokens,skip_special_tokens=False)
                    answer['text']=self.tokenizer.decode(clean,skip_special_tokens=False)
                    slot['state']='captured'
                except BaseException as error:
                    slot['answer']=getattr(error,'baseline_partial_answer',slot['answer'])
                    if slot['answer'] is not None:
                        try:
                            tokens=slot['answer']['tokens']
                            clean=tokens[:-1] if tokens and tokens[-1] in self.config['eos_token_ids'] else tokens
                            slot['answer']['raw_text']=self.tokenizer.decode(tokens,skip_special_tokens=False)
                            slot['answer']['text']=self.tokenizer.decode(clean,skip_special_tokens=False)
                        except Exception as decode_error:
                            slot['partial_decode_error']=dict(type=type(decode_error).__name__,message=str(decode_error))
                    slot.update(state='failed',error=dict(type=type(error).__name__,message=str(error)))
                    raise
                finally:
                    slot.update(operation_elapsed_seconds=time.monotonic()-start,
                        deadline_overshoot_seconds=max(0.,time.monotonic()-start-self.config['response_seconds_cap']))
                    self.memory(); self.save_status('evaluating')
            self.clear(); self.frozen('after_evaluation_'+condition)

    def run(self):
        try:
            self.preflight(); self.update(1); self.update(1,replay=True)
            for count in range(2,self.config['retained_updates']+1): self.update(count)
            self.export_adapter(); self.evaluate()
            with self.phase('final_restore',self.config['restore_seconds']): self.restore('initial')
            self.frozen('final_initial_restored')
            result=dict(status='complete',physical_optimizer_calls=self.physical_calls,replay_optimizer_calls=self.replay_calls,
                retained_updates=self.retained,retained_target_presentations=self.retained*self.config['targets_per_update'],
                physical_update_target_presentations=self.physical_calls*self.config['targets_per_update'],
                production_backward_target_contributions_attempted=self.backward_targets,
                gradient_reference_target_presentations_attempted=self.reference_targets,
                initial_work_fork_restored=True,recovery_verified=self.replay_verified,memory=self.memory(),
                timings=self.timings,capability_success_claimed=False)
            atomic(self.root/'formation-result.json',result); self.save_status('complete'); return result
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL,0); self.clear()
            rollback='unavailable_before_initial'
            if self.valid and self.transactions is not None and self.transactions.valid and 'initial' in self.checkpoints:
                try:
                    with self.phase('failure_restore',self.config['restore_seconds']): self.restore('initial')
                    rollback='verified_initial_restored'
                except BaseException as recovery_error:
                    rollback='failed_process_invalid:'+type(recovery_error).__name__; self.valid=False
            result=dict(status='failed',error_type=type(error).__name__,error=str(error),rollback=rollback,
                physical_optimizer_calls=self.physical_calls,replay_optimizer_calls=self.replay_calls,retained_updates=self.retained,
                matched_endpoint_complete=False,production_backward_target_contributions_attempted=self.backward_targets,
                gradient_reference_target_presentations_attempted=self.reference_targets,
                transaction_owner_valid=None if self.transactions is None else self.transactions.valid,
                no_retry=True,memory=self.memory(),timings=self.timings)
            atomic(self.root/'formation-failure.json',result); self.save_status('failed',failure=result)
            raise
