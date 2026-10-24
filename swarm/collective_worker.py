"""Two-arm complete-round controller with charged replay and full recovery."""
from contextlib import contextmanager
import random
import resource
import signal
import time
from pathlib import Path

import torch

from effective_state_guard import capture_effective_state
from guarded_transactions import _same
from muon_optimizer import Muon
from swarm.collective_contract import ARMS, atomic, append, require, reserve, slots
from swarm.collective_recovery import CollectiveTransactions, bind
from swarm.population_objective import recomputed_backward, selected
from swarm.population_runtime import Population


class Collective:
    def __init__(self, runtime, gate, tokenizer, config, rows, prompts, root, deadline, *,
                 frozen_modules, arithmetic_modules, numerical_check, clock=time.monotonic,
                 optimizer_factory=Muon, forward=None, guard_factory=capture_effective_state):
        self.runtime, self.gate, self.tokenizer = runtime, gate, tokenizer
        self.config, self.rows, self.prompts = config, rows, prompts
        self.root, self.deadline, self.clock = Path(root), deadline, clock
        require(config['fixture'] or all(p.device.type == 'cuda' for g in runtime.groups for p in g.values()), 'Pretrained CPU work forbidden')
        require(config['fixture'] or (optimizer_factory is Muon and forward is None and guard_factory is capture_effective_state), 'Production implementation override forbidden')
        self.frozen_modules, self.arithmetic_modules = frozen_modules, arithmetic_modules
        self.numerical_check, self.optimizer_factory, self.guard_factory = numerical_check, optimizer_factory, guard_factory
        self.forward = forward or (lambda member, row:selected(runtime, member, row))
        self.started = clock(); self.transactions = None; self.current_arm = None
        self.active_arm = None
        self.common = {}; self.ahead = {}; self.receipts = {}; self.durations = []
        self.answer_slots = slots(prompts)
        self.report = dict(schema='hive-collective-run-v1', status='unstarted', completed_rounds=0,
            physical_optimizer_calls=0, replay_optimizer_calls=0, arm_update_attempts=0,
            forward_attempts=0, forward_completed=0, backward_target_contributions_attempted=0,
            per_arm={arm:dict(physical_optimizer_calls=0, replay_optimizer_calls=0,
                successful_update_attempts=0, retained_rounds=0, retained_target_tokens=0) for arm in ARMS},
            last_common={}, unfinished={}, generation_costs=[], memory={}, source_identity={k:config[k] for k in
                ('recipe_sha256','code_sha256','training_records_sha256','formed_parents_sha256','prompt_sha256')})
        self.persist()

    def event(self, stage, **fields):
        append(self.root/'events.jsonl', dict(stage=stage, elapsed_seconds=self.clock()-self.started,
            physical_optimizer_calls=self.report['physical_optimizer_calls'],
            activation_generation=self.runtime.checkpoint_generation, **fields))

    def memory(self):
        self.report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            row = self.report['memory'].setdefault(str(i), dict(peak_allocated=0, peak_reserved=0))
            row['peak_allocated'] = max(row['peak_allocated'], torch.cuda.max_memory_allocated(i))
            row['peak_reserved'] = max(row['peak_reserved'], torch.cuda.max_memory_reserved(i))

    def persist(self):
        self.report['last_common'] = self.common
        self.report['elapsed_seconds'] = self.clock()-self.started
        atomic(self.root/'collective-report.json', self.report)
        atomic(self.root/'slots.json', dict(schema='hive-collective-slots-v1', status=self.report['status'], slots=self.answer_slots))
        atomic(self.root/'checkpoint-manifest.json', dict(receipts=self.receipts, last_common=self.common,
            incomplete_round=self.ahead, purpose='only last_common is eligible for matched evaluation'))

    @contextmanager
    def phase(self, name, seconds, *, end=None):
        start = self.clock(); stop = min(start+seconds, self.deadline if end is None else end)
        reserve(start, stop, .001, name)
        previous = None
        if not self.config['fixture']:
            previous = (signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL))
            signal.signal(signal.SIGALRM, lambda *args:(_ for _ in ()).throw(TimeoutError(name)))
            signal.setitimer(signal.ITIMER_REAL, stop-start)
        self.event(name+'_started', deadline=stop)
        try:
            yield
            require(self.clock() <= stop, 'Operation exceeded frozen allowance: '+name)
        finally:
            if previous is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous[0])
                delay, interval = previous[1]
                if delay: signal.setitimer(signal.ITIMER_REAL, max(.001, delay-(self.clock()-start)), interval)
            self.memory()
            self.event(name+'_finished', seconds=self.clock()-start, overshoot_seconds=max(0.,self.clock()-stop))

    def clear(self):
        for p in self.runtime.model.parameters(): p.grad = None
        if hasattr(self, 'binding'):
            for p in self.binding.parameters.values(): p.requires_grad_(True)

    def cursor(self, arm, round_number, phase='complete'):
        return dict(arm=arm, completed_rounds=round_number, next_record=0, phase=phase,
            record_order=self.config['record_order'], training_records_sha256=self.config['training_records_sha256'],
            target_tokens=round_number*self.config['target_tokens_per_arm_update'])

    def save(self, label, cursor):
        self.clear()
        receipt = self.transactions.save(self.root/(label+'.pt'), cursor=cursor, label=label)
        self.receipts[label] = receipt; self.persist()
        return receipt

    def restore(self, receipt, label):
        self.clear()
        result=self.transactions.restore(receipt['path'], receipt['sha256'], expected_cursor=receipt['cursor'], label=label)
        self.active_arm=receipt['cursor']['arm']
        return result

    def setup(self):
        self.clear(); self.runtime.model.requires_grad_(False)
        self.runtime.model.train(); self.runtime.model.config.use_cache = False
        if hasattr(self.runtime.model, 'gradient_checkpointing_enable'):
            self.runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.runtime.model.enable_input_require_grads()
        self.binding = bind(self.runtime, self.gate, frozen_modules=self.frozen_modules)
        options = self.config['optimizer']
        self.optimizers = [self.optimizer_factory(list(group.values()), lr=options['lr'], momentum=options['momentum'],
            weight_decay=options['weight_decay']) for group in [*self.runtime.groups, {'mix_gate.weight':self.gate.weight}]]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda step:1.) for o in self.optimizers]
        self.guard, info = self.guard_factory(self.runtime, frozen_fingerprints=self.binding.frozen_fingerprints,
            source_files=self.config['guard_source_files'], arithmetic_modules=self.arithmetic_modules,
            expected_counts=None if self.config['fixture'] else dict(nf4_linears=376,nf4_expert_tensors=92,sparse_views=5888))
        atomic(self.root/'effective-state-capture.json', info)
        self.transactions = CollectiveTransactions(self.runtime, self.gate, self.binding, self.optimizers,
            self.schedulers, effective_snapshot=self.guard, identity=self.report['source_identity'],
            frozen_modules=self.frozen_modules, event=self.event)
        random.seed(self.config['seed']); torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.config['seed'])
        for arm in ARMS: self.common[arm] = self.save('initial_'+arm, self.cursor(arm, 0))
        left = torch.load(self.common[ARMS[0]]['path'], map_location='cpu', weights_only=True)
        right = torch.load(self.common[ARMS[1]]['path'], map_location='cpu', weights_only=True)
        require(all(_same(left[k],right[k]) for k in left if k!='cursor'), 'Paired initial state differs')
        require(all(not o.state for o in self.optimizers), 'Optimizer state must start fresh')
        self.event('identical_initial_states_verified')

    def update(self, arm, *, replay):
        require(self.active_arm==arm,'Attempted arm is not the fully restored active arm')
        self.current_arm = arm; self.clear(); self.runtime.model.train()
        self.report['arm_update_attempts'] += 1
        self.runtime.checkpoint_generation += 1
        self.event('arm_update_started', arm=arm, replay=replay)
        try:
            for row in self.rows:
                def forward(member):
                    self.report['forward_attempts'] += 1
                    if torch.is_grad_enabled():
                        self.report['backward_target_contributions_attempted'] += row['encoded']['supervised_tokens']
                    value = self.forward(member, row)
                    self.report['forward_completed'] += 1
                    return value
                self.event('record_backward_started', arm=arm, record_id=row['record_id'], replay=replay)
                recomputed_backward(forward, members=(0,1,2,3), mode=arm, gate=self.gate,
                    denominator=self.config['target_tokens_per_arm_update'])
            parameters = list(self.binding.parameters.values())
            require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in parameters), 'Invalid intended gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.config['optimizer']['grad_clip'], error_if_nonfinite=True))
            for index, optimizer in enumerate(self.optimizers):
                require(self.report['physical_optimizer_calls'] < self.config['physical_optimizer_call_cap'], 'Physical call cap exhausted')
                self.event('optimizer_call_started', arm=arm, optimizer=index, replay=replay,
                    next_physical_call=self.report['physical_optimizer_calls']+1)
                self.report['physical_optimizer_calls'] += 1
                self.report['per_arm'][arm]['physical_optimizer_calls'] += 1
                if replay:
                    self.report['replay_optimizer_calls'] += 1
                    self.report['per_arm'][arm]['replay_optimizer_calls'] += 1
                optimizer.step()
                self.event('optimizer_call_returned', arm=arm, optimizer=index, replay=replay)
            for scheduler in self.schedulers: scheduler.step()
            require(all(torch.isfinite(p).all() for p in parameters), 'Nonfinite updated parameter')
            self.report['per_arm'][arm]['successful_update_attempts'] += 1
            self.event('arm_update_returned', arm=arm, replay=replay, gradient_norm_before_clip=norm)
        finally:
            self.clear(); self.persist()

    def operation(self, arm, before, round_number, *, replay=False, expected=None):
        start = self.clock(); label = f'round{round_number}_{arm}'+('_replay' if replay else '')
        with self.phase(label, self.config['operation_seconds'], end=self.train_end):
            self.restore(before, 'before_'+label)
            self.update(arm, replay=replay)
            result = self.save(label, self.cursor(arm, round_number))
            if expected is not None:
                a = torch.load(expected['path'], map_location='cpu', weights_only=True)
                b = torch.load(result['path'], map_location='cpu', weights_only=True)
                require(_same(a,b), 'Full optimizer/scheduler/RNG/cursor replay differs')
                self.event('complete_replay_verified', arm=arm, round=round_number)
        self.durations.append(self.clock()-start)
        return result

    def evaluate(self):
        by_id = {p['record_id']:p for p in self.prompts}
        for arm in ARMS:
            with self.phase('evaluation_restore_'+arm, self.config['restore_seconds']):
                self.restore(self.common[arm], 'evaluation_'+arm)
            self.runtime.model.eval()
            for slot in (s for s in self.answer_slots if s['condition']==arm):
                if self.deadline-self.clock() < self.config['restore_seconds']+.001: return
                prompt = by_id[slot['record_id']]; slot['state']='started'; self.persist()
                start=self.clock(); self.memory()
                try:
                    answer=Population(self.runtime,self.gate,members=(0,1,2,3)).generate(prompt['input_ids'],
                        eos_ids=self.config['eos_token_ids'], max_new_tokens=self.config['max_new_tokens'],
                        deadline=min(self.clock()+self.config['response_seconds_cap'],self.deadline-self.config['restore_seconds']))
                    slot['answer']=answer
                    slot['state']='captured'
                except BaseException as error:
                    slot['answer']=getattr(error,'baseline_partial_answer',None)
                    slot.update(state='failed',error=dict(type=type(error).__name__,message=str(error)))
                finally:
                    answer=slot.get('answer')
                    if answer is not None:
                        try:
                            tokens=answer['tokens']; clean=tokens[:-1] if tokens and tokens[-1] in self.config['eos_token_ids'] else tokens
                            answer['raw_text']=self.tokenizer.decode(tokens,skip_special_tokens=False)
                            answer['text']=self.tokenizer.decode(clean,skip_special_tokens=False)
                        except BaseException as error:
                            slot.update(state='failed',decode_error=dict(type=type(error).__name__,message=str(error)))
                    slot['operation_elapsed_seconds']=self.clock()-start
                    self.memory(); self.persist()

    def run(self):
        self.report['status']='setup';self.persist()
        try:
            with self.phase('nonmutating_numerical_admission', self.config['budget']['numerical']):
                self.setup()
                self.numerical_check(self)
                self.restore(self.common[ARMS[0]], 'after_numerical_admission')
            self.report['status']='training';self.persist()
            self.train_end=min(self.clock()+self.config['budget']['training'],
                self.deadline-self.config['budget']['evaluation']-self.config['budget']['recovery'])
            for round_number in range(1,self.config['max_rounds']+1):
                required=(4 if round_number==1 else 2)*self.config['operation_seconds']
                if self.train_end-self.clock() < required:
                    self.report['training_stop']='reserve_complete_pair_and_endpoint';break
                if round_number>1 and max(self.durations)*1.25>self.config['operation_seconds']:
                    self.report['training_stop']='measured_complete_operation_cost';break
                self.ahead={}; before=dict(self.common)
                try:
                    for arm in ARMS:
                        self.current_arm=arm
                        result=self.operation(arm,before[arm],round_number)
                        self.ahead[arm]=result;self.persist()
                        if round_number==1: self.operation(arm,before[arm],round_number,replay=True,expected=result)
                    self.common=dict(self.ahead);self.ahead={}
                    self.report['completed_rounds']=round_number
                    for arm in ARMS:
                        self.report['per_arm'][arm].update(retained_rounds=round_number,
                            retained_target_tokens=round_number*self.config['target_tokens_per_arm_update'])
                    self.event('complete_common_round',round=round_number);self.persist()
                except BaseException as error:
                    self.report['training_failure']=dict(type=type(error).__name__,message=str(error),arm=self.current_arm,
                        active_state_arm=self.active_arm,round=round_number,owner_valid=self.transactions.valid)
                    if self.active_arm!=self.current_arm:
                        self.report['unfinished_note']='Attempted arm was never activated; previous arm remains represented by its ahead receipt.'
                    elif self.transactions.valid and all(torch.isfinite(p).all() for p in self.binding.parameters.values()):
                        try:
                            with self.phase('unfinished_capture',self.config['operation_seconds'],end=self.train_end):
                                self.report['unfinished'][self.current_arm]=self.save('unfinished_'+self.current_arm,
                                    {**self.cursor(self.current_arm,self.report['completed_rounds'],'incomplete_not_matched'),
                                     'attempted_round':round_number})
                        except BaseException as evidence_error:
                            self.report['unfinished_capture_error']=str(evidence_error)
                    self.persist();break
            require(self.transactions.valid,'Invalid transaction owner forbids further model work')
            self.report['status']='evaluating';self.persist();self.evaluate()
            self.report['status']='complete' if self.report['completed_rounds']==self.config['max_rounds'] and all(
                s['state']=='captured' for s in self.answer_slots) else 'partial'
        except BaseException as error:
            self.report.update(status='failed',error_type=type(error).__name__,error=str(error))
        finally:
            self.clear()
            self.report['final_recovery']='unavailable_before_initial_or_invalid_owner'
            if self.transactions is not None and self.transactions.valid and self.common:
                try:
                    with self.phase('final_recovery',self.config['restore_seconds']):
                        self.restore(next(iter(self.common.values())),'final_common_recovery')
                    self.report['final_recovery']='verified_last_common_restored'
                except BaseException as error:
                    self.report.update(status='failed',final_recovery='failed_process_invalid',recovery_error=str(error))
            self.memory();self.persist()
        return self.report
