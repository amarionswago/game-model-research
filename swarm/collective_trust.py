"""Versioned post-update trust guard; accepted training/recovery code is inherited."""
from contextlib import contextmanager
import copy
from pathlib import Path
import random
from types import MappingProxyType

import torch

from formation_contract import digest
from swarm.collective_contract import atomic, require, sha
from swarm.collective_worker import Collective
from swarm.population_objective import selected, mixture
from swarm.population_runtime import combine, synchronize

MEMBERS = (0, 1, 2, 3)
DISTRIBUTIONS = ('member0', 'member1', 'member2', 'member3', 'mixture')
LIMITS = MappingProxyType(dict(max_target_logp_change=1., mean_target_logp_change=.25,
              max_policy_KL=.01, max_policy_probability_change=.05))
SCHEMA = 'hive-collective-update-trust-v1'


def layout(owner):
    """Bind every selected causal position and both sample entries per record."""
    require(all(type(member) is int for member in owner.gate.members)
            and tuple(owner.gate.members) == MEMBERS and len(owner.runtime.groups) == 4,
            'Trust gate/member alignment differs')
    rows = []
    for row in owner.rows:
        e = row['encoded']; ids, labels = e['input_ids'], e['labels']
        require(len(ids) == len(labels) and 1 < len(ids) <= owner.config['max_sequence'], 'Invalid native length')
        require(all(type(x) is int and x >= 0 for x in ids) and labels[0] == -100, 'Invalid native token identity')
        positions = [i-1 for i in range(1,len(labels)) if labels[i] != -100]
        require(positions and len(positions) == e['supervised_tokens']
                and positions == list(range(e['prompt_tokens']-1,len(ids)-1))
                and all(labels[i+1] == ids[i+1] for i in positions), 'Missing/reordered complete causal targets')
        require(all(x == -100 for x in labels[:e['prompt_tokens']]), 'Prompt labels must be masked')
        rows.append(dict(record_id=row['record_id'], native_sha256=digest(e),
            positions=positions, target_ids=[labels[i+1] for i in positions],
            samples=[dict(target_index=j, causal_position=positions[j], target_id=labels[positions[j]+1])
                     for j in (0,len(positions)-1)]))
    ids = [r['record_id'] for r in rows]
    require(ids == owner.config['record_order'] and len(set(ids)) == len(ids) and rows, 'Frozen row order differs')
    total = sum(len(r['positions']) for r in rows)
    require(total == owner.config['target_tokens_per_arm_update'] and total > 0, 'Complete target denominator differs')
    return dict(schema=SCHEMA, members=list(MEMBERS), distributions=list(DISTRIBUTIONS), rows=rows,
        target_tokens=total, sample_positions=2*len(rows), vocabulary=int(owner.runtime.head.weight.shape[0]))


@contextmanager
def evaluation_state(runtime):
    """Both snapshots use eval semantics without changing RNG, flags or gradients."""
    modes = [(m,m.training) for m in runtime.model.modules()]
    flags = [(p,p.requires_grad) for p in runtime.model.parameters()]
    adapter = copy.deepcopy(runtime.model.active_adapter)
    python_rng, cpu_rng = random.getstate(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        runtime.model.eval()
        with torch.no_grad(): yield
    finally:
        runtime.model.set_adapter(adapter)
        for parameter, flag in flags: parameter.requires_grad_(flag)
        for module, mode in modes: module.training = mode
        random.setstate(python_rng); torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)


def snapshot(owner, identity, stage, journal):
    """Only one record's hidden states/policies reside on device at a time.

    Retained CPU tensors: [5,all_targets] and [5,2*rows,vocabulary]. Full
    vocabulary samples are never averaged across records or members.
    """
    started = owner.clock()
    counts = journal.setdefault('snapshots', {}).setdefault(stage, dict(status='started', counts={}))
    counters = counts['counts']; current = {}; hooks = []; logs_all, policies_all = [], []
    def count(name, state, amount=1):
        key = name+'_'+state; counters[key] = counters.get(key,0)+amount
    def head_started(module, args):
        n = int(args[0].shape[0]); current['head_positions'] = n
        count(current['kind']+'_calls','attempted')
        count(current['kind']+'_positions','attempted',n)
        count(current['kind']+'_vocabulary_values','attempted',n*identity['vocabulary'])
    def head_returned(module, args, result):
        n = current['head_positions']; count(current['kind']+'_calls','completed')
        count(current['kind']+'_positions','completed',n)
        count(current['kind']+'_vocabulary_values','completed',n*identity['vocabulary'])
    try:
        synchronize()
        require(layout(owner) == identity, 'Trust input identity changed')
        hooks = [owner.runtime.head.register_forward_pre_hook(head_started),
                 owner.runtime.head.register_forward_hook(head_returned)]
        with evaluation_state(owner.runtime):
            for row, expected in zip(owner.rows, identity['rows']):
                target_count = len(expected['positions']); logs, hidden, policies = [], [], []
                for member in MEMBERS:
                    counts['active_call'] = dict(record_id=row['record_id'], member=member, kind='selected')
                    count('forwards','attempted'); count('target_presentations','attempted',target_count)
                    current['kind'] = 'target_head'
                    lp, hs = selected(owner.runtime, member, row)
                    count('forwards','completed'); count('target_presentations','completed',target_count)
                    require(lp.shape == (target_count,) and hs.shape == (target_count,owner.gate.hidden_size),
                            'Selected causal logp/hidden shape differs')
                    require(torch.isfinite(lp).all() and torch.isfinite(hs).all(), 'Nonfinite selected trust input')
                    logs.append(lp.detach()); hidden.append(hs.detach())
                    counts['active_call']['kind'] = 'sample_head'; current['kind'] = 'sample_head'
                    policy = owner.runtime.head(hs[[0,-1]]).float().log_softmax(-1)
                    require(policy.shape == (2,identity['vocabulary']) and torch.isfinite(policy).all(),
                            'Nonfinite/misaligned full vocabulary sample')
                    policies.append(policy.detach())
                member_logs, member_hidden = torch.stack(logs), torch.stack(hidden)
                counts['active_call'] = dict(record_id=row['record_id'],kind='mixture_target')
                count('mixture_target_calls','attempted')
                count('mixture_target_positions','attempted',target_count)
                deployed = mixture(member_logs,member_hidden,owner.gate)[0]
                count('mixture_target_calls','completed')
                count('mixture_target_positions','completed',target_count)
                member_policies = torch.stack(policies)
                counts['active_call']['kind'] = 'mixture_policy'; count('mixture_policy_calls','attempted')
                count('mixture_sample_positions','attempted',2)
                count('mixture_sample_vocabulary_values','attempted',2*identity['vocabulary'])
                deployed_policy, weights = combine(member_policies,member_hidden[:,[0,-1]],owner.gate)
                count('mixture_policy_calls','completed')
                count('mixture_sample_positions','completed',2)
                count('mixture_sample_vocabulary_values','completed',2*identity['vocabulary'])
                require(weights.shape == (2,4) and torch.isfinite(weights).all(), 'Invalid deployed gate weights')
                logs_all.append(torch.cat([member_logs,deployed[None]]).detach().cpu().clone())
                policies_all.append(torch.cat([member_policies,deployed_policy[None]]).detach().cpu().clone())
                del lp,hs,policy,logs,hidden,policies,member_logs,member_hidden,member_policies,deployed,deployed_policy,weights
        require(layout(owner) == identity, 'Trust input mutated during snapshot')
        result = dict(identity=copy.deepcopy(identity), targets=torch.cat(logs_all,dim=1),
                      policies=torch.cat(policies_all,dim=1))
        validate_snapshot(result,identity)
        counts['status'] = 'returned'
        return result
    except BaseException as error:
        counts.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        for hook in hooks: hook.remove()
        try: synchronize()
        except BaseException as error:
            counts.update(status='failed',synchronization_error=dict(type=type(error).__name__,message=str(error)))
            raise
        finally: counts['seconds'] = owner.clock()-started


def validate_snapshot(value, identity):
    require(all(type(member) is int for member in identity['members'])
            and all(type(member) is int for member in value['identity']['members'])
            and value['identity'] == identity and identity['members'] == list(MEMBERS)
            and identity['distributions'] == list(DISTRIBUTIONS), 'Trust distribution/record/sample identities differ')
    require(identity['target_tokens'] == sum(len(r['positions']) for r in identity['rows'])
            and identity['sample_positions'] == 2*len(identity['rows']), 'Trust coverage differs')
    for key, shape in [('targets',(5,identity['target_tokens'])),
                       ('policies',(5,identity['sample_positions'],identity['vocabulary']))]:
        x = value[key]
        require(isinstance(x,torch.Tensor) and tuple(x.shape) == shape and x.device.type == 'cpu'
                and x.is_floating_point() and not x.requires_grad and x.grad_fn is None
                and torch.isfinite(x).all(), 'Nonfinite/misaligned/differentiable trust '+key)
    require(float(value['policies'].double().logsumexp(-1).abs().max()) <= 1e-4,
            'Invalid normalized full-vocabulary policy')


def passes_limits(metrics):
    return all(metrics[k] <= v for k,v in LIMITS.items())


def compare_snapshots(before, after, identity):
    validate_snapshot(before,identity); validate_snapshot(after,identity)
    results = {}
    for i, name in enumerate(DISTRIBUTIONS):
        delta = (after['targets'][i].double()-before['targets'][i].double()).abs()
        p, q = before['policies'][i].double(), after['policies'][i].double()
        kl = (p.exp()*(p-q)).sum(-1)
        probability = (p.exp()-q.exp()).abs()
        require(torch.isfinite(kl).all() and torch.isfinite(probability).all(), 'Nonfinite policy trust metric')
        metrics = dict(max_target_logp_change=float(delta.max()),mean_target_logp_change=float(delta.mean()),
                       max_policy_KL=float(kl.max()),max_policy_probability_change=float(probability.max()))
        metrics['passed'] = passes_limits(metrics)
        results[name] = metrics
    return dict(passed=all(r['passed'] for r in results.values()),limits=dict(LIMITS),
        distributions=results,identity=identity,
        scope='complete target-token-weighted change; first/last sampled policies, not global distribution stability')


class TrustedCollective(Collective):
    """The inherited bounded operation saves only after this guard returns."""
    def update(self, arm, *, replay):
        require(self.active_arm == arm, 'Trust requires the activated pre-update arm')
        require(self.config['guard_source_files'].get(str(Path(__file__).resolve())) == sha(__file__),
                'Trust helper must be bound into effective source manifest')
        trust = self.report.setdefault('trust',dict(schema=SCHEMA,limits=dict(LIMITS),operations=[],accepted_updates=0,
            successful_update_attempts_meaning='Inherited counter is optimizer/update return; trust acceptance is separate'))
        ordinal = len(trust['operations'])+1
        journal = dict(ordinal=ordinal,arm=arm,replay=replay,status='started',accepted=False,
                       physical_calls_before=self.report['physical_optimizer_calls'])
        trust['operations'].append(journal)
        started = self.clock()
        try:
            identity = layout(self); journal['identity'] = identity
            before = snapshot(self,identity,'before',journal)
            update_started = self.clock()
            try: super().update(arm,replay=replay)
            finally: journal['inherited_update_seconds'] = self.clock()-update_started
            after = snapshot(self,identity,'after',journal)
            comparison_started = self.clock()
            try: journal['metrics'] = compare_snapshots(before,after,identity)
            finally: journal['comparison_seconds'] = self.clock()-comparison_started
            require(journal['metrics']['passed'], 'Member or deployed-mixture trust boundary exceeded')
            journal.update(status='accepted',accepted=True); trust['accepted_updates'] += 1
        except BaseException as error:
            journal.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            journal.update(guarded_update_seconds=self.clock()-started,physical_calls_after=self.report['physical_optimizer_calls'])
            journal['trust_seconds_excluding_diagnostic_write'] = journal['guarded_update_seconds']-journal.get('inherited_update_seconds',0.)
            atomic(self.root/f'trust-{ordinal:04d}-{arm}.json',journal)
            self.persist()
