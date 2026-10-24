"""Inference-only GPU recovery; corrected reference provenance; bounded joint-model sample."""
from pathlib import Path
import gc, hashlib, importlib.metadata, json, math, os, resource, signal, time, traceback
import numpy as np
import torch
from tokenizers import Tokenizer
from gpu_loader import load_base, wrap_population, restore, file_sha
from gpu_policy import Population
from four_conditions import CheckpointBank
from rl_numerics import policy_numerics
from sparse_experts import install_sparse_experts
from nf4_linear import verify_nf4_linears

def locate(suffix):
    matches=[p for p in Path('/kaggle/input').rglob(Path(suffix).name) if str(p).endswith(suffix)]
    assert len(matches)==1, ('Missing/ambiguous input',suffix)
    return matches[0]

def placement(name):
    if name.startswith('model.layers.'):
        return 'cuda:0' if int(name.split('.')[2])<23 else 'cuda:1'
    return 'cuda:0' if name.startswith(('model.embed_tokens','model.rotary_emb')) else 'cuda:1'

def admit(config, admission_deadline, root):
    started=time.monotonic();fixtures=json.loads((root/'fixtures.json').read_text())
    probe={'status':'running','stage':'devices','optimizer_updates':0,'checks':[],
           'admitted_to_benchmark':False,'limits':fixtures['limits'],'max_sequence':config['max_sequence']}
    def save(stage=None,**fields):
        if stage:probe['stage']=stage
        probe.update(fields,elapsed_seconds=time.monotonic()-started,
                     host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        from baseline_controller import atomic
        atomic(root/'probe-report.json',probe)
    def alarm(*args):raise TimeoutError('Registered numerical admission deadline')
    previous=signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,max(.001,admission_deadline-time.monotonic()))
    try:
        assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in (0,1))
        versions={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','accelerate','bitsandbytes')}
        assert versions=={'torch':'2.11.0+cu128','transformers':'5.18.0','peft':'0.21.2','accelerate':'1.15.0','bitsandbytes':'0.50.2'},versions
        torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        save('load_exact_nf4',devices=[torch.cuda.get_device_name(i) for i in (0,1)],versions=versions)
        folder=locate('nf4-base/manifest.json').parent
        assert file_sha(folder/'manifest.json')==config['export_manifest_sha256']
        base,manifest=load_base(folder,placement=placement,progress=lambda **kw:save(**kw))
        assert manifest['model_revision']==fixtures['reference_model_revision']
        runtime=wrap_population(base,placement=placement)
        assert [sum(p.numel() for p in g.values()) for g in runtime.groups]==[21031936,21031936]
        save('sparse_expert_checks',all_effective_base_tensors_match_exported_GPU_checksums=True)
        probe['sparse_experts']=install_sparse_experts(runtime)
        with policy_numerics(runtime):
            save('nf4_linear_checks')
            probe['nf4_linear_verification']=verify_nf4_linears(runtime.model)
            bank=CheckpointBank(runtime)
            for mode in ('baseline','joint_gate','independent_gate'):
                save('load_adapters',condition=mode)
                gate=restore(runtime,locate('gated-cooperation/'+mode+'.pt'),fixtures['checkpoints'][mode]['sha256'])
                population=Population(runtime,gate)
                probe.setdefault('restored_checkpoints',{})[mode]={'sha256':fixtures['checkpoints'][mode]['sha256'],
                    'cursor':runtime.restored_cursor,'identity':runtime.restored_identity}
                if mode=='joint_gate':
                    assert runtime.restored_cursor['completed_updates']==7
                    assert runtime.restored_identity['stage']=='joint_gate' and gate is not None
                    assert torch.count_nonzero(gate.weight)>0
                if mode=='baseline':
                    save('corrected_independent_full_vocabulary_reference')
                    binding=json.loads(Path('reference-fixture.json').read_text())
                    refpath=locate('repaired-baseline-code.npz')
                    assert file_sha(refpath)==binding['npz_sha256']
                    fixture=next(x for x in fixtures['fixtures'] if x['id']=='baseline-code')
                    sequence=fixture['prompt']+fixture['tokens'][:-1]
                    positions=binding['positions']
                    logs,hidden,policy,alpha=population.forced(sequence,positions,deadline=admission_deadline)
                    with np.load(refpath,allow_pickle=False) as package:full=package['reference_logps']
                    assert logs.shape==full.shape and np.isfinite(logs).all()
                    probability=float(np.max(np.abs(np.exp(logs)-np.exp(full))))
                    kl=float(np.max(np.sum(np.exp(full)*(full-logs),axis=-1)))
                    top1=float(np.mean(logs.argmax(-1)==full.argmax(-1)))
                    check={'id':'saved_independent_repaired_baseline_code','source':binding['source'],
                        'positions':positions,'max_probability_error':probability,'max_KL':kl,'top1_agreement':top1,
                        'passed':probability<=fixtures['limits']['max_probability'] and kl<=fixtures['limits']['max_KL'] and top1==1.}
                    probe['checks'].append(check);save('independent_reference_result',latest_check=check)
                    assert check['passed'],'Correctly bound independent full-vocabulary reference failed'
                    del logs,hidden,policy,alpha,full;gc.collect()
                for f in (f for f in fixtures['fixtures'] if f['condition']==mode):
                    save('numerical_reference',fixture=f['id'])
                    sequence=f['prompt']+f['tokens'][:-1];positions=list(range(len(f['prompt'])-1,len(sequence)))
                    logs,hidden,policy,alpha=population.forced(sequence,positions,deadline=admission_deadline)
                    targets=f['tokens'];selected=np.stack([logs[m,np.arange(len(targets)),targets] for m in (0,1)],axis=-1)
                    deltas=np.max(np.abs(selected-np.asarray(f['reference_logps'])),axis=0)
                    gate_delta=float(np.max(np.abs(alpha-np.asarray(f['reference_gate']))))
                    check={'id':f['id'],'condition':mode,'member_selected_logp_errors':deltas.tolist(),'gate_weight_error':gate_delta}
                    check['passed']=bool(max(deltas)<=fixtures['limits']['selected_logp'] and gate_delta<=fixtures['limits']['gate_weight'])
                    probe['checks'].append(check);save('numerical_reference_result',latest_check=check)
                    assert check['passed'],'Exact pretrained GPU references failed: '+f['id']
                    if mode=='joint_gate':
                        for member in (0,1):
                            solo=Population(runtime,None,members=(member,))
                            solo_values=solo.forced(sequence,positions,deadline=admission_deadline)
                            error=float(np.max(np.abs(solo_values[0][0]-logs[member])))
                            check={'id':f['id']+'_solo_'+str(member),'max_all_vocabulary_logp_error':error,
                                   'passed':error<=fixtures['limits']['selected_logp']}
                            probe['checks'].append(check);save('solo_saved_fixture_result',latest_check=check)
                            assert check['passed'],'Solo member differs from paired reference on math/code fixture'
                            del solo_values
                    del logs,hidden,policy,alpha;gc.collect()
                if mode != 'baseline':
                    bank.capture(mode,gate,sha256=fixtures['checkpoints'][mode]['sha256'])
            population,activation=bank.activate('joint_gate')
            save('long_cache_partition_check')
            seed=fixtures['fixtures'][0]['prompt'];span=config['max_sequence'];tokens=(seed*((span+len(seed)-1)//len(seed)))[:span]
            positions=[63,64,767,768,1534,1535]
            a=population.forced(tokens,positions,chunk_size=64,deadline=admission_deadline)
            b=population.forced(tokens,positions,chunk_size=63,deadline=admission_deadline)
            cache_error=float(np.max(np.abs(a[0]-b[0])));gate_error=float(np.max(np.abs(a[3]-b[3])))
            check={'id':'long_cache_partitions_64_vs_63','prefix_tokens':span,'condition':'joint_gate','cache_storage':'exact FP32 on GPU',
                   'max_all_vocabulary_logp_error':cache_error,'gate_weight_error':gate_error,
                   'passed':cache_error<=fixtures['limits']['selected_logp'] and gate_error<=fixtures['limits']['gate_weight']}
            probe['checks'].append(check);save('long_cache_result',latest_check=check)
            assert check['passed'],'Long cache partition check failed'
            del a,b;gc.collect()
            save('throughput_probe')
            sample=population.generate(seed,eos_ids=[],max_new_tokens=32,deadline=min(admission_deadline,time.monotonic()+600))
            assert len(sample['tokens'])==32,'Incomplete fixed throughput probe'
            rate=32/sample['seconds'];assert math.isfinite(rate) and rate>0
            probe['throughput']={'tokens':32,'seconds':sample['seconds'],'collective_tokens_per_second':rate,'includes_prefill':True}
            save('actual_joint_generation_replay')
            sequence=seed+sample['tokens'][:-1];positions=list(range(len(seed)-1,len(sequence)))
            logs,hidden,policy,alpha=population.forced(sequence,positions,deadline=admission_deadline)
            selected=np.stack([logs[m,np.arange(32),sample['tokens']] for m in (0,1)],axis=-1)
            error=float(np.max(np.abs(selected-np.asarray(sample['member_logps']))))
            gate_error=float(np.max(np.abs(alpha-np.asarray(sample['member0_weights']))))
            check={'id':'joint_generated_tokens_vs_chunked_replay','tokens':32,'selected_logp_error':error,
                   'gate_weight_error':gate_error,'passed':error<=fixtures['limits']['selected_logp'] and gate_error<=fixtures['limits']['gate_weight']}
            probe['checks'].append(check);save('actual_joint_generation_replay_result',latest_check=check)
            assert check['passed'],'Actual joint generation/replay check failed'
            assert np.array_equal(policy.argmax(-1),sample['tokens']),'Joint greedy tokens changed on replay'
            del logs,hidden,policy,alpha;gc.collect()
            for member in (0,1):
                save('solo_member_admission',member=member)
                solo,activation=bank.activate('member'+str(member))
                generated=solo.generate(seed,eos_ids=[],max_new_tokens=8,deadline=admission_deadline)
                assert len(generated['tokens'])==8 and generated['member_ids']==[member]
                assert set(generated['forward_calls'])=={str(member)} and set(generated['cache'])=={str(member)}
                assert generated['member0_weights'] is None
                sequence=seed+generated['tokens'][:-1]
                positions=list(range(len(seed)-1,len(sequence)))
                own=solo.forced(sequence,positions,deadline=admission_deadline)
                paired,_=bank.activate('joint_gate')
                paired_values=paired.forced(sequence,positions,deadline=admission_deadline)
                all_logp_error=float(np.max(np.abs(own[0][0]-paired_values[0][member])))
                selected=own[0][0,np.arange(8),generated['tokens']]
                replay_error=float(np.max(np.abs(selected-np.asarray(generated['member_logps'])[:,0])))
                assert np.array_equal(own[2].argmax(-1),generated['tokens']),'Solo greedy tokens changed on replay'
                check={'id':'solo_member_'+str(member)+'_paired_and_generation_replay',
                       'member':member,'tokens':8,'max_all_vocabulary_logp_error':all_logp_error,
                       'selected_logp_replay_error':replay_error,'forward_calls':generated['forward_calls'],
                       'cache':generated['cache'],'passed':all_logp_error<=fixtures['limits']['selected_logp'] and replay_error<=fixtures['limits']['selected_logp']}
                probe['checks'].append(check);save('solo_member_admission_result',latest_check=check)
                assert check['passed'],'Solo member numerical/call-count admission failed'
                del own,paired_values,generated;gc.collect()
            for mode in ('independent_gate','joint_gate'):
                population,activation=bank.activate(mode)
                fixture=next(f for f in fixtures['fixtures'] if f['condition']==mode)
                sequence=fixture['prompt']+fixture['tokens'][:-1]
                positions=list(range(len(fixture['prompt'])-1,len(sequence)))
                logs,hidden,policy,alpha=population.forced(sequence,positions,deadline=admission_deadline)
                selected=np.stack([logs[m,np.arange(len(fixture['tokens'])),fixture['tokens']] for m in (0,1)],axis=-1)
                error=float(np.max(np.abs(selected-np.asarray(fixture['reference_logps']))))
                gate_error=float(np.max(np.abs(alpha-np.asarray(fixture['reference_gate']))))
                check={'id':'checkpoint_bank_'+mode+'_reference','selected_logp_error':error,
                       'gate_weight_error':gate_error,'switch_seconds':activation['checkpoint_switch_seconds'],
                       'passed':error<=fixtures['limits']['selected_logp'] and gate_error<=fixtures['limits']['gate_weight']}
                probe['checks'].append(check);save('checkpoint_bank_result',latest_check=check)
                assert check['passed'],'Checkpoint bank switch changed admitted policy'
                del logs,hidden,policy,alpha;gc.collect()
                if mode=='independent_gate':
                    generated=population.generate(seed,eos_ids=[],max_new_tokens=8,deadline=admission_deadline)
                    assert len(generated['tokens'])==8
                    sequence=seed+generated['tokens'][:-1]
                    positions=list(range(len(seed)-1,len(sequence)))
                    logs,hidden,policy,alpha=population.forced(sequence,positions,deadline=admission_deadline)
                    selected=np.stack([logs[m,np.arange(8),generated['tokens']] for m in (0,1)],axis=-1)
                    error=float(np.max(np.abs(selected-np.asarray(generated['member_logps']))))
                    gate_error=float(np.max(np.abs(alpha-np.asarray(generated['member0_weights']))))
                    greedy_equal=bool(np.array_equal(policy.argmax(-1),generated['tokens']))
                    check={'id':'independent_generated_tokens_vs_chunked_replay','tokens':8,
                           'selected_logp_error':error,'gate_weight_error':gate_error,'greedy_tokens_equal':greedy_equal,
                           'passed':error<=fixtures['limits']['selected_logp'] and gate_error<=fixtures['limits']['gate_weight'] and greedy_equal}
                    probe['checks'].append(check);save('independent_generation_replay_result',latest_check=check)
                    assert check['passed'],'Independent generation/replay admission failed'
                    del logs,hidden,policy,alpha,generated;gc.collect()
            save('admitted',status='passed',admitted_to_benchmark=True, gpu_peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in (0,1)])
        population,activation=bank.activate('member0')
        assert activation['checkpoint_sha256']==config['checkpoint_sha256']
        assert runtime.restored_cursor['completed_updates']==7 and runtime.restored_identity['stage']=='joint_gate'
        tokenizer_path=locate('model-config/tokenizer.json')
        assert file_sha(tokenizer_path)==config['tokenizer_sha256']
        assert importlib.metadata.version('tokenizers')==config['decoding']['tokenizers_version']
        return ParentRuntime(population,Tokenizer.from_file(str(tokenizer_path)),activation)
    except BaseException as error:
        save('failed',status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


class ParentRuntime:
    def __init__(self,population,tokenizer,activation):
        self.population,self.tokenizer=population,tokenizer
        self.identity={'checkpoint_sha256':activation['checkpoint_sha256'],'member_ids':[0]}
    def generate(self,prompt,**kwargs):
        with policy_numerics(self.population.runtime):
            return self.population.generate(prompt,**kwargs)
    def decode(self,tokens,*,skip_special_tokens):
        return self.tokenizer.decode(tokens,skip_special_tokens=skip_special_tokens)
