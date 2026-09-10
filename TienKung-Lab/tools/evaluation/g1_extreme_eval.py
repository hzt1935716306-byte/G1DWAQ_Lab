"""Four-model sequential adaptive exploration and independent formal evaluation."""
import argparse,os,signal,subprocess,sys,time
from pathlib import Path
from g1_recovery_protocol import LAB,read_json,write_json,sha256,inspect_checkpoint,digest
from g1_paper_execution import current_runtime,verify_run
from g1_paper_recovery import ROOT as OLD
from g1_extreme_plan import MODELS,SLOPES,GROUPS,START,EASY,conditions,key,scenario,curves,scan_level,formal_levels
ROOT=LAB/'experiments/g1_extreme_d50_v1'
SCRIPT='tools/evaluation/g1_extreme_eval.py'
SOURCES=[SCRIPT,'tools/evaluation/g1_extreme_plan.py','tools/evaluation/g1_extreme_d50.py','tools/evaluation/g1_extreme_report.py','tools/evaluation/g1_paper_execution.py','tools/evaluation/g1_paper_sim.py','tools/evaluation/g1_paper_recovery.py']

def immutable(p,obj):
    if p.exists() and read_json(p)!=obj:raise ValueError('Frozen object changed: '+str(p))
    if not p.exists():write_json(p,obj)

def prepare():
    ROOT.mkdir(exist_ok=True,parents=True)
    if (ROOT/'identity.json').exists():
        frozen=read_json(ROOT/'identity.json')
        if frozen['runtime']!=current_runtime() or any(sha256(LAB/p)!=h for p,h in frozen['sources'].items()):raise ValueError('Execution code changed')
        if sha256(ROOT/'protocol_frozen.json')!=frozen['protocol_sha256']:raise ValueError('Frozen protocol changed')
        if sha256(ROOT/'checkpoint_identities.json')!=frozen['checkpoint_identities_sha256']:raise ValueError('Frozen checkpoint identities changed')
        return
    identities=read_json(OLD/'checkpoint_identities.json');identities.update(read_json(LAB/'experiments/g1_paper_dwaq_v2_fixed_time_v1/checkpoint_identities.json'))
    for m,i in identities.items():
        fresh=inspect_checkpoint(i['task_name'],LAB/i['checkpoint_path'],m,'final',estimator=LAB/i['estimator_path'] if i.get('estimator_path') else None)
        for k in ['checkpoint_sha256','env_config_sha256','agent_config_sha256','native_configuration_sha256','resources']:
            if fresh[k]!=i[k]:raise ValueError('Checkpoint/native snapshot changed: '+m+' '+k)
    protocol=read_json(OLD/'protocol_frozen.json');protocol.update(slopes_deg=list(SLOPES),terrain_seed=26101000,role='extreme_d50',exploration_models=['ppo'],exploration_repeats=20,formal_repeats=20,scan_factor=1.25,stop_ppo_recovery_at_or_below=.3,max_upward_rounds=16,formal_nonmonotone_supplement_repeats=0,baseline_repeats=20,max_formal_levels=3,method_names=list(MODELS),stage_order=['original','extended'],seed_namespace='g1-extreme-v1',note='No teacher gate; fixed existing recovery judge. +/-20 outside configured training range, +/-15 boundary. Exploration and formal seeds disjoint. At scan budget exhaustion report censored, never extrapolate.')
    protocol.pop('formal_counts_per_model',None)
    protocol['formal_status']='reduced_sample_validation_not_final_paper_precision'
    protocol['automatic_supplement']=False
    immutable(ROOT/'checkpoint_identities.json',identities);immutable(ROOT/'protocol_frozen.json',protocol)
    immutable(ROOT/'identity.json',dict(runtime=current_runtime(),sources={p:sha256(LAB/p) for p in SOURCES},protocol_sha256=sha256(ROOT/'protocol_frozen.json'),checkpoint_identities_sha256=sha256(ROOT/'checkpoint_identities.json'),checkpoint_sha256={m:i['checkpoint_sha256'] for m,i in identities.items()}))

def wave_rows(phase,group,model,max_round=None,scan_only=False):
    rows=[]
    for p in sorted((ROOT/'waves').glob(f'{group}_{phase}_*/formal/{model}/all/attempt-001/completion.json')):
        suffix=p.relative_to(ROOT/'waves').parts[0].rsplit('_',1)[-1]
        if (scan_only or max_round is not None) and not suffix.isdigit():continue
        if max_round is not None and int(suffix)>=max_round:continue
        c=read_json(p);rows.extend(read_json(p.parent/'records'/f) for f in c['records_sha256'])
    return rows

def wave(name,rows,models=MODELS):
    if not rows:return
    if not models or any(m not in MODELS for m in models):raise ValueError('Invalid model selection')
    prepare();root=ROOT/'waves'/name;root.mkdir(exist_ok=True,parents=True)
    immutable(root/'wave_models.json',list(models))
    immutable(root/'formal_frozen_manifest.json',rows)
    for f in ['protocol_frozen.json','checkpoint_identities.json']:immutable(root/f,read_json(ROOT/f))
    runtime=read_json(ROOT/'identity.json')['runtime']
    for model in models:
        out=root/'formal'/model/'all/attempt-001';job=dict(runtime=runtime,checkpoint_sha256=read_json(ROOT/'identity.json')['checkpoint_sha256'][model],manifest_hash=digest(rows),protocol_sha256=sha256(root/'protocol_frozen.json'),num_envs=64,episodes=len(rows))
        if (out/'completion.json').exists():verify_run(out,job,True);continue
        log=root/(model+'.log')
        if log.exists():raise RuntimeError('Incomplete prior wave; preserve and inspect '+str(log))
        with log.open('xb',buffering=0) as f:
            env=os.environ.copy();env['PYTHONPATH']=str(LAB/'rsl_rl')+':'+str(LAB)
            child=subprocess.Popen([sys.executable,'-B',SCRIPT,'--worker',name,'--model',model],cwd=LAB,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);sealed=None
            while child.poll() is None:
                failure=(out/'failure.json').exists();complete=(out/'completion.json').exists()
                if complete and sealed is None:sealed=time.monotonic()
                if failure or (complete and time.monotonic()-sealed>20):
                    if complete:verify_run(out,job,True)
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                write_json(ROOT/'status.json',dict(status='RUNNING',wave=name,model=model,pid=child.pid,progress=read_json(out/'progress.json') if (out/'progress.json').exists() else None,log=str(log)))
                time.sleep(10)
            if not (out/'completion.json').exists() or (out/'failure.json').exists():raise RuntimeError('Evaluation contract error: '+str(log))
            verify_run(out,job,True)
    immutable(root/'WAVE_COMPLETE.json',dict(episodes_per_model=len(rows),models=list(models)))
    from g1_extreme_report import report
    report(ROOT)

def worker(name,model):
    prepare()
    if Path(name).name!=name:raise ValueError('Invalid wave name')
    import g1_paper_sim as sim
    sim.ROOT=ROOT/'waves'/name
    sim.execute(argparse.Namespace(model=model,stage='formal',suite=None,offset=0,limit=None,attempt='attempt-001',num_envs=64,device='cuda:0',snapshot_root=ROOT/'snapshots',contract_root=ROOT))

def run_group(group):
    cc=conditions(GROUPS[group])
    # Added slopes must have an independent unperturbed baseline before pushes.
    if group=='extended':
        a=[scenario((s,v,0,'A'),0.,rep,'baseline') for s in GROUPS[group] for v in (.5,1.) for rep in range(20)]
        wave(group+'_baseline_00',a)
    # Starting maximum plus a fixed previously used easy reference, no adaptation by method.
    wave(group+'_explore_00',[scenario(c,m,rep,'explore') for c in cc for m in [EASY[c[3]],START[c[3]]] for rep in range(20)],models=('ppo',))
    for n in range(1,16):
        pp=curves(wave_rows('explore',group,'ppo',max_round=n));active=[c for c in cc if pp[c][-1].pushed and pp[c][-1].probability>.3]
        if not active:break
        wave(group+f'_explore_{n:02d}',[scenario(c,scan_level(n,c[3]),rep,'explore') for c in active for rep in range(20)],models=('ppo',))
    pp=curves(wave_rows('explore',group,'ppo'));selection={key(c):formal_levels(pp[c],c[3]) for c in cc}
    frozen=ROOT/(group+'_formal_levels.json');immutable(frozen,selection)
    rows=[scenario(c,m,rep,'formal') for c in cc for m in selection[key(c)] for rep in range(20)]
    wave(group+'_formal_00',rows)
    write_json(ROOT/(group+'_COMPLETE.json'),dict(status='COMPLETE',formal_levels=selection,note='Reduced 20-repeat comparison. No automatic expansion; ambiguous D50 remains unresolved.'))

def main():
    prepare()
    for group in GROUPS:run_group(group)
    from g1_extreme_report import report
    report(ROOT);write_json(ROOT/'status.json',dict(status='COMPLETE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--model',choices=MODELS);a=p.parse_args()
    try:worker(a.worker,a.model) if a.worker else main()
    except BaseException as e:
        if not a.worker:write_json(ROOT/'status.json',dict(status='EVALUATION_ERROR_STOP',error=str(e)))
        raise
