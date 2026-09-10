"""Add the fixed DWAQ V2 checkpoint without modifying archived paper runs."""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path
from g1_paper_recovery import ROOT as BASE
from g1_recovery_protocol import LAB, read_json, write_json, sha256, inspect_checkpoint, digest
from g1_paper_execution import current_runtime, verify_run
ROOT=LAB/'experiments/g1_paper_dwaq_v2_fixed_time_v1'
SUITES=['A','B1','B2','C_force','C_load']
SCRIPT='tools/evaluation/g1_paper_dwaq_v2.py'

def immutable(path, obj):
    if path.exists() and read_json(path)!=obj:raise ValueError('Frozen input changed: '+str(path))
    if not path.exists():write_json(path,obj)

def prepare():
    source=next(r for r in read_json(LAB/'experiments/g1_dwaq_reward_ablation_eval_v1/report/new_sources.json') if r['model']=='dwaq_v2' and r['suite']=='standard')['identity']
    identity=inspect_checkpoint('g1_dwaq_slope_nosys_d_matched_v2',LAB/source['checkpoint_path'],'dwaq_v2','final')
    for k in ['checkpoint_sha256','env_config_sha256','agent_config_sha256']:
        if identity[k]!=source[k]:raise ValueError('DWAQ V2 provenance mismatch: '+k)
    ROOT.mkdir(exist_ok=True,parents=True)
    for f in ['formal_frozen_manifest.json','protocol_frozen.json','paired_physics_64.json']:
        immutable(ROOT/f,read_json(BASE/f))
    immutable(ROOT/'checkpoint_identities.json',{'dwaq_v2':identity})
    runtime=current_runtime();baseline=read_json(BASE/'execution_continuation.json')
    reference=next(j for j in baseline['jobs'] if j['suite']=='B1' and j['model']=='dwaq')['runtime']
    for k in ['evaluation_runtime_sha256','paper_sources']:
        if runtime[k]!=reference[k]:raise ValueError('Shared simulation/judge code changed: '+k)
    plans=read_json(ROOT/'formal_frozen_manifest.json')
    jobs=[dict(suite=s,run=str(ROOT/'formal/dwaq_v2'/s/'attempt-001'),runtime=runtime,checkpoint_sha256=identity['checkpoint_sha256'],manifest_hash=digest([p for p in plans if p['suite']==s]),protocol_sha256=sha256(ROOT/'protocol_frozen.json'),num_envs=64,episodes=sum(p['suite']==s for p in plans)) for s in SUITES]
    frozen=dict(jobs=jobs,wrapper_sha256=sha256(LAB/SCRIPT),source_plan_sha256=sha256(BASE/'formal_frozen_manifest.json'),physics_sha256=sha256(BASE/'paired_physics_64.json'),note='Same simulator and judge source hashes as repaired original B/C; extension wrapper/commit recorded separately. Original A judge/reset equivalence checked in source continuation index.')
    immutable(ROOT/'execution.json',frozen)
    return frozen

def worker(suite):
    import g1_paper_sim as sim
    frozen=read_json(ROOT/'execution.json');job=next(j for j in frozen['jobs'] if j['suite']==suite)
    if sha256(LAB/SCRIPT)!=frozen['wrapper_sha256'] or current_runtime()!=job['runtime']:raise ValueError('Frozen extension runtime changed')
    if sha256(ROOT/'formal_frozen_manifest.json')!=frozen['source_plan_sha256']:raise ValueError('Paired plan changed')
    sim.ROOT=ROOT
    sim.execute(argparse.Namespace(model='dwaq_v2',stage='formal',suite=suite,offset=0,limit=None,attempt='attempt-001',num_envs=64,device='cuda:0'))

def report():
    import g1_paper_report as reporter
    data=reporter.load('formal')  # checks the original explicit three-model index
    rows=[]
    for job in read_json(ROOT/'execution.json')['jobs']:
        run=Path(job['run']);c=verify_run(run,job,True)
        rows.extend({**read_json(run/'records'/f),'run':str(run)} for f in c['records_sha256'])
    data['dwaq_v2']=rows
    reporter.ROOT=ROOT;reporter.load=lambda stage:data
    reporter.report('formal')

def run():
    frozen=prepare();start=time.monotonic();env=os.environ.copy();env['PYTHONPATH']=str(LAB/'rsl_rl')+':'+str(LAB)
    for index,job in enumerate(frozen['jobs']):
        out=Path(job['run']);suite=job['suite']
        if (out/'completion.json').exists():verify_run(out,job,True);continue
        log=ROOT/(suite+'.log')
        with log.open('xb',buffering=0) as f:
            child=subprocess.Popen([sys.executable,'-B',SCRIPT,'--worker',suite],cwd=LAB,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);sealed=None
            while child.poll() is None:
                failure=(out/'failure.json').exists();complete=(out/'completion.json').exists()
                if complete and sealed is None:sealed=time.monotonic()
                if failure or (complete and time.monotonic()-sealed>20):
                    if complete:verify_run(out,job,True)
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                write_json(ROOT/'status.json',dict(status='RUNNING',suite=suite,pid=child.pid,queue_index=index,progress=read_json(out/'progress.json') if (out/'progress.json').exists() else None))
                time.sleep(10)
            if not (out/'completion.json').exists() or (out/'failure.json').exists():
                write_json(ROOT/'status.json',dict(status='EVALUATION_ERROR_STOP',suite=suite,log=str(log)));raise RuntimeError('Evaluation error; stop and preserve attempt')
            verify_run(out,job,True)
    report();write_json(ROOT/'status.json',dict(status='COMPLETE',episodes=6420,elapsed_s=time.monotonic()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=SUITES);p.add_argument('--report',action='store_true');a=p.parse_args()
    worker(a.worker) if a.worker else report() if a.report else run()
