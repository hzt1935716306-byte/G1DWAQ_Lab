"""Sequential formal queue; explicit completion seals, periodic progress, stop on software errors."""
import os,signal,subprocess,sys,time
from pathlib import Path
from g1_paper_recovery import ROOT
from g1_paper_execution import prepare,current_runtime,verify_run
from g1_recovery_protocol import LAB,read_json,write_json,sha256

def main():
    # Only explicit cross-model pilot admission unlocks formal execution.
    gate=read_json(ROOT/'pilot_admission.json')
    if not gate['physical_contract_passed'] or not gate['formal_matrix_frozen']:raise ValueError('Pilot not admitted')
    if sha256(ROOT/'protocol_frozen.json') != gate['protocol_sha256'] or sha256(ROOT/'formal_frozen_manifest.json') != gate['formal_manifest_sha256']:
        raise ValueError('Admitted protocol or paired plan changed')
    plan=prepare();queue=plan['jobs']
    if sha256(ROOT/'paired_physics_64.json')!=plan['physical_contract_sha256']:raise ValueError('Physical contract changed')
    env=os.environ.copy();env['PYTHONPATH']=str(LAB/'rsl_rl')+':'+str(LAB)+':'+env.get('PYTHONPATH','')
    start=time.monotonic()
    for index,job in enumerate(queue):
        suite,model=job['suite'],job['model'];out=Path(job['run'])
        if (out/'completion.json').exists():
            verify_run(out,job,True);continue
        if current_runtime()!=job['runtime']:raise ValueError('Frozen runtime changed before launch')
        command=[sys.executable,'-B','tools/evaluation/g1_paper_sim.py','--model',model,'--stage','formal','--suite',suite,'--attempt',job['attempt']]
        log=ROOT/f"formal_{suite}_{model}_{job['attempt']}.log"
        if log.exists():raise ValueError('Unsealed previous attempt must be reviewed: '+str(log))
        with log.open('xb',buffering=0) as f:
            child=subprocess.Popen(command,cwd=LAB,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            sealed_at=None
            while child.poll() is None:
                if (out/'failure.json').exists():
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                seal=out/'completion.json'
                if seal.exists():
                    if sealed_at is None:sealed_at=time.monotonic()
                    if time.monotonic()-sealed_at>20:
                        c=read_json(seal)
                        if not all(sha256(out/'records'/f)==h for f,h in c['records_sha256'].items()):raise ValueError('Completion seal mismatch')
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                        write_json(out/'shutdown_cleanup.json',{'reason':'records sealed; app.close did not exit within20s','owned_process_group':child.pid})
                        break
                progress=read_json(out/'progress.json') if (out/'progress.json').exists() else None
                write_json(ROOT/'status.json',dict(status='RUNNING',suite=suite,model=model,pid=child.pid,queue_index=index,queue_total=len(queue),progress=progress,elapsed_s=time.monotonic()-start,log=str(log)))
                time.sleep(10)
            if not (out/'completion.json').exists() or (out/'failure.json').exists():
                write_json(ROOT/'status.json',dict(status='EVALUATION_ERROR_STOP',suite=suite,model=model,returncode=child.returncode,log=str(log)))
                raise RuntimeError('Simulation contract error; queue stopped')
        verify_run(out,job,True)
    subprocess.run([sys.executable,'-B','tools/evaluation/g1_paper_report.py','--stage','formal'],cwd=LAB,env=env,check=True)
    write_json(ROOT/'status.json',dict(status='COMPLETE',episodes=19260,elapsed_s=time.monotonic()-start))
if __name__=='__main__':main()
