"""One-GPU host queue for frozen PENDING_EXECUTION; never fills old quotas."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import time
from g1_reduced_budget import ROOT
from g1_reduced_compatibility import verify_proof
from g1_recovery_protocol import LAB, read_json, sha256, write_json, code_identity
from g1_robustness_protocol import MODEL_ORDER, load_prepared


def main():
    root=ROOT;dev=root/'development';status=dev/'execution_status.json'
    proof=verify_proof(root);pin=proof['execution_commit']
    state=dict(status='RUNNING_REDUCED_BUDGET',driver_pid=os.getpid(),execution_commit=pin,models=[],started_unix=time.time())
    save=lambda:write_json(status,state)
    queues=read_json(root/'pending_execution_queues.json');save()
    try:
        for model in MODEL_ORDER:
            if code_identity()['evaluation_code_commit']!=pin:raise ValueError('Execution commit changed')
            verify_proof(root)
            if not queues[model]:
                state['models'].append(dict(model=model,status='REUSED_ALL',new_trials=0));save();continue
            execution=root/'execution'/model;load_prepared(execution)
            reference=execution/'completed_models'/f'{model}.json'
            def sealed():
                if not reference.exists():return None
                ref=read_json(reference);run=execution/'runs'/ref['evaluation_id']
                if sha256(run/'completion.json')!=ref['completion_sha256'] or read_json(run/'completion.json')['status']!='COMPLETE':
                    raise ValueError('Completion seal mismatch')
                return run
            if sealed():
                state['models'].append(dict(model=model,status='COMPLETE',new_trials=len(queues[model]),evaluation_id=sealed().name));save();continue
            command=[sys.executable,'-u','-B',str(LAB/'tools/evaluation/g1_complete_robustness.py'),'run',
                     '--output_root',str(execution),'--model',model,'--device','cuda:0']
            with (dev/f'{model}.log').open('xb',buffering=0) as log:
                proc=subprocess.Popen(command,cwd=LAB,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item=dict(model=model,status='RUNNING',pid=proc.pid,new_trials=len(queues[model]),started_unix=time.time());state['models'].append(item);save();sealed_at=None
                while proc.poll() is None:
                    failures=list((execution/'runs').glob('*/failure.json'))
                    if failures:raise ValueError('Execution error preserved: '+str(failures))
                    if (root/'STOP_AFTER_BATCH').exists() and not (execution/'STOP_AFTER_BATCH').exists():
                        (execution/'STOP_AFTER_BATCH').write_text('User requested stop at next saved batch\n')
                    if sealed():
                        if sealed_at is None:sealed_at=time.time()
                        if time.time()-sealed_at>30:
                            proc.terminate()
                            try:proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:proc.kill();proc.wait()
                            break
                    time.sleep(5)
                run=sealed()
                if run is None:raise ValueError('Run paused or exited PARTIAL; source data preserved')
                item.update(status='COMPLETE',evaluation_id=run.name,finished_unix=time.time());save()
        state['status']='ALL_REDUCED_PHYSICS_COMPLETE_FINAL_AUDIT';save()
        from g1_reduced_sources import finalize_sources, full_primary_audit
        finalize_sources(root)
        full_primary_audit(root)
        state['status']='ALL_PRIMARY_AUDITS_PASSED_REPORT_PENDING';save()
        # Reporting stays separate from all physics and detector definitions.
        from g1_reduced_report import build_report
        build_report(root)
        state.update(status='REPORT_BUILT_PENDING_VISUAL_REVIEW',finished_unix=time.time());save()
    except BaseException as exc:
        state.update(status='STOPPED_ERROR',error=repr(exc),stopped_unix=time.time());save();raise


if __name__=='__main__':main()
