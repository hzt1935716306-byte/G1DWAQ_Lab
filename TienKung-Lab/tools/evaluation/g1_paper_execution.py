"""Explicit runtime cohorts for retained A and repaired disturbance experiments."""
import ast, subprocess
from g1_paper_recovery import ROOT
from g1_recovery_protocol import LAB, read_json, write_json, sha256, digest, code_identity
PLAN=ROOT/'execution_continuation.json'
SOURCES=['tools/evaluation/g1_paper_recovery.py','tools/evaluation/g1_paper_sim.py']

def runtime_key(binding):
    r=binding['runtime']
    return {k:r[k] for k in ['evaluation_code_commit','evaluation_runtime_sha256','paper_sources']}

def current_runtime():
    return runtime_key({'runtime':{**code_identity(),'paper_sources':{p:sha256(LAB/p) for p in SOURCES}}})

def verify_run(run,expected,complete=False):
    b=read_json(run/'binding.json')
    if runtime_key(b)!=expected['runtime']:raise ValueError('Unexpected execution runtime: '+str(run))
    for k in ['checkpoint_sha256','manifest_hash','protocol_sha256','num_envs']:
        if b[k]!=expected[k]:raise ValueError('Execution input differs: '+k)
    if complete:
        if (run/'failure.json').exists():raise ValueError('Failed run cannot be admitted')
        c=read_json(run/'completion.json')
        if c['status']!='COMPLETE' or c['episodes']!=expected['episodes'] or c['binding_sha256']!=sha256(run/'binding.json'):raise ValueError('Incomplete seal')
        if len(c['records_sha256'])!=expected['episodes']:raise ValueError('Wrong sealed record count')
        for f,h in c['records_sha256'].items():
            if sha256(run/'records'/f)!=h:raise ValueError('Sealed record changed')
        return c

def prepare():
    if PLAN.exists():return read_json(PLAN)
    # A has no applied force. Confirm the judge, reset plan and force vectors
    # are exactly the same AST as the previously sealed implementation.
    old=subprocess.check_output(['git','show','81bf4cb:./tools/evaluation/g1_paper_recovery.py'],cwd=LAB,text=True)
    def selected(s):return [ast.dump(n,include_attributes=False) for n in ast.parse(s).body if getattr(n,'name',None) in ['Trial','plans','direction']]
    if selected(old)!=selected((LAB/SOURCES[0]).read_text()):raise ValueError('Judge/reset/direction changed; cannot retain A')
    proof=ROOT/'validation/ppo/B1/reset-impulse-fix-001/completion.json'
    c=read_json(proof)
    if c['episodes']!=64 or not all(sha256(proof.parent/'records'/f)==h for f,h in c['records_sha256'].items()):raise ValueError('Missing repair validation')
    gate=read_json(ROOT/'pilot_admission.json');rows=read_json(ROOT/'formal_frozen_manifest.json');identities=read_json(ROOT/'checkpoint_identities.json');jobs=[];new=current_runtime();old_runtime=None
    for suite in ['A','B1','B2','C_force','C_load']:
        for model in ['ppo','dwaq','ours']:
            attempt='attempt-002' if (suite,model)==('B1','ppo') else 'attempt-001'
            run=ROOT/'formal'/model/suite/attempt;subset=[r for r in rows if r['suite']==suite]
            runtime=runtime_key(read_json(run/'binding.json')) if suite=='A' else new
            e=dict(suite=suite,model=model,attempt=attempt,run=str(run),runtime=runtime,checkpoint_sha256=identities[model]['checkpoint_sha256'],manifest_hash=digest(subset),protocol_sha256=gate['protocol_sha256'],num_envs=64,episodes=len(subset))
            if suite=='A':
                verify_run(run,e,True)
                if old_runtime is not None and runtime!=old_runtime:raise ValueError('A methods have different runtimes')
                old_runtime=runtime
            jobs.append(e)
    plan=dict(jobs=jobs,protocol_sha256=gate['protocol_sha256'],manifest_sha256=gate['formal_manifest_sha256'],physical_contract_sha256=sha256(ROOT/'paired_physics_64.json'),note='A: retained complete same-runtime three-model cohort, unchanged judge/reset AST. B/C: same repaired runtime for all three models. No old incomplete B1 or diagnostics admitted; no cross-suite pooling.')
    write_json(PLAN,plan);return plan
if __name__=='__main__':prepare()
