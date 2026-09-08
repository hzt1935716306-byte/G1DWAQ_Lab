"""Exact allowlisted queue-only source transition; physical definitions stay fixed."""
from __future__ import annotations
import hashlib
from pathlib import Path
from g1_recovery_protocol import LAB, EVALUATION_RUNTIME_SOURCES, code_identity, sha256, digest, read_json
from g1_audit_compatibility import source_at
from g1_reduced_budget import BASE_COMMIT, ROOT, immutable

NEW_SOURCE='tools/evaluation/g1_reduced_budget.py'


def expected_queue_source(name,old):
    """Exact reviewed additions only; an unrelated one-character edit is rejected."""
    if name=='tools/evaluation/g1_recovery_protocol.py':
        return old.replace('EVALUATION_RUNTIME_SOURCES = (\n',"EVALUATION_RUNTIME_SOURCES = (\n    'tools/evaluation/g1_reduced_budget.py',\n")
    if name=='tools/evaluation/g1_robustness_protocol.py':
        return old.replace("    if info.get('validation_mode'):\n","    if info.get('budget_mode'):\n        if info['budget_mode']!='reduced_budget_v1':raise ValueError('Unknown reduced budget mode')\n        from g1_reduced_budget import validate_pending\n        validate_pending(root,p,rows,info)\n    if info.get('validation_mode'):\n")
    if name!='tools/evaluation/g1_complete_robustness.py':return old
    changes=[
        ("    args.task=identity['task_name'];args.num_envs=prepared['num_envs'] if smoke else p['robustness']['num_envs'];args.headless=True",
         "    if prepared.get('budget_mode')=='reduced_budget_v1':\n        identity.update(subset='reduced_budget_v1_pending',execution_batch_size=64,\n            reduced_manifest_sha256=prepared['reduced_manifest_sha256'])\n    args.task=identity['task_name'];args.num_envs=prepared['num_envs'] if smoke else p['robustness']['num_envs'];args.headless=True"),
        ("                    selected=manifest[offset:offset+args.num_envs]\n",
         "                    selected=manifest[offset:offset+args.num_envs]\n                    active_count=len(selected)\n                    if prepared.get('budget_mode')=='reduced_budget_v1':\n                        from g1_reduced_budget import execution_rows\n                        selected,active_count=execution_rows(manifest,offset,args.num_envs)\n"),
        ("                    machines=[RobustnessTrial(r,p) for r in selected]\n",
         "                    machines=[RobustnessTrial(r,p) for r in selected]\n                    for inactive in machines[active_count:]:inactive.status='NOT_SCHEDULED'\n"),
        ("                    for i,m in enumerate(machines):\n                        state=",
         "                    for i,m in enumerate(machines):\n                        if i>=active_count:continue\n                        state="),
        ("                    write_json(run/'progress.json',progress);print(progress,flush=True)\n",
         "                    write_json(run/'progress.json',progress);print(progress,flush=True)\n                    if (root/'STOP_AFTER_BATCH').exists():\n                        controller.clear()\n                        return {'status':'PARTIAL','evaluation_id':run.name,'completed':len(done)}\n")]
    for before,after in changes:
        if old.count(before)!=1:raise ValueError('Baseline queue boundary changed')
        old=old.replace(before,after)
    return old


def create_proof(root=ROOT):
    code=code_identity()
    if code['evaluation_runtime_dirty']:raise ValueError('Commit queue implementation first')
    before={};after={};changed=[]
    for name in EVALUATION_RUNTIME_SOURCES:
        after[name]=sha256(LAB/name)
        if name==NEW_SOURCE:continue
        old=source_at(BASE_COMMIT,name);before[name]=hashlib.sha256(old.encode()).hexdigest()
        if (LAB/name).read_text()!=expected_queue_source(name,old):raise ValueError('Non-queue runtime changed: '+name)
        if before[name]!=after[name]:changed.append(name)
    proof=dict(scope='BUDGET_SELECTION_QUEUE_AND_PROVENANCE_ONLY',base_commit=BASE_COMMIT,
        execution_commit=code['evaluation_code_commit'],baseline_runtime_sources=before,
        execution_runtime_sources=after,evaluation_runtime_sha256=digest(after),changed_sources=changed,
        added_selection_source=NEW_SOURCE,execution_batch_size=64,
        unchanged=['checkpoints','native inputs','actor/critic/history','env.step/actions','wrench math/frame',
                   'physical intervention timing','Common detector','threshold/W/H','old manifests and sealed records'],
        packing='Pending target IDs sorted identically for each model;64 slots; inactive final slots receive no intervention or trial outcome',
        compatibility_limit='Equal trial inputs and physical/judge code semantics; no assertion of bitwise trajectory replay')
    immutable(Path(root)/'reduced_budget_runtime_compatibility.json',proof)
    return proof


def verify_proof(root=ROOT):
    proof=read_json(Path(root)/'reduced_budget_runtime_compatibility.json')
    for name,expected in proof['execution_runtime_sources'].items():
        if sha256(LAB/name)!=expected:raise ValueError('Execution source changed: '+name)
        if name!=NEW_SOURCE and (LAB/name).read_text()!=expected_queue_source(name,source_at(BASE_COMMIT,name)):
            raise ValueError('Queue-only proof no longer holds')
    return proof
