"""Deterministic budget-only subset, source references and pending execution queues.

Selection never reads results. Original trial dictionaries are preserved verbatim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import time
import yaml

from g1_recovery_protocol import LAB, atomic_write, digest, read_json, read_jsonl, sha256, write_json, jsonl
from g1_robustness_protocol import MODEL_ORDER, load_prepared, COUNTS

SOURCE = LAB/'experiments/g1_complete_robustness_v2/cohorts/wrench_fix_96e85ee1'
ROOT = LAB/'experiments/g1_complete_robustness_v2/reduced_budget_v1'
TAG = 'reduced_budget_v1'
BASE_COMMIT = 'c8cff3bed7b181e27a7d411591b50ed2f16fa941'
TARGET_COUNTS = {'extreme_main':1192, 'extreme_ood':128, 'velocity_ood':60,
                 'force_pulse':180, 'constant_force':100, 'repeated_impulse':120,
                 'random_force':120, 'wrench_pulse':60}


def immutable(path, value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Immutable reduced input changed: '+str(path))
    else:write_json(path,value)


def score(row):
    return hashlib.sha256((TAG+row['condition_id']+row['trial_id']).encode()).hexdigest()


def rotation(condition, n, purpose='balance'):
    return int(hashlib.sha256((TAG+purpose+condition).encode()).hexdigest(),16)%n


def balanced(rows, n, condition):
    """Fixed cyclic foot/phase quotas; exact specified hash ranks within each cell."""
    phases=sorted({r['target_phase'] for r in rows})
    if len(phases)!=2:raise ValueError('Expected the two unchanged reference phases')
    cells=[('left',phases[0]),('right',phases[1]),('left',phases[1]),('right',phases[0])]
    start=rotation(condition,4);cells=[cells[j] for j in (start,start^1,(start+2)%4,((start+2)%4)^1)]
    pools={c:sorted([r for r in rows if (r['reference_touchdown_foot'],r['target_phase'])==c],key=score) for c in cells}
    chosen=[]
    for j in range(n):
        cell=cells[j%4]
        if not pools[cell]:raise ValueError('Missing fixed foot/phase quota: '+condition+str(cell))
        chosen.append(pools[cell].pop(0))
    return chosen


def select(rows):
    """Pure function of the old manifest; no model, status or availability inputs."""
    groups=defaultdict(list)
    for r in rows:groups[r['condition_id']].append(r)
    chosen=[];coverage=[]
    for condition,cell in sorted(groups.items()):
        r=cell[0];suite=r['suite'];extra=[]
        if suite.startswith('extreme_'):
            slope=r['slope_deg'];dv=r['push_magnitude']
            if suite=='extreme_ood':n=2 if dv in (0.,1.,2.,3.) else 0
            elif slope==0:n=5
            else:n=3 if dv in (0.,.5,1.,1.5,2.,2.5,3.) else 0
            picked=balanced(cell,n,condition) if n else []
        elif suite in ('force_pulse','constant_force','wrench_pulse'):
            directions=sorted({x['direction_deg'] for x in cell})
            if directions!=list(range(0,360,45)):raise ValueError('Expected eight fixed directions')
            start=rotation(condition,8,'directions');extra=[directions[(start+j)%8] for j in range(4)]
            picked=[]
            for angle in directions:
                picked+=balanced([x for x in cell if x['direction_deg']==angle],2+(angle in extra),condition+f'/direction={angle}')
        else:
            picked=sorted(cell,key=score)[:20]
            if len(picked)!=20:raise ValueError('Missing random-plan quota: '+condition)
        chosen+=picked
        if picked:
            coverage.append(dict(condition_id=condition,suite=suite,n=len(picked),extra_directions=extra,
                directions=dict(Counter(str(x['direction_deg']) for x in picked)),
                foot_phase=dict(Counter(f"{x['reference_touchdown_foot']}/{x['target_phase']}" for x in picked)),
                selected_trial_ids=[x['trial_id'] for x in picked]))
    chosen=sorted(chosen,key=lambda r:r['trial_id'])
    if dict(Counter(r['suite'] for r in chosen))!=TARGET_COUNTS:raise ValueError('Reduced quotas do not match')
    if len({r['trial_id'] for r in chosen})!=1960:raise ValueError('Duplicate/missing reduced trial')
    return chosen,coverage


def prepare(root=ROOT, source=SOURCE):
    root=Path(root);source=Path(source);p,full,prepared=load_prepared(source)
    selected,coverage=select(full)
    manifest=dict(schema_version=1,budget_version=TAG,source_manifest_sha256=sha256(source/'manifest.jsonl'),
        source_root=str(source),protocol_hash=digest(p),trial_count=1960,counts_per_model=TARGET_COUNTS,
        selection_score='SHA256("reduced_budget_v1" + condition_id + trial_id)',trials=selected)
    immutable(root/'reduced_budget_manifest_v1.json',manifest)
    audit_path=root/'reduced_budget_selection_audit.json'
    if not audit_path.exists():
        immutable(audit_path,dict(budget_version=TAG,changed_at_unix=time.time(),base_commit=BASE_COMMIT,
            original_counts_per_model=COUNTS,new_counts_per_model=TARGET_COUNTS,standard_per_model=390,
            timing_disclosure='Budget reduced after collection began; not a preregistration from the start',
            selection_reads_outcomes=False,selection_reads_availability=False,
            phase_foot_rule='Four-cell cyclic quotas rotated by condition-only hash; prescribed hash ranks within cells',
            random_rule='Unstratified prescribed hash top20; no waveform/distribution editing',coverage=coverage,
            supplemental_plans_required=0,shortfalls=[]))
    index=read_json(source/'standard_benchmark_five_model_index.json')
    immutable(root/'standard_benchmark_five_model_index.json',index)
    return manifest


def validate_pending(root,p,rows,info):
    """Explicit new prepare contract, rather than disabling existing checks."""
    root=Path(root);master=Path(info['reduced_budget_root'])
    target=read_json(master/'reduced_budget_manifest_v1.json')
    if sha256(master/'reduced_budget_manifest_v1.json')!=info['reduced_manifest_sha256']:raise ValueError('Reduced master changed')
    if target['protocol_hash']!=digest(p):raise ValueError('Budget changed physical protocol')
    source=Path(target['source_root'])
    if sha256(source/'manifest.jsonl')!=target['source_manifest_sha256']:raise ValueError('Original manifest changed')
    expected,_=select(read_jsonl(source/'manifest.jsonl'))
    if target['trials']!=expected:raise ValueError('Reduced selection changed')
    for name,key in [('pending_execution_queues.json','execution_queue_sha256'),('reduced_budget_reuse_index.json','reuse_index_sha256')]:
        if sha256(master/name)!=info[key]:raise ValueError('Frozen budget queue/source index changed')
    queue=read_json(master/'pending_execution_queues.json')[info['reduced_model']]
    pending=[r['target_trial_id'] for r in read_json(master/'reduced_budget_reuse_index.json')['entries']
             if r['model']==info['reduced_model'] and r['reuse_decision']=='PENDING_EXECUTION']
    if queue!=pending:raise ValueError('Queue includes a reused or unassigned trial')
    lookup={r['trial_id']:r for r in expected}
    if rows!=[lookup[tid] for tid in queue]:raise ValueError('Pending manifest differs from frozen queue')
    if info.get('execution_batch_size')!=64:raise ValueError('Budget change must keep64 environments')


def execution_rows(manifest, offset, num_envs):
    """Keep64 native slots; incomplete final batch has unscored inactive slots."""
    active=manifest[offset:offset+num_envs]
    if not active:raise ValueError('Empty execution batch')
    return active+[active[-1]]*(num_envs-len(active)),len(active)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare'])
    args=parser.parse_args();print(json.dumps({k:v for k,v in prepare().items() if k!='trials'},indent=2))
