"""One common B2 intensity adjustment, preserving initial pilot inputs and records."""
import argparse,copy
from collections import Counter
from g1_paper_recovery import ROOT,MODELS
from g1_paper_report import load
from g1_recovery_protocol import read_json,write_json,sha256

def freeze():
    data=load('pilot')
    if any(len(r)!=180 for r in data.values()):raise ValueError('All three180-episode pilots required')
    for rr in data.values():
        b=[r for r in rr if r['suite']=='B2']
        if len(b)!=72 or not all(r['recovered'] for r in b):raise ValueError('Predetermined all-easy adjustment trigger not met')
    p=read_json(ROOT/'protocol.json');p.update(B1_strengths=[.3,.6,.9],B2_strengths=[.5,1.5,2.5],calibration_adjustment_count=1,calibration_reason='All three methods recovered72/72 B2 episodes at initial0.3/0.6/0.9. Increase B2 uniformly once; keep B1 unchanged.')
    objects={'protocol_frozen.json':p}
    for stage,source in [('calibration','pilot'),('formal_frozen','formal')]:
        rows=read_json(ROOT/(source+'_manifest.json'))
        if stage=='calibration':rows=[r for r in rows if r['suite']=='B2']
        for r in rows:
            if r['suite']=='B2':
                old=r['strength'];new={.3:.5,.6:1.5,.9:2.5}[old];r['strength']=new;r['trial_id']=r['trial_id'].replace(f'_a{old:g}_',f'_a{new:g}_')
        objects[stage+'_manifest.json']=rows
    for name,obj in objects.items():
        target=ROOT/name
        if target.exists() and read_json(target)!=obj:raise ValueError('Frozen calibration inputs changed')
        write_json(target,obj)

def admit():
    data=load('calibration')
    if any(len(r)!=72 for r in data.values()):raise ValueError('Each model must complete72 adjusted B2 trials')
    expected={r['trial_id'] for r in read_json(ROOT/'calibration_manifest.json')}
    for rows in data.values():
        if {r['trial_id'] for r in rows}!=expected:raise ValueError('Unpaired calibration')
        if any(r['status'] not in ['COMPLETED','FELL','OUT_OF_TEST_AREA','ABNORMAL_TIMEOUT'] for r in rows):raise ValueError('Calibration software error')
    summary={m:{str(s):dict(n=len(a),recovered=sum(r['recovered'] for r in a),fell=sum(r['fell'] for r in a),pre_push_failure=sum(r['pre_push_failure'] for r in a)) for s in [.5,1.5,2.5] for a in [[r for r in rows if r['strength']==s]]} for m,rows in data.items()}
    write_json(ROOT/'pilot_admission.json',dict(physical_contract_passed=True,formal_matrix_frozen=True,adjustment_count=1,B2_calibration=summary,protocol_sha256=sha256(ROOT/'protocol_frozen.json'),formal_manifest_sha256=sha256(ROOT/'formal_frozen_manifest.json'),note='No further intensity tuning. Initial pilots/adjusted calibration excluded from formal stats. Physics checks embedded in simulator; startup failures preserved separately.'))
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','admit']);a=p.parse_args();freeze() if a.action=='freeze' else admit()
