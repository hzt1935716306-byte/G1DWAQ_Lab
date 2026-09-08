"""Explicit audit-only source transition, without rewriting historical run identities."""
from __future__ import annotations
import ast,copy,subprocess
from pathlib import Path
from g1_recovery_protocol import LAB,read_json,write_json,digest,sha256

# Reviewed audit/registry/CLI boundaries only. Actual simulation loops, inference,
# physics adapters, detector mathematics, resources and all other sources are exact.
AUDIT_BOUNDARIES={
 'tools/evaluation/g1_recovery_protocol.py':{'RunStore.validate','RunStore._validate_light','RunStore.complete','register_result','code_identity'},
 'tools/evaluation/g1_robustness_store.py':{'validate_trial','RobustnessStore.validate','RobustnessStore._validate_light','RobustnessStore.complete'},
 'tools/evaluation/g1_recovery_eval.py':{'main'},
 'tools/evaluation/g1_complete_robustness.py':{'immutable_json'},
}

def projection(source,path):
    tree=ast.parse(source);allowed=AUDIT_BOUNDARIES.get(path,set())
    def filter_body(body,prefix=''):
        result=[]
        for node in body:
            name=getattr(node,'name',None);qualified=prefix+name if name else None
            if qualified in allowed:continue
            if isinstance(node,ast.ClassDef):node.body=filter_body(node.body,node.name+'.')
            result.append(node)
        return result
    tree.body=filter_body(tree.body)
    return ast.dump(tree,include_attributes=False)

def source_at(commit,path):
    prefix=subprocess.check_output(['git','-C',str(LAB),'rev-parse','--show-prefix'],text=True).strip()
    return subprocess.check_output(['git','-C',str(LAB),'show',commit+':'+prefix+path],text=True)

def make_transition(root):
    from g1_recovery_protocol import code_identity
    root=Path(root);old=read_json(root/'runtime_commit.json');new=code_identity()
    if new['evaluation_runtime_dirty']:raise ValueError('Commit audit policy changes before admission')
    before=old['evaluation_runtime_sources'];after=new['evaluation_runtime_sources']
    if before.keys()!=after.keys():raise ValueError('Runtime allowlist must not be weakened/changed')
    changed=[];projections={}
    for name in before:
        if before[name]==after[name]:continue
        if name not in AUDIT_BOUNDARIES:raise ValueError('Non-audit runtime changed: '+name)
        original=source_at(old['evaluation_code_commit'],name)
        import hashlib
        if hashlib.sha256(original.encode()).hexdigest()!=before[name]:raise ValueError('Baseline source does not match archived runtime')
        a=projection(original,name);b=projection((LAB/name).read_text(),name)
        if a!=b:raise ValueError('Physical/non-audit code changed: '+name)
        changed.append(name);projections[name]=digest(a)
    proof=dict(schema_version=1,scope='EXPLICIT_AUDIT_POLICY_ONLY',
        predecessor=old,successor={k:new[k] for k in ('evaluation_code_commit','evaluation_runtime_sha256','evaluation_runtime_sources')},
        changed_runtime_files=changed,unchanged_non_audit_ast_sha256=projections,
        allowed_audit_boundaries={k:sorted(v) for k,v in AUDIT_BOUNDARIES.items()},
        audit_code_sha256=new.get('audit_code_sha256'),
        statements=['Physical simulation loops and all non-audit runtime definitions unchanged',
                    'Native policy/checkpoint/certificate/detector/threshold/W/H unchanged',
                    'Actual physics/environment/manifest/initial-state equality remains mandatory',
                    'Full-file runtime hashes remain distinct and reported; no historical identity rewritten'])
    path=root/'audit_policy_transition.json'
    if path.exists() and read_json(path)!=proof:raise ValueError('Audit transition is immutable')
    if not path.exists():write_json(path,proof)
    return proof

def verify_transition(root):
    root=Path(root);saved=read_json(root/'audit_policy_transition.json')
    from g1_recovery_protocol import code_identity
    if code_identity().get('audit_code_sha256')!=saved['audit_code_sha256']:raise ValueError('Audit backend differs from approved transition')
    before=saved['predecessor']['evaluation_runtime_sources'];after=saved['successor']['evaluation_runtime_sources']
    if before.keys()!=after.keys():raise ValueError('Runtime allowlist changed')
    import hashlib
    for name,expected in after.items():
        if sha256(LAB/name)!=expected:raise ValueError('Current source differs from approved audit transition: '+name)
        if before[name]!=expected:
            old=source_at(saved['predecessor']['evaluation_code_commit'],name)
            if hashlib.sha256(old.encode()).hexdigest()!=before[name]:raise ValueError('Archived source mismatch')
            if name not in AUDIT_BOUNDARIES or projection(old,name)!=projection((LAB/name).read_text(),name):raise ValueError('Non-audit source changed')
    return saved


def admit_immutable(path,value):
    """Return true only for separately proven audit-only metadata admission."""
    path=Path(path)
    if path.name not in ('runtime_commit.json','paired_physics_identity.json'):return False
    if not (path.parent/'audit_policy_transition.json').exists():return False
    proof=verify_transition(path.parent);old=read_json(path)
    if old['evaluation_runtime_sha256']!=proof['predecessor']['evaluation_runtime_sha256'] or value['evaluation_runtime_sha256']!=proof['successor']['evaluation_runtime_sha256']:raise ValueError('Unapproved runtime transition')
    if path.name=='runtime_commit.json':
        if old!=proof['predecessor'] or value!=proof['successor']:raise ValueError('Runtime source transition differs')
    else:
        ignore={'evaluation_code_commit','evaluation_runtime_sha256'}
        if {k:v for k,v in old.items() if k not in ignore}!={k:v for k,v in value.items() if k not in ignore}:raise ValueError('Physical paired identity changed across audit transition')
    target=path.parent/'audit_runtime_admissions'/value['evaluation_runtime_sha256']/path.name
    if target.exists() and read_json(target)!=value:raise ValueError('Audit admission changed')
    if not target.exists():write_json(target,value)
    return True

def validate_cohort_runtime(root,identities):
    hashes={i['evaluation_runtime_sha256'] for i in identities}
    commits={i['evaluation_code_commit'] for i in identities}
    if len(hashes)==len(commits)==1:return {'status':'IDENTICAL_FULL_RUNTIME'}
    proof=verify_transition(root)
    permitted={proof[k]['evaluation_runtime_sha256']:proof[k] for k in ('predecessor','successor')}
    for i in identities:
        p=permitted.get(i['evaluation_runtime_sha256'])
        if p is None or i['evaluation_code_commit']!=p['evaluation_code_commit'] or i['evaluation_runtime_sources']!=p['evaluation_runtime_sources']:raise ValueError('Unapproved code/runtime identity')
    return dict(status='AUDIT_ONLY_REVISION_EXPLICITLY_VERIFIED',transition_sha256=sha256(Path(root)/'audit_policy_transition.json'),runtime_hashes=sorted(hashes),code_commits=sorted(commits))
