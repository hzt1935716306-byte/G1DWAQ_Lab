"""Catch unintended ablation changes without requiring Isaac imports."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/recovery'))
from reward_shaping_contract import compare,canonical,digest


def configs():
    env={'terrain':{'seed':42},'reward':{'idle':{'weight':-2,'params':{'threshold':.2}},'track':{'weight':2}}}
    agent={'resume':False,'run_name':'parent','algorithm':{'lr':.001},'policy':{'encoder':[128,64]}}
    child=copy.deepcopy(env);child['reward']['idle']=None
    ca=copy.deepcopy(agent);ca['run_name']='child'
    return env,child,agent,ca


def test_only_requested_disabled_term_passes():
    assert set(compare(*configs(),{'idle'}))=={'idle'}


@pytest.mark.parametrize('mutation',['reward','terrain','optimizer','vae','resume'])
def test_unintended_differences_are_rejected(mutation):
    e,c,a,ca=configs()
    if mutation=='reward':c['reward']['track']['weight']=1
    if mutation=='terrain':c['terrain']['seed']=43
    if mutation=='optimizer':ca['algorithm']['lr']=.002
    if mutation=='vae':ca['policy']['encoder']=[64,32]
    if mutation=='resume':ca['resume']=True
    with pytest.raises(ValueError):compare(e,c,a,ca,{'idle'})


def test_config_hash_handles_infinite_limits_and_slices_stably():
    value={'limit':float('inf'),'ids':slice(None)}
    assert digest(value)==digest(canonical(value))


@pytest.mark.parametrize('defect',[None,'iterations','warm_start','budget','config_hash','nonfinite'])
def test_final_checkpoint_validation(tmp_path,defect):
    import torch
    from reward_shaping_contract import DWAQ,file_sha
    from run_reward_shaping_training import verify_checkpoint
    params=tmp_path/'params';params.mkdir()
    for kind in ('env','agent'):(params/(kind+'.yaml')).write_text('seed: 42\n')
    task=DWAQ+'_v2'
    provenance={'task_name':task,'parent_training_run_id':None,'training_transitions':4096*24*10000,
                'env_config_sha256':file_sha(params/'env.yaml'),'agent_config_sha256':file_sha(params/'agent.yaml')}
    checkpoint={'iter':9999,'training_provenance':provenance,
                'optimizer_state_dict':{'state':{0:{'step':2}}},'model_state_dict':{'weight':torch.ones(1)}}
    if defect=='iterations':checkpoint['iter']=1
    if defect=='warm_start':provenance['parent_training_run_id']='old-run'
    if defect=='budget':provenance['training_transitions']=4096*24*9999
    if defect=='config_hash':provenance['env_config_sha256']='wrong'
    if defect=='nonfinite':checkpoint['model_state_dict']['weight'][0]=float('nan')
    path=tmp_path/'model_9999.pt';torch.save(checkpoint,path)
    if defect is None:
        assert verify_checkpoint(path,task)['checkpoint_sha256']==file_sha(path)
    else:
        with pytest.raises(AssertionError):verify_checkpoint(path,task)
