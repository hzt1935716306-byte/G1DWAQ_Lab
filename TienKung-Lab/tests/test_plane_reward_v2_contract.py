"""Reject confounded reward-on/off ablations without importing Isaac."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/recovery'))
from plane_reward_v2_contract import compare_control


def pair():
    e={'plane_v1_reward':{'enabled':False,'certificate_progress_weight':.25},
       'reward':{'idle_penalty':{'weight':-.1},'feet_swing_height':{'weight':-.2}},
       'terrain':{'curriculum':True},'recovery_context':{'mode':'certificate'}}
    n=copy.deepcopy(e);n['plane_v1_reward']['enabled']=True
    a={'resume':False,'run_name':'off','algorithm':{'learning_rate':.001}}
    b=copy.deepcopy(a);b['run_name']='on'
    return e,n,a,b


def test_only_enabled_and_logging_names_differ():
    values=pair();before=copy.deepcopy(values)
    assert compare_control(*values)==['plane_v1_reward.enabled']
    assert values==before


@pytest.mark.parametrize('change',['progress_weight','locomotion','curriculum','context','ppo','resume'])
def test_other_differences_rejected(change):
    e,n,a,b=pair()
    if change=='progress_weight':n['plane_v1_reward']['certificate_progress_weight']=.5
    if change=='locomotion':n['reward']['idle_penalty']['weight']=-2.
    if change=='curriculum':n['terrain']['curriculum']=False
    if change=='context':n['recovery_context']['mode']='other'
    if change=='ppo':b['algorithm']['learning_rate']=.002
    if change=='resume':b['resume']=True
    with pytest.raises(ValueError):compare_control(e,n,a,b)
