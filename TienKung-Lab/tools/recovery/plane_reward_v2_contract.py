"""Full configuration audit for the reward-on Plane locomotion ablation."""
import copy
import json
from reward_shaping_contract import LAB, canonical, digest, compare, file_sha, ESTIMATOR, ESTIMATOR_SHA

PARENT='g1_plane_v1_estimator_context_reward_matched'
TASK=PARENT+'_v2'
CONTROL='g1_plane_v1_estimator_context_no_reward_matched_v2'
ROOT=LAB/'experiments/g1_plane_reward_v2_training'


def compare_control(control_env,new_env,control_agent,new_agent):
    left,right=copy.deepcopy(canonical(control_env)),canonical(new_env)
    assert left['plane_v1_reward']['enabled'] is False and right['plane_v1_reward']['enabled'] is True
    left['plane_v1_reward']['enabled']=True
    if left!=right:raise ValueError('Control differs beyond plane_v1_reward.enabled')
    ignore={'experiment_name','wandb_project','run_name'}
    ca,na=canonical(control_agent),canonical(new_agent)
    if {k:v for k,v in ca.items() if k not in ignore}!={k:v for k,v in na.items() if k not in ignore}:
        raise ValueError('Control training configuration differs')
    return ['plane_v1_reward.enabled']


def audit(registry,before):
    for name,old in before['tasks'].items():
        e,a=registry.get_cfgs(name)
        if digest(e)!=old['env_sha256'] or digest(a)!=old['agent_sha256']:
            raise ValueError('Existing task changed: '+name)
    pe,pa=registry.get_cfgs(PARENT);e,a=registry.get_cfgs(TASK);ce,ca=registry.get_cfgs(CONTROL)
    changes=compare(pe,e,pa,a,{'idle_penalty','feet_swing_height'})
    control_diff=compare_control(ce,e,ca,a)
    assert canonical(pe.plane_v1_reward)==canonical(e.plane_v1_reward)==dict(enabled=True,
        certificate_progress_weight=.25,certificate_delta_phi_clip=2.,unrecovered_touchdown_cost=-.05,
        td5_unrecovered_penalty=-.25,certificate_target_n_max=1,certificate_horizon_touchdowns=5)
    assert e.com_velocity_source=='estimator' and e.recovery_context.enabled and e.recovery_context.mode=='certificate'
    assert not e.stage2_reward.enabled and not e.stage2_reward.enable_certificate_reward and not e.stage2_reward.enable_shared_event_reward
    assert e.reward.idle_penalty.weight==-.1 and e.reward.feet_swing_height.weight==-.2
    assert e.reward.track_lin_vel_xy_exp.weight==2 and e.reward.track_ang_vel_z_exp.weight==2 and e.reward.joint_deviation_hip.weight==-.3
    assert a.resume is False and (a.seed,e.scene.num_envs,a.num_steps_per_env,a.max_iterations)==(42,4096,24,10000)
    assert a.algorithm.symmetry_cfg.use_data_augmentation and a.algorithm.symmetry_cfg.use_mirror_loss
    assert a.experiment_name==a.wandb_project==TASK
    assert registry.get_task_class(TASK) is registry.get_task_class(PARENT)
    assert file_sha(ESTIMATOR)==ESTIMATOR_SHA
    for name,old in before['tasks'].items():
        x,y=registry.get_cfgs(name);assert (digest(x),digest(y))==(old['env_sha256'],old['agent_sha256'])
    return dict(status='PASS',task=TASK,parent=PARENT,reward_diff=changes,control_env_diff=control_diff,
        recoverability_reward=canonical(e.plane_v1_reward),env_sha256=digest(e),agent_sha256=digest(a),
        original_tasks_unchanged=True,estimator_sha256=ESTIMATOR_SHA,resume=False)
