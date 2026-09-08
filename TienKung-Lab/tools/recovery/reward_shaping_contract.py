"""Strict, CPU-only checks for the three requested reward ablations."""
import dataclasses
import hashlib
import json
import math
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
BASE_COMMIT = '6a5ad21710d878129970266084565f9f03094055'
PLANE = 'g1_plane_v1_estimator_context_no_reward_matched'
DWAQ = 'g1_dwaq_slope_nosys_d_matched'
TASKS = {PLANE+'_v2': (PLANE, {'idle_penalty', 'feet_swing_height'}),
         DWAQ+'_v2': (DWAQ, {'idle_penalty'}), DWAQ+'_v3': (DWAQ, {'feet_swing_height'})}
RUN_NAMES = ['estimator_context_no_reward_matched_v2_idle01_swing02',
             'dwaq_nosys_matched_no_idle', 'dwaq_nosys_matched_no_swing_height']
ESTIMATOR = LAB/'logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed/com_velocity_estimator_v2_long_best.pt'
ESTIMATOR_SHA = '8574d845f28cbc7908e437250de46ba865898056483741e8fb72a81281f9e319'
BASELINE = LAB/'tools/recovery/configs/reward_shaping_parent_6a5ad21.json'


def canonical(x):
    if isinstance(x, float) and not math.isfinite(x): return {'__float__': str(x)}
    if x is dataclasses.MISSING: return {'__missing__': True}
    if callable(x): return {'__callable__': x.__module__+'.'+x.__qualname__}
    if isinstance(x, slice): return {'__slice__': [x.start, x.stop, x.step]}
    if hasattr(x, 'to_dict'): return canonical(x.to_dict())
    if isinstance(x, dict): return {str(k): canonical(v) for k,v in sorted(x.items())}
    if isinstance(x, (tuple, list)): return [canonical(v) for v in x]
    if hasattr(x, 'tolist'): return canonical(x.tolist())
    if isinstance(x, Path): return str(x)
    if x is None or isinstance(x, (str, bool, int, float)): return x
    raise TypeError(type(x))


def digest(x):
    return hashlib.sha256(json.dumps(canonical(x), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def active_rewards(rewards):
    return {k:v for k,v in rewards.items() if isinstance(v,dict) and v.get('weight',0)!=0}


def compare(parent_env, child_env, parent_agent, child_agent, expected_terms):
    p,c,pa,ca=map(canonical,(parent_env,child_env,parent_agent,child_agent))
    pr,cr=p['reward'],c['reward']
    changed={k for k in active_rewards(pr).keys() | active_rewards(cr).keys()
             if active_rewards(pr).get(k)!=active_rewards(cr).get(k)}
    if changed != set(expected_terms):raise ValueError(f'Unexpected active reward diff: {changed}')
    if {k:v for k,v in pr.items() if k not in expected_terms}!={k:v for k,v in cr.items() if k not in expected_terms}:
        raise ValueError('Other reward configuration changed')
    if {k:v for k,v in p.items() if k!='reward'}!={k:v for k,v in c.items() if k!='reward'}:
        raise ValueError('Non-reward environment configuration changed')
    allowed={'experiment_name','wandb_project','run_name','resume'}
    if {k:v for k,v in pa.items() if k not in allowed}!={k:v for k,v in ca.items() if k not in allowed}:
        raise ValueError('Training algorithm/network/normalization configuration changed')
    if ca['resume'] is not False:raise ValueError('Resume forbidden')
    return {k:{'before':pr.get(k),'after':cr.get(k)} for k in sorted(changed)}


def audit_registry(registry):
    before=json.loads(BASELINE.read_text());parents={};results={}
    if before['base_commit']!=BASE_COMMIT:raise ValueError('Wrong baseline snapshot')
    for name,old in before['tasks'].items():
        env,agent=registry.get_cfgs(name);e,a=canonical(env),canonical(agent)
        if e!=old['env'] or a!=old['agent']:raise ValueError('Original task changed: '+name)
        if digest(e)!=old['env_sha256'] or digest(a)!=old['agent_sha256']:raise ValueError('Parent hash changed')
        parents[name]={'env_sha256':digest(e),'agent_sha256':digest(a)}
    for (name,(parent,terms)),run_name in zip(TASKS.items(),RUN_NAMES):
        pe,pa=registry.get_cfgs(parent);env,agent=registry.get_cfgs(name)
        diff=compare(pe,env,pa,agent,terms)
        if registry.get_task_class(name) is not registry.get_task_class(parent):raise ValueError('Native environment class changed')
        assert agent.experiment_name==agent.wandb_project==name and agent.run_name==run_name
        assert agent.seed==42 and agent.num_steps_per_env==24 and agent.max_iterations==10000
        assert env.scene.num_envs==4096
        if parent==PLANE:
            assert env.com_velocity_source=='estimator'
            assert env.recovery_context.enabled and env.recovery_context.mode=='certificate'
            assert not env.plane_v1_reward.enabled and not env.stage2_reward.enabled
            assert not env.stage2_reward.enable_certificate_reward and not env.stage2_reward.enable_shared_event_reward
            assert agent.algorithm.symmetry_cfg.use_data_augmentation and agent.algorithm.symmetry_cfg.use_mirror_loss
            assert env.reward.idle_penalty.weight==-.1
            assert env.reward.idle_penalty.params=={'cmd_threshold':.2,'vel_threshold':.1}
            assert env.reward.feet_swing_height.weight==-.2
            assert env.reward.feet_swing_height.params['target_height']==.08
            for key in ('sensor_cfg','asset_cfg'):
                assert env.reward.feet_swing_height.params[key].body_names=='.*ankle_roll.*'
            assert env.reward.track_lin_vel_xy_exp.weight==2 and env.reward.track_ang_vel_z_exp.weight==2
            assert env.reward.joint_deviation_hip.weight==-.30
        else:
            assert env.reward.alive.weight==.15 and env.reward.joint_deviation_ankle.weight==-.2
            assert env.reward.gait_phase_contact is None
            if name.endswith('_v2'):
                assert env.reward.idle_penalty is None
                assert env.reward.feet_swing_height.weight==-.2
                assert env.reward.feet_swing_height.params['target_height']==.08
            else:
                assert env.reward.feet_swing_height is None
                assert env.reward.idle_penalty.weight==-2
                assert env.reward.idle_penalty.params=={'cmd_threshold':.2,'vel_threshold':.1}
        results[name]={'parent':parent,'reward_diff':diff,'env_sha256':digest(env),'agent_sha256':digest(agent),
                       'reward_table':canonical(env.reward),'num_envs':env.scene.num_envs,
                       'steps_per_env':agent.num_steps_per_env,'iterations':agent.max_iterations,'seed':agent.seed,'resume':agent.resume}
    if file_sha(ESTIMATOR)!=ESTIMATOR_SHA:raise ValueError('Estimator SHA mismatch')
    # Instantiating and comparing the children must not mutate registered parents.
    for name,old in before['tasks'].items():
        e,a=registry.get_cfgs(name)
        assert digest(e)==old['env_sha256'] and digest(a)==old['agent_sha256']
    return {'status':'PASS','base_commit':BASE_COMMIT,'parents':parents,'tasks':results,'estimator_sha256':ESTIMATOR_SHA}
