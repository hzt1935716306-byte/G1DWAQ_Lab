"""Two native training iterations, followed by a native push integration probe."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import faulthandler
import traceback
import inspect
from reward_shaping_contract import TASKS,ESTIMATOR,ESTIMATOR_SHA,audit_registry,canonical,file_sha,LAB
sys.path.insert(0,str(LAB/'rsl_rl'))

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task',choices=TASKS,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.argv=['legged_lab/scripts/train.py','--task',a.task,'--num_envs','32','--max_iterations','2',
          '--seed','42','--headless','--device','cuda:0','--experiment_name',a.task+'_smoke',
          '--run_name','SMOKE_ONLY_NOT_FORMAL']
if a.task.startswith('g1_plane'):
    sys.argv+=['--estimator_checkpoint_path',str(ESTIMATOR)]
app=runpy.run_path(str(LAB/'legged_lab/scripts/train.py'),run_name='smoke_train')
g=app['train'].__globals__
import torch
faulthandler.dump_traceback_later(90,repeat=True)

def finite(x):
    assert torch.isfinite(x).all(), 'Nonfinite smoke tensor'

try:
    audit_registry(g['task_registry'])
    ec,ac=g['task_registry'].get_cfgs(a.task)
    runner_name=ac.runner_class_name
    NativeRunner=g[runner_name]
    print('SMOKE_RUNNER_SOURCE '+inspect.getfile(NativeRunner),flush=True)
    assert Path(inspect.getfile(NativeRunner)).resolve().is_relative_to(LAB/'rsl_rl')
    # DWAQ critic includes 12 foot pose/velocity, 6 contact-force and 1 root
    # height features and the parent's 187 height-scan rays in addition to
    # its 101 basic privileged observations. The actor remains blind.
    expected=(483,1010) if a.task.startswith('g1_plane') else (115,307)

    class SmokeRunner(NativeRunner):
        def load(self,*args,**kwargs):
            raise RuntimeError('Smoke must start from random policy, never load a checkpoint')

        def learn(self,num_learning_iterations,**kwargs):
            print('SMOKE_NATIVE_LEARN_ENTER',flush=True)
            assert num_learning_iterations==2 and self.current_learning_iteration==0
            assert not self.alg.optimizer.state
            assert self.training_provenance['parent_training_run_id'] is None
            env=self.env;policy=self.alg.policy
            print('SMOKE_NETWORK_INPUTS',policy.actor[0].in_features,policy.critic[0].in_features,flush=True)
            assert (policy.actor[0].in_features,policy.critic[0].in_features)==expected
            assert env.num_actions==29 and env.num_envs==32
            active=list(env.reward_manager.active_terms)
            assert ('idle_penalty' in active)==(not a.task.endswith('matched_v2') or a.task.startswith('g1_plane'))
            assert ('feet_swing_height' in active)==(not a.task.endswith('_v3'))
            super().learn(num_learning_iterations=num_learning_iterations,**kwargs)
            print('SMOKE_NATIVE_LEARN_FINISHED',flush=True)
            assert self.current_learning_iteration==1 and self.alg.optimizer.state
            for x in policy.parameters():finite(x)
            # Explicit test-only trigger, preserving the native function and XY
            # distribution. No shortened push timing or terrain schedule is used.
            event=env.event_manager.get_term_cfg('push_robot')
            before=env.robot.data.root_vel_w[:,:2].clone()
            ids=torch.arange(env.num_envs,device=env.device)
            event.func(env,ids,**event.params)
            after=env.robot.data.root_vel_w[:,:2].clone()
            delta=after-before
            # The installed native Isaac event uses vel_w += sampled velocity.
            # Check the actual increment, preserving the parent implementation.
            assert delta.abs().max()<=1.0001 and torch.any(delta!=0)
            obs,_=env.get_observations();extras=env.extras;shape=list(obs.shape)
            policy.eval()
            for _ in range(32):
                with torch.inference_mode():
                    if hasattr(policy,'cenet_forward'):
                        actions=policy.act_inference(obs,extras['observations']['obs_hist'])
                    else:actions=self.get_inference_policy(device=env.device)(obs)
                    finite(actions)
                    obs,reward,done,extras=env.step(actions)
                    finite(obs);finite(reward);finite(extras['observations']['critic'])
            data={'status':'PASS','role':'SMOKE_ONLY_NOT_FORMAL','task':a.task,
                  'run_directory':str(self.log_dir),'iterations':2,'start_iteration':0,'resume':False,
                  'num_envs':32,'actor_observation_shape':shape,'actor_input':expected[0],
                  'critic_input':expected[1],'action_dim':29,'active_rewards':active,
                  'reward_table':canonical(env.cfg.reward),'push_event':canonical(ec.domain_rand.events.push_robot),
                  'native_push_probe_count':32,'native_push_delta_xy':delta.cpu().tolist(),
                  'native_push_before_xy':before.cpu().tolist(),'native_push_after_xy':after.cpu().tolist(),
                  'terrain_type':env.cfg.scene.terrain_type,'terrain_levels_present':hasattr(env.scene.terrain,'terrain_levels'),
                  'post_training_probe_steps':32,'finite':True,'policy_class':type(policy).__name__,
                  'algorithm_class':type(self.alg).__name__}
            if a.task.startswith('g1_plane'):
                assert file_sha(ESTIMATOR)==ESTIMATOR_SHA
                assert not env._estimator.training and all(not p.requires_grad for p in env._estimator.parameters())
                assert env._estimator_forward_count>0 and env._context_refresh_evaluations>0
                assert env._recovery_context.shape==(32,3)
                data.update(estimator_sha256=ESTIMATOR_SHA,estimator_forwards=env._estimator_forward_count,
                            certificate_queries=env._context_refresh_evaluations,
                            certificate_valid_queries=env._context_valid_evaluations,
                            context_shape=list(env._recovery_context.shape),certificate_reward_enabled=env._plane_v1_reward_enabled)
            else:
                assert policy.encoder[0].in_features==480
                assert self.alg.optimizer.state[policy.encoder[0].weight]['step']>0
                assert self.alg.optimizer.state[policy.decoder[0].weight]['step']>0
                data.update(encoder=str(policy.encoder),decoder=str(policy.decoder),
                            encoder_input=policy.encoder[0].in_features,vae_optimizer_updated=True)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            with a.output.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
            print('REWARD_SHAPING_SMOKE_PASS '+a.task,flush=True)
    g[runner_name]=SmokeRunner
    app['train']()
except BaseException:
    traceback.print_exc()
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    app['simulation_app'].close()
