"""Two fresh native PPO iterations plus bounded, separately labelled event probes."""
import argparse
import inspect
import json
from pathlib import Path
import runpy
import sys
import traceback
from plane_reward_v2_contract import TASK,ROOT,audit
from reward_shaping_contract import LAB,ESTIMATOR,ESTIMATOR_SHA,file_sha,canonical

sys.path.insert(0,str(LAB/'rsl_rl'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.argv=['legged_lab/scripts/train.py','--task',TASK,'--num_envs','32','--max_iterations','2','--seed','42',
          '--headless','--device','cuda:0','--estimator_checkpoint_path',str(ESTIMATOR),
          '--experiment_name',TASK+'_smoke','--run_name','SMOKE_ONLY_NOT_FORMAL']
app=runpy.run_path(str(LAB/'legged_lab/scripts/train.py'),run_name='smoke_launcher')
g=app['train'].__globals__
import torch

def finite(value):assert torch.isfinite(value).all(), 'NaN/Inf in smoke'

try:
    result=audit(g['task_registry'],json.loads((LAB/'tools/recovery/configs/plane_reward_v2_parents_d9c034d.json').read_text()))
    (ROOT/'config_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    Native=g['OnPolicyRunner'];assert Path(inspect.getfile(Native)).resolve().is_relative_to(LAB/'rsl_rl')
    class Smoke(Native):
        def load(self,*args,**kwargs):raise RuntimeError('Smoke policy loading forbidden')
        def learn(self,num_learning_iterations,**kwargs):
            assert num_learning_iterations==2 and self.current_learning_iteration==0 and not self.alg.optimizer.state
            assert self.training_provenance['parent_training_run_id'] is None
            env=self.env;policy=self.alg.policy
            assert (policy.actor[0].in_features,policy.critic[0].in_features,env.num_actions)==(483,1010,29)
            assert env._plane_v1_reward_enabled
            counts=dict(idle=0,swing=0,recoverability=0);peaks={k:0. for k in counts}
            native_step=env.step
            def measured_step(actions):
                values=native_step(actions)
                obs,reward,_,extras=values
                finite(obs);finite(reward);finite(extras['observations']['critic'])
                terms={'recoverability':env._v1_event_total}
                for key,name in [('idle','idle_penalty'),('swing','feet_swing_height')]:
                    term=env.reward_manager.get_term_cfg(name)
                    terms[key]=term.func(env,**term.params)*term.weight
                for key,value in terms.items():
                    finite(value);counts[key]+=int((value!=0).sum());peaks[key]=max(peaks[key],float(value.abs().max()))
                return values
            env.step=measured_step
            super().learn(num_learning_iterations=num_learning_iterations,**kwargs)
            assert self.current_learning_iteration==1 and self.alg.optimizer.state
            for value in policy.parameters():finite(value)
            for state in self.alg.optimizer.state.values():
                for value in state.values():
                    if isinstance(value,torch.Tensor):finite(value)
            # Controlled idle probe after training, restored before native rollouts.
            command=env.command_generator.command.clone();velocity=env.robot.data.root_vel_w.clone()
            try:
                env.command_generator.command[:,:2]=0.;env.command_generator.command[:,0]=.4
                still=velocity.clone();still[:,:2]=0.;env.robot.write_root_velocity_to_sim(still)
                term=env.reward_manager.get_term_cfg('idle_penalty')
                idle=term.func(env,**term.params)*term.weight
                assert torch.allclose(idle,torch.full_like(idle,-.1))
            finally:
                env.command_generator.command[:]=command;env.robot.write_root_velocity_to_sim(velocity)
            event=env.event_manager.get_term_cfg('push_robot')
            before=env.robot.data.root_vel_w[:,:2].clone()
            event.func(env,torch.arange(32,device=env.device),**event.params)
            delta=env.robot.data.root_vel_w[:,:2]-before
            assert delta.abs().max()<=1.0001 and torch.any(delta!=0)
            infer=self.get_inference_policy(device=env.device);obs,_=env.get_observations()
            with torch.inference_mode():
                for step in range(750):
                    actions=infer(obs);finite(actions);obs,_,_,_=env.step(actions)
                    if step%100==0:print('EVENT_PROBE',step,counts,flush=True)
                    if counts['recoverability'] and counts['swing']:break
            assert counts['recoverability']>0, 'No real recoverability event in bounded smoke'
            assert counts['swing']>0, 'No nonzero swing-height reward'
            assert file_sha(ESTIMATOR)==ESTIMATOR_SHA
            assert not env._estimator.training and all(not x.requires_grad for x in env._estimator.parameters())
            assert env._estimator_forward_count>0 and env._context_refresh_evaluations>0
            assert env._recovery_context.shape==(32,3)
            data=dict(status='PASS',role='SMOKE_ONLY_NOT_FORMAL',task=TASK,resume=False,start_iteration=0,iterations=2,
                num_envs=32,actor=483,critic=1010,action=29,run_directory=str(self.log_dir),
                config_audit=result,estimator_sha256=ESTIMATOR_SHA,estimator_forwards=env._estimator_forward_count,
                certificate_queries=env._context_refresh_evaluations,certificate_valid_queries=env._context_valid_evaluations,
                real_step_nonzero_counts=counts,real_step_absolute_peaks=peaks,idle_controlled_probe=idle.cpu().tolist(),
                post_training_probe_steps=step+1,terrain=env.cfg.scene.terrain_type,
                push_config=canonical(env.cfg.domain_rand.events.push_robot),finite_parameters_and_optimizer=True)
            with a.output.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
            print('PLANE_REWARD_V2_SMOKE_PASS',flush=True)
    g['OnPolicyRunner']=Smoke
    app['train']()
except BaseException:
    traceback.print_exc();raise
finally:
    app['simulation_app'].close()
