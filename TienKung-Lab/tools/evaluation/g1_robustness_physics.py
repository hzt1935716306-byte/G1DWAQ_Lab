"""Substep external wrench adapter; native actor/estimator histories are untouched."""
from __future__ import annotations
import math
import numpy as np
from g1_robustness_protocol import waveform, waveform_sha
from g1_common_task_detector import EPS


def vector_world(plan, yaw):
    if plan['family']=='velocity_ood':return np.asarray(plan['delta_v_world_xy'],dtype=float)
    d=np.asarray(plan['push_direction_heading'],dtype=float)*plan['push_magnitude']
    if plan['vector_frame']=='heading_at_onset':
        c,s=math.cos(yaw),math.sin(yaw)
        return np.array([c*d[0]-s*d[1],s*d[0]+c*d[1]])
    return d


def force_and_arm(plan, mass, elapsed, wave=None, dt=.005):
    """Requested world force and offset from actual whole-robot CoM.

    Force is held on [onset,end), including exact non-policy-aligned 0.05 s.
    Wrench torque about whole-robot CoM equals cross(arm,force).
    """
    force=np.zeros(3);arm=np.zeros(3)
    if elapsed < -EPS or elapsed >= plan['duration_s']-EPS:return force,arm
    family=plan['family'];d=np.asarray(plan['push_direction_heading'],dtype=float)
    if family in ('force_pulse','wrench_pulse'):
        force[:2]=mass*plan['equivalent_delta_v_mps']/plan['duration_s']*d
    elif family=='constant_force':force[:2]=mass*plan['acceleration_mps2']*d
    elif family=='random_force':
        if wave is None:raise ValueError('Random force requires the pinned waveform')
        index=int(round(elapsed/dt))
        if not 0<=index<len(wave):raise ValueError('Waveform substep out of bounds')
        force[:2]=mass*wave[index]
    if family=='wrench_pulse':
        length=plan['moment_arm_m'];mode=plan['wrench_mode']
        if mode=='upper_pitch':arm[2]=length
        elif mode=='lateral_yaw':arm[:2]=plan['torque_sign']*length*np.array([d[1],-d[0]])
    return force,arm


class PhysicalInterventions:
    def __init__(self, env, identity, protocol):
        import torch
        self.torch=torch;self.env=env;self.identity=identity;self.p=protocol
        self.dt=float(env.physics_dt)
        if abs(self.dt-protocol['robustness']['physics_dt_s'])>EPS:
            raise ValueError('Unexpected physical substep duration')
        ids,names=env.robot.find_bodies('torso_link',preserve_order=True)
        if names!=['torso_link']:raise ValueError('Unique torso application link required')
        self.body_id=int(ids[0]);self.enabled=False;self.machines=[];self.waves={};self.pending=[]
        self.original_write=env.scene.write_data_to_sim
        self.original_update=env.scene.update
        env.scene.write_data_to_sim=self.write
        env.scene.update=self.update

    def begin(self,machines):
        self.machines=machines;self.base_counter=int(self.env.sim_step_counter)
        self.waves={};self.next_impact={};self.endpoint_frames={};self.enabled=True
        for i,m in enumerate(machines):
            if m.plan['family']=='random_force':
                w=waveform(m.plan,self.dt)
                if waveform_sha(w)!=m.plan['waveform_sha256']:raise ValueError('Waveform hash mismatch')
                self.waves[i]=w
            self.next_impact[i]=0

    def clear(self):
        self.enabled=False
        self.env.robot.permanent_wrench_composer.reset()

    def start(self,index,machine,frame):
        env=self.env;mass=float(env.eval_masses[index].sum().item())
        machine.start_intervention(dict(time=machine.t,duration_s=machine.plan['duration_s'],
            mass_kg=mass,heading_yaw=frame['yaw'],application_body='torso_link',
            vector_frame=machine.plan['vector_frame'],force_application=self.p['robustness']['force_application']))
        if machine.plan['family'] in ('velocity_jump','velocity_ood'):
            self.impact(index,machine,vector_world(machine.plan,frame['yaw']),machine.t,0)
        elif machine.plan['family']=='repeated_impulse':
            self.impact(index,machine,np.asarray(machine.plan['impacts'][0]['delta_v_world_xy']),machine.t,0)
            self.next_impact[index]=1

    def impact(self,index,machine,delta,time,impact_index):
        torch=self.torch;env=self.env
        before=env.robot.data.root_vel_w[index].detach().cpu().numpy().copy()
        expected=before.copy();expected[:2]+=delta
        ids=torch.tensor([index],device=env.device)
        env.robot.write_root_velocity_to_sim(torch.tensor(expected[None],dtype=torch.float32,device=env.device),env_ids=ids)
        actual=env.robot.data.root_vel_w[index].detach().cpu().numpy().copy()
        if not np.allclose(actual,expected,atol=5e-6,rtol=0):raise ValueError('Actual velocity jump differs from paired input')
        notified=False
        if self.identity['method'].startswith('context') and not bool(env.recovery_active[index]):
            env._on_curriculum_push(ids,torch.tensor(np.asarray(delta)[None],dtype=torch.float32,device=env.device),torch.zeros(1,dtype=torch.long,device=env.device))
            notified=True
            if not np.array_equal(env.robot.data.root_vel_w[index].detach().cpu().numpy(),actual):
                raise ValueError('Native notification caused a second physical jump')
        # Native overlapping episodes are prohibited. Later impacts do not clear
        # or restart an active native episode; continuous context refresh is native.
        machine.add_impact(dict(time=float(time),impact_index=impact_index,
            delta_v_world_xy=np.asarray(delta).tolist(),velocity_before=before.tolist(),velocity_after=actual.tolist(),
            native_notification=notified))

    def write(self):
        env=self.env;torch=self.torch
        if not self.enabled:return self.original_write()
        before_history=env.eval_history_calls
        time=(int(env.sim_step_counter)-self.base_counter-1)*self.dt
        forces=np.zeros((env.num_envs,1,3),dtype=np.float32)
        points=np.zeros_like(forces);self.pending=[]
        com=((env.eval_masses[...,None]*env.robot.data.body_com_pos_w).sum(1)/env.eval_masses.sum(1,keepdim=True)).detach().cpu().numpy()
        link=env.robot.data.body_link_pos_w[:,self.body_id].detach().cpu().numpy()
        for i,m in enumerate(self.machines):
            if m.status or m.onset is None or i in self.endpoint_frames:continue
            plan=m.plan;elapsed=time-m.onset;mass=float(env.eval_masses[i].sum().item())
            if plan['family']=='repeated_impulse' and m.first_terminal_physics_time is None:
                j=self.next_impact[i]
                if j<len(plan['impacts']) and abs(elapsed-plan['impacts'][j]['offset_s'])<=EPS:
                    self.impact(i,m,np.asarray(plan['impacts'][j]['delta_v_world_xy']),time,j);self.next_impact[i]+=1
                elif j<len(plan['impacts']) and elapsed>plan['impacts'][j]['offset_s']+EPS:
                    raise ValueError('Missed scheduled repeated impact')
            force,arm=force_and_arm(plan,mass,elapsed,self.waves.get(i),self.dt)
            if m.first_terminal_physics_time is not None:force[:]=0
            forces[i,0]=force;points[i,0]=com[i]+arm
            self.pending.append((i,dict(time=time+self.dt,start_time=time,dt_s=self.dt,
                force_active=bool(np.any(force)),force_world_n=forces[i,0].astype(float).tolist(),
                application_point_world_m=points[i,0].astype(float).tolist(),whole_robot_com_world_m=com[i].astype(float).tolist(),
                application_link_position_world_m=link[i].astype(float).tolist(),
                torque_about_com_world_nm=np.cross(points[i,0]-com[i],forces[i,0]).astype(float).tolist(),
                force_requested_world_n=force.tolist(),mass_kg=mass)))
        composer=env.robot.permanent_wrench_composer
        # Global wrenches must be re-expressed using the current link orientation.
        # The installed composer caches poses until reset; reset every substep.
        composer.reset()
        composer.set_forces_and_torques(forces=torch.tensor(forces,device=env.device),
            torques=torch.zeros_like(torch.tensor(forces,device=env.device)),
            positions=torch.tensor(points,device=env.device),body_ids=[self.body_id],is_global=True)
        from isaaclab.utils.math import quat_apply
        q=env.robot.data.body_link_quat_w[:,self.body_id]
        actual_f=quat_apply(q,composer.composed_force_as_torch[:,self.body_id]).detach().cpu().numpy()
        actual_tau=quat_apply(q,composer.composed_torque_as_torch[:,self.body_id]).detach().cpu().numpy()
        for i,sample in self.pending:
            expected_tau=np.cross(points[i,0]-link[i],forces[i,0])
            tolerance=max(.002,float(np.linalg.norm(forces[i,0]))*2e-4)
            if not np.allclose(actual_f[i],forces[i,0],atol=tolerance,rtol=0) or not np.allclose(actual_tau[i],expected_tau,atol=tolerance,rtol=0):
                raise ValueError('Applied world wrench differs from declared force/application point')
            sample['composed_force_world_n']=actual_f[i].astype(float).tolist()
            sample['composed_torque_about_link_world_nm']=actual_tau[i].astype(float).tolist()
        result=self.original_write()
        if env.eval_history_calls!=before_history:raise ValueError('External wrench advanced actor history')
        return result

    def update(self,*args,**kwargs):
        result=self.original_update(*args,**kwargs)
        if self.enabled and self.pending:
            env=self.env;torch=self.torch
            # Read the same native torso contact history without requesting observations.
            history=env.contact_sensor.data.net_forces_w_history[:, :, env.termination_contact_cfg.body_ids]
            terminal=torch.any(torch.max(torch.linalg.vector_norm(history,dim=-1),dim=1)[0]>1.,dim=1).detach().cpu().numpy()
            for i,sample in self.pending:
                sample['terminal']=bool(terminal[i]);m=self.machines[i]
                # Archive every disturbance substep plus the first zero-force
                # interval and any later first terminal event. Post-release GT
                # trajectories remain complete at native 50 Hz.
                if sample['start_time']<=m.release_time+EPS or (terminal[i] and m.first_terminal_physics_time is None):
                    m.add_physics(sample)
            endpoints=[i for i,m in enumerate(self.machines) if m.onset is not None and not m.status
                and abs((int(env.sim_step_counter)-self.base_counter)*self.dt-(m.release_time+10))<=EPS
                and abs((m.release_time+10)/env.step_dt-round((m.release_time+10)/env.step_dt))>EPS]
            if endpoints:
                fell=torch.as_tensor(terminal,device=env.device)
                snapshots=env.physical_snapshot(fell,torch.zeros(env.num_envs,dtype=torch.bool,device=env.device))
                for i in endpoints:
                    self.endpoint_frames[i]={**snapshots[i],'time':self.machines[i].release_time+10,
                        'plane':env.plane_diagnostics(i) if not terminal[i] else None}
            self.pending=[]
        return result
