"""Independent fixed-time paper experiment; CPU plan and confirmation endpoint.

Does not change common_task_window_v1 or any native task/reward.
"""
from pathlib import Path
import argparse,copy,json,math,random
import numpy as np
from g1_recovery_protocol import LAB,read_json,write_json,sha256,inspect_checkpoint,load_yaml
from g1_common_task_detector import PhysicalTouchdowns
ROOT=LAB/'experiments/g1_paper_fixed_time_v1'
MODELS={'ppo':'ppo_plain','dwaq':'dwaq_v3','ours':'context_only_v2_final'}
VERSION='paper_fixed_time_confirmation_v1_candidate'


def direction(plan,tangent=False):
    a=math.radians(plan['direction_deg']);v=np.array([math.cos(a),math.sin(a),0.])
    if tangent:
        s=math.radians(plan['slope_deg']);v=np.array([v[0]*math.cos(s),v[1],v[0]*math.sin(s)])
    return v


def plans(pilot=False):
    rows=[]
    for suite,repeats in [('A',1 if pilot else 100),('B1',1 if pilot else 30),('B2',1 if pilot else 30),('C_force',1 if pilot else 50),('C_load',1 if pilot else 50)]:
        for slope in [-10,0,10]:
            for speed in ([.5,1.] if suite in ('A','B1','B2') else [.5]):
                for angle in ([0,90,180,270] if suite in ('B1','B2','C_force') else [0]):
                    for strength in ([.3,.6,.9] if suite in ('B1','B2') else [.05,.1] if suite.startswith('C') else [0]):
                        for rep in range(repeats):
                            # Same reset across slope/speed/family for a repeat, independent of outcome.
                            seed=26101000+rep+(10000 if pilot else 0);rng=random.Random(seed)
                            onset_rng=random.Random(seed+442)
                            sampled=onset_rng.uniform(3,5);onset=math.ceil(sampled/.02-1e-9)*.02
                            if suite.startswith('C'):onset=3.
                            tid=f'{suite}_s{slope:+d}_v{speed:g}_d{angle}_a{strength:g}_r{rep:03d}'
                            rows.append(dict(trial_id=tid,suite=suite,slope_deg=slope,command_name='+x',command_vx=speed,command_vy=0.,command_yaw=0.,direction_deg=angle,strength=strength,reset_seed=seed,repeat_id=rep,sampled_onset=sampled if suite.startswith('B') else None,onset=onset if suite!='A' and suite!='C_load' else None,release=onset+(.2 if suite=='B1' else 0) if suite.startswith('B') else None,episode_s=12.,
                                initial_pose_parameters=dict(x=rng.uniform(-.2,.2),y=rng.uniform(-.2,.2),z=0.,roll=0.,pitch=0.,yaw=0.),initial_velocity_parameters=dict(x=0.,y=0.,z=0.,roll=0.,pitch=0.,yaw=0.),initial_joint_parameters=dict(position_scale=[rng.uniform(.9,1.1) for _ in range(29)],velocity=[0.]*29)))
    return rows


class PhysicsImpulseLedger:
    """Integrate only completed simulation steps, never reset/forward writes."""
    def __init__(self, num_envs, dt, initial_counter):
        self.dt=dt;self.last_counter=initial_counter
        self.pending=np.zeros((num_envs,3));self.impulse=np.zeros_like(self.pending)
        self.steps=np.zeros(num_envs,dtype=int)
    def write(self, forces):
        self.pending[:]=forces
    def advance(self, counter):
        if counter!=self.last_counter+1:
            raise ValueError('Impulse ledger requires exactly one callback per physical step')
        self.impulse+=self.pending*self.dt
        self.steps+=np.any(self.pending!=0,axis=1)
        self.last_counter=counter


class Trial:
    def __init__(self,plan):
        self.plan=plan;self.contact=PhysicalTouchdowns(5.,3.,2,.08);self.frames=[];self.events=[];self.status=None;self.onset=None;self.release=None;self.hold_start=None;self.confirmation=None;self.steps=None;self.pre_state=None;self.count_start=None;self.last_t=None
    def applied(self,frame,metadata):
        if self.onset is not None:raise ValueError('Duplicate intervention')
        self.onset=float(frame['time']);self.release=self.onset+(.2 if self.plan['suite']=='B1' else 0.)
        self.pre_state={k:np.asarray(frame[k]).tolist() for k in ['com_velocity','root_velocity','root_position','roll_pitch','yaw','forces','root_quaternion_wxyz']}
        self.pre_state['contact']=self.contact.contact.tolist();self.events.append(dict(event='intervention',time=self.onset,**metadata))
    def feed(self,f):
        if self.status:return False
        t=float(f['time'])
        if self.last_t is not None and abs(t-self.last_t-.02)>1e-7:raise ValueError('Missing/nonuniform control sample')
        self.last_t=t
        for k in ['com_velocity','roll_pitch','command','angular_velocity_world_z','forces']:
            if not np.isfinite(f[k]).all():raise ValueError('Invalid GT '+k)
        if not f['data_valid']:raise ValueError('Invalid physical measurement')
        physical,_=self.contact.update(t,f['forces']);self.events.extend(physical)
        f=dict(f);f['physical_contact']=self.contact.contact.copy();f['touchdown_flags']=np.array([any(e['foot']==s for e in physical) for s in ['left','right']]);self.frames.append(f)
        if f['fell'] or f['out_of_test_area'] or f['timeout']:
            self.status='FELL' if f['fell'] else 'OUT_OF_TEST_AREA' if f['out_of_test_area'] else 'ABNORMAL_TIMEOUT';return False
        if self.onset is not None and self.plan['suite'] in ('B1','B2') and t>=self.release-1e-9:
            # Count events strictly after release, through the confirmation sample.
            error=float(np.linalg.norm(np.asarray(f['com_velocity'])[:2]-np.asarray(f['command'])[:2]))
            good=error<=.2 and bool(np.all(np.abs(f['roll_pitch'])<=math.radians(15)))
            if good:
                if self.hold_start is None:self.hold_start=t
                if self.confirmation is None and t-self.hold_start>=.3-1e-9 and t<=self.release+5+1e-9:
                    self.confirmation=t;self.steps=sum(e['event']=='physical_touchdown' and self.release+1e-9<e['time']<=t+1e-9 for e in self.events)
                    self.events.append(dict(event='recovery_confirmed',time=t,steps=self.steps))
            else:self.hold_start=None
        if t>=12-1e-9:self.status='COMPLETED';return False
        return self.onset is None and self.plan['suite'] in ('B1','B2','C_force') and t>=self.plan['onset']-1e-9
    def result(self):
        success=self.status=='COMPLETED';b=self.plan['suite'] in ('B1','B2');recovered=b and self.confirmation is not None and success
        ts=np.array([f['time'] for f in self.frames]);mask=ts>=2-1e-9;t=ts[mask]
        error=np.array([np.asarray(f['com_velocity'])[:2]-np.asarray(f['command'])[:2] for f in self.frames])[mask];yaw=np.array([f['angular_velocity_world_z'] for f in self.frames])[mask]
        duration=float(t[-1]-t[0]) if len(t)>1 else 0.
        def rms(v):return float(np.sqrt(np.trapz(v,t)/duration)) if duration else None
        return {**self.plan,'metrics_version':VERSION,'status':self.status,'actually_perturbed':self.onset is not None,'pre_push_failure':b and self.onset is None,'fell':self.status=='FELL','abnormal_termination':self.status not in ('COMPLETED','FELL'),'episode_success':success,'survived_after_push':success if self.onset is not None else None,'recovered':recovered if b else None,'recovery_timeout':b and self.onset is not None and self.confirmation is None and success,'recovery_time':self.confirmation-self.release if recovered else None,'recovery_steps':self.steps if recovered else None,'first_confirmation':self.confirmation,'first_confirmation_steps':self.steps,'within_5_touchdowns':recovered and self.steps<=5,'actual_onset':self.onset,'actual_release':self.release if b else None,'pre_push_state':self.pre_state,'tracking_duration_s':duration,'velocity_rmse_mps':rms((error**2).sum(1)),'yaw_rate_rmse_radps':rms(yaw**2),'observation_end':float(ts[-1])}


def prepare():
    from collections import Counter
    ROOT.mkdir(parents=True,exist_ok=True);source=read_json(LAB/'experiments/g1_context_v2_final_eval/report/sources.json');identities={};inventory={}
    for name,old in MODELS.items():
        r=Path(source[old]['standard']['run']);previous=read_json(r/'identity.json');cp=LAB/previous['checkpoint_path']
        i=inspect_checkpoint(previous['task_name'],cp,name,'final',estimator=(LAB/previous['estimator_path']) if previous.get('estimator_path') else None)
        if i['checkpoint_sha256']!=previous['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        identities[name]=i
        inventory[name]=[str(p.relative_to(LAB)) for p in cp.parent.parent.glob('*/model_9999.pt')]
    protocol={'version':VERSION,'status':'candidate_unvalidated','slopes_deg':[-10,0,10],'commands_mps':[.5,1.],'episode_s':12,'tracking_start_s':2,'recovery_velocity_error_mps':.2,'roll_pitch_abs_deg':15,'confirmation_s':.3,'recovery_deadline_s':5,'contact_on_n':5,'contact_off_n':3,'contact_stable_frames':2,'contact_debounce_s':.08,'readiness_gate':None,'onset':'uniform3to5 rounded up to 0.02s, shared across all models','force_frame':'initial heading slope tangent, torso CoM point; magnitude m*dv/0.2','velocity_jump_frame':'initial heading horizontal world XY, angular unchanged','payload':'added solid-sphere equivalent r=0.1m centered at torso rigid-body CoM, inertia increment 2/5*m*r² I; fixed from reset','num_envs':64,'seeds':inventory,'D':'SKIPPED: no matching reward-on V2 checkpoint found; no V1 substitution','formal_counts_per_model':dict(Counter(p['suite'] for p in plans()))}
    for file,obj in [('protocol.json',protocol),('checkpoint_identities.json',identities),('pilot_manifest.json',plans(True)),('formal_manifest.json',plans(False))]:
        f=ROOT/file
        if f.exists() and read_json(f)!=obj:raise ValueError('Prepared inputs already frozen and differ: '+file)
        write_json(f,obj)
    print(json.dumps(protocol,indent=2))

if __name__=='__main__':prepare()
