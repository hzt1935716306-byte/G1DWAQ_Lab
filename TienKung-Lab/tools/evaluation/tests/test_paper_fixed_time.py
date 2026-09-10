import copy
import numpy as np
from g1_paper_recovery import plans,Trial,direction

def frame(t,v=.5,forces=(10,10),fell=False):
 return dict(time=t,com_velocity=np.array([v,0,0.]),root_velocity=np.array([v,0,0.]),root_position=np.array([0,0,.8]),root_quaternion_wxyz=np.array([1.,0,0,0]),roll_pitch=np.array([0.,0.]),yaw=0.,forces=np.array(forces),command=np.array([.5,0,0]),angular_velocity_world_z=0.,data_valid=True,fell=fell,timeout=False,out_of_test_area=False)
def test_plan_counts_and_cross_slope_state_pairing():
 from collections import Counter
 p=plans();assert Counter(r['suite'] for r in p)==dict(A=600,B1=2160,B2=2160,C_force=1200,C_load=300)
 assert len({r['trial_id'] for r in p})==6420
 b=[r for r in p if r['suite']=='B1' and r['repeat_id']==0];assert len({str(r['initial_joint_parameters']) for r in b})==1
 assert all(3<=r['onset']<=5 for r in p if r['suite'].startswith('B'))
def test_confirm_and_zero_steps_are_allowed():
 p=next(r for r in plans(True) if r['suite']=='B2');m=Trial(p);m.feed(frame(0));m.applied(frame(0),{})
 for i in range(1,17):m.feed(frame(i*.02))
 assert m.confirmation==.32 and m.steps==0
 for i in range(17,601):m.feed(frame(i*.02))
 assert m.result()['recovery_time']==.32 and m.result()['recovery_steps']==0

def test_fall_after_confirmation_nulls_main_result():
 p=next(r for r in plans(True) if r['suite']=='B2');m=Trial(p);m.feed(frame(0));m.applied(frame(0),{})
 for i in range(1,17):m.feed(frame(i*.02))
 m.feed(frame(.34,fell=True));r=m.result();assert r['first_confirmation']==.32 and not r['recovered'];assert r['recovery_time'] is None and r['recovery_steps'] is None

def test_two_feet_touchdown_and_release_boundary():
 p=next(r for r in plans(True) if r['suite']=='B1');m=Trial(p);m.feed(frame(0));m.applied(frame(0),{})
 for i in range(1,11):m.feed(frame(i*.02,forces=(0,0)))
 for i in range(11,26):m.feed(frame(i*.02))
 assert m.confirmation==.5 and m.steps==2

def test_pre_push_failure_no_recovery_and_no_readiness_selection():
 p=next(r for r in plans(True) if r['suite']=='B2');m=Trial(p);m.feed(frame(0,fell=True));r=m.result();assert r['pre_push_failure'];assert r['survived_after_push'] is None;assert r['recovery_steps'] is None

def test_force_tangent_and_jump_horizontal():
 p=dict(direction_deg=0,slope_deg=10);assert direction(p,True)[2]>0;assert direction(p)[2]==0;assert np.isclose(np.linalg.norm(direction(p,True)),1)

def test_velocity_oscillation_does_not_cancel():
 p=next(r for r in plans(True) if r['suite']=='B2');m=Trial(p);m.feed(frame(0));m.applied(frame(0),{})
 for i in range(1,601):m.feed(frame(i*.02,v=.5+(-1)**i*.3))
 assert not m.result()['recovered'] and m.result()['recovery_time'] is None
