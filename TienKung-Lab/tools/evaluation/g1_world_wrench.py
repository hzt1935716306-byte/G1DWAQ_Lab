"""Evaluation-only NumPy wrench algebra. Quaternions are WXYZ, link to world."""
from __future__ import annotations
import numpy as np


def rotation_world_from_link(quaternion):
    q=np.asarray(quaternion,dtype=np.float64)
    if q.shape[-1:]!=(4,) or not np.isfinite(q).all():raise ValueError('Finite WXYZ quaternion required')
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(abs(norm-1)>2e-6):raise ValueError('Unit WXYZ quaternion required')
    q=q/norm  # Remove only accepted float32 normalization roundoff.
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))


def _vector(value):
    a=np.asarray(value,dtype=np.float64)
    if a.shape[-1:]!=(3,) or not np.isfinite(a).all():raise ValueError('Finite three-vector required')
    return a


def rotate(rotation,vector):return np.einsum('...ij,...j->...i',rotation,_vector(vector))


def world_wrench_at_point_to_link_wrench(force_world,application_point_world,free_torque_world,
                                         link_position_world,link_quaternion_world):
    f,p,t,l=map(_vector,(force_world,application_point_world,free_torque_world,link_position_world))
    rt=np.swapaxes(rotation_world_from_link(link_quaternion_world),-1,-2)
    r=p-l;arm=np.cross(r,f);fl=rotate(rt,f)
    lhs=np.cross(rotate(rt,r),fl);rhs=rotate(rt,arm)
    scale=np.maximum(1,np.linalg.norm(r,axis=-1)*np.linalg.norm(f,axis=-1))
    if np.any(np.linalg.norm(lhs-rhs,axis=-1)>64*np.finfo(np.float64).eps*scale):
        raise ValueError('Wrench covariance identity failed')
    return fl,rotate(rt,t+arm)


def application_point_world(semantics,link_position,link_quaternion,*,declared_offset=None,
                            declared_world_point=None,whole_robot_com=None):
    """Resolve distinct point semantics; a CoM-following point is not world-fixed."""
    if semantics=='world_fixed_point':return _vector(declared_world_point)
    offset=_vector(declared_offset)
    if semantics=='link_fixed_offset':return _vector(link_position)+rotate(rotation_world_from_link(link_quaternion),offset)
    if semantics=='whole_robot_com_plus_world_offset':return _vector(whole_robot_com)+offset
    raise ValueError('Explicit application point semantics required')


def float32_close(actual,expected,*,name='wrench'):
    """64 float32 eps times vector scale: numerical tolerance, not physical slack."""
    a,e=_vector(actual),_vector(expected)
    tolerance=64*np.finfo(np.float32).eps*np.maximum(1,np.linalg.norm(e,axis=-1))
    if np.any(np.linalg.norm(a-e,axis=-1)>tolerance):raise ValueError('Applied world wrench differs from declared force/application point: '+name)


def set_world_wrenches(composer,force_world,points_world,free_torque_world,link_positions_world,
                       link_quaternions_world,body_id,device):
    """Overwrite permanent composer each substep; never delegate moment arms to it.

    Permanent composer persists until overwritten/reset. Caller clears after
    release, fall, reset and exceptions. No observations or physics are advanced.
    """
    import torch
    f,p,t,l=map(_vector,(force_world,points_world,free_torque_world,link_positions_world))
    q=np.asarray(link_quaternions_world,dtype=float);r=p-l;arm=np.cross(r,f);equivalent=t+arm
    fl,tl=world_wrench_at_point_to_link_wrench(f,p,t,l,q)
    fl=fl.astype(np.float32);tl=tl.astype(np.float32)
    rotation=rotation_world_from_link(q)
    float32_close(rotate(rotation,fl),f,name='force before composer')
    float32_close(rotate(rotation,tl),equivalent,name='torque before composer')
    composer.reset()
    try:
        composer.set_forces_and_torques(forces=torch.as_tensor(fl[:,None,:],device=device),
            torques=torch.as_tensor(tl[:,None,:],device=device),positions=None,body_ids=[body_id],is_global=False)
        cf=composer.composed_force_as_torch[:,body_id].detach().cpu().numpy().copy()
        ct=composer.composed_torque_as_torch[:,body_id].detach().cpu().numpy().copy()
        float32_close(cf,fl,name='composer local force buffer')
        float32_close(ct,tl,name='composer local torque buffer')
        fw,tw=rotate(rotation,cf),rotate(rotation,ct)
        float32_close(fw,f,name='composer reconstructed world force')
        float32_close(tw,equivalent,name='composer reconstructed world torque')
    except BaseException:
        composer.reset()
        raise
    return [dict(force_requested_world_n=f[i].tolist(),force_world_n=f[i].tolist(),
        application_point_world_m=p[i].tolist(),application_link_position_world_m=l[i].tolist(),
        application_link_quaternion_wxyz=q[i].tolist(),lever_arm_world_m=r[i].tolist(),
        arm_torque_world_nm=arm[i].tolist(),free_torque_world_nm=t[i].tolist(),
        equivalent_torque_world_nm=equivalent[i].tolist(),force_link_n=fl[i].astype(float).tolist(),
        equivalent_torque_link_nm=tl[i].astype(float).tolist(),
        composed_force_link_n=cf[i].astype(float).tolist(),composed_torque_link_nm=ct[i].astype(float).tolist(),
        composed_force_world_n=fw[i].tolist(),composed_torque_about_link_world_nm=tw[i].tolist(),
        composer_positions_none=True,composer_is_global=False) for i in range(len(f))]


def clear_wrench(composer,env_ids=None):
    composer.reset(env_ids=env_ids)
    for name in ('composed_force_as_torch','composed_torque_as_torch'):
        values=getattr(composer,name)
        if env_ids is not None:values=values[env_ids]
        if bool((values!=0).any()):raise ValueError('Residual wrench after explicit cleanup: '+name)


def verify_wrench_sample(s):
    """Independent replay of all three frame contracts; missing fields fail closed."""
    if s['composer_positions_none'] is not True or s['composer_is_global'] is not False:
        raise ValueError('Composer must receive positions=None, is_global=False')
    f,p,t,l=map(_vector,(s['force_requested_world_n'],s['application_point_world_m'],
        s['free_torque_world_nm'],s['application_link_position_world_m']))
    q=s['application_link_quaternion_wxyz'];r=p-l;arm=np.cross(r,f);tau=t+arm
    fl,tl=world_wrench_at_point_to_link_wrench(f,p,t,l,q);rot=rotation_world_from_link(q)
    for key,value in [('lever_arm_world_m',r),('arm_torque_world_nm',arm),
        ('equivalent_torque_world_nm',tau),('force_link_n',fl),('equivalent_torque_link_nm',tl),
        ('composed_force_link_n',fl),('composed_torque_link_nm',tl),
        ('composed_force_world_n',f),('composed_torque_about_link_world_nm',tau)]:
        float32_close(s[key],value,name=key)
    float32_close(rotate(rot,s['composed_force_link_n']),f,name='replayed world force')
    float32_close(rotate(rot,s['composed_torque_link_nm']),tau,name='replayed world torque')
    if s['application_point_semantics']!='whole_robot_com_plus_world_offset':
        raise ValueError('Existing robustness protocol requires CoM-following world offset')
    float32_close(p-_vector(s['whole_robot_com_world_m']),s['declared_world_offset_m'],name='declared CoM offset')
