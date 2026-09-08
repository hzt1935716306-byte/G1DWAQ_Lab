"""Serial fresh-training queue for the three audited reward ablations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from reward_shaping_contract import LAB,TASKS,ESTIMATOR,ESTIMATOR_SHA,file_sha

ROOT=LAB/'experiments/g1_reward_shaping_ablation_v1'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def verify_checkpoint(path,task):
    import torch
    c=torch.load(path,map_location='cpu',weights_only=False)
    assert c['iter']==9999
    p=c['training_provenance']
    assert p['task_name']==task and p['parent_training_run_id'] is None
    assert p['training_transitions']==4096*24*10000
    params=path.parent/'params'
    for kind in ('env','agent'):
        assert p[kind+'_config_sha256']==file_sha(params/(kind+'.yaml'))
    if task.startswith('g1_plane'):
        assert p['resources']['estimator']['sha256']==ESTIMATOR_SHA
    assert c['optimizer_state_dict']['state']
    assert c['model_state_dict']
    for value in c['model_state_dict'].values():
        assert torch.isfinite(value).all(), 'Nonfinite final policy weights'
    return dict(task=task,run_directory=str(path.parent),checkpoint=str(path),checkpoint_sha256=file_sha(path),
                training_seed=42,iteration=9999,completed_iterations=10000,training_transitions=p['training_transitions'],
                agent_config_sha256=p['agent_config_sha256'],env_config_sha256=p['env_config_sha256'],resume=False)


def report(state):
    audit=json.loads((ROOT/'audit/config_diff.json').read_text())
    rows=[('Original DWAQ','-2.0','-0.2','原始checkpoint保留'),
          ('DWAQ no-idle','关闭','-0.2','g1_dwaq_slope_nosys_d_matched_v2'),
          ('DWAQ no-swing','-2.0','关闭','g1_dwaq_slope_nosys_d_matched_v3'),
          ('Original Plane Context-only','未启用','未启用','原始checkpoint保留'),
          ('Plane Context + idle/swing','-0.1','-0.2','g1_plane_v1_estimator_context_no_reward_matched_v2')]
    lines=['# Reward Shaping Ablation','',f"状态：{state['status']}。训练commit：`{state['training_commit']}`。",
           '', '三个新任务从iteration 0随机初始化，seed 42；4096环境、24 steps/env、10000 iterations。',
           '只改变指定reward；尚未进行这些新模型的性能评测，不能据训练完成声称抗扰改善。',
           '', '| 模型 | idle weight | swing height weight | 任务/来源 |','|---|---:|---:|---|']
    lines += ['| '+' | '.join(row)+' |' for row in rows]
    lines += ['', 'idle阈值：command 0.2 m/s、velocity 0.1 m/s；swing目标高度0.08 m。',
              f'Estimator SHA256：`{ESTIMATOR_SHA}`。', '', '## 训练记录', '']
    for item in state['models']:
        lines += ['### '+item['task'],'', '```json',json.dumps(item,indent=2,ensure_ascii=False),'```','']
    lines += ['## 审计与完整reward表','', '原任务实例化env/agent hash保持不变。',
              '32环境、2次迭代 smoke独立保存，不加载到正式训练。',
              '完整审计：`audit/config_diff.json`；smoke：`smoke/`；固定队列：`training_plan.json`。','']
    for task,data in audit['tasks'].items():
        lines += ['### '+task,'','| Reward | Weight |','|---|---:|']
        lines += [f"| {name} | {term['weight']} |" for name,term in data['reward_table'].items() if isinstance(term,dict) and 'weight' in term]
        lines += ['']
    (ROOT/'REWARD_SHAPING_ABLATION.md').write_text('\n'.join(lines)+'\n')


def main(plan_path):
    plan=json.loads(Path(plan_path).read_text());state_path=ROOT/'training_status.json'
    lock=(ROOT/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not state_path.exists(), 'Existing training queue must be inspected; no automatic resume/restart'
    for task in TASKS:
        smoke=json.loads(Path(plan['smokes'][task]).read_text())
        assert smoke['status']=='PASS' and smoke['task']==task
        assert smoke['resume'] is False and smoke['iterations']==2 and smoke['num_envs']==32
    state={'status':'RUNNING','training_commit':plan['training_commit'],'driver_pid':os.getpid(),'models':[],'started_unix':time.time()}
    save=lambda:(write(state_path,state),report(state))
    save()
    try:
        for task in TASKS:
            assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=LAB,text=True).strip()==plan['training_commit']
            for name,sha in plan['source_sha256'].items():
                assert file_sha(LAB/name)==sha, 'Pinned source changed: '+name
            assert file_sha(ESTIMATOR)==ESTIMATOR_SHA
            assert json.loads(Path(plan['smokes'][task]).read_text())['status']=='PASS'
            output=LAB/'logs'/task
            before=set(output.glob('*'))
            if before:raise RuntimeError('Formal log root already contains runs: '+str(output))
            command=plan['commands'][task]
            logfile=ROOT/'training_logs'/f'{task}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
            with logfile.open('xb',buffering=0) as stream:
                process_env=os.environ.copy()
                process_env['PYTHONPATH']=str(LAB/'rsl_rl')+os.pathsep+str(LAB)+os.pathsep+process_env.get('PYTHONPATH','')
                proc=subprocess.Popen(command,cwd=LAB,env=process_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item={'task':task,'status':'RUNNING','pid':proc.pid,'command':command,'log':str(logfile),'started_unix':time.time()}
                state['models'].append(item);save();final=None;sealed_at=None
                while True:
                    time.sleep(20)
                    with logfile.open('rb') as f:
                        f.seek(max(0,logfile.stat().st_size-24000));tail=f.read().decode(errors='replace')
                    iterations=re.findall(r'Learning iteration\s+(\d+)/10000',tail)
                    if iterations:item['last_logged_iteration']=int(iterations[-1])
                    if 'Traceback (most recent call last)' in tail:raise RuntimeError('Training exception retained: '+str(logfile))
                    candidates=list(output.glob('*/model_9999.pt'))
                    if len(candidates)>1:raise RuntimeError('Ambiguous final checkpoint')
                    if candidates and final is None and time.time()-candidates[0].stat().st_mtime>20:
                        final=verify_checkpoint(candidates[0],task);sealed_at=time.time()
                    code=proc.poll()
                    if final is not None and (code is not None or time.time()-sealed_at>40):
                        if code is None:
                            # The verified final checkpoint proves training ended;
                            # only terminate a stalled Isaac shutdown, never learning.
                            proc.terminate()
                            try:proc.wait(timeout=20)
                            except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        item.update(final,status='COMPLETE',finished_unix=time.time(),training_commit=plan['training_commit']);save();break
                    if code is not None and not candidates:raise RuntimeError(f'Training exited {code} before final checkpoint')
                    save()
        state.update(status='COMPLETE',finished_unix=time.time());save()
    except BaseException as exc:
        state.update(status='STOPPED_ERROR',error=repr(exc),stopped_unix=time.time());save()
        if 'proc' in locals() and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True)
    main(p.parse_args().plan)
