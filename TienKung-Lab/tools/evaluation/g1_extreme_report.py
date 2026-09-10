"""Small, read-only reports from sealed reduced-sample extreme experiments."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from g1_recovery_protocol import read_json,write_json,sha256,digest
from g1_paper_execution import verify_run
from g1_extreme_plan import MODELS,cell,key,curves
from g1_extreme_d50 import RecoveryPoint,summarize_d50
NAMES={'ppo':'PPO','dwaq_v2':'DWAQ V2','dwaq':'DWAQ V3','ours':'Ours (Context-only V2)'}


def load(root):
    """Admit only whole completed waves, verify records, never replay traces."""
    identity=read_json(root/'identity.json');data=[];seen=set()
    for seal in sorted((root/'waves').glob('*/WAVE_COMPLETE.json')):
        wave=seal.parent;meta=read_json(seal);models=read_json(wave/'wave_models.json')
        if models!=meta['models']:raise ValueError('Wave model selection changed')
        phase='explore' if '_explore_' in wave.name else 'baseline' if '_baseline_' in wave.name else 'formal'
        if models!=(['ppo'] if phase=='explore' else list(MODELS)):raise ValueError('Unpaired wave admission')
        plan=read_json(wave/'formal_frozen_manifest.json');expected_ids={r['trial_id'] for r in plan}
        if len(expected_ids)!=len(plan):raise ValueError('Duplicate plan IDs')
        for model in models:
            run=wave/'formal'/model/'all/attempt-001'
            job=dict(runtime=identity['runtime'],checkpoint_sha256=identity['checkpoint_sha256'][model],manifest_hash=digest(plan),protocol_sha256=identity['protocol_sha256'],num_envs=64,episodes=len(plan))
            c=verify_run(run,job,True)
            rr=[read_json(run/'records'/f) for f in c['records_sha256']]
            if {r['trial_id'] for r in rr}!=expected_ids:raise ValueError('Record IDs differ from paired plan')
            indexed={r['trial_id']:r for r in plan}
            for r in rr:
                if any(r.get(k)!=v for k,v in indexed[r['trial_id']].items()):raise ValueError('Realized record plan differs')
                if r['checkpoint_sha256']!=job['checkpoint_sha256']:raise ValueError('Record checkpoint differs')
                token=(model,r['trial_id'])
                if token in seen:raise ValueError('Duplicate episode across waves')
                seen.add(token);data.append(dict(r,model=model,phase=phase,group=wave.name.split('_')[0],run=str(run)))
    return data


def summary(rows):
    pushed=[r for r in rows if r['actually_perturbed']];good=[r for r in pushed if r['recovered']]
    def dist(field):
        a=[r[field] for r in good]
        if any(v is None for v in a):raise ValueError('Recovered episode missing endpoint')
        return {field+'_'+k:float(fn(a)) if a else None for k,fn in [('mean',np.mean),('median',np.median),('q25',lambda a:np.percentile(a,25)),('q75',lambda a:np.percentile(a,75))]}
    n=len(pushed);k=len(good);ci=RecoveryPoint(1.,len(rows),n,k).interval
    rmse=[r['velocity_rmse_mps'] for r in rows if r['velocity_rmse_mps'] is not None]
    return dict(designated=len(rows),pushed=n,pre_push_failure=sum(bool(r['pre_push_failure']) for r in rows),pre_onset_fall=sum(r['fell'] and not r['actually_perturbed'] for r in rows),episode_success=sum(r['episode_success'] for r in rows),post_push_fall=sum(r['fell'] for r in pushed),out_of_area=sum(r['status']=='OUT_OF_TEST_AREA' for r in rows),survived=sum(bool(r['survived_after_push']) for r in pushed),recovered=k,within5=sum(r['within_5_touchdowns'] for r in pushed),recovery_rate=k/n if n else None,survival_rate=sum(bool(r['survived_after_push']) for r in pushed)/n if n else None,recovery_ci_low=ci[0] if ci else None,recovery_ci_high=ci[1] if ci else None,velocity_rmse_mean=float(np.mean(rmse)) if rmse else None,tracking_duration_mean=float(np.mean([r['tracking_duration_s'] for r in rows])) if rows else None,**dist('recovery_time'),**dist('recovery_steps'))


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(root):
    root=Path(root);data=load(root);out=root/'report';out.mkdir(exist_ok=True)
    grouped=defaultdict(list)
    for r in data:grouped[(r['phase'],r['model'],cell(r),r['strength'])].append(r)
    cells=[]
    for (phase,m,c,a),rr in sorted(grouped.items()):
        cells.append(dict(phase=phase,model=m,slope_deg=c[0],command_mps=c[1],direction_deg=c[2],family=c[3],magnitude_mps=a,**summary(rr)))
    write_csv(out/'condition_summary.csv',cells)
    d50=[]
    for phase in ['explore','formal']:
        for m in MODELS:
            for c,points in sorted(curves([r for r in data if r['phase']==phase and r['model']==m and r['suite'] in ('B1','B2')]).items()):
                s=summarize_d50(points)
                d50.append(dict(phase=phase,model=m,slope_deg=c[0],command_mps=c[1],direction_deg=c[2],family=c[3],status=s['status'],d50_mps=s.get('d50'),bound_mps=s.get('bound'),reason=s.get('reason'),min_tested=points[0].magnitude,max_tested=points[-1].magnitude))
    write_csv(out/'D50.csv',d50);write_json(out/'summary.json',dict(cells=cells,d50=d50,completed_episodes=len(data),completed_groups=[g for g in ['original','extended'] if (root/(g+'_COMPLETE.json')).exists()]))
    # Use a single shared actual tested level nearest PPO's formal crossing.
    near=[]
    for d in d50:
        if d['phase']!='formal' or d['model']!='ppo' or d['d50_mps'] is None:continue
        matched=[c for c in cells if c['phase']=='formal' and all(c[k]==d[k] for k in ['slope_deg','command_mps','direction_deg','family'])]
        level=min({c['magnitude_mps'] for c in matched},key=lambda x:(abs(x-d['d50_mps']),x))
        near.extend(c for c in matched if c['magnitude_mps']==level)
    write_csv(out/'near_PPO_D50.csv',near)
    def rate(k,n):return f'{k}/{n} ({100*k/n:.1f}%)' if n else 'N/A'
    def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
    lines=['# 四模型极限扰动：缩减样本对比','',
        'PPO 单独探索；四模型仅在冻结的共同强度点比较。每格 20 次，独立于探索的新种子。没有自动补样本。',
        '恢复判据未变：CoM heading XY 速度误差 ≤0.2 m/s，世界绝对 roll/pitch ≤15°，连续0.3 s确认，扰动结束后5 s内确认且12 s结束前未跌倒或异常终止。时间与落脚次数均截至确认时刻。',
        'T/K 只统计成功恢复样本；恢复和生存分母为真正受扰数，受扰前失败单列。无受扰数据为 N/A。独立物理落脚、不强制交替、允许0步。',
        '每个模型仅训练种子42。曲线95% Wilson区间仅描述评测样本不确定性，不代表训练种子间变化。D50 为粗略区间内估计，不远距离外推；不明确的曲线保持待确认。',
        '±20° 超出配置训练范围，为未见坡度泛化；±15° 是配置边界。重建训练地形范围约 −14.9325° 至 +14.9454°，不代表已证明训练访问过每个坡度。世界姿态15°门限不随坡面调整，较大坡度结果包含该固定任务条件的限制。',
        '新实验独立目录、共同新物理环境；不合并旧实验计数。旧 −10/0/+10° 无扰动结果保留于原报告；新增四坡度另做无扰动基线。',
        f'已完成并核验的 episode：{len(data)}；完整阶段：'+(', '.join(g for g in ['original','extended'] if (root/(g+'_COMPLETE.json')).exists()) or '尚未完成，以下为进度结果。'),'']
    def table(headers,rr):
        lines.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        lines.extend('| '+' | '.join(str(x) for x in r)+' |' for r in rr);lines.append('')
    lines+=['## 四模型固定强度对比','每个条件的强度由 PPO 探索确定；不同坡度的强度可能不同，不能据此把混合恢复率当作坡度难度曲线。','']
    for group in ['original','extended']:
        for suite in ['B1','B2']:
            rr=[r for r in data if r['phase']=='formal' and r['group']==group and r['suite']==suite]
            if not rr:continue
            lines+=[f'### {group} / {suite}','']
            table(['模型','指定/受扰前失败','恢复/受扰','存活/受扰','T中位数 [Q1,Q3] s','K平均/中位数','≤5落脚/受扰'],[[NAMES[m],f"{q['designated']}/{q['pre_push_failure']}",rate(q['recovered'],q['pushed']),rate(q['survived'],q['pushed']),f"{fmt(q['recovery_time_median'])} [{fmt(q['recovery_time_q25'])}, {fmt(q['recovery_time_q75'])}]",f"{fmt(q['recovery_steps_mean'])}/{fmt(q['recovery_steps_median'])}",rate(q['within5'],q['pushed'])] for m in MODELS for q in [summary([r for r in rr if r['model']==m])]])
    baseline=[c for c in cells if c['phase']=='baseline']
    if baseline:
        lines+=['## 新增坡度无扰动行走','失败在这里表示基础行走受限，不能归为抗扰恢复失败。','']
        table(['模型','坡度°','命令m/s','完整行走','速度RMSE m/s'],[[NAMES[c['model']],c['slope_deg'],c['command_mps'],rate(c['episode_success'],c['designated']),fmt(c['velocity_rmse_mean'])] for c in baseline])
    formal=[d for d in d50 if d['phase']=='formal']
    if formal:
        lines+=['## D50：按相同坡度、速度、方向分别比较','完整数值见 [D50.csv](D50.csv)，同强度对比见 [near_PPO_D50.csv](near_PPO_D50.csv)。大于/小于号只表示当前测试范围边界，不是实际极限。','']
        table(['模型','坡度°','速度m/s','方向°','类型','D50 m/s','状态'],[[NAMES[d['model']],d['slope_deg'],d['command_mps'],d['direction_deg'],d['family'],fmt(d['d50_mps']) if d['d50_mps'] is not None else ('>' if d['status']=='above_tested_range' else '<' if d['status']=='below_tested_range' else '')+fmt(d['bound_mps']),d['status']] for d in formal])
    lines+=['## 复现与原始数据','`../identity.json` 绑定运行commit、评测源码、协议和checkpoint SHA；`../waves/` 保留每条记录、trace、event及completion。`../paired_physics_64.json` 为四模型各波次共同物理契约。',
            '强度扫描最多16档作为执行上限；若仍未降至30%，记为未找到上限。其他方法只测 PPO 决定的范围，不宣称已找到它们的真实最大值。',
            '详细逐条件分母：[condition_summary.csv](condition_summary.csv)。探索结果与正式小样本结果在所有文件中分开。','']
    # Plot only paired formal waves. Exploration never joins four-model curves.
    if formal:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for slope in sorted({d['slope_deg'] for d in formal}):
            for family in ['B1','B2']:
                selected=[c for c in cells if c['phase']=='formal' and c['slope_deg']==slope and c['family']==family]
                if not selected:continue
                fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
                for i,v in enumerate([.5,1.]):
                    for j,d in enumerate([0,90,180,270]):
                        ax=axes[i,j]
                        for m in MODELS:
                            pts=sorted([c for c in selected if c['model']==m and c['command_mps']==v and c['direction_deg']==d],key=lambda c:c['magnitude_mps'])
                            if not pts:continue
                            val=lambda k:[p[k] if p[k] is not None else np.nan for p in pts]
                            x=val('magnitude_mps');line=ax.plot(x,val('recovery_rate'),'o-',label=NAMES[m])[0]
                            ax.fill_between(x,val('recovery_ci_low'),val('recovery_ci_high'),alpha=.12,color=line.get_color())
                        ax.axhline(.5,color='gray',ls='--');ax.set(title=f'v={v} m/s, direction={d} deg',xlabel='Equivalent delta-v (m/s)' if family=='B1' else 'Velocity jump (m/s)',ylabel='Recovered / actually pushed',ylim=(0,1.05))
                handles,labels=axes[0,0].get_legend_handles_labels()
                if handles:fig.legend(handles,labels,loc='upper center',ncol=4)
                fig.suptitle(f'{family}, slope {slope:+g} deg'+(' (OOD)' if abs(slope)==20 else ''),y=1.02)
                fig.tight_layout(rect=(0,0,1,.95));name=f'curve_{family}_s{slope:+g}.png';fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig)
                lines.append(f'![{family} {slope:+g}°]({name})')
    (out/'COMPARISON_ZH.md').write_text('\n'.join(lines)+'\n')
    return dict(episodes=len(data),formal_episodes=sum(r['phase']=='formal' for r in data))
