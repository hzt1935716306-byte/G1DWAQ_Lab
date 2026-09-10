"""Read-only final Context-only V2 comparison with the frozen reduced benchmark."""
from pathlib import Path
from collections import Counter
import numpy as np
from g1_recovery_protocol import LAB, read_json, read_jsonl, write_json, sha256, atomic_write, RunStore
from g1_robustness_store import RobustnessStore
from g1_reduced_budget import ROOT as OLD, TARGET_COUNTS
from g1_reduced_sources import verify_reference
from g1_dwaq_ablation_report import paired_records
from g1_robustness_analysis import aggregate, curve, boundary, holm
from g1_recovery_report import markdown, word_document
from g1_reduced_report import NAMES, rate_text, fmt

ROOT=LAB/'experiments/g1_context_v2_final_eval'
NEW='context_only_v2_final'
LABELS={'ppo_plain':'PPO','dwaq':'DWAQ','rl_only':'RL-only','context_only':'Context-only',
        'context_reward':'Context+Reward','dwaq_v2':'DWAQ V2','dwaq_v3':'DWAQ V3',NEW:'Context-only V2 final'}

def read_sealed(run, robustness=False):
    run=Path(run)
    if read_json(run/'completion.json')['status']!='COMPLETE':raise ValueError('Unsealed run')
    (RobustnessStore(run) if robustness else RunStore(run)).validate(validation_level='light')
    return [read_json(p) for p in sorted((run/'trial_records').glob('*.json'))]

def build_report():
    report=ROOT/'report';report.mkdir(exist_ok=True)
    standard={};robust={m:[] for m in LABELS};sources={}
    prior=read_json(LAB/'experiments/g1_context_v2_7500_validation/report/comparison.json')
    for m in LABELS:
        if m==NEW:continue
        ref=prior['sources'][m];run=Path(ref['run'])
        if sha256(run/'completion.json')!=ref['completion_sha256']:raise ValueError('Old standard seal changed')
        standard[m]=[read_json(p) for p in sorted((run/'trial_records').glob('*.json'))]
        sources[m]={'standard':ref}
    audit=read_json(OLD/'audits/final_primary_full_audit.json')
    assert audit['status']=='AUDIT_PASSED'
    assert sha256(OLD/'primary_matched_source_index.json')==audit['primary_source_index_sha256']
    cache=set()
    for ref in read_json(OLD/'primary_matched_source_index.json')['entries']:
        robust[ref['model']].append(verify_reference(ref,cache))
    oldnew=read_json(LAB/'experiments/g1_dwaq_reward_ablation_eval_v1/report/new_sources.json')
    for ref in oldnew:
        if ref['suite']!='robustness':continue
        run=Path(ref['run']);assert sha256(run/'completion.json')==ref['completion_sha256']
        robust[ref['model']]=read_sealed(run,True);sources[ref['model']]['robustness']=ref
    sr=next((ROOT/'standard/runs').glob('*/completion.json')).parent
    rr=next((ROOT/'robustness/execution/context_only/runs').glob('*/completion.json')).parent
    standard[NEW]=read_sealed(sr);robust[NEW]=read_sealed(rr,True)
    sources[NEW]={k:{'run':str(r),'completion_sha256':sha256(r/'completion.json'),'identity':read_json(r/'identity.json')} for k,r in [('standard',sr),('robustness',rr)]}
    for m in LABELS:
        paired_records(standard[m],standard['context_only']);paired_records(robust[m],robust['context_only'])
        assert len(standard[m])==390 and dict(Counter(r['suite'] for r in robust[m]))==TARGET_COUNTS
    groups={'标准恢复':{m:[r for r in standard[m] if r['experiment']=='E1'] for m in LABELS},
            '域内抗扰':{m:[r for r in robust[m] if r['suite']!='extreme_ood'] for m in LABELS}}
    groups.update({name:{m:[r for r in robust[m] if r['suite']==suite] for m in LABELS} for suite,name in NAMES.items()})
    summaries={name:{m:aggregate(rows) for m,rows in data.items()} for name,data in groups.items()}
    blocks=[('heading',1,'Context-only V2 最终模型：八模型对比'),('text',
        '每模型同一390标准与1960缩减抗扰清单。标准8环境、抗扰64环境；公共判据candidate/unvalidated，阈值和W/H未改。'
        '旧结果只读复用；7500 checkpoint单独列作训练进展。生存、恢复条件率的实际受扰子集可能不同；不可用条件率掩盖准备失败。')]
    def table(headers,rows):blocks.append(('table',headers,[[str(v) for v in row] for row in rows]))
    for name,data in summaries.items():
        blocks.append(('heading',2,name+('（域外探索，不与域内混排）' if name==NAMES['extreme_ood'] else '')))
        table(['模型','实际施扰/指定','生存/实际施扰','持续恢复/指定','≤5落脚/指定','平均落脚','中位落脚','中位时间(s)'],
              [[LABELS[m],f"{a['actually_pushed']}/{a['designated_real']}",rate_text(a['survived']['count'],a['actually_pushed']),
                rate_text(a['recovered_sustained_and_survived']['count'],a['designated_real']),
                rate_text(a['recovery_within_5_touchdowns']['count'],a['designated_real']),fmt(a['recovery_steps']['mean']),
                fmt(a['recovery_steps']['median']),fmt(a['recovery_time']['median'])] for m,a in data.items()])
    blocks.append(('text','时间和落脚分布仅统计成功试验。持续扰动的恢复从释放/最后冲击结束开始计算；sham不进入真实扰动分母。'))
    blocks.append(('heading',2,'持续扰动期间：生存与任务质量'))
    for suite in ('force_pulse','constant_force','repeated_impulse','random_force','wrench_pulse'):
        table([NAMES[suite],'扰动期生存/实际施扰','任务条件占用率','CoM速度RMSE(m/s)'],
              [[LABELS[m],rate_text(a['disturbance_phase_survived'],a['actually_pushed']),
                fmt(a['disturbance_task_domain_occupancy']['mean']),fmt(a['disturbance_com_velocity_rmse_mps']['mean'])]
               for m,a in summaries[NAMES[suite]].items()])
    blocks.append(('text','占用率为扰动期任务窗口合格的时间比例，按试验取平均；扰动期未合格不直接等于扰动结束后恢复失败。缺失数据为N/A。'))
    blocks.append(('heading',2,'无扰动跟踪'))
    e0=[]
    for m,rows in standard.items():
        data=[r for r in rows if r['experiment']=='E0']
        means=[]
        for command in ('+x','-x','+y','-y','standing'):
            values=[r['com_xy_velocity_rmse'] for r in data if r['command_name']==command and r.get('com_xy_velocity_rmse') is not None]
            means.append(fmt(float(np.mean(values)),4) if values else 'N/A')
        e0.append([LABELS[m],sum(r['status']=='E0_COMPLETED' for r in data),*means])
    table(['模型','完成/150','前进RMSE','后退RMSE','+y RMSE','-y RMSE','站立RMSE'],e0)
    blocks.append(('text','RMSE单位m/s，为各试验质心XY速度RMSE的算术平均；E0不产生真实扰动恢复结论。'))
    blocks.append(('heading',2,'7500 → 9999：标准恢复训练进展'))
    table(['checkpoint','持续恢复/240','平均落脚','中位落脚','P90落脚','≤5落脚/240','中位时间(s)'],
          [[label,a['recovered_sustained_and_survived']['count'],fmt(a['recovery_steps']['mean']),fmt(a['recovery_steps']['median']),
            fmt(a['recovery_steps']['P90']),a['recovery_within_5_touchdowns']['count'],fmt(a['recovery_time']['median'])]
           for label,a in [('7500 intermediate',prior['summary']['context_only_v2_7500']),('9999 final',summaries['标准恢复'][NEW])]])
    import g1_robustness_analysis as analysis
    saved=analysis.PAIR_COMPARISONS
    try:
        analysis.PAIR_COMPARISONS=tuple((NEW,m) for m in ('context_only','context_reward','rl_only','dwaq'))
        tests=[t for name,data in groups.items() for t in analysis.paired_tests(data,name)]
    finally:analysis.PAIR_COMPARISONS=saved
    holm(tests)
    blocks.append(('heading',2,'域内持续恢复配对差异'))
    table(['对照','差值(百分点)','95% CI','Holm p'],
          [[LABELS[t['model_b']],fmt(100*t['effect']),str([round(100*x,2) for x in t['ci95']]),'<0.0001' if t['holm_p']<.0001 else fmt(t['holm_p'],4)]
           for t in tests if t['cohort']=='域内抗扰' and t['endpoint']=='recovered_sustained_and_survived'])
    blocks.append(('text','二元指标采用exact McNemar，连续指标只比较同一试验双方均成功者；单训练seed不代表跨训练seed优势。'))
    limits={field:{m:{str(q):boundary(curve([r for r in robust[m] if r['suite']=='extreme_main'],field),q) for q in (.9,.5)} for m in LABELS}
            for field in ('survived','recovered_sustained_and_survived')}
    def bound(v):return {'left':'<','right':'≥','none':'','gap':'缺档 ','unidentified':'N/A '}.get(v['censor'],'')+fmt(v['value_mps'])
    blocks.append(('heading',2,'域内速度扰动曲线边界'))
    table(['模型','90%生存(m/s)','90%恢复(m/s)','50%恢复(m/s)'],
          [[LABELS[m],bound(limits['survived'][m]['0.9']),bound(limits['recovered_sustained_and_survived'][m]['0.9']),bound(limits['recovered_sustained_and_survived'][m]['0.5'])] for m in LABELS])
    blocks.append(('text','边界按指定试验分母、从最低非零档连续计算，档间线性插值；准备失败会影响边界，≥3仅表示未测出更高边界。'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field,title in [('survived','Survival'),('recovered_sustained_and_survived','Sustained recovery')]:
        fig,ax=plt.subplots(figsize=(10,4))
        for m in LABELS:
            points=curve([r for r in robust[m] if r['suite']=='extreme_main'],field)
            ax.plot([v['delta_v_mps'] for v in points],[100*v['rate'] for v in points],label=LABELS[m],marker='.')
        ax.set(xlabel='Velocity jump (m/s)',ylabel='Designated-trial success (%)',ylim=(0,105),title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        path=report/(field+'.png');fig.savefig(path,dpi=150);plt.close(fig)
        blocks.append(('image',str(path),'域内固定清单曲线：包含未就绪和未恢复，排除sham。'))
    write_json(report/'summary.json',summaries);write_json(report/'sources.json',sources)
    write_json(report/'paired_statistics.json',tests);write_json(report/'envelope_boundaries.json',limits)
    atomic_write(report/'CONTEXT_V2_FINAL_COMPARISON.md',markdown(blocks,report))
    atomic_write(report/'CONTEXT_V2_FINAL_COMPARISON.docx',word_document(blocks))
    return summaries

if __name__=='__main__':build_report()
