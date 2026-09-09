"""Seven-model comparison from sealed results; original reports stay unchanged."""
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from g1_recovery_protocol import LAB,read_json,read_jsonl,write_json,sha256,atomic_write
from g1_recovery_report import markdown,word_document
from g1_reduced_budget import ROOT as OLD,TARGET_COUNTS
from g1_reduced_sources import verify_reference
from g1_robustness_protocol import MODEL_ORDER,LABELS
from g1_robustness_analysis import aggregate,curve,boundary,holm
from g1_reduced_report import rate_text,fmt,NAMES
from g1_dwaq_ablation_execute import ROOT,RUNS,sealed,check_physics,SOURCE


def paired_records(rows,expected):
    if len(rows)!=len(expected) or len({r['trial_id'] for r in rows})!=len(rows):raise ValueError('Missing/duplicate trials')
    if {r['trial_id'] for r in rows}!={r['trial_id'] for r in expected}:raise ValueError('Unpaired trials')
    if any(r['status']=='EVALUATION_ERROR' for r in rows):raise ValueError('Evaluation error excluded from scientific ranking')


def build_report(root=ROOT):
    root=Path(root);report=root/'report';report.mkdir(exist_ok=True)
    order=(*MODEL_ORDER,*RUNS);labels={**LABELS,'dwaq_v2':'DWAQ V2（无idle）','dwaq_v3':'DWAQ V3（无swing）'}
    standard={};robust={m:[] for m in order};sources=[]
    oldindex=read_json(OLD/'standard_benchmark_five_model_index.json')
    for entry in oldindex['models']:
        run=LAB/'experiments/g1_recovery_eval_v2/runs'/entry['evaluation_id']
        if sha256(run/'completion.json')!=entry['completion_sha256']:raise ValueError('Old standard seal changed')
        standard[entry['model']]=[read_json(p) for p in sorted((run/'trial_records').glob('*.json'))]
    oldaudit=read_json(OLD/'audits/final_primary_full_audit.json')
    assert oldaudit['status']=='AUDIT_PASSED'
    assert sha256(OLD/'primary_matched_source_index.json')==oldaudit['primary_source_index_sha256']
    cache=set()
    for entry in read_json(OLD/'primary_matched_source_index.json')['entries']:
        robust[entry['model']].append(verify_reference(entry,cache))
    for model in RUNS:
        sr=sealed(root/model/'standard');rr=sealed(root/model/'execution/dwaq')
        if sr is None or rr is None:raise ValueError('New model incomplete')
        from g1_recovery_protocol import RunStore
        from g1_robustness_store import RobustnessStore
        RunStore(sr).validate(validation_level='light')
        RobustnessStore(rr).validate(validation_level='light')
        for run,key in [(sr,'standard'),(rr,'robustness')]:
            rows=[read_json(p) for p in sorted((run/'trial_records').glob('*.json'))]
            paired_records(rows,standard['dwaq'] if key=='standard' else robust['dwaq'])
            if key=='standard':standard[model]=rows
            else:robust[model]=rows
            sources.append(dict(model=model,suite=key,run=str(run),completion_sha256=sha256(run/'completion.json'),identity=read_json(run/'identity.json')))
    for m in order:
        paired_records(standard[m],standard['dwaq']);paired_records(robust[m],robust['dwaq'])
        assert len(standard[m])==390 and dict(Counter(r['suite'] for r in robust[m]))==TARGET_COUNTS
    groups={'标准E1':{m:[r for r in standard[m] if r['experiment']=='E1'] for m in order},
            '域内抗扰':{m:[r for r in robust[m] if r['suite']!='extreme_ood'] for m in order}}
    groups.update({name:{m:[r for r in robust[m] if r['suite']==suite] for m in order} for suite,name in NAMES.items()})
    summaries={name:{m:aggregate(rows) for m,rows in models.items()} for name,models in groups.items()}
    blocks=[('heading',1,'DWAQ奖励消融：加入原五模型的统一对比'),('text',
        '每模型同一390标准清单和1960抗扰清单；原五模型结果保留复用，新DWAQ V2/V3各新增2350条。'
        '标准8环境、抗扰64环境。此次与训练共享GPU，运行耗时不用于模型性能比较。'
        '公共判据仍为candidate/unvalidated；单训练seed结果，不代表跨训练seed稳定优势。')]
    def text(s):blocks.append(('text',s))
    def table(h,r):blocks.append(('table',h,[[str(v) for v in row] for row in r]))
    def rate(a,k):return rate_text(a[k]['count'],a['designated_real'])
    for name,data in summaries.items():
        blocks.append(('heading',2,name+('（OOD探索，不计入域内排名）' if name==NAMES['extreme_ood'] else '')))
        table(['模型','指定真实扰动','实际施扰','生存','持续恢复','≤5落脚恢复','中位时间(s)','中位落脚数'],
            [[labels[m],d['designated_real'],d['actually_pushed'],rate(d,'survived'),rate(d,'recovered_sustained_and_survived'),
              rate(d,'recovery_within_5_touchdowns'),fmt(d['recovery_time']['median']),fmt(d['recovery_steps']['median'])] for m,d in data.items()])
        if name in ('标准E1','域内抗扰'):
            parent=data['dwaq']
            for m in RUNS:
                delta=100*(data[m]['recovered_sustained_and_survived']['designated_rate']-parent['recovered_sustained_and_survived']['designated_rate'])
                text(f"{labels[m]}相对原DWAQ：持续恢复率差值{delta:+.2f}个百分点。恢复时间/步数仅统计实际成功者。")
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    for name in ('标准E1','域内抗扰'):
        data=summaries[name];fig,ax=plt.subplots(figsize=(11,4))
        x=np.arange(len(order));a=[100*data[m]['survived']['designated_rate'] for m in order];b=[100*data[m]['recovered_sustained_and_survived']['designated_rate'] for m in order]
        ax.bar(x-.18,a,.36,label='生存');ax.bar(x+.18,b,.36,label='持续恢复')
        ax.set(xticks=x,xticklabels=[labels[m] for m in order],ylabel='指定真实扰动中的比例 (%)',ylim=(0,110),title=name);ax.legend();plt.xticks(rotation=15)
        fig.tight_layout();path=report/(name+'.png');fig.savefig(path,dpi=160);plt.close(fig)
        blocks.append(('image',str(path),'分母包含未就绪和未恢复结果；sham不计入真实扰动分母。'))
    limits={}
    for field,title in [('survived','域内极限生存'),('recovered_sustained_and_survived','域内极限持续恢复')]:
        fig,ax=plt.subplots(figsize=(10,4));limits[field]={}
        for m in order:
            pts=curve([r for r in robust[m] if r['suite']=='extreme_main'],field)
            ax.plot([p['delta_v_mps'] for p in pts],[100*p['rate'] for p in pts],marker='.',label=labels[m])
            limits[field][m]={str(q):boundary(pts,q) for q in (1.,.9,.5)}
        ax.set(xlabel='速度扰动 Δv (m/s)',ylabel='指定真实扰动中的比例 (%)',title=title,ylim=(0,105));ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();path=report/(field+'_envelope.png');fig.savefig(path,dpi=160);plt.close(fig)
        blocks.append(('image',str(path),'使用同一固定幅值清单；曲线分母包含未恢复结果，0 m/s sham单独排除。'))
    write_json(report/'envelope_boundaries.json',limits)
    blocks.append(('heading',2,'域内极限边界'))
    def bound(v):
        prefix={'left':'<','right':'≥','none':'','gap':'缺档，最后合格点 ','unidentified':'N/A '}.get(v['censor'],'')
        return prefix+fmt(v['value_mps'])
    table(['模型','90%生存 (m/s)','50%生存 (m/s)','90%恢复 (m/s)','50%恢复 (m/s)'],
        [[labels[m],*[bound(limits[f][m][str(q)]) for f,q in [('survived',.9),('survived',.5),('recovered_sustained_and_survived',.9),('recovered_sustained_and_survived',.5)]]] for m in order])
    import g1_robustness_analysis as analysis
    previous=analysis.PAIR_COMPARISONS
    try:
        analysis.PAIR_COMPARISONS=(('dwaq_v2','dwaq'),('dwaq_v3','dwaq'),('dwaq_v2','dwaq_v3'))
        tests=[t for name,models in groups.items() for t in analysis.paired_tests(models,name)]
    finally:analysis.PAIR_COMPARISONS=previous
    holm(tests);write_json(report/'paired_statistics.json',tests)
    blocks.append(('heading',2,'配对统计：域内持续恢复'))
    table(['对比','差值(百分点)','95% CI','Holm p'],[[labels[t['model_a']]+' vs '+labels[t['model_b']],fmt(100*t['effect']),[fmt(100*x) if x is not None else 'N/A' for x in t['ci95']],fmt(t['holm_p'],4)] for t in tests if t['cohort']=='域内抗扰' and t['endpoint']=='recovered_sustained_and_survived'])
    blocks.append(('text','统计比较包含原DWAQ、无idle与无swing；成功率使用exact McNemar，时间/步数仅配对双方均成功的trial使用Wilcoxon。完整数据包含效应量、95%区间及Holm校正。'))
    write_json(report/'summary.json',summaries);write_json(report/'new_sources.json',sources)
    atomic_write(report/'DWAQ_REWARD_ABLATION_COMPARISON.md',markdown(blocks,report))
    atomic_write(report/'DWAQ_REWARD_ABLATION_COMPARISON.docx',word_document(blocks))
    return summaries

if __name__=='__main__':build_report()
