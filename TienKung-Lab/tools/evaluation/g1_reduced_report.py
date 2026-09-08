"""Chinese five-model report for the fixed Primary matched cohort only."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import math
import time
import io
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from g1_recovery_protocol import LAB,read_json,read_jsonl,write_json,atomic_write,sha256,csv_bytes
from g1_recovery_report import markdown,word_document
from g1_reduced_budget import ROOT,TARGET_COUNTS
from g1_reduced_sources import verify_reference
from g1_robustness_protocol import MODEL_ORDER,LABELS,FAMILIES
from g1_robustness_analysis import aggregate,distribution,curve,boundary,paired_tests,holm,native_diagnostics,association

COLORS=['#4477AA','#EE7733','#228833','#AA3377','#CC3311']
NAMES={'extreme_main':'域内极限','extreme_ood':'OOD ±20°','velocity_ood':'随机速度跳变',
 'force_pulse':'有限时长力脉冲','constant_force':'5秒持续力','repeated_impulse':'8秒重复冲击',
 'random_force':'10秒随机力','wrench_pulse':'力/力矩脉冲'}


def fmt(value,digits=2):
    return 'N/A' if value is None or not math.isfinite(float(value)) else f'{value:.{digits}f}'


def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return min(p,max(0.,center-half)),max(p,min(1.,center+half))


def rate_text(k,n,ci=False):
    if not n:return 'N/A（分母0）'
    text=f'{k}/{n} ({100*k/n:.1f}%)'
    if ci:
        lo,hi=wilson(k,n);text+=f'；95%区间 {100*lo:.1f}–{100*hi:.1f}%'
    return text


def endpoint(rows,field='recovered_sustained_and_survived',conditional=False):
    real=[r for r in rows if not r.get('sham',False)]
    pushed=[r for r in real if r['push_applied']]
    k=sum(bool(r.get(field)) for r in pushed);n=len(pushed) if conditional else len(real)
    return dict(designated=len(rows),executed=len(rows),real_designated=len(real),pushed=len(pushed),
                success=k,denominator=n,rate=k/n if n else None,ci95=wilson(k,n))


def cumulative_recovery(rows,times,conditional=False):
    eligible=[r for r in rows if not r.get('sham',False) and (not conditional or r['push_applied'])]
    counts=[sum(bool(r['push_applied'] and r['recovered_sustained_and_survived'] and
                r['recovery_time'] is not None and r['recovery_time']<=t) for r in eligible) for t in times]
    n=len(eligible)
    return dict(time_s=list(times),success=counts,denominator=n,
                rate=[k/n if n else None for k in counts],ci95=[wilson(k,n) for k in counts])


def readable_word_document(blocks):
    """Keep table rows together and fit dense comparison tables without clipping."""
    namespace='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    tag=lambda name:'{'+namespace+'}'+name
    stream=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(word_document(blocks))) as source, zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as output:
        for item in source.infolist():
            data=source.read(item.filename)
            if item.filename=='word/document.xml':
                doc=ET.fromstring(data)
                for row in doc.iter(tag('tr')):
                    props=row.find(tag('trPr'))
                    if props is None:props=ET.Element(tag('trPr'));row.insert(0,props)
                    ET.SubElement(props,tag('cantSplit'))
                    for run in row.iter(tag('r')):
                        rp=ET.Element(tag('rPr'));run.insert(0,rp)
                        ET.SubElement(rp,tag('sz'),{tag('val'):'16'})
                data=ET.tostring(doc,encoding='utf-8',xml_declaration=True)
            output.writestr(item,data)
    return stream.getvalue()


def boundary_text(points,q):
    b=boundary(points,q)
    if b['value_mps'] is None:return 'N/A：未识别'
    if b['censor']=='right':return f"≥{b['value_mps']:g}；测试范围内未到界"
    if b['censor']=='left':return f"<{b['value_mps']:g}；下界不确定"
    if b['censor']=='gap':return '数据缺口，边界不确定'
    crossing=next(j for j,p in enumerate(points) if p['rate'] is not None and p['rate']<q)
    lo=points[crossing-1]['delta_v_mps'];hi=points[crossing]['delta_v_mps']
    return f'[{lo:g}, {hi:g}]；小样本不确定'


def build_report(root=ROOT):
    root=Path(root);report=root/'report';figdir=report/'figures';figdir.mkdir(parents=True,exist_ok=True)
    audit=read_json(root/'audits/final_primary_full_audit.json')
    if audit['status']!='AUDIT_PASSED':raise ValueError('Final primary FULL audit must pass before reporting')
    source_index=read_json(root/'primary_matched_source_index.json')
    if sha256(root/'primary_matched_source_index.json')!=audit['primary_source_index_sha256']:raise ValueError('Audited source index changed')
    models={m:[] for m in MODEL_ORDER};sources={};headers=set()
    for e in source_index['entries']:
        row=verify_reference(e,headers);models[e['model']].append(row);sources[(e['model'],row['trial_id'])]=Path(e['source_run_path'])
    wanted={r['trial_id'] for r in read_json(root/'reduced_budget_manifest_v1.json')['trials']}
    for m,rows in models.items():
        if len(rows)!=1960 or {r['trial_id'] for r in rows}!=wanted:raise ValueError('Incomplete primary pairing')
        if dict(Counter(r['suite'] for r in rows))!=TARGET_COUNTS:raise ValueError('Wrong primary quota')
    index=read_json(root/'standard_benchmark_five_model_index.json');standard={}
    for e in index['models']:
        run=LAB/'experiments/g1_recovery_eval_v2/runs'/e['evaluation_id']
        if sha256(run/'completion.json')!=e['completion_sha256']:raise ValueError('Standard source changed')
        standard[e['model']]=[read_json(p) for p in sorted((run/'trial_records').glob('*.json'))]
        if len(standard[e['model']])!=390:raise ValueError('Standard390 changed')
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    blocks=[('heading',1,'G1 人形机器人恢复与抗扰：五模型统一对比'),
        ('text','Primary matched cohort｜预算缩减 v1｜protocol 2.0-dev｜candidate、unvalidated。固定顺序：PPO、DWAQ、RL-only、Context-only、Context+Reward。')]
    figure_data={};summaries={}
    def text(s):blocks.append(('text',s))
    def heading(s):blocks.append(('heading',2,s))
    def table(h,rows):blocks.append(('table',h,[[str(x) for x in r] for r in rows]))
    def savefig(name,fig,caption,data=None):
        fig.tight_layout();path=figdir/f'{name}.png';fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
        blocks.append(('image',str(path),caption));figure_data[name]=data
    def n_caption(groups,field='recovered_sustained_and_survived'):
        parts=[]
        for m in MODEL_ORDER:
            d=endpoint(groups[m],field);parts.append(f"{LABELS[m]}：指定/执行={d['designated']}/{d['executed']}，实际施扰={d['pushed']}，成功={d['success']}/{d['denominator']}")
        return '；'.join(parts)+'。sham单列；区间为Wilson 95%区间。'
    def bars(name,title,groups,field='recovered_sustained_and_survived'):
        data={m:endpoint(groups[m],field) for m in MODEL_ORDER};fig,ax=plt.subplots(figsize=(10,3.7))
        for j,m in enumerate(MODEL_ORDER):
            d=data[m];v=d['rate'];lo,hi=d['ci95']
            if v is None:ax.text(j,.04,'N/A',ha='center');continue
            ax.bar(j,v,color=COLORS[j],width=.65);ax.errorbar(j,v,yerr=[[v-lo],[hi-v]],color='black',capsize=4,fmt='none')
            ax.text(j,min(1.08,hi+.035),f"{d['success']}/{d['denominator']}",ha='center')
        ax.set(xticks=range(5),xticklabels=[LABELS[m] for m in MODEL_ORDER],ylim=(0,1.16),ylabel='成功比例（越大越好）',title=title);ax.grid(axis='y',alpha=.2)
        savefig(name,fig,n_caption(groups,field),data)
    main={m:[r for r in rows if r['suite']!='extreme_ood'] for m,rows in models.items()}
    e1={m:[r for r in rows if r['experiment']=='E1'] for m,rows in standard.items()}
    std={m:aggregate(e1[m]) for m in MODEL_ORDER};primary={m:aggregate(main[m]) for m in MODEL_ORDER}
    grouped={s:{m:[r for r in models[m] if r['suite']==s] for m in MODEL_ORDER} for s in TARGET_COUNTS}
    testgroups={'标准390中的E1':e1,'域内共同子集':main,**{NAMES[s]:grouped[s] for s in TARGET_COUNTS}}
    tests=[]
    for scope,group in testgroups.items():tests.extend(paired_tests(group,scope))
    holm(tests);write_json(report/'paired_statistics.json',tests);atomic_write(report/'paired_statistics.csv',csv_bytes(tests))
    heading('1. 一页结果摘要')
    table(['模型','标准持续恢复/240','标准≤5落脚/240','成功者中位时间(s) ↓','成功者中位落脚数 ↓','域内共同子集持续恢复','域内生存'],[
        [LABELS[m],rate_text(std[m]['recovered_sustained_and_survived']['count'],std[m]['designated_real']),
         rate_text(std[m]['recovery_within_5_touchdowns']['count'],std[m]['designated_real']),fmt(std[m]['recovery_time']['median']),
         fmt(std[m]['recovery_steps']['median']),rate_text(primary[m]['recovered_sustained_and_survived']['count'],primary[m]['designated_real']),
         rate_text(primary[m]['survived']['count'],primary[m]['designated_real'])] for m in MODEL_ORDER])
    for a,b,question in [('context_only','rl_only','Context 本身'),('context_reward','context_only','Reward 在 Context 之上'),('context_reward','ppo_plain','完整方法相对 PPO'),('context_reward','dwaq','完整方法相对 DWAQ')]:
        result=next(t for t in tests if t['cohort']=='域内共同子集' and t['model_a']==a and t['model_b']==b and t['endpoint']=='recovered_sustained_and_survived')
        delta=result['effect'];claim='校正后差异有统计支持' if result['holm_p']<.05 else '当前数据不支持校正后的明确差异'
        direction='提高' if delta>0 else '降低' if delta<0 else '不变'
        text(f"{question}：域内持续恢复比例{direction}（差值 {100*delta:+.2f} 个百分点），配对95%区间 [{100*result['ci95'][0]:+.2f}, {100*result['ci95'][1]:+.2f}]；{claim}。")
    best_steps=max(std[m]['recovery_within_5_touchdowns']['count'] for m in MODEL_ORDER)
    text('标准≤5落脚恢复点估计最高：'+ '、'.join(LABELS[m] for m in MODEL_ORDER if std[m]['recovery_within_5_touchdowns']['count']==best_steps)+f'（{best_steps}/240）。这是当前指定试验的观察排名；配对检验和不确定性见第4节。')
    text('这些结论限定于指定checkpoint、0.4 m/s前进任务和当前候选判据。成功者时间/步数不能代表全部指定试验的恢复速度；失败保留在成功比例和累计恢复曲线的分母中。')
    bars('01_standard_recovery','标准恢复：五模型持续恢复',e1)
    heading('2. 完成情况与预算变化')
    reuse=read_json(root/'reduced_budget_reuse_index.json')['entries'];extended=read_json(root/'extended_available_index.json')['sources'];completion=[]
    for m in MODEL_ORDER:
        completion.append(dict(model=m,suite='standard',target=390,reused=390,new=0,missing=0,extra=0))
        for s,target in TARGET_COUNTS.items():
            reused=sum(e['model']==m and e['suite']==s and e['reuse_decision']=='REUSE_EXISTING' for e in reuse)
            extra=sum(e['extra_counts'].get(s,0) for e in extended if e['model']==m)
            completion.append(dict(model=m,suite=s,target=target,reused=reused,new=target-reused,missing=0,extra=extra))
    table(['模型','类别','目标','复用','新增','缺失','额外保留'],[[LABELS[r['model']],NAMES.get(r['suite'],'标准Benchmark'),r['target'],r['reused'],r['new'],r['missing'],r['extra']] for r in completion])
    selection=read_json(root/'reduced_budget_selection_audit.json')
    text(f"预算调整时间（UTC）：{time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(selection['changed_at_unix']))}。原robustness配额16128/模型，调整为1960/模型；部分数据在调整前已采集。本方案不是从实验开始预注册的小样本设计。")
    text('Primary matched cohort：标准1950条＋robustness9800条＝11750条；其中新增4592条。旧额外数据留在原run，仅列为Extended available data，不重复计入主表。')
    heading('3. 标准390：原样保留的五模型比较')
    table(['模型','E0正常/150','readiness/240','实际施扰/240','生存/指定真实','一次恢复/指定真实','持续恢复/指定真实','≤3落脚','≤5落脚','复发次数'],[
        [LABELS[m],sum(r['status']=='E0_COMPLETED' for r in standard[m]),std[m]['readiness_passed'],std[m]['actually_pushed'],
         *[rate_text(std[m][k]['count'],std[m]['designated_real']) for k in ('survived','recovered_once_and_survived','recovered_sustained_and_survived','recovery_within_3_touchdowns','recovery_within_5_touchdowns')],std[m]['relapse_count']] for m in MODEL_ORDER])
    table(['模型','时间：均值/中位/P90 (s)','落脚数：均值/中位/P90','entry中位(s)','confirmation中位(s)'],[
        [LABELS[m],'/'.join(fmt(std[m]['recovery_time'][k]) for k in ('mean','median','P90')),
         '/'.join(fmt(std[m]['recovery_steps'][k]) for k in ('mean','median','P90')),
         fmt(std[m]['first_recovery_entry_latency_s']['median']),fmt(std[m]['first_confirmation_latency_s']['median'])] for m in MODEL_ORDER])
    table(['模型','移动E0 CoM RMSE中位(m/s)','站立E0 CoM RMSE中位(m/s)','移动E0 root RMSE中位(m/s)'],[
        [LABELS[m],*[fmt(distribution([r.get(field) for r in standard[m] if r['experiment']=='E0' and (abs(r['command_vx'])+abs(r['command_vy'])>0)==moving])['median']) for moving,field in [(True,'com_xy_velocity_rmse'),(False,'com_xy_velocity_rmse'),(True,'root_xy_velocity_rmse')]]] for m in MODEL_ORDER])
    text('E0是无扰动跟踪观察，正常完成不称为“恢复失败”。站立E0单列，不宣称实现站立恢复。entry与confirmation分开；未恢复的主时间和步数保留null。')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8));time_data={}
    for ax,field,label in zip(axes,['recovery_time','recovery_steps'],['恢复时间(s，越小越好)','物理落脚数(越少越好)']):
        time_data[field]={}
        for j,m in enumerate(MODEL_ORDER):
            values=[r[field] for r in e1[m] if r['recovered_sustained_and_survived'] and r[field] is not None]
            time_data[field][m]=distribution(values)
            if values:ax.boxplot([values],positions=[j],widths=.55,patch_artist=True,boxprops={'facecolor':COLORS[j]},showfliers=False)
        ax.set(xticks=range(5),xticklabels=[LABELS[m] for m in MODEL_ORDER],ylabel=label);ax.tick_params(axis='x',rotation=20)
    savefig('02_standard_time_steps',fig,'仅成功者的分布，箱体为四分位区间，不是置信区间。'+n_caption(e1),time_data)
    fig,axes=plt.subplots(1,2,figsize=(12,4));cumulative={}
    for ax,conditional in zip(axes,(False,True)):
        key='conditional_pushed' if conditional else 'designated_real';cumulative[key]={}
        for j,m in enumerate(MODEL_ORDER):
            xs=np.linspace(0,10,101);d=cumulative_recovery(main[m],xs,conditional);cumulative[key][m]=d
            if not d['denominator']:ax.plot([],[],label=LABELS[m]+' N/A',color=COLORS[j]);continue
            ax.plot(xs,d['rate'],label=f"{LABELS[m]} n={d['denominator']}",color=COLORS[j])
            ax.fill_between(xs,[v[0] for v in d['ci95']],[v[1] for v in d['ci95']],color=COLORS[j],alpha=.07)
        ax.set(xlabel='扰动结束后的恢复entry时间(s)',ylabel='累计持续恢复 / '+('实际施扰试验' if conditional else '全部指定真实试验'),ylim=(0,1.02));ax.legend(fontsize=7);ax.grid(alpha=.2)
    savefig('03_cumulative_recovery',fig,'未恢复结果始终留在分母；不以8或10秒伪造恢复事件。'+n_caption(main),cumulative)
    table(['模型','CoM速度RMSE中位(m/s)','重力倾角峰值中位(deg)','航向偏移峰值中位(deg)','路径偏移峰值中位(m)'],[
        [LABELS[m],*[fmt(distribution([None if r.get(key) is None else r[key]*scale for r in main[m] if r['push_applied']])['median']) for key,scale in [('com_xy_velocity_rmse',1),('gravity_tilt_peak_rad',180/math.pi),('heading_offset_peak_rad',180/math.pi),('path_deviation_peak_m',1)]]] for m in MODEL_ORDER])
    text('连续指标包含已施扰的失败结果，按各自实际观察时段计算；提前跌倒会缩短观测，不能脱离生存结果把较小误差解释为更好。航向/路径偏移不加入主恢复gate。')
    heading('4. 核心消融与配对统计')
    bars('04_ablation','域内共同子集：RL-only → Context-only → Context+Reward',main)
    chosen=[t for t in tests if t['endpoint']=='recovered_sustained_and_survived' and t['cohort'] in ('标准390中的E1','域内共同子集')]
    table(['范围','A 对 B','配对n','成功比例差A−B','95%区间','raw p','Holm p'],[
        [t['cohort'],LABELS[t['model_a']]+' / '+LABELS[t['model_b']],t['n'],fmt(t['effect']),str([round(x,4) if x is not None else None for x in t['ci95']]),fmt(t['raw_p'],4),fmt(t['holm_p'],4)] for t in chosen])
    text('二元终点：exact McNemar；连续终点：仅双方同一trial均持续恢复时做配对Wilcoxon。完整机器表保留效应量、配对bootstrap区间、rank-biserial效应、原始p和全表Holm校正p。不能因“差一点显著”追加样本。')
    heading('5. 极限扰动：幅值、坡度与方向')
    curve_tables={};boundaries={}
    def envelope(name,group,title):
        fig,axes=plt.subplots(1,2,figsize=(12,4.2));data={};bounds={}
        for ax,field,label in zip(axes,['survived','recovered_sustained_and_survived'],['生存比例','持续恢复比例']):
            data[field]={};bounds[field]={}
            for j,m in enumerate(MODEL_ORDER):
                points=curve(group[m],field);data[field][m]=points
                for point in points:
                    point['ci95']=wilson(point['numerator'],point['denominator'])
                    cell=[r for r in group[m] if r['push_magnitude']==point['delta_v_mps'] and not r.get('sham',False)]
                    point.update(designated=len(cell),executed=len(cell),actually_pushed=sum(r['push_applied'] for r in cell))
                xs=[p['delta_v_mps'] for p in points];ys=[p['rate'] for p in points]
                ax.plot(xs,ys,'o-',color=COLORS[j],label=LABELS[m],markersize=3)
                ax.fill_between(xs,[p['ci95'][0] for p in points],[p['ci95'][1] for p in points],color=COLORS[j],alpha=.07)
                bounds[field][m]={str(q):boundary(points,q) for q in (.9,.5)}
            ax.set(xlabel='瞬时速度扰动 Δv (m/s)',ylabel=label+'（越大越好）',ylim=(0,1.03));ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(title);savefig(name,fig,n_caption(group)+'每幅值详细指定/施扰/成功数见figure_data.json。',data)
        curve_tables[name]=data;boundaries[name]=bounds
    flat={m:[r for r in grouped['extreme_main'][m] if r['slope_deg']==0] for m in MODEL_ORDER}
    envelope('05_flat_envelope',flat,'平地极限：每幅值40条；零幅值sham单列')
    for slope in (-15,-10,10,15):
        group={m:[r for r in grouped['extreme_main'][m] if r['slope_deg']==slope] for m in MODEL_ORDER}
        envelope(f'06_slope_{slope:+d}',group,f'坡度{slope:+d}°：每幅值24条；零幅值sham单列')
    table(['模型','平地90%生存边界(m/s)','平地90%恢复边界(m/s)','平地50%恢复边界(m/s)','首次跌倒幅值(m/s)'],[
        [LABELS[m],boundary_text(curve(flat[m],'survived'),.9),boundary_text(curve(flat[m],'recovered_sustained_and_survived'),.9),
         boundary_text(curve(flat[m],'recovered_sustained_and_survived'),.5),fmt(min([r['push_magnitude'] for r in flat[m] if r['status']=='FELL' and r['push_applied']],default=None))] for m in MODEL_ORDER])
    text('边界沿最小真实测试幅值起的连续成功率曲线计算，保持既定线性跨越规则；显示相邻测试幅值区间，不用单次最大成功充当极限。小样本点估计不等于精确总体90%边界；样本内100%存活不等于真实100%可靠性。曲线中的0 m/s不作为真实push。')
    fig,ax=plt.subplots(figsize=(11,5));slopes=[-15,-10,0,10,15];heat=np.zeros((5,5));slope_data={}
    for j,m in enumerate(MODEL_ORDER):
        for k,slope in enumerate(slopes):
            d=endpoint([r for r in grouped['extreme_main'][m] if r['slope_deg']==slope]);heat[j,k]=d['rate'];slope_data[f'{m}/{slope}']=d
    im=ax.imshow(heat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');fig.colorbar(im,ax=ax,label='持续恢复比例 ↑')
    for j,m in enumerate(MODEL_ORDER):
        for k,slope in enumerate(slopes):
            d=slope_data[f'{m}/{slope}'];ax.text(k,j,f"{d['success']}/{d['denominator']}\nP={d['pushed']}\n[{d['ci95'][0]:.2f},{d['ci95'][1]:.2f}]",ha='center',va='center',fontsize=8,color='white' if heat[j,k]>.65 else 'black')
    ax.set(xticks=range(5),xticklabels=[f'{s}°' for s in slopes],yticks=range(5),yticklabels=[LABELS[m] for m in MODEL_ORDER])
    savefig('07_slope_heatmap',fig,'单元格：成功/指定真实、P=实际施扰、Wilson95%区间；各坡度指定=执行，0°为520，其余各168（含sham）。',slope_data)
    fig,ax=plt.subplots(figsize=(7,5),subplot_kw={'projection':'polar'});angles=np.deg2rad(np.arange(0,360,45));direction_data={}
    for j,m in enumerate(MODEL_ORDER):
        ds=[endpoint([r for r in grouped['extreme_main'][m] if r['direction_deg']==a]) for a in range(0,360,45)];vals=[d['rate'] for d in ds]
        ax.plot(np.r_[angles,angles[0]],vals+[vals[0]],'o-',color=COLORS[j],label=LABELS[m])
        ax.fill_between(np.r_[angles,angles[0]],[d['ci95'][0] for d in ds]+[ds[0]['ci95'][0]],[d['ci95'][1] for d in ds]+[ds[0]['ci95'][1]],color=COLORS[j],alpha=.08)
        direction_data[m]=ds
    ax.set_ylim(0,1);ax.set_title('8方向持续恢复比例 ↑（heading frame）');ax.legend(bbox_to_anchor=(1.2,1.05),fontsize=8)
    savefig('08_direction',fig,'每方向指定=执行149条（含17条sham）；各模型实际施扰、成功数和95%区间见figure_data。'+n_caption(grouped['extreme_main']),direction_data)
    table(['模型']+[f'{a}°：成功/132；P' for a in range(0,360,45)],[
        [LABELS[m]]+[f"{d['success']}/{d['denominator']}；P={d['pushed']}" for d in direction_data[m]] for m in MODEL_ORDER])
    heading('6. 六类扰动：持续扰动期间与释放后分开')
    family_summary={s:{m:aggregate(grouped[s][m]) for m in MODEL_ORDER} for s in FAMILIES}
    table(['类别','模型','指定=执行','实际施扰','扰动期生存/施扰','释放后持续恢复/指定','释放后持续恢复/施扰','时间中位(s)','落脚中位'],[
        [NAMES[s],LABELS[m],a['designated'],a['actually_pushed'],rate_text(a['disturbance_phase_survived'],a['actually_pushed']) if s!='velocity_ood' else 'N/A（瞬时）',
         rate_text(a['recovered_sustained_and_survived']['count'],a['designated_real']),rate_text(a['recovered_sustained_and_survived']['count'],a['actually_pushed']),fmt(a['recovery_time']['median']),fmt(a['recovery_steps']['median'])]
        for s in FAMILIES for m,a in family_summary[s].items()])
    text('持续力、重复冲击、随机力与wrench在扰动结束后启动恢复时钟。扰动期间任务条件不满足，不直接判作释放后恢复失败。每次完整随机/重复序列算一个trial，未把脉冲数或帧数当作独立样本。')
    for s in FAMILIES:
        groups=grouped[s];conditions=sorted({r['condition_id'] for r in groups['ppo_plain']});fig,ax=plt.subplots(figsize=(max(9,len(conditions)*1.25),5));matrix=np.zeros((5,len(conditions)));data={}
        for j,m in enumerate(MODEL_ORDER):
            for k,c in enumerate(conditions):
                d=endpoint([r for r in groups[m] if r['condition_id']==c]);matrix[j,k]=d['rate'];data[f'{m}/{c}']=d
        image=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');fig.colorbar(image,ax=ax,label='释放后持续恢复比例 ↑')
        for j,m in enumerate(MODEL_ORDER):
            for k,c in enumerate(conditions):
                d=data[f'{m}/{c}'];ax.text(k,j,f"{d['success']}/20 P={d['pushed']}\n[{d['ci95'][0]:.2f},{d['ci95'][1]:.2f}]",ha='center',va='center',fontsize=8,color='white' if matrix[j,k]>.65 else 'black')
        ax.set(xticks=range(len(conditions)),xticklabels=[c.removeprefix(s+'_') for c in conditions],yticks=range(5),yticklabels=[LABELS[m] for m in MODEL_ORDER],title=NAMES[s]+'：每condition指定=执行20条')
        ax.tick_params(axis='x',rotation=25)
        savefig('09_'+s,fig,'格内为成功/20、P=实际施扰、Wilson95%区间。dv/平方分量bound单位m/s，持续时间/period/tau单位s，a/RMS单位m/s²；force按实际质量换算，完整物理量在来源记录中。',data)
    heading('7. N / margin机制与失败模式')
    mechanism={m:[] for m in MODEL_ORDER};evolution={m:np.zeros((101,5)) for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        if not m.startswith('context'):continue
        for row in main[m]:
            if not row['push_applied']:continue
            with np.load(sources[(m,row['trial_id'])]/'traces'/f"{row['trial_id']}.npz",allow_pickle=False) as trace:
                d=native_diagnostics(trace,row)
            d.update(suite=row['suite'],slope_deg=row['slope_deg']);mechanism[m].append(d)
            refresh=d.get('refresh_trajectory',[]);cursor=-1
            for j,elapsed in enumerate(np.linspace(0,10,101)):
                t=d['disturbance_start_time']+elapsed
                if t>row['observed_terminal_time']:break
                while cursor+1<len(refresh) and refresh[cursor+1]['time']<=t+1e-9:cursor+=1
                if cursor>=0 and refresh[cursor]['context_valid']:
                    evolution[m][j]+=[refresh[cursor]['N'],refresh[cursor]['margin'],1,refresh[cursor]['N']**2,refresh[cursor]['margin']**2]
        print('Mechanism loaded '+m,flush=True)
    assoc={m:{key:association(mechanism[m],key) for key in ('N_post','margin_post')} for m in MODEL_ORDER}
    table(['模型','N可用trial','N与真实落脚ρ','N与时间ρ','N预测恢复AUROC','margin与时间ρ','margin预测恢复AUROC'],[
        [LABELS[m],assoc[m]['N_post']['available'] if m.startswith('context') else 'N/A',fmt(assoc[m]['N_post']['recovery_steps_spearman']),fmt(assoc[m]['N_post']['recovery_time_spearman']),fmt(assoc[m]['N_post']['sustained_AUROC']),fmt(assoc[m]['margin_post']['recovery_time_spearman']),fmt(assoc[m]['margin_post']['sustained_AUROC'])] for m in MODEL_ORDER])
    text('PPO、DWAQ、RL-only没有原生Context证书输入，N/margin为N/A。相关性只描述已保存轨迹，不能证明机制因果关系；N是证书量，不是公共裁判的恢复步数。首个post-refresh无效时保留无效诊断，不用之后更有利查询替换。')
    table(['模型','扰动后查询数','查询有效比例','查询失败数','N下降率中位(/s)','margin改善率中位(1/s)','首次N=0落脚中位'],[
        [LABELS[m],sum(r.get('queries_after_onset',0) for r in mechanism[m]) if m.startswith('context') else 'N/A',
         rate_text(sum(r.get('valid_queries_after_onset',0) for r in mechanism[m]),sum(r.get('queries_after_onset',0) for r in mechanism[m])),
         sum(r.get('query_failures',0) for r in mechanism[m]) if m.startswith('context') else 'N/A',
         *[fmt(distribution([r.get(k) for r in mechanism[m]])['median']) for k in ('N_decrease_rate_per_s','margin_improvement_rate_per_s','touchdowns_to_first_N_zero')]] for m in MODEL_ORDER])
    table(['模型','N桶','n','持续恢复/桶n（95%区间）','落脚中位/P90','时间中位(s)'],[
        [LABELS[m],b['lower'],b['count'],rate_text(round(b['success_rate']*b['count']),b['count'],True),fmt(b['steps']['median'])+'/'+fmt(b['steps']['P90']),fmt(b['time']['median'])]
        for m in ('context_only','context_reward') for b in assoc[m]['N_post']['calibration']])
    fig,axes=plt.subplots(1,2,figsize=(11,4));mechanism_data={}
    for j,m in enumerate(MODEL_ORDER):
        good=[r for r in mechanism[m] if r.get('N_post') is not None and r.get('recovery_steps') is not None]
        axes[0].scatter([r['N_post'] for r in good],[r['recovery_steps'] for r in good],s=9,alpha=.25,color=COLORS[j],label=LABELS[m]+(f' n={len(good)}' if m.startswith('context') else ' N/A'))
        bins=[b for b in assoc[m]['margin_post']['calibration'] if b['count']];centers=[(b['lower']+b['upper'])/2 for b in bins];rates=[b['success_rate'] for b in bins]
        axes[1].plot(centers,rates,'o-',color=COLORS[j],label=LABELS[m]+(' N/A' if not m.startswith('context') else ''))
        if bins:
            cis=[wilson(round(b['success_rate']*b['count']),b['count']) for b in bins]
            axes[1].errorbar(centers,rates,yerr=[[v-c[0] for v,c in zip(rates,cis)],[c[1]-v for v,c in zip(rates,cis)]],fmt='none',color=COLORS[j],capsize=3)
    axes[0].set(xlabel='首次post-push N（证书步数）',ylabel='实际恢复物理落脚数');axes[1].set(xlabel='原生margin（归一化、无量纲）',ylabel='持续恢复比例 ↑',ylim=(0,1.05))
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    savefig('10_mechanism',fig,'散点仅包含观测到持续恢复且N可用的trial；margin为原生分位桶。可用/缺失数量、桶n及成功/时间/步数统计另存mechanism_summary.json。',assoc)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,m in enumerate(MODEL_ORDER):
        counts=evolution[m][:,2]
        for k,ax in enumerate(axes):
            vals=np.divide(evolution[m][:,k],counts,out=np.full(101,np.nan),where=counts>0)
            ax.plot(np.linspace(0,10,101),vals,color=COLORS[j],label=LABELS[m]+(' N/A' if not m.startswith('context') else ''))
            variance=np.divide(evolution[m][:,3+k]-counts*vals**2,counts-1,out=np.full(101,np.nan),where=counts>1)
            se=np.sqrt(np.maximum(0,variance)/np.maximum(counts,1))
            ax.fill_between(np.linspace(0,10,101),vals-1.96*se,vals+1.96*se,color=COLORS[j],alpha=.1)
    for ax,label in zip(axes,['平均N（证书步数）','平均margin（无量纲）']):ax.set(xlabel='从扰动开始经过(s)',ylabel=label);ax.legend(fontsize=7)
    savefig('11_native_evolution',fig,'仅持有当时已获得的有效查询，无未来信息；曲线受存活者/有效查询选择影响，不能直接视为reward的因果效果。带状区域是跨trial均值的近似95%区间，各时刻可用n另存。',{m:a.tolist() for m,a in evolution.items()})
    failures=[];representatives={};categories=['PRECONDITION_FAILED','FELL','OUT_OF_TEST_AREA','ALIVE_NOT_RECOVERED','RECOVERED_ONCE_BUT_RELAPSED','RECOVERED_SUSTAINED']
    table(['模型','准备失败','跌倒','越界','存活未恢复','恢复后复发','持续恢复'],[[LABELS[m],*[sum(r['failure_category']==c for r in main[m]) for c in categories]] for m in MODEL_ORDER])
    for m,rows in models.items():
        groups=defaultdict(list)
        for r in rows:groups[(r['suite'],r['slope_deg'],r['direction_deg'],r['push_magnitude'],r['failure_category'])].append(r['trial_id'])
        for keys,ids in groups.items():failures.append(dict(model=m,suite=keys[0],slope_deg=keys[1],direction_deg=keys[2],magnitude=keys[3],category=keys[4],n=len(ids),representatives=ids[:3]))
        representatives[m]={name:[r['trial_id'] for r in sorted(rows,key=lambda r:r['trial_id']) if pred(r)][:3] for name,pred in {
            'success':lambda r:r['recovered_sustained_and_survived'],'unrecovered':lambda r:r['status']=='ALIVE_NOT_RECOVERED','fall':lambda r:r['status']=='FELL'}.items()}
    for m in MODEL_ORDER:
        target=boundary(curve(flat[m],'recovered_sustained_and_survived'),.5)['value_mps']
        near=sorted([r for r in flat[m] if r['recovered_sustained_and_survived']],key=lambda r:(abs(r['push_magnitude']-(target if target is not None else 3.)),r['trial_id']))
        representatives[m]['boundary_success']=[r['trial_id'] for r in near[:3]]
    heading('8. OOD ±20°：探索性附录')
    text('OOD exploratory appendix；不与±15°以内主结果混排。不扩大Plane的原生理论参数域；域外查询失败如实保留。每坡度、每幅值仅16条，探索性不确定性较大。')
    for slope in (-20,20):
        envelope(f'12_ood_{slope:+d}',{m:[r for r in grouped['extreme_ood'][m] if r['slope_deg']==slope] for m in MODEL_ORDER},f'OOD {slope:+d}°：仅探索，不进入主排名')
    heading('9. 样本量、不确定性与结论边界')
    text('每condition的20条包含所有指定合法结果，绝非20条成功结果。小样本区间较宽，不能声称20次一定能检出差异。五模型各使用一个固定训练seed/checkpoint，不能据此宣称多训练seed下稳定优势。参数与W/H仍为candidate、unvalidated，未冻结为正式2.0。')
    text('主表只使用固定共同子集；合法准备失败、摔倒、未恢复均保留。没有实际施扰时，条件于施扰的生存/恢复率为N/A。成功条件下的时间/落脚数统计有选择性，必须与包含失败的累计恢复比例一起理解。统计相关性与消融对比支持的是当前数据范围的观察。')
    heading('10. 原始数据与审计索引')
    table(['索引','位置'],[['固定共同清单','../reduced_budget_manifest_v1.json'],['选样与覆盖审计','../reduced_budget_selection_audit.json'],['初始复用决定','../reduced_budget_reuse_index.json'],['最终逐trial来源','../primary_matched_source_index.json'],['额外数据','../extended_available_index.json'],['代码兼容证据','../reduced_budget_runtime_compatibility.json'],['最终FULL审计','../audits/final_primary_full_audit.json'],['标准390索引','../standard_benchmark_five_model_index.json']])
    proof=read_json(root/'reduced_budget_runtime_compatibility.json')
    text(f"本轮执行commit：{proof['execution_commit']}。完整checkpoint、runtime、protocol/physics与原生输入SHA保留在来源索引及源identity中，不在正文堆长哈希。每个新run只做LIGHT+SAMPLED；最后对11750条主对比来源做一次FULL，不对历史额外数据重复全量回放。")
    text('旧数据删除/覆盖：NO；threshold/W/H/物理扰动定义改变：NO；更换checkpoint或重训：NO。完整原始run保留，旧RL-only大清单仍为PARTIAL，不冒充COMPLETE。')
    table(['模型','sham指定','sham正常完成','实际真实施扰','指定真实试验'],[[LABELS[m],sum(r.get('sham',False) for r in models[m]),sum(r.get('sham',False) and r['status']=='SHAM_COMPLETED' for r in models[m]),sum(r['push_applied'] for r in models[m]),sum(not r.get('sham',False) for r in models[m])] for m in MODEL_ORDER])
    write_json(report/'completion_counts.json',completion);write_json(report/'figure_data.json',figure_data)
    write_json(report/'mechanism_summary.json',assoc);write_json(report/'mechanism_trials.json',mechanism)
    write_json(report/'failure_groups.json',failures);write_json(report/'representative_trials.json',representatives)
    write_json(report/'boundaries.json',boundaries)
    write_json(report/'summary.json',dict(standard=std,primary=primary,families=family_summary,execution_commit=proof['execution_commit'],
        report_source_sha256=sha256(Path(__file__)),primary_source_index_sha256=sha256(root/'primary_matched_source_index.json')))
    md=report/'G1_FIVE_MODEL_COMPARISON.md';doc=md.with_suffix('.docx')
    atomic_write(md,markdown(blocks,report));atomic_write(doc,readable_word_document(blocks))
    return {'markdown':str(md),'docx':str(doc),'primary_trials':11750}

if __name__=='__main__':print(build_report())
