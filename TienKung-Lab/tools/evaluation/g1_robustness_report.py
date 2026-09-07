"""Five-model figures and compact Markdown/DOCX, retaining detailed machine tables."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from g1_recovery_protocol import LAB, read_json, read_jsonl, write_json, sha256, atomic_write, csv_bytes
from g1_recovery_report import markdown, word_document
from g1_robustness_protocol import MODEL_ORDER, LABELS, FAMILIES, COUNTS
from g1_robustness_analysis import aggregate, curve, boundary, paired_tests, holm, native_diagnostics, association, distribution


def fmt(v,percent=False):return 'N/A' if v is None else f'{100*v:.1f}%' if percent else f'{v:.3g}'
def limit(v):return ('<' if v['censor']=='left' else '≥' if v['censor']=='right' else '~' if v['censor']=='gap' else '')+fmt(v['value_mps'])

def build_report(root):
    root=Path(root);report=root/'report';report.mkdir(parents=True,exist_ok=True);figures=report/'figures';figures.mkdir(exist_ok=True)
    index=read_json(root/'standard_benchmark_five_model_index.json');models={};standard={};identities={};mechanisms={}
    for model in MODEL_ORDER:
        entry=read_json(root/'completed_models'/f'{model}.json');run=root/'runs'/entry['evaluation_id']
        if sha256(run/'completion.json')!=entry['completion_sha256']:raise ValueError('New completion changed')
        completion=read_json(run/'completion.json')
        # Verify every archived byte before interpreting any outcome.
        for name,h in completion['files'].items():
            if sha256(run/name)!=h:raise ValueError('Sealed evidence changed: '+name)
        identities[model]=read_json(run/'identity.json')
        models[model]=[read_json(x) for x in sorted((run/'trial_records').glob('*.json'))]
        old=next(x for x in index['models'] if x['model']==model);oldrun=LAB/'experiments/g1_recovery_eval_v2/runs'/old['evaluation_id']
        if sha256(oldrun/'completion.json')!=old['completion_sha256']:raise ValueError('Standard completion changed')
        standard[model]=[read_json(x) for x in sorted((oldrun/'trial_records').glob('*.json'))]
        mechanisms[model]=[]
        for r in models[model]:
            if not r['push_applied']:continue
            with np.load(run/'traces'/f'{r["trial_id"]}.npz',allow_pickle=False) as trace:
                m=native_diagnostics(trace,r)
            m.update(suite=r['suite'],slope_deg=r['slope_deg'],direction_deg=r['direction_deg'],push_magnitude=r['push_magnitude'])
            atomic_write(report/'mechanism_trials'/model/f'{r["trial_id"]}.json',__import__('json').dumps(m,allow_nan=False))
            mechanisms[model].append({k:v for k,v in m.items() if k not in ('touchdowns','refresh_trajectory')})
        print('Report loaded '+model,flush=True)
    for key in ('evaluation_code_commit','evaluation_runtime_sha256','actual_physics_hash','realized_environment_hash','candidate_parameters_sha256','protocol_hash','manifest_hash','metrics_config_hash'):
        if len({i[key] for i in identities.values()})!=1:raise ValueError('Cannot pool unequal identity: '+key)
    for model,rows in models.items():
        if len(rows)!=sum(COUNTS.values()):raise ValueError('Incomplete paired suite')
        if any(r['status']=='EVALUATION_ERROR' for r in rows):raise ValueError('Evaluation error in main results')
    blocks=[('heading',1,'G1 Humanoid Recoverability and Robustness'),('heading',2,'Five-Model Comprehensive Evaluation'),
        ('text','Protocol 2.0-dev • common_task_window_v1 • candidate_unvalidated. Development research results; not a frozen, validated final paper benchmark. Units: m/s, s, physical touchdowns. All designated failures remain in denominators; sham is not a real push.')]
    def heading(n,title):blocks.append(('heading',2,f'{n}. {title}'))
    def table(headers,rows):blocks.append(('table',headers,[[str(v) for v in row] for row in rows]))
    def figure(name,caption,draw):
        fig=draw();fig.tight_layout();path=figures/(name+'.png');fig.savefig(path,dpi=160);plt.close(fig);blocks.append(('image',str(path),caption))
    def bars(values,ylabel):
        fig,ax=plt.subplots(figsize=(9,4));ax.bar([LABELS[m] for m in MODEL_ORDER],[np.nan if v is None else v for v in values]);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.25);return fig
    standards={m:aggregate([r for r in rows if r['experiment']=='E1']) for m,rows in standard.items()}
    suites={s:{m:aggregate([r for r in rows if r['suite']==s]) for m,rows in models.items()} for s in COUNTS}
    limits={};curves={}
    for scope,selection in [('overall',lambda r:r['suite']=='extreme_main')]+[(f'slope_{s}',lambda r,s=s:r['family']=='velocity_jump' and r['slope_deg']==s) for s in [-20,-15,-10,0,10,15,20]]+[(f'direction_{d}',lambda r,d=d:r['suite']=='extreme_main' and r['direction_deg']==d) for d in range(0,360,45)]:
        curves[scope]={};limits[scope]={}
        for m,rows in models.items():
            selected=[r for r in rows if selection(r)];curves[scope][m]={};limits[scope][m]={}
            for field in ('survived','recovered_sustained_and_survived','recovery_within_5_touchdowns'):
                c=curve(selected,field);curves[scope][m][field]={'designated':c,'pushed':curve(selected,field,True)}
                limits[scope][m][field]={str(q):boundary(c,q) for q in ([1.,.9,.5] if field=='survived' else [.9,.5])}
            falls=[r['push_magnitude'] for r in selected if r['push_applied'] and r['status']=='FELL']
            limits[scope][m]['first_fall_magnitude_mps']=min(falls) if falls else None
    write_json(report/'probability_curves.json',curves);write_json(report/'boundaries.json',limits)
    heading(1,'Executive Summary')
    headers=['Model','Std sustained','≤5 TD','Median s','Median TD','Survival90 Δv','Recovery90 Δv','Recovery50 Δv','Strong survival','Pulse recovery','Constant survival','Repeated survival','Random survival','Wrench recovery']
    main=[]
    for m in MODEL_ORDER:
        a=standards[m];l=limits['overall'][m]
        strong=aggregate([r for r in models[m] if r['suite']=='extreme_main' and r['push_magnitude']>=1.25])
        main.append([LABELS[m],fmt(a['recovered_sustained_and_survived']['designated_rate'],True),fmt(a['recovery_within_5_touchdowns']['designated_rate'],True),fmt(a['recovery_time']['median']),fmt(a['recovery_steps']['median']),
            limit(l['survived']['0.9']),limit(l['recovered_sustained_and_survived']['0.9']),limit(l['recovered_sustained_and_survived']['0.5']),fmt(strong['survived']['designated_rate'],True),
            fmt(suites['force_pulse'][m]['recovered_sustained_and_survived']['designated_rate'],True),*[fmt(suites[s][m]['disturbance_phase_survival_rate'],True) for s in ('constant_force','repeated_impulse','random_force')],
            fmt(suites['wrench_pulse'][m]['recovered_sustained_and_survived']['designated_rate'],True)])
    table(headers,main)
    blocks.append(('text','Recovery rates use designated real trials. Sustained-force phase survival columns condition on actual onset (counts below). Strong = in-domain Δv ≥1.25 m/s. Boundaries use designated success curves; < and ≥ identify tested-range censoring.'))
    heading(2,'Standard Recovery Benchmark')
    table(['Model','E0 normal/150','Readiness/240','Pushed/240','Survival','Once','Sustained','≤3 TD','≤5 TD','Relapses'],[
        [LABELS[m],sum(r['status']=='E0_COMPLETED' for r in standard[m]),standards[m]['readiness_passed'],standards[m]['actually_pushed'],*[standards[m][k]['count'] for k in ('survived','recovered_once_and_survived','recovered_sustained_and_survived','recovery_within_3_touchdowns','recovery_within_5_touchdowns')],standards[m]['relapse_count']] for m in MODEL_ORDER])
    table(['Model','Moving E0 CoM RMSE median (m/s)','Moving E0 root RMSE median (m/s)','Standing E0 CoM RMSE median (m/s)'],
        [[LABELS[m],*[fmt(distribution([r.get(field) for r in standard[m] if r['experiment']=='E0' and
            (r['command_name']=='standing')==standing])['median']) for field,standing in
            [('com_xy_velocity_rmse',False),('root_xy_velocity_rmse',False),('com_xy_velocity_rmse',True)]]] for m in MODEL_ORDER])
    figure('01_standard','Standard sustained recovery; denominator 240 designated real trials/model.',lambda:bars([standards[m]['recovered_sustained_and_survived']['designated_rate'] for m in MODEL_ORDER],'Sustained recovery probability'))
    for field,unit in [('recovery_time','s'),('recovery_steps','physical touchdowns')]:figure('02_'+field,'Standard successful-only median; full distribution in summaries.json.',lambda field=field,unit=unit:bars([standards[m][field]['median'] for m in MODEL_ORDER],f'Median recovery ({unit})'))
    heading(3,'Main Ablation')
    table(['Cohort','RL-only sustained','Context-only sustained','Context+Reward sustained'],[[s,*[fmt((standards if s=='standard' else suites[s])[m]['recovered_sustained_and_survived']['designated_rate'],True) for m in ('rl_only','context_only','context_reward')]] for s in ['standard',*COUNTS]])
    figure('15_ablation','RL-only → Context-only → Context+Reward; envelope main designated denominator.',lambda:bars([suites['extreme_main'][m]['recovered_sustained_and_survived']['designated_rate'] for m in MODEL_ORDER],'Envelope sustained recovery probability'))
    heading(4,'Extreme Disturbance Envelope')
    def plot_curve(scope,field):
        fig,ax=plt.subplots(figsize=(9,4.8))
        for m in MODEL_ORDER:
            pts=curves[scope][m][field]['designated'];ax.plot([x['delta_v_mps'] for x in pts],[x['rate'] for x in pts],marker='.',label=LABELS[m])
        ax.set(xlabel='Instantaneous Δv (m/s)',ylabel='Probability / designated real trials',ylim=(-.03,1.03));ax.legend();ax.grid(alpha=.25);return fig
    for field,name in [('survived','03_survival'),('recovered_sustained_and_survived','04_recovery'),('recovery_within_5_touchdowns','04_five_touchdown')]:figure(name,'Main slopes only; sham excluded. Conditional-pushed versions archived in probability_curves.json.',lambda field=field:plot_curve('overall',field))
    table(['Model','Survival100','Survival90','Survival50','Recovery90','Recovery50','≤5 TD90','≤5 TD50','First fall Δv'],[[LABELS[m],*[limit(limits['overall'][m][f][str(q)]) for f,q in [('survived',1.),('survived',.9),('survived',.5),('recovered_sustained_and_survived',.9),('recovered_sustained_and_survived',.5),('recovery_within_5_touchdowns',.9),('recovery_within_5_touchdowns',.5)]],fmt(limits['overall'][m]['first_fall_magnitude_mps'])] for m in MODEL_ORDER])
    figure('05_boundary90','90% sustained boundary; read censoring symbols in table.',lambda:bars([limits['overall'][m]['recovered_sustained_and_survived']['0.9']['value_mps'] for m in MODEL_ORDER],'90% recovery Δv boundary (m/s)'))
    heading(5,'Slope Robustness')
    def heat_matrix(labels,values,xlabel):
        fig,ax=plt.subplots(figsize=(10,max(4,len(labels)*.35)));im=ax.imshow(np.asarray(values,dtype=float),vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(5),[LABELS[m] for m in MODEL_ORDER]);ax.set_yticks(range(len(labels)),labels);ax.set_ylabel(xlabel);fig.colorbar(im,ax=ax,label='Sustained recovery / designated');return fig
    slopes=[-15,-10,0,10,15]
    figure('06_slope','In-domain slopes; all real magnitudes pooled equally.',lambda:heat_matrix(slopes,[[aggregate([r for r in models[m] if r['suite']=='extreme_main' and r['slope_deg']==s])['recovered_sustained_and_survived']['designated_rate'] for m in MODEL_ORDER] for s in slopes],'Slope (deg)'))
    for s in [-20,20]:
        blocks.append(('heading',3,f'OUT-OF-DOMAIN SLOPE EXTRAPOLATION: {s:+d} deg — excluded from main ranking'))
        figure(f'ood_{s}','OOD appendix; not directly pooled with in-domain results.',lambda s=s:plot_curve(f'slope_{s}','recovered_sustained_and_survived'))
    heading(6,'Direction Robustness')
    def polar():
        fig,ax=plt.subplots(figsize=(7,6),subplot_kw={'projection':'polar'});angles=np.deg2rad(np.arange(0,361,45))
        for m in MODEL_ORDER:
            v=[aggregate([r for r in models[m] if r['suite']=='extreme_main' and r['direction_deg']==d])['recovered_sustained_and_survived']['designated_rate'] for d in range(0,360,45)]
            ax.plot(angles,v+[v[0]],label=LABELS[m])
        ax.set_ylim(0,1);ax.legend(loc='upper right',bbox_to_anchor=(1.45,1.1));ax.set_title('Heading direction (deg); designated recovery probability');return fig
    figure('07_direction','Eight onset-heading directions; real magnitudes, main slopes.',polar)
    for n,s,title in [(7,'force_pulse','Force Pulse'),(8,'constant_force','Sustained Force'),(9,'repeated_impulse','Repeated Impact'),(10,'random_force','Random Force'),(11,'wrench_pulse','Wrench Robustness')]:
        heading(n,title);conds=sorted({r['condition_id'] for r in models[MODEL_ORDER[0]] if r['suite']==s})
        table(['Model','Designated','Readiness','Onset','Phase survival/onset','Once/designated','Sustained/designated','Median s','Median TD'],[[LABELS[m],suites[s][m]['designated'],suites[s][m]['readiness_passed'],suites[s][m]['actually_pushed'],fmt(suites[s][m]['disturbance_phase_survival_rate'],True),fmt(suites[s][m]['recovered_once_and_survived']['designated_rate'],True),fmt(suites[s][m]['recovered_sustained_and_survived']['designated_rate'],True),fmt(suites[s][m]['recovery_time']['median']),fmt(suites[s][m]['recovery_steps']['median'])] for m in MODEL_ORDER])
        figure(f'{n+1:02d}_{s}','Condition labels encode Δv (m/s), duration/period/tau (s), acceleration/RMS (m/s²); recovery starts at release.',lambda s=s,conds=conds:heat_matrix(conds,[[aggregate([r for r in models[m] if r['condition_id']==c])['recovered_sustained_and_survived']['designated_rate'] for m in MODEL_ORDER] for c in conds],'Disturbance condition'))
    # Duration/strength grids retain each model separately; curves use explicit units.
    for suite,xkey,ykey in [('force_pulse','duration_s','equivalent_delta_v_mps'),
                            ('repeated_impulse','period_s','single_delta_v_mps'),
                            ('random_force','correlation_time_s','rms_acceleration_mps2')]:
        for model in MODEL_ORDER:
            rows=[r for r in models[model] if r['suite']==suite]
            xs=sorted({r[xkey] for r in rows});ys=sorted({r[ykey] for r in rows})
            def grid(rows=rows,xs=xs,ys=ys,xkey=xkey,ykey=ykey,model=model):
                fig,ax=plt.subplots(figsize=(6,4))
                values=[[aggregate([r for r in rows if r[xkey]==x and r[ykey]==y])['recovered_sustained_and_survived']['designated_rate'] for x in xs] for y in ys]
                im=ax.imshow(np.asarray(values,dtype=float),vmin=0,vmax=1,aspect='auto')
                ax.set_xticks(range(len(xs)),xs);ax.set_yticks(range(len(ys)),ys)
                ax.set(xlabel=xkey+' (s)',ylabel=ykey+' (m/s²)' if 'acceleration' in ykey else ykey+' (m/s)',title=LABELS[model]);fig.colorbar(im,ax=ax,label='Designated sustained probability');return fig
            figure(suite+'_'+model,'Paired strength × duration/period/correlation grid.',grid)
    def constant_curve():
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for m in MODEL_ORDER:
            rows=[r for r in models[m] if r['suite']=='constant_force'];xs=sorted({r['acceleration_mps2'] for r in rows})
            cells=[aggregate([r for r in rows if r['acceleration_mps2']==x]) for x in xs]
            axes[0].plot(xs,[c['disturbance_phase_survival_rate'] for c in cells],marker='o',label=LABELS[m])
            axes[1].plot(xs,[c['recovered_sustained_and_survived']['designated_rate'] for c in cells],marker='o',label=LABELS[m])
        for ax,title in zip(axes,['5 s force-phase survival / actual onset','Post-release sustained / designated']):
            ax.set(xlabel='Horizontal acceleration (m/s²)',ylabel='Probability',title=title,ylim=(-.03,1.03));ax.legend(fontsize=8)
        return fig
    figure('09_constant_force_curve','Force N = actual mass kg × acceleration m/s²; mass and force retained per trial.',constant_curve)
    heading(12,'Recoverability Mechanism')
    assoc={m:{k:association([r for r in mechanisms[m] if r['suite']!='extreme_ood'],k) for k in ('N_post','margin_post')} for m in MODEL_ORDER}
    write_json(report/'mechanism_summary.json',assoc)
    table(['Model','N available','N missing','N AUROC','N vs TD rho','Margin AUROC','Margin vs s rho'],[[LABELS[m],assoc[m]['N_post']['available'],assoc[m]['N_post']['missing'],fmt(assoc[m]['N_post']['sustained_AUROC']),fmt(assoc[m]['N_post']['recovery_steps_spearman']),fmt(assoc[m]['margin_post']['sustained_AUROC']),fmt(assoc[m]['margin_post']['recovery_time_spearman'])] for m in MODEL_ORDER])
    blocks.append(('text','PPO/DWAQ/RL-only have no native certificate context; diagnostics are N/A, never zero. First post-onset refresh is retained even when invalid. Correlations with time/steps condition on observed sustained recovery; success AUROC includes failures with available predictor.'))
    def scatter():
        fig,ax=plt.subplots(figsize=(9,5))
        for m in ('context_only','context_reward'):
            r=[x for x in mechanisms[m] if x['suite']!='extreme_ood' and x.get('N_post') is not None and x['recovery_steps'] is not None];ax.scatter([x['N_post'] for x in r],[x['recovery_steps'] for x in r],s=8,alpha=.12,label=LABELS[m])
        ax.set(xlabel='First post-onset N (certificate steps)',ylabel='Actual recovery (physical touchdowns)');ax.legend();return fig
    figure('13_N_steps','Observed sustained successes only; missing diagnostics excluded and counted.',scatter)
    def margin_plot():
        fig,ax=plt.subplots(figsize=(9,4))
        for m in ('context_only','context_reward'):
            b=assoc[m]['margin_post']['calibration'];ax.plot([(x['lower']+x['upper'])/2 for x in b],[x['success_rate'] for x in b],marker='o',label=LABELS[m])
        ax.set(xlabel='Native margin (native certificate units)',ylabel='Sustained success probability');ax.legend();return fig
    figure('14_margin','Native margin quantile calibration; counts in mechanism_summary.json.',margin_plot)
    heading(13,'Statistical Comparisons')
    tests=paired_tests({m:[r for r in standard[m] if r['experiment']=='E1'] for m in MODEL_ORDER},'standard')
    for s in COUNTS:tests.extend(paired_tests({m:[r for r in models[m] if r['suite']==s] for m in MODEL_ORDER},s))
    holm(tests);write_json(report/'paired_statistics.json',tests);atomic_write(report/'paired_statistics.csv',csv_bytes(tests))
    table(['Cohort','A','B','N','Risk difference A−B','95% CI','Raw p','Holm p'],[[r['cohort'],LABELS[r['model_a']],LABELS[r['model_b']],r['n'],fmt(r['effect']),str(r['ci95']),fmt(r['raw_p']),fmt(r['holm_p'])] for r in tests if r['endpoint']=='recovered_sustained_and_survived' and r['cohort'] in ('standard','extreme_main')])
    blocks.append(('text','Exact McNemar uses all designated real paired trials. Paired Wilcoxon uses only identical trial IDs successful in both methods. Full endpoint tables include paired mean differences, bootstrap 95% CI, rank-biserial effects, raw p and globally Holm-adjusted p. No unpaired successful-only comparison.'))
    heading(14,'Failure Analysis');failures=[];representatives={};groups=defaultdict(list)
    for m,rows in models.items():
        for r in rows:groups[(m,r['suite'],r['slope_deg'],r['direction_deg'],r['push_magnitude'],r['failure_category'])].append(r['trial_id'])
        success=[r for r in rows if r['recovered_sustained_and_survived']];falls=[r for r in rows if r['push_applied'] and r['status']=='FELL'];unrecovered=[r for r in rows if r['push_applied'] and r['status']=='ALIVE_NOT_RECOVERED']
        near=sorted([r for r in success if r['suite']=='extreme_main'],key=lambda r:(abs(r['push_magnitude']-(limits['overall'][m]['recovered_sustained_and_survived']['0.5']['value_mps'] or 0)),r['trial_id']))
        representatives[m]={name:[r['trial_id'] for r in group[:3]] for name,group in [('typical_success',success),('boundary_success',near),('unrecovered',unrecovered),('fall',falls)]}
    for key,ids in groups.items():failures.append(dict(zip(('model','suite','slope_deg','direction_deg','magnitude','category'),key),count=len(ids),representative_trial_ids=ids[:3]))
    write_json(report/'failure_groups.json',failures);write_json(report/'representative_trials.json',representatives)
    table(['Model','Precondition failed','Fell','Out of area','Alive not recovered','Once but relapsed','Sustained'],[[LABELS[m],*[sum(r['failure_category']==c for r in models[m]) for c in ('PRECONDITION_FAILED','FELL','OUT_OF_TEST_AREA','ALIVE_NOT_RECOVERED','RECOVERED_ONCE_BUT_RELAPSED','RECOVERED_SUSTAINED')]] for m in MODEL_ORDER])
    heading(15,'Conclusions')
    for a,b in [('context_only','rl_only'),('context_reward','context_only'),('context_reward','ppo_plain'),('context_reward','dwaq')]:
        x=suites['extreme_main'][a];y=suites['extreme_main'][b]
        blocks.append(('text',f"{LABELS[a]} vs {LABELS[b]}: main-envelope designated sustained recovery difference {100*(x['recovered_sustained_and_survived']['designated_rate']-y['recovered_sustained_and_survived']['designated_rate']):+.2f} percentage points; 90% recovery limits {limit(limits['overall'][a]['recovered_sustained_and_survived']['0.9'])} vs {limit(limits['overall'][b]['recovered_sustained_and_survived']['0.9'])} m/s. Standard time/steps and all family results above determine whether the benefit is ordinary tracking, strong-disturbance success, or both."))
    heading(16,'Experimental Integrity')
    table(['Model','Commit','Checkpoint SHA','Runtime hash','Realized environment hash'],[[LABELS[m],identities[m]['evaluation_code_commit'],identities[m]['checkpoint_sha256'],identities[m]['evaluation_runtime_sha256'],identities[m]['realized_environment_hash']] for m in MODEL_ORDER])
    blocks.append(('text',f'Raw standard root: {LAB / "experiments/g1_recovery_eval_v2"}. New raw root: {root}. Standard and new runtimes are distinct cohorts; never pool trial outcomes across these protocols. OOD ±20 deg is a separate appendix. Threshold changed: NO. W/H changed: NO. Retraining: NO. Historical deletion: NO. New robustness trials: 80,640; standard supplement: 390.'))
    write_json(report/'summaries.json',{'standard':standards,'suites':suites,'identities':identities,'main_table':{'columns':headers,'rows':main}})
    # Full condition/direction/magnitude distributions remain machine-readable.
    cells=[]
    for m,rows in models.items():
        grouped=defaultdict(list)
        for r in rows:grouped[(r['suite'],r['condition_id'],r['slope_deg'],r['direction_deg'],r['push_magnitude'])].append(r)
        for keys,rs in grouped.items():cells.append({'model':m,**dict(zip(('suite','condition_id','slope_deg','direction_deg','magnitude'),keys)),**aggregate(rs)})
    write_json(report/'cell_metrics.json',cells)
    md=report/'G1_COMPLETE_FIVE_MODEL_COMPARISON.md';doc=md.with_suffix('.docx')
    atomic_write(md,markdown(blocks,report));atomic_write(doc,word_document(blocks))
    return {'markdown':str(md),'docx':str(doc),'models':5,'robustness_trials':80640}
