"""Read-only equal-scenario summaries and paper figures for the fixed-time experiment."""
from collections import defaultdict,Counter
from pathlib import Path
import argparse,json,csv
import numpy as np
from g1_paper_recovery import ROOT,MODELS
from g1_recovery_protocol import read_json,write_json,sha256

def load(stage):
    data={}
    jobs=None
    if stage=='formal':
        from g1_paper_execution import PLAN,verify_run
        jobs=read_json(PLAN)['jobs']
    for m in MODELS:
        records=[];seen=set()
        seals=sorted((ROOT/stage/m).glob('**/completion.json')) if jobs is None else [Path(j['run'])/'completion.json' for j in jobs if j['model']==m]
        for seal in seals:
            run=seal.parent;c=read_json(seal)
            binding=read_json(run/'binding.json')
            if jobs is not None:
                expected=next(j for j in jobs if Path(j['run'])==run)
                verify_run(run,expected,True)
            for f,h in c['records_sha256'].items():
                path=run/'records'/f
                if sha256(path)!=h:raise ValueError('Sealed record changed')
                r=read_json(path)
                if r['trial_id'] in seen:raise ValueError('Duplicate completed trial across attempts')
                seen.add(r['trial_id']);records.append({**r,'run':str(run)})
        data[m]=records
    return data

def stats(rows):
    pushed=[r for r in rows if r['actually_perturbed']];good=[r for r in pushed if r['recovered']];cells=defaultdict(list)
    for r in rows:cells[(r['slope_deg'],r['command_vx'],r['direction_deg'],r['strength'])].append(r)
    conditional=[]
    for c in cells.values():
        q=[r for r in c if r['actually_perturbed']];conditional.append(sum(bool(r['recovered']) for r in q)/len(q) if q else None)
    distribution=lambda k:{'median':float(np.median([r[k] for r in good])),'mean':float(np.mean([r[k] for r in good])),'q25':float(np.percentile([r[k] for r in good],25)),'q75':float(np.percentile([r[k] for r in good],75))} if good else None
    rng=np.random.default_rng(261010);boot=[]
    # Evaluate-scenario resampling only, not training-seed uncertainty.
    applicable=bool(rows) and rows[0]['suite'] in ('B1','B2')
    if pushed and applicable:
        blocks=defaultdict(list)
        for r in pushed:blocks[r['repeat_id']].append(r)
        counts=np.array([[sum(bool(r['recovered']) for r in b),len(b)] for b in blocks.values()])
        draws=counts[rng.integers(0,len(counts),size=(1000,len(counts)))].sum(1)
        boot=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist()
    return dict(designated=len(rows),pushed=len(pushed),pre_push_failure=sum(r['pre_push_failure'] for r in rows),episode_success=sum(r['episode_success'] for r in rows),post_push_fall=sum(r['fell'] for r in pushed),survived=sum(r['survived_after_push'] for r in pushed),recovered=len(good) if applicable else None,within5=sum(r['within_5_touchdowns'] for r in pushed) if applicable else None,time=distribution('recovery_time'),steps=distribution('recovery_steps'),conditional_recovery_equal_cells=float(np.mean(conditional)) if applicable and conditional and all(v is not None for v in conditional) else None,bootstrap_pushed_ci=boot or None)

def report(stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=load(stage)
    if stage=='formal':
        expected={r['trial_id'] for r in read_json(ROOT/'formal_frozen_manifest.json')}
        if len(expected)!=6420 or any({r['trial_id'] for r in rr}!=expected for rr in data.values()):
            raise ValueError('Formal report requires all6420 paired episodes per method')
    out=ROOT/stage/'report';out.mkdir(parents=True,exist_ok=True);summ={};lines=['# G1 fixed-time slope recovery: PPO / DWAQ V3 / Ours','',f'Stage: {stage}; independent candidate protocol. Training seed42 only; repeat-block evaluation bootstrap does not measure training-seed uncertainty. No readiness selection. All failures retained.','']
    def table(headers,rows):
        lines.append('| '+' | '.join(headers)+' |');lines.append('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows:lines.append('| '+' | '.join(str(x) for x in row)+' |')
        lines.append('')
    rate=lambda k,n:f'{k}/{n} ({100*k/n:.1f}%)' if n else 'N/A'
    fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
    for m,rr in data.items():
        summ[m]={s:stats([r for r in rr if r['suite']==s]) for s in ['A','B1','B2','C_force','C_load']}
    if stage=='formal':lines+=['A retains the complete pre-fix runtime cohort. B/C use a common repaired runtime across methods. Judge/reset definitions are unchanged; runtime hashes are in execution_continuation.json. No cross-suite pooling.','']
    lines+=['## A: normal walking','']
    table(['Method','slope deg','command m/s','success','XY velocity RMSE m/s','yaw RMSE rad/s','tracking duration mean s'],[[m,s,v,rate(sum(r['episode_success'] for r in a),len(a)),fmt(np.mean([r['velocity_rmse_mps'] for r in a if r['velocity_rmse_mps'] is not None])) if a else 'N/A',fmt(np.mean([r['yaw_rate_rmse_radps'] for r in a if r['yaw_rate_rmse_radps'] is not None])) if a else 'N/A',fmt(np.mean([r['tracking_duration_s'] for r in a])) if a else 'N/A'] for m,rr in data.items() for s in [-10,0,10] for v in [.5,1.] for a in [[r for r in rr if r['suite']=='A' and r['slope_deg']==s and r['command_vx']==v]]])
    for suite in ['B1','B2']:
        lines+=['## '+suite,'']
        table(['Method','designated / pushed / pre-failed','recovery/pushed','survival/pushed','median T [Q1,Q3] s','mean K','median K','<=5/pushed','equal-cell conditional recovery'],[[m,f"{a['designated']}/{a['pushed']}/{a['pre_push_failure']}",rate(a['recovered'],a['pushed']),rate(a['survived'],a['pushed']),(f"{a['time']['median']:.3f} [{a['time']['q25']:.3f}, {a['time']['q75']:.3f}]" if a['time'] else 'N/A'),fmt(a['steps']['mean']) if a['steps'] else 'N/A',fmt(a['steps']['median']) if a['steps'] else 'N/A',rate(a['within5'],a['pushed']),fmt(a['conditional_recovery_equal_cells'])] for m in data for a in [summ[m][suite]]])
    lines+=['Successful recovery only for T/K distributions. First0.3s confirmation within5s after release; later fall/abnormal termination nulls primary T/K. Recovery need not persist as a gate after confirmation, but episode must complete. Conditional denominators can differ after pre-push failure; designated and per-cell rates are retained.','## C: continuous force and payload','']
    table(['Method','suite','slope deg','weight fraction','direction deg','success','XY RMSE m/s'],[[m,suite,s,strength,d,rate(sum(r['episode_success'] for r in a),len(a)),fmt(np.mean([r['velocity_rmse_mps'] for r in a if r['velocity_rmse_mps'] is not None])) if a else 'N/A'] for m,rr in data.items() for suite in ['C_force','C_load'] for s in [-10,0,10] for strength in [.05,.1] for d in ([0,90,180,270] if suite=='C_force' else [0]) for a in [[r for r in rr if r['suite']==suite and r['slope_deg']==s and r['strength']==strength and r['direction_deg']==d]]])
    # Equal allocated scene grid; report unidentified cells instead of removing them.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,suite in zip(axes,['B1','B2']):
        for m,rr in data.items():
            yy=[]
            for strength in sorted({r['strength'] for rr in data.values() for r in rr if r['suite']==suite}):
                a=[r for r in rr if r['suite']==suite and r['strength']==strength];q=stats(a);yy.append(q['conditional_recovery_equal_cells'] if q['conditional_recovery_equal_cells'] is not None else np.nan)
            ax.plot(sorted({r['strength'] for rr in data.values() for r in rr if r['suite']==suite}),yy,'o-',label=m)
        ax.set(title=suite,xlabel='Equivalent delta-v / delta-v (m/s)',ylabel='Equal-cell recovery probability',ylim=(0,1.05));ax.legend()
    fig.tight_layout();fig.savefig(out/'fig1_strength.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));t=np.linspace(0,5,251)
    for ax,suite in zip(axes,['B1','B2']):
        for m,rr in data.items():
            p=[r for r in rr if r['suite']==suite and r['actually_perturbed']]
            ax.plot(t,[sum(r['recovered'] and r['recovery_time']<=x for r in p)/len(p) if p else np.nan for x in t],label=f'{m} n={len(p)}')
        ax.set(title=suite,xlabel='Time since disturbance release (s)',ylabel='Recovered and survived / pushed',ylim=(0,1.05));ax.legend()
    fig.tight_layout();fig.savefig(out/'fig2_cumulative.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for row,suite in enumerate(['B1','B2']):
        for col,field in enumerate(['recovery_time','recovery_steps']):
            values=[[r[field] for r in rr if r['suite']==suite and r['recovered']] for rr in data.values()];axes[row,col].boxplot([a or [np.nan] for a in values],tick_labels=list(data));axes[row,col].set(title=suite,ylabel='Recovery time (s)' if col==0 else 'Physical touchdowns (count)')
    fig.tight_layout();fig.savefig(out/'fig3_time_steps.png',dpi=160);plt.close(fig)
    colors={'ppo':'tab:blue','dwaq':'tab:orange','ours':'tab:green'}
    tid='B1_s+0_v0.5_d0_a0.6_r000';fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
    for m,rr in data.items():
        r=next((r for r in rr if r['trial_id']==tid),None)
        if r is None:continue
        with np.load(Path(r['run'])/'traces'/(tid+'.npz')) as z:
            ts=z['time'];axes[0].plot(ts,np.linalg.norm(z['com_velocity'][:,:2]-z['command'][:,:2],axis=1),label=m);axes[1].plot(ts,np.rad2deg(z['roll_pitch'][:,0]),label=m+' roll',color=colors[m]);axes[1].plot(ts,np.rad2deg(z['roll_pitch'][:,1]),'--',label=m+' pitch',color=colors[m])
            for j in range(2):axes[j+2].step(ts,z['physical_contact'][:,j].astype(float)+list(data).index(m)*1.3,label=m,where='post')
        for ax in axes:
            if r['actual_onset'] is not None:ax.axvspan(r['actual_onset'],r['actual_release'],alpha=.07)
            if r['first_confirmation'] is not None:ax.axvline(r['first_confirmation'],alpha=.7,linestyle='--',color=colors[m],label=m+' confirmation' if ax is axes[0] else None)
    for ax,label in zip(axes,['XY velocity error (m/s)','Roll / pitch (deg)','Left foot contact','Right foot contact']):ax.set_ylabel(label);ax.legend(fontsize=7)
    for ax in axes[2:]:ax.set_yticks([.5,1.8,3.1],list(data))
    axes[0].axhline(.2,color='gray',linestyle=':',label='velocity threshold')
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel('Episode time (s)');fig.tight_layout();fig.savefig(out/'fig4_trace.png',dpi=160);plt.close(fig)
    lines+=['## Figures','',*[f'![{name}]({name}.png)' for name in ['fig1_strength','fig2_cumulative','fig3_time_steps','fig4_trace']],'','Representative trace fixed before results: '+tid+'. Same ID for all methods; failures/censoring shown without replacement.','D skipped: no matched reward-on V2 checkpoint available.','']
    write_json(out/'summary.json',summ);(out/'RESULTS.md').write_text('\n'.join(lines))
    allrows=[r for rr in data.values() for r in rr]
    if allrows:
        fields=list(allrows[0]);
        with (out/'episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
            for r in allrows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    conditions=[]
    for m,rr in data.items():
        groups=defaultdict(list)
        for r in rr:groups[(r['suite'],r['slope_deg'],r['command_vx'],r['direction_deg'],r['strength'])].append(r)
        for key,rows in sorted(groups.items()):
            conditions.append(dict(model=m,**dict(zip(['suite','slope_deg','command_vx','direction_deg','strength'],key)),**stats(rows)))
    write_json(out/'condition_summary.json',conditions)
    if conditions:
        with (out/'condition_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(conditions[0]));w.writeheader()
            for r in conditions:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    print(json.dumps({m:{s:(a['designated'],a['pushed'],a['recovered']) for s,a in ss.items()} for m,ss in summ.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='formal',choices=['pilot','calibration','formal']);report(p.parse_args().stage)
