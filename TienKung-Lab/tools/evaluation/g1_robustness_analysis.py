"""Paired endpoint statistics, censored curve limits and native diagnostic analysis."""
from __future__ import annotations
from collections import Counter
import json
import math
import numpy as np
from scipy import stats
from g1_robustness_protocol import PAIR_COMPARISONS

BINARY=('survived','recovered_once_and_survived','recovered_sustained_and_survived',
        'recovery_within_3_touchdowns','recovery_within_5_touchdowns')


def distribution(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        **{name:float(np.percentile(a,q)) if len(a) else None for name,q in [('median',50),('P75',75),('P90',90)]})


def aggregate(rows):
    pushed=[r for r in rows if r['push_applied']]
    real=[r for r in rows if not r.get('sham',False)]
    result={'designated':len(rows),'designated_real':len(real),'readiness_passed':sum(r['precondition_passed'] for r in rows),
        'actually_pushed':len(pushed),'sham_designated':len(rows)-len(real),
        'relapse_count':sum(r['relapse_count'] for r in pushed),
        'disturbance_phase_survived':sum(r.get('disturbance_phase_survived') is True for r in pushed)}
    for field in BINARY:
        n=sum(bool(r[field]) for r in pushed)
        result[field]={'count':n,'designated_rate':n/len(real) if real else None,
            'pushed_rate':n/len(pushed) if pushed else None}
    result['disturbance_phase_survival_rate']=result['disturbance_phase_survived']/len(pushed) if pushed else None
    for field in ('recovery_time','recovery_steps','first_recovery_entry','first_confirmation',
                  'disturbance_task_domain_occupancy','disturbance_com_velocity_rmse_mps'):
        result[field]=distribution([r.get(field) for r in pushed])
    return result


def curve(rows,field,conditional=False):
    output=[]
    for dv in sorted({r['push_magnitude'] for r in rows if not r.get('sham',False)}):
        cell=[r for r in rows if r['push_magnitude']==dv and not r.get('sham',False)]
        eligible=[r for r in cell if r['push_applied']] if conditional else cell
        count=sum(bool(r['push_applied'] and r[field]) for r in eligible)
        output.append({'delta_v_mps':dv,'numerator':count,'denominator':len(eligible),
            'rate':count/len(eligible) if eligible else None,'denominator_role':'actually_pushed' if conditional else 'designated_real'})
    return output


def boundary(points,target):
    """Contiguous from smallest tested real magnitude; no cherry-picked later island."""
    if not points or points[0]['rate'] is None:return {'value_mps':None,'censor':'unidentified','target':target}
    if points[0]['rate']<target:return {'value_mps':points[0]['delta_v_mps'],'censor':'left','target':target}
    for a,b in zip(points,points[1:]):
        if b['rate'] is None:return {'value_mps':a['delta_v_mps'],'censor':'gap','target':target}
        if b['rate']<target:
            value=a['delta_v_mps']+(b['delta_v_mps']-a['delta_v_mps'])*(a['rate']-target)/(a['rate']-b['rate'])
            return {'value_mps':value,'censor':'none','target':target}
    return {'value_mps':points[-1]['delta_v_mps'],'censor':'right','target':target}


def bootstrap_ci(difference,seed=20260908,repeats=2000):
    d=np.asarray(difference,dtype=float)
    if not len(d):return [None,None]
    if np.all(d==d[0]):return [float(d[0]),float(d[0])]
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,repeats,50):
        means.extend(d[rng.integers(0,len(d),size=(min(50,repeats-start),len(d)))].mean(axis=1))
    return np.percentile(means,[2.5,97.5]).tolist()


def paired_tests(by_model,cohort):
    results=[]
    for a,b in PAIR_COMPARISONS:
        left={r['trial_id']:r for r in by_model[a] if not r.get('sham',False)}
        right={r['trial_id']:r for r in by_model[b] if not r.get('sham',False)}
        if left.keys()!=right.keys():raise ValueError('Incomplete paired trial assignment')
        ids=sorted(left)
        for field in BINARY:
            x=np.array([bool(left[k]['push_applied'] and left[k][field]) for k in ids],dtype=int)
            y=np.array([bool(right[k]['push_applied'] and right[k][field]) for k in ids],dtype=int)
            win=int(np.sum((x==1)&(y==0)));loss=int(np.sum((x==0)&(y==1)))
            results.append(dict(cohort=cohort,model_a=a,model_b=b,endpoint=field,test='exact McNemar',n=len(ids),
                discordant_a_only=win,discordant_b_only=loss,effect=float((x-y).mean()) if len(ids) else None,
                effect_definition='paired designated success risk difference A-B',ci95=bootstrap_ci(x-y),
                raw_p=float(stats.binomtest(win,win+loss,.5).pvalue) if win+loss else 1.))
        both=[k for k in ids if left[k]['recovered_sustained_and_survived'] and right[k]['recovered_sustained_and_survived']]
        for field in ('recovery_time','recovery_steps'):
            d=np.array([left[k][field]-right[k][field] for k in both]);nz=d[d!=0]
            ranks=stats.rankdata(abs(nz)) if len(nz) else np.array([])
            results.append(dict(cohort=cohort,model_a=a,model_b=b,endpoint=field,test='paired Wilcoxon',n=len(d),
                effect=float(d.mean()) if len(d) else None,effect_definition='paired successful-only mean difference A-B',
                rank_biserial=float(np.dot(np.sign(nz),ranks)/ranks.sum()) if len(nz) else 0. if len(d) else None,
                ci95=bootstrap_ci(d),raw_p=float(stats.wilcoxon(d,zero_method='wilcox',method='auto').pvalue) if len(nz) else 1. if len(d) else None))
    return results


def holm(results):
    ordered=sorted([r for r in results if r['raw_p'] is not None],key=lambda r:r['raw_p']);previous=0.
    for k,r in enumerate(ordered):
        previous=max(previous,min(1.,r['raw_p']*(len(ordered)-k)));r['holm_p']=previous
    for r in results:r.setdefault('holm_p',None)
    return results


def native_diagnostics(trace,record):
    """First post-onset refresh retained even if invalid; no teacher judge inputs."""
    t=np.asarray(trace['time']);planes=[json.loads(str(x)) for x in trace['plane_json']]
    onset=record.get('disturbance_start_time',record.get('push_start_time'))
    base=dict(trial_id=record['trial_id'],diagnostics_available=any(p is not None for p in planes),
        sustained_success=record['recovered_sustained_and_survived'],recovery_steps=record['recovery_steps'],recovery_time=record['recovery_time'])
    if onset is None:return {**base,'real_push':False}
    refresh=[(float(time),p) for time,p in zip(t,planes) if p and p.get('query_event')]
    # Query IDs can persist across policy frames: count each native query once.
    unique=[];seen=set()
    for time,p in refresh:
        key=p.get('query_id')
        if key not in seen:unique.append((time,p));seen.add(key)
    pre=[(float(time),p) for time,p in zip(t,planes) if time<=onset and p and p.get('context_valid')]
    post=next(((time,p) for time,p in unique if time>onset),None)
    after=[(time,p) for time,p in unique if time>onset];valid=[(time,p) for time,p in after if p['context_valid']]
    td=[]
    for idx in np.flatnonzero(np.any(trace['physical_touchdown_flags'],axis=1)):
        if t[idx]<=onset:continue
        latest=next(((time,p) for time,p in reversed(unique) if time<=t[idx]),None)
        p=planes[idx]
        td.append({'time':float(t[idx]),'physical_count':int(np.sum(trace['physical_touchdown_flags'][idx])),
            'N':p.get('N') if p else None,'margin':p.get('margin') if p else None,
            'certificate_valid':p.get('context_valid') if p else None,
            'query_age_s':float(t[idx]-latest[0]) if latest else None,'query_event':p.get('query_event') if p else None})
    firstzero=next((time for time,p in valid if p['N']==0),None)
    rate=lambda key: (valid[-1][1][key]-valid[0][1][key])/(valid[-1][0]-valid[0][0]) if len(valid)>1 else None
    return {**base,'real_push':True,'N_pre':pre[-1][1]['N'] if pre else None,'margin_pre':pre[-1][1]['margin'] if pre else None,
        'N_post':post[1]['N'] if post else None,'margin_post':post[1]['margin'] if post else None,
        'first_post_refresh_time':post[0] if post else None,'first_post_refresh_valid':post[1]['context_valid'] if post else None,
        'queries_after_onset':len(after),'valid_queries_after_onset':len(valid),
        'query_failures':sum(bool(p['query_failed']) for _,p in after),
        'query_categories':dict(Counter(p['query_category'] for _,p in after)),
        'N_decrease_rate_per_s':-rate('N') if rate('N') is not None else None,'margin_improvement_rate_per_s':rate('margin'),
        'touchdowns_to_first_N_zero':sum(x['physical_count'] for x in td if x['time']<=firstzero) if firstzero is not None else None,
        'touchdowns':td,'refresh_trajectory':[{'time':time,**p} for time,p in after]}


def association(rows,key):
    eligible=[r for r in rows if r.get(key) is not None]
    x=np.array([r[key] for r in eligible]);y=np.array([r['sustained_success'] for r in eligible],dtype=int)
    auc=None
    if len(set(y))==2:
        score=-x if key=='N_post' else x;ranks=stats.rankdata(score);n1=y.sum();n0=len(y)-n1
        auc=float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
    out={'available':len(eligible),'missing':len(rows)-len(eligible),'sustained_AUROC':auc}
    for target in ('recovery_steps','recovery_time'):
        good=[r for r in eligible if r[target] is not None]
        out[target+'_spearman'] = float(stats.spearmanr([r[key] for r in good],[r[target] for r in good]).statistic) if len(good)>2 and len({r[key] for r in good})>1 and len({r[target] for r in good})>1 else None
    cuts=sorted(set(x)) if key=='N_post' else np.unique(np.quantile(x,np.linspace(0,1,6))).tolist() if len(x) else []
    buckets=[]
    for i,c in enumerate(cuts if key=='N_post' else cuts[:-1]):
        group=[r for r in eligible if r[key]==c] if key=='N_post' else [r for r in eligible if c<=r[key] and (r[key]<cuts[i+1] or i==len(cuts)-2)]
        buckets.append(dict(lower=c,upper=c if key=='N_post' else cuts[i+1],count=len(group),
            success_rate=sum(r['sustained_success'] for r in group)/len(group) if group else None,
            steps=distribution([r['recovery_steps'] for r in group]),time=distribution([r['recovery_time'] for r in group])))
    out['calibration']=buckets
    return out
