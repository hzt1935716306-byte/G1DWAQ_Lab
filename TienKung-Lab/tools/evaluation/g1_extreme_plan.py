"""Deterministic, method-independent extreme experiment plans and scan decisions."""
import math,random
from collections import defaultdict
from g1_extreme_d50 import RecoveryPoint,summarize_d50,refinement_magnitude
MODELS=('ppo','dwaq_v2','dwaq','ours')
SLOPES=(-20,-15,-10,0,10,15,20)
GROUPS={'original':(-10,0,10),'extended':(-20,-15,15,20)}
START={'B1':.9,'B2':2.5};EASY={'B1':.3,'B2':.5}

def conditions(slopes):return [(s,v,d,f) for s in slopes for v in (.5,1.) for d in (0,90,180,270) for f in ('B1','B2')]
def key(c):return f'{c[3]}_s{c[0]:+g}_v{c[1]:g}_d{c[2]:g}'
def cell(r):return (r['slope_deg'],r['command_vx'],r['direction_deg'],r['suite'])

def scenario(c,magnitude,rep,phase):
    # Separate phase namespaces; the same condition/repeat reset and onset
    # are reused across intensity levels and all methods, never keyed by outcome.
    if phase not in ('explore','formal','baseline') or not 0 <= rep < 1000:
        raise ValueError('Unknown phase or repeat outside frozen seed allocation')
    # Collision-free phase/condition/repeat allocation, within uint32.
    cells=[(s,v,d,f) for s in SLOPES for v in (.5,1.) for d in (0,90,180,270) for f in ('A','B1','B2')]
    seed=202610000+('explore','formal','baseline').index(phase)*1000000+cells.index(c)*1000+rep
    rng=random.Random(seed);timing=random.Random(seed^0x615A);sampled=timing.uniform(3,5);onset=math.ceil(sampled/.02)*.02
    slope,speed,direction,suite=c
    return dict(trial_id=f'{phase}_{key(c)}_a{magnitude:.12g}_r{rep:04d}',suite=suite,slope_deg=slope,command_name='+x',command_vx=speed,command_vy=0.,command_yaw=0.,direction_deg=direction,strength=float(magnitude),reset_seed=seed,repeat_id=rep,sampled_onset=None if suite=='A' else sampled,onset=None if suite=='A' else onset,release=None if suite=='A' else onset+(.2 if suite=='B1' else 0),episode_s=12.,initial_pose_parameters=dict(x=rng.uniform(-.2,.2),y=rng.uniform(-.2,.2),z=0.,roll=0.,pitch=0.,yaw=0.),initial_velocity_parameters=dict(x=0.,y=0.,z=0.,roll=0.,pitch=0.,yaw=0.),initial_joint_parameters=dict(position_scale=[rng.uniform(.9,1.1) for _ in range(29)],velocity=[0.]*29))

def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(cell(r),r['strength'])].append(r)
    return {k:RecoveryPoint(k[1],len(a),sum(r['actually_perturbed'] for r in a),sum(bool(r['recovered']) for r in a)) for k,a in grouped.items()}

def curves(rows):
    result=defaultdict(list)
    for (c,_),p in aggregate(rows).items():result[c].append(p)
    return {c:sorted(a,key=lambda p:p.magnitude) for c,a in result.items()}

def scan_level(round_id,family):return float(f'{START[family]*1.25**round_id:.12g}')

def formal_levels(points,family):
    """At most three fixed common levels: easy, near PPO 50%, PPO upper limit.

    Exploration only locates the common test range. Never search other methods'
    maxima. Formal repeats use fresh seeds; noisy D50 remains unresolved.
    """
    points=sorted(points,key=lambda p:p.magnitude)
    if not points:raise ValueError('Exploration missing')
    easy=EASY[family];upper=points[-1].magnitude
    crossing=next(((a,b) for a,b in zip(points,points[1:]) if a.probability is not None and b.probability is not None and a.probability>=.5>=b.probability and a.probability>b.probability),None)
    if crossing:
        a,b=crossing
        middle=a.magnitude+(.5-a.probability)*(b.magnitude-a.magnitude)/(b.probability-a.probability)
    else:
        middle=refinement_magnitude(easy,upper) if easy<upper else easy
    return sorted({easy,float(f'{middle:.12g}'),upper})
