"""CPU-only D50 summaries; never alter the shared recovery judge."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RecoveryPoint:
    magnitude: float
    designated: int
    pushed: int
    recovered: int
    def __post_init__(self):
        if not math.isfinite(self.magnitude) or self.magnitude<=0 or not 0<=self.recovered<=self.pushed<=self.designated:
            raise ValueError('Invalid recovery counts or disturbance magnitude')
    @property
    def probability(self):return self.recovered/self.pushed if self.pushed else None
    @property
    def interval(self):
        if not self.pushed:return None
        z=1.959963984540054;n=self.pushed;p=self.probability;d=1+z*z/n
        c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        return max(0.,c-h),min(1.,c+h)

def summarize_d50(points):
    """For ONE slope/speed/direction/family and ONE checkpoint only.

    Interpolate only inside an observed downward crossing. Missing pushed
    samples, multiple crossings and conspicuous upward reversals are unresolved.
    Wilson intervals describe evaluation sampling, not training-seed variation.
    """
    points=sorted(points,key=lambda p:p.magnitude)
    if len({p.magnitude for p in points})!=len(points):raise ValueError('Aggregate repeated magnitudes before D50')
    if not points or any(p.pushed==0 for p in points):return dict(status='N/A',d50=None,reason='No valid pushed denominator at one or more levels')
    if len(points)<2:return dict(status='insufficient_levels',d50=None)
    rates=[p.probability for p in points]
    reversals=any(b.interval[0]>a.interval[1] for i,a in enumerate(points) for b in points[i+1:])
    crossings=[(a,b) for a,b in zip(points,points[1:]) if a.probability>.5>b.probability]
    exact=[p for p in points if p.probability==.5]
    upward=any(a.probability<.5<b.probability for i,a in enumerate(points) for b in points[i+1:])
    if reversals or upward or len(crossings)+len(exact)>1:
        return dict(status='needs_more_trials',d50=None,reason='Non-monotone or ambiguous 50% crossing; no smoothing or extrapolation')
    if all(v>.5 for v in rates):return dict(status='above_tested_range',d50=None,bound=points[-1].magnitude)
    if all(v<.5 for v in rates):return dict(status='below_tested_range',d50=None,bound=points[0].magnitude)
    if exact:return dict(status='observed',d50=exact[0].magnitude)
    if len(crossings)!=1:return dict(status='needs_more_trials',d50=None)
    a,b=crossings[0];d=a.magnitude+(.5-a.probability)*(b.magnitude-a.magnitude)/(b.probability-a.probability)
    return dict(status='interpolated',d50=d,bracket=[a.magnitude,b.magnitude],interpolation='linear probability within tested adjacent levels')

def refinement_magnitude(lower,upper):
    if not 0<lower<upper:raise ValueError('Invalid positive bracket')
    return math.sqrt(lower*upper)

def next_magnitude(current):
    if not math.isfinite(current) or current<=0:raise ValueError('Invalid magnitude')
    return current*1.25
