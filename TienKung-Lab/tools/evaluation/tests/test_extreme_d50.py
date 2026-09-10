import pytest
from g1_extreme_d50 import RecoveryPoint as P,summarize_d50,next_magnitude,refinement_magnitude

def test_bracketed_d50_and_geometric_scan():
 r=summarize_d50([P(1,20,20,16),P(2,20,20,4)])
 assert r['d50']==1.5 and r['bracket']==[1,2]
 assert next_magnitude(.9)==1.125
 assert refinement_magnitude(1,4)==2

def test_censored_d50_does_not_extrapolate():
 assert summarize_d50([P(1,100,100,90),P(2,100,100,60)])==dict(status='above_tested_range',d50=None,bound=2)
 assert summarize_d50([P(1,100,100,40),P(2,100,100,10)])==dict(status='below_tested_range',d50=None,bound=1)

def test_pre_onset_failures_not_recovery_failures_and_empty_is_na():
 assert P(1,100,20,10).probability==.5
 assert summarize_d50([P(1,100,0,0),P(2,100,10,1)])['status']=='N/A'

def test_no_fit_through_oscillating_curve():
 assert summarize_d50([P(1,100,100,80),P(2,100,100,20),P(3,100,100,75)])['status']=='needs_more_trials'

def test_low_n_intervals_and_invalid_counts():
 lo,hi=P(1,20,20,10).interval;assert lo<.5<hi and hi-lo>.3
 with pytest.raises(ValueError):P(1,10,11,5)
 with pytest.raises(ValueError):summarize_d50([P(1,20,20,10),P(1,20,20,10)])
