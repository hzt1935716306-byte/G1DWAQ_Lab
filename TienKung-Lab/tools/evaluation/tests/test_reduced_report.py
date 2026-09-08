"""Reporting denominators and uncertainty, independent of real outcomes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_reduced_report import endpoint,rate_text,wilson,boundary_text,cumulative_recovery


def test_no_real_push_conditional_rate_is_na():
    rows=[dict(push_applied=False,recovered_sustained_and_survived=False)]*20
    assert endpoint(rows)['rate']==0
    assert endpoint(rows,conditional=True)['rate'] is None
    assert endpoint(rows)['denominator']==20
    assert rate_text(0,0).startswith('N/A')


def test_sham_never_enters_real_denominator():
    rows=[dict(sham=True,push_applied=False,recovered_sustained_and_survived=True),
          dict(push_applied=True,recovered_sustained_and_survived=True),
          dict(push_applied=True,recovered_sustained_and_survived=False)]
    value=endpoint(rows)
    assert value['designated']==value['executed']==3
    assert value['pushed']==value['denominator']==2 and value['success']==1


def test_observed_twenty_successes_are_not_certain_population_success():
    lo,hi=wilson(20,20)
    assert .8<lo<.9 and hi==1
    assert wilson(0,0)==(None,None)


def test_tested_range_boundary_is_explicitly_censored():
    points=[dict(delta_v_mps=x,rate=1.) for x in [.25,.5,1.,3.]]
    assert '测试范围内未到界' in boundary_text(points,.9)


def test_boundary_does_not_invent_zero_width_certainty_at_exact_rate():
    points=[dict(delta_v_mps=.5,rate=.9),dict(delta_v_mps=1.,rate=.8)]
    assert '[0.5, 1]' in boundary_text(points,.9)
    assert '不确定' in boundary_text(points,.9)


def test_cumulative_recovery_keeps_failures_and_excludes_sham():
    rows=[dict(push_applied=True,recovered_sustained_and_survived=True,recovery_time=2.),
          dict(push_applied=True,recovered_sustained_and_survived=False,recovery_time=None),
          dict(push_applied=False,recovered_sustained_and_survived=False,recovery_time=None),
          dict(sham=True,push_applied=False,recovered_sustained_and_survived=True,recovery_time=0.)]
    assert cumulative_recovery(rows,[0,1.99,2,10])['rate']==[0,0,1/3,1/3]
    assert cumulative_recovery(rows,[2,10],True)['rate']==[.5,.5]
    assert cumulative_recovery(rows[2:],[10],True)['rate']==[None]


def test_wilson_endpoint_errorbars_never_negative():
    for n in (1,20,132,1696):
        for k in (0,n):
            lo,hi=wilson(k,n)
            assert 0<=lo<=k/n<=hi<=1
