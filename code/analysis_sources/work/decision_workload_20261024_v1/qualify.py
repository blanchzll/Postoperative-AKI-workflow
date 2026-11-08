from pathlib import Path
import sys,json,hashlib
import numpy as np
from decision_math import *
import common as base
OUT=ROOT/'outputs/decision_workload_20261024_v1'
def main():
    y=np.array([1,0,1,0,0,0,1,0,0,0]);p=np.array([.8,.8,.7,.6,.5,.4,.3,.2,.1,.1])
    g=['a','a','b','c','d','e','f','g','h','i'];r=list(map(str,range(10)));c=['1']*5+['2']*5
    q=actions(y,p,g,r,{'training_capacity20':.8})
    assert q['training_capacity20']['reviewed_n']==2 and q['assessment_capacity20']['reviewed_n']==2
    # Locked >= threshold retains ties, capacity imposes exact ceil allocation.
    q2=actions(y,p,g,r,{'training_capacity10':.8})
    assert q2['training_capacity10']['reviewed_n']==2 and q2['assessment_capacity10']['reviewed_n']==1
    w=workload(q['training_capacity20'],.5,15)
    assert w['expected_additional_tests']==1 and w['review_hours']==.5
    e=effect(q['training_capacity20'],0,1);assert e['hypothetical_events_averted']==0
    e=effect(q['training_capacity20'],-.1,1);assert e['hypothetical_events_averted']==-.1
    assert not effect(dict(n=10,reviewed_n=1,captured_n=1),-.1,1)['probability_assumption_compatible']
    assert scale_interval(2,1,3,-.1)==[-.2,-.30000000000000004,-.1]
    assert matched_nontargeted(q['training_capacity20'])['expected_captured_n']==.6
    # Empty alerts/zero events remain undefined for corresponding predictive fractions.
    z=actions(np.zeros(10,int),p,g,r,{'training_none':1.})['training_none']
    assert z['PPV'] is None and z['sensitivity'] is None and workload(z,1,10)['planned_reviews']==0
    th={'a':{'training_capacity20':.8},'b':{'training_capacity20':.8}}
    result=threshold_bootstrap(y,{'a':p,'b':p},g,r,c,{},th,[('a','b')],replicates=2000)
    assert all(v['estimate'] in [None,0] and v['lower95'] in [None,0] and v['upper95'] in [None,0] for v in result['paired_differences'])
    # Permuting source order must not alter group bootstrap or estimates.
    rev=np.arange(9,-1,-1)
    rr=threshold_bootstrap(y[rev],{'a':p[rev],'b':p[rev]},np.array(g)[rev],np.array(r)[rev],np.array(c)[rev],{},th,[('a','b')],replicates=2000)
    assert result==rr
    base.atomic(OUT/'QUALIFICATION.json',{'status':'PASS_synthetic_workload_and_threshold_intervals','checks':[
    'locked_threshold_ties_vs_capacity','additional_test_minutes_arithmetic','zero_harm_probability_admissibility','negative_factor_interval_order',
    'analytic_random_expected_capture','zero_event_and_empty_alerts','paired_equal_models_zero_difference','2000_group_bootstraps_order_invariance'],
    'clinical_results_generated':False,'no_model_or_treatment_fitting':True,'resample_sha256':result['receipt']['resample_weight_sha256']})
if __name__=='__main__':main()
