"""Frozen clinical workload algebra; no predictive or treatment-effect fitting."""
from pathlib import Path
import sys,math,hashlib
import numpy as np
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
from evaluation_stats import FixedRisk,_design,group_weights,expanded,route,counts
def actions(y,p,g,r,thresholds):
    y=np.asarray(y);p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid probability')
    if set(np.unique(y))-set([0,1]):raise ValueError('nonbinary outcome')
    result={}
    f=FixedRisk(y,p,g,r)
    for fraction in [.1,.2]:
        result['assessment_capacity'+str(int(fraction*100))]=counts(y,f.review_counts(fraction).astype(bool))
    for name,t in thresholds.items():
        if name.startswith('training_') and isinstance(t,(int,float)):
            if not 0<=t<=1:raise ValueError('invalid locked threshold')
            result[name]=counts(y,p>=t)
    return result
def workload(q,additional,minutes):
    if additional not in [0,.5,1] or minutes not in [5,10,15]:raise ValueError('outside frozen grid')
    n=q['n'];a=q['reviewed_n']
    return {'planned_reviews':a,'planned_tests':a,'expected_additional_tests':a*additional,
        'review_hours':a*minutes/60,'reviews_per1000':1000*a/n,
        'additional_tests_per1000':1000*a*additional/n,'review_hours_per1000':1000*a*minutes/(60*n)}
def effect(q,rrr,uptake):
    if rrr not in [-.1,0,.1,.2,.3] or uptake not in [.5,.75,1]:raise ValueError('outside frozen grid')
    tp=q['captured_n'];a=q['reviewed_n']
    risk=tp/a if a else None
    compatible=(risk is None or risk*(1-rrr)<=1)
    return {'probability_assumption_compatible':compatible,
        'hypothetical_events_averted':tp*uptake*rrr if compatible else None,
        'hypothetical_remaining_alert_group_events':tp*(1-uptake*rrr) if compatible else None,
        'hypothetical_events_averted_per1000':1000*tp*uptake*rrr/q['n'] if compatible else None,
        'no_observed_treatment_effect':True}
def scale_interval(estimate,lo,hi,factor):
    if estimate is None:return [None,None,None]
    a=None if lo is None else lo*factor;b=None if hi is None else hi*factor
    return [estimate*factor,None if a is None or b is None else min(a,b),None if a is None or b is None else max(a,b)]
def matched_nontargeted(q):
    n=q['n'];a=q['reviewed_n'];e=q['events']
    return {'expected_captured_n':a*e/n,'expected_unflagged_AKI_n':e-a*e/n,
        'expected_false_alerts_n':a*(n-e)/n,'analytic_expectation_not_realised_random_trial':True}
def threshold_bootstrap(y,predictions,g,r,centers,policies,thresholds,comparisons,replicates=2000,seed=20261017):
    """Only new locked-threshold burden intervals; existing capacity intervals reused elsewhere."""
    yy,gg,rr,cc,strata,models=_design(y,g,r,centers,predictions)
    pp={k:m.p for k,m in models.items()};rng=np.random.default_rng(seed);proof=hashlib.sha256()
    def score(idx,groups,records):
        pr={nm:p[idx] for nm,p in pp.items()}
        for nm,it in policies.items():_,pr[nm],_=route(it,pp[it['base']][idx],pp[it['upgrade']][idx],groups,records)
        return {(nm,rule):counts(yy[idx],p>=t) for nm,p in pr.items() for rule,t in thresholds[nm].items()
                if rule.startswith('training_') and isinstance(t,(int,float))}
    point=score(np.arange(len(yy)),gg,rr)
    fields=['reviewed_per1000','captured_per1000','missed_per1000','false_alerts_per1000','PPV','sensitivity']
    samples={k:{f:[] for f in fields} for k in point};paired={}
    for a,b in comparisons:
        for rule in thresholds[a].keys() & thresholds[b].keys():
            if (a,rule) in point and (b,rule) in point:paired[(a,b,rule)]={f:[] for f in fields}
    for _ in range(replicates):
        w=group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes());idx,gs,rs=expanded(w,gg,rr)
        q=score(idx,gs,rs)
        for key,v in q.items():
            for f in fields:
                if v[f] is not None and np.isfinite(v[f]):samples[key][f].append(v[f])
        for (a,b,rule),cols in paired.items():
            for f in fields:
                va=q[(a,rule)][f];vb=q[(b,rule)][f]
                if va is not None and vb is not None:cols[f].append(va-vb)
    rows=[];deltas=[]
    for (nm,rule),cols in samples.items():
        for f,v in cols.items():
            ci=np.quantile(v,[.025,.975]).tolist() if v else [None,None]
            rows.append(dict(model=nm,rule=rule,metric=f,estimate=point[(nm,rule)][f],lower95=ci[0],upper95=ci[1],valid_replicates=len(v)))
    for (a,b,rule),cols in paired.items():
        for f,v in cols.items():
            va=point[(a,rule)][f];vb=point[(b,rule)][f];ci=np.quantile(v,[.025,.975]).tolist() if v else [None,None]
            deltas.append(dict(comparison=a+' MINUS '+b,rule=rule,metric=f,estimate=None if va is None or vb is None else va-vb,lower95=ci[0],upper95=ci[1],valid_replicates=len(v)))
    return {'point':[dict(model=k[0],rule=k[1],**v) for k,v in point.items()],'intervals':rows,'paired_differences':deltas,
        'receipt':{'replicates':replicates,'seed':seed,'resample_weight_sha256':proof.hexdigest(),
        'hospital_stratified_patient_groups':True,'policies_rerouted_within_each_resample':True,'new_threshold_intervals_only':True}}
