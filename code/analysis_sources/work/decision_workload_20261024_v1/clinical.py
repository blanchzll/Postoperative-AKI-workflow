"""Finite clinical scenario batch. Requires a separate stage-release receipt."""
import sys,resource,time
import pandas as pd
from clinical_adapter import *
from decision_math import workload,effect,matched_nontargeted,scale_interval,threshold_bootstrap
from threadpoolctl import threadpool_limits
CODE=Path(__file__).resolve().parent
def run(line,cohort):
    assert line in ['geography','temporal'] and cohort in ['preop','paired']
    for n,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/n)==h
    for p,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(p)==h
    assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_synthetic_workload_and_threshold_intervals'
    assert base.read(OUT/'SOURCE_ADAPTER_QUALIFICATION.json')['status']=='PASS_26_model_and_24_policy_frozen_predictions_replayed'
    release=base.read(OUT/'RELEASE_FOR_CLINICAL_SCENARIOS.json')
    assert release['status']=='PASS_required_preceding_stages_accounted_for'
    assert release['code_lock_sha256']==base.sha(OUT/'CODE_LOCK.json')
    assert release['nonlinear_LR_status_explicit'] and release['public_and_necessary_supporting_stage_statuses']
    m,pr,th,pol,eligible,comparisons,refs=load(line,cohort)
    dest=OUT/(line+'_'+cohort);dest.mkdir(parents=True,exist_ok=True)
    signature={'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'release_sha256':base.sha(OUT/'RELEASE_FOR_CLINICAL_SCENARIOS.json'),'prediction_hashes':refs}
    if (dest/'COMPLETE.json').exists():
        r=base.read(dest/'COMPLETE.json');assert r['signature']==signature
        for f,h in r['files'].items():assert base.sha(dest/f)==h
        return
    y=m.y.to_numpy();g=m.group_hash.astype(str).tolist();r=m.excel_row.astype(str).tolist()
    final=base.read(base.OUT/(line+'_'+cohort)/'EVALUATION.json')
    checkpoint=dest/'NEW_THRESHOLD_INTERVALS.json'
    if checkpoint.exists():
        obj=base.read(checkpoint);assert obj['signature']==signature;new=obj['result']
    else:
        new=threshold_bootstrap(y,pr,g,r,m.center.astype(str).tolist(),pol,th,comparisons)
        assert new['receipt']['resample_weight_sha256']==final['receipt']['resample_weight_sha256']
        base.atomic(checkpoint,{'signature':signature,'result':new})
    full=dict(pr)
    for nm,it in pol.items():_,full[nm],_=route(it,pr[it['base']],pr[it['upgrade']],g,r)
    points={};intervals={}
    for nm,p in full.items():
        for rule,q in actions(y,p,g,r,th[nm]).items():points[(nm,rule)]=q
    for z in new['intervals']:intervals[(z['model'],z['rule'],z['metric'])]=z
    for z in final['intervals']:
        if z['model'] not in full:continue
        for frac in [10,20]:
            prefix='capacity'+str(frac)+'_'
            if z['metric'].startswith(prefix):
                field=z['metric'][len(prefix):];rule='assessment_capacity'+str(frac)
                intervals[(z['model'],rule,field)]=z
    rows=[];eff=[];baselines=[]
    for (nm,rule),q in points.items():
        for fraction in [0,.5,1]:
            for minutes in [5,10,15]:
                row={'model':nm,'rule':rule,'additional_fraction':fraction,'minutes_per_review':minutes,**q,**workload(q,fraction,minutes)}
                z=intervals[(nm,rule,'reviewed_per1000')]
                for field,factor in [('reviews_per1000',1),('additional_tests_per1000',fraction),('review_hours_per1000',minutes/60)]:
                    _,lo,hi=scale_interval(z['estimate'],z['lower95'],z['upper95'],factor)
                    row[field+'_lower95']=lo;row[field+'_upper95']=hi
                rows.append(row)
        baselines.append({'model':nm,'rule':rule,'n':q['n'],'events':q['events'],'algorithm_alerts':q['reviewed_n'],**matched_nontargeted(q)})
        if eligible[nm]:
            z=intervals[(nm,rule,'captured_per1000')]
            for rrr in [-.1,0,.1,.2,.3]:
                for uptake in [.5,.75,1]:
                    e=effect(q,rrr,uptake)
                    _,lo,hi=scale_interval(z['estimate'],z['lower95'],z['upper95'],rrr*uptake)
                    eff.append({'model':nm,'rule':rule,'assumed_RRR':rrr,'assumed_uptake':uptake,**e,
                      'conditional_sampling_lower95':lo if e['probability_assumption_compatible'] else None,
                      'conditional_sampling_upper95':hi if e['probability_assumption_compatible'] else None,
                      'assumption_uncertainty_not_in_interval':True})
    pd.DataFrame([dict(model=nm,rule=rule,**q) for (nm,rule),q in points.items()]).to_csv(dest/'OPERATING_BURDEN.csv',index=False)
    pd.DataFrame(rows).to_csv(dest/'WORKLOAD_SCENARIOS.csv',index=False)
    pd.DataFrame(eff).to_csv(dest/'HYPOTHETICAL_PREOPERATIVE_EFFECT_ONLY.csv',index=False)
    pd.DataFrame(baselines).to_csv(dest/'MATCHED_NONTARGETED_ANALYTIC_EXPECTATION.csv',index=False)
    ref=[{'comparator':'no_additional_algorithm_flags','n':len(y),'observed_AKI':int(y.sum()),'reviewed_n':0,'algorithm_captured_n':0,'routine_care_continues':True},
         {'comparator':'flag_everyone_resource_reference','n':len(y),'observed_AKI':int(y.sum()),'reviewed_n':len(y),'algorithm_captured_n':int(y.sum()),'routine_care_continues':True}]
    pd.DataFrame(ref).to_csv(dest/'RESOURCE_BOUND_REFERENCES.csv',index=False)
    # Preserve existing DCA, calibrated/un-calibrated comparisons and exact resample identity.
    base.atomic(dest/'REUSED_CAPACITY_DCA_AND_PAIRWISE_RESULTS.json',{'source':str(base.OUT/(line+'_'+cohort)/'EVALUATION.json'),
      'sha256':base.sha(base.OUT/(line+'_'+cohort)/'EVALUATION.json'),'capacity_intervals_and_DCA_reused_not_recomputed':True,
      'new_threshold_intervals':str(checkpoint),'cross_time_or_stage_counts_not_pooled':True})
    files=[p.name for p in dest.iterdir() if p.suffix in ['.csv','.json'] and p.name!='COMPLETE.json']
    base.atomic(dest/'COMPLETE.json',{'status':'PASS_frozen_workload_scenarios_not_clinical_efficacy','signature':signature,
      'files':{f:base.sha(dest/f) for f in files},'models':len(pr),'policies':len(pol),'n':len(y),'events':int(y.sum()),
      'no_predictive_fitting':True,'no_optimisation_of_rules_or_assumptions':True,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      'whole_project_complete':False})
if __name__=='__main__':
    with threadpool_limits(limits=1):run(sys.argv[1],sys.argv[2])
