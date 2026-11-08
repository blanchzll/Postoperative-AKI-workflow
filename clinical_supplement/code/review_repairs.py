from pathlib import Path
import json,shutil,datetime,hashlib
import numpy as np,pandas as pd
from statsmodels.stats.proportion import proportion_confint
R=Path(__file__).resolve().parents[1];before=R/'audit/pre_scientific_review_repairs';before.mkdir(exist_ok=True)
names=['DAILY_POTENTIAL_DEMAND.csv','DAILY_SUMMARY.csv','DAILY_TIME_SCENARIOS.csv','MIMIC_DELETION_SENSITIVITY.csv','HARD_OUTCOME_DISCRIMINATION.csv']
for name in names:shutil.copy2(R/'results'/name,before/name)
d=pd.read_csv(R/'results/DAILY_POTENTIAL_DEMAND.csv');d=d.rename(columns={'review_requests':'requests_if_both_model_reviews_performed'});d['planned_final_reviews']=d.policy_alert;d['actual_completed_reviews']=np.nan;d['actual_alert_arrivals']=np.nan;d.to_csv(R/'results/DAILY_POTENTIAL_DEMAND.csv',index=False)
s=pd.read_csv(R/'results/DAILY_SUMMARY.csv');s.loc[s.metric.eq('review_requests'),'metric']='requests_if_both_model_reviews_performed';rows=[];rng=np.random.default_rng(20261108)
for (ctx,center),q in d.groupby(['context','center']):
 q=q.sort_values('surgery_date');L=len(q);starts=rng.integers(0,L,(2000,int(np.ceil(L/7))));idx=((starts[:,:,None]+np.arange(7))%L).reshape(2000,-1)[:,:L];v=q.policy_alert.to_numpy();a=q.operations.to_numpy()>0;vals=v[idx];active=a[idx]
 for denom in ['all_calendar_days','active_operation_days']:
  ss=s.loc[s.context.eq(ctx)&s.center.eq(center)&s.denominator.eq(denom)&s.metric.eq('policy_alert')].copy();assert len(ss)==1;ss['metric']='planned_final_reviews';ss['scope']='study cohort operation-date-indexed demand; not whole-hospital arrivals or staffing';arr=vals.astype(float)
  if denom=='active_operation_days':arr[~active]=np.nan
  pct=np.nanquantile(arr,.95,axis=1);ss['p95_block7_lo']=np.quantile(pct,.025);ss['p95_block7_hi']=np.quantile(pct,.975)
  for cap in [1,2,5,10,20]:
   x=np.nanmean(np.where(np.isnan(arr),np.nan,arr>cap),axis=1);obs=q.loc[a,'policy_alert'] if denom=='active_operation_days' else q.policy_alert;ss['overload_above'+str(cap)+'_fraction']=float((obs>cap).mean());ss['overload_above'+str(cap)+'_lo']=np.quantile(x,.025);ss['overload_above'+str(cap)+'_hi']=np.quantile(x,.975)
  rows.append(ss)
s=pd.concat([s]+rows,ignore_index=True);s.to_csv(R/'results/DAILY_SUMMARY.csv',index=False)
t=pd.read_csv(R/'results/DAILY_TIME_SCENARIOS.csv');rr=[]
for row in t.to_dict('records'):
 q=d.loc[d.context.eq(row['context'])&d.center.eq(row['center'])];minutes=q.policy_alert*row['review_minutes']+q.missing_chart_requests*row['missing_chart_field_minutes']+q.missing_lab_requests*row['missing_lab_minutes'];row.update(total_minutes=float(minutes.sum()),mean_calendar_day_minutes=float(minutes.mean()),p95_calendar_day_minutes=float(minutes.quantile(.95)),max_day_minutes=float(minutes.max()),fraction_days_over60minutes=float((minutes>60).mean()),review_basis='one final fixed-policy review; baseline and updated alternative reviews not summed',staff_substitution='collection and clinical review roles not measured; task-specific substitutability is scenario only');rr.append(row)
pd.DataFrame(rr).to_csv(R/'results/DAILY_TIME_SCENARIOS.csv',index=False)
x=pd.read_csv(R/'results/MIMIC_DELETION_SENSITIVITY.csv');empty=x.N.eq(0);x.loc[empty,['deleted_fraction_lo','deleted_fraction_hi','lost_known_fraction_lo','lost_known_fraction_hi']]=np.nan
for i,row in x.iterrows():
 n=int(row.original_known_events);a=int(row.lost_known_events_lower_bound)
 if n:
  lo,hi=proportion_confint(a,n,method='wilson');x.loc[i,'lost_event_Wilson_lo']=lo;x.loc[i,'lost_event_Wilson_hi']=hi
x['zero_loss_CI_note']='empirical bootstrap can degenerate at zero; Wilson bound and known event denominator retained; not stopping safety'
x.to_csv(R/'results/MIMIC_DELETION_SENSITIVITY.csv',index=False)
x=pd.read_csv(R/'results/HARD_OUTCOME_DISCRIMINATION.csv');tiny=x.events.lt(5);x.loc[tiny,['AUROC_lo','AUROC_hi']]=np.nan;x['precision_qualification']=np.where(tiny,'fewer than5 events: patient bootstrap CI suppressed; point descriptive only','conditional on frozen AKI predictions; rare outcome uncertainty retained');x.to_csv(R/'results/HARD_OUTCOME_DISCRIMINATION.csv',index=False)
(R/'audit/SCIENTIFIC_REVIEW_REMEDIATIONS.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'Codex single reviewer with independent numeric implementations; not human/external independent panel','issues':[{'issue':'Potential summing alternative base and updated reviews as primary daily demand','action':'Use one final-policy review as primary; summed baseline+updated reviews remain explicitly conditional hypothetical sequence; recompute time scenarios from final reviews'},{'issue':'Empty MIMIC temporal low-risk stratum produced100% reduction CI by denominator guard','action':'Undefined intervals set missing; no threshold change or group substitution'},{'issue':'Zero empirically lost events can give degenerate bootstrap interval','action':'Add Wilson interval using original known-event denominator; do not imply zero possible future loss'},{'issue':'Single/fewer5 hard outcome events yield spuriously tight conditional bootstrap AUC intervals','action':'Retain point/count/valid replicates, suppress inferential AUC CI with<5 events; no replacement algorithm'}],'original_tables_preserved':str(before),'no_source_or_frozen_result_change':True},indent=2))
print('SCIENTIFIC_REPAIRS_SAVED')

# FINAL_ZERO_EVENT_CENTER_PRECISION_RULE
x=pd.read_csv(R/'results/CENTER_METRICS.csv')
for i,row in x.iterrows():
 lo,hi=proportion_confint(int(row.events),int(row.N),method='wilson');x.loc[i,'prevalence_Wilson_lo']=lo;x.loc[i,'prevalence_Wilson_hi']=hi
 if row.events==0:x.loc[i,['O_E_lo','O_E_hi']]=np.nan
x['zero_event_interval_note']='zero-event bootstrap O/E interval suppressed; prevalence Wilson interval provided; center not a randomized hospital sample'
x.to_csv(R/'results/CENTER_METRICS.csv',index=False)
