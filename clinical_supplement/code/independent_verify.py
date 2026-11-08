from pathlib import Path
import os,json,hashlib,time,concurrent.futures
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,pandas as pd
from scipy.special import logit
from scipy.stats import rankdata,spearmanr
from fractions import Fraction
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1];P=R.parents[1];O=P/'correction_runs/full_data_20261025_postslot/outputs';start=time.time()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def check(a,b):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9,equal_nan=True)
def verify_context(tag):
 z=np.load(R/'private'/f'{tag}_RESAMPLING.npz',allow_pickle=False);w=z['weights'];p=z['pbase'];up=z['pupgrade'];y=z['y'];keys=z['records'];g=read(R/'private'/f'{tag}_DECISIONS.csv').group_hash.to_numpy(str);line,cohort=tag.split('_');lock=json.loads((O/f'selective_policy_lock_20261023_v1/locked_policies/{line}/{cohort}/0.25/capacity/selective/POLICY_LOCK.json').read_text());bl=json.loads(Path(lock['base_model']['path']).read_text());gate=lock['gate'];dist=np.abs(logit(np.clip(p,gate['clip'],1-gate['clip']))-logit(gate['threshold']));near=np.abs(p-bl['training_thresholds']['training_capacity20']);rnd=np.array([int(hashlib.sha256(f'20261108|{a}|{b}'.encode()).hexdigest()[:15],16) for a,b in zip(g,keys)]);canon=np.array(sorted(range(len(y)),key=lambda i:(str(g[i]),str(keys[i]))));priority={'existing_selective25':dist,'highest_base_risk25':-p,'near_training_cutoff25':near,'random25':rnd};rows=read(R/'results/POLICY_EQUAL_BUDGET.csv');rows=rows[rows.context.eq(tag)&rows.cost_ratio.eq(.1)];dif=read(R/'results/POLICY_PAIRED_DIFFERENCES.csv');dif=dif[dif.context.eq(tag)&dif.cost_ratio.eq(.1)];values={};pointchecked=0;maxerr=0
 # Independent algorithm: materialize individual copies, rerank each patient list, then direct sum outcome labels.
 for mode in ['no_collection','existing_selective25','highest_base_risk25','near_training_cutoff25','random25','all_collection']:
  caps=[]
  for wi,weight in enumerate(w):
   idx=np.repeat(canon,weight[canon]);n=len(idx);K=int(np.ceil(.2*n));U=int(np.ceil(.25*n));mask=np.zeros(n,bool)
   if mode=='all_collection':mask[:]=True
   elif mode!='no_collection':
    chosen=np.argsort(priority[mode][idx],kind='stable')[:U];mask[chosen]=True
   score=np.where(mask,up[idx],p[idx]);order=np.argsort(-score,kind='stable');budget=K+.1*U;review=int(np.floor(budget-.1*mask.sum()+1e-9));review=max(0,min(n,review));tp=int(y[idx[order[:review]]].sum());caps.append(tp/n*1000)
   if wi==0:
    row=rows[rows.policy.eq(mode)].iloc[0];check([tp,mask.sum(),review],[row.captured_events,row.upgraded_n,row.reviewed_n]);pointchecked+=3
  a=np.asarray(caps);values[mode]=a;row=rows[rows.policy.eq(mode)].iloc[0];check(a[0],row.captured_per1000);check(np.quantile(a[1:],[.025,.975]),[row.captured_per1000_lo,row.captured_per1000_hi])
 for mode,a in values.items():
  delta=a-values['no_collection'];q=dif[dif.policy.eq(mode)].iloc[0];check(delta[0],q.delta_captured_per1000);check(np.quantile(delta[1:],[.025,.975]),[q.lo,q.hi])
 # Center counts, probabilities and fixed source thresholds, including both frozen roles.
 c=read(R/'results/CENTER_METRICS.csv');c=c[c.context.eq(tag)];meta=read(R/'private'/f'{tag}_DECISIONS.csv')
 for _,row in c.iterrows():
  pp=p if row.role=='base' else up;ll=bl if row.role=='base' else json.loads(Path(lock['upgrade_model']['path']).read_text());mask=np.ones(len(y),bool) if str(row.center)=='all' else meta.center.astype(str).eq(str(row.center)).to_numpy();yy=y[mask];pr=pp[mask];alert=pr>=ll['training_thresholds'][row.rule];check([len(yy),yy.sum(),pr.sum(),np.square(yy-pr).mean(),alert.sum(),(alert*yy).sum()],[row.N,row.events,row.expected_events,row.Brier,row.reviewed_n,row.captured_events])
  if len(np.unique(yy))==2:
   ranks=rankdata(pr);n1=yy.sum();n0=len(yy)-n1;auc=(ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0);check(auc,row.AUROC)
 return {'context':tag,'replicates_rebuilt':len(w)-1,'policies':6,'equal_budget_primary_point_and_CI':'PASS_expanded_records','center_rows_checked':len(c),'counts_checked':pointchecked,'new_fits':0}
def verify_mimic():
 M=O/'novelty_strengthening_20261101_v1/mimic_calendar_v2';meta=read(M/'PRIVATE_METADATA.csv').set_index('record_key');lab=pd.read_csv(M/'PRIVATE_SELECTED_LAB_EVENTS.csv',dtype=str);con=json.loads((O/'novelty_strengthening_20261101_v1/qualification/MIMIC_LAB_AND_ENDPOINT_PRELOCK.json').read_text());ids={str(x['itemid']) for x in con['selected_dictionary_entries'] if x['label']=='Creatinine'};lab=lab[lab.itemid.isin(ids)].dropna(subset=['valuenum','valueuom','charttime']).copy();lab['chart']=pd.to_datetime(lab.charttime);lab['store']=pd.to_datetime(lab.storetime);lab['value']=lab.apply(lambda v:Fraction(v.valuenum)/(Fraction('88.4') if v.valueuom.lower().replace('µ','u').replace('μ','u')=='umol/l' else 1),axis=1);private=read(R/'private/MIMIC_CLOCK_REPLAY.csv').set_index('record_key');de=read(R/'private/MIMIC_MEASUREMENT_DELETION.csv');checks=0
 for key,m in meta.iterrows():
  cut=pd.Timestamp(m.procedure_date);q=lab[lab.subject_id.eq(str(m.subject_id))].sort_values(['chart','store','labevent_id']);q=q[q.value>0];base=q[(q.chart>=cut-pd.Timedelta(days=30))&(q.chart<cut)&(q.store<cut)].sort_values(['chart','store']).iloc[-1];post=q[(q.chart>=cut+pd.Timedelta(days=1))&(q.chart<cut+pd.Timedelta(days=8))].copy();post['day']=(post.chart-cut).dt.days;pos=post[post.value>=Fraction(3,2)*base.value];assert bool(len(pos))==bool(m.y_scr_calendar)
  absolute=relative=0
  for _,v in post.iterrows():
   a=q[(q.chart<v.chart)&(q.chart>=v.chart-pd.Timedelta(hours=48))];b=q[(q.chart<v.chart)&(q.chart>=v.chart-pd.Timedelta(days=7))];absolute|=int(any(v.value-u>=Fraction(3,10) for u in a.value));relative|=int(any(v.value>=Fraction(3,2)*u for u in b.value))
  check([len(post),absolute,relative,int(bool(absolute or relative))],[private.loc[key,'post_n'],private.loc[key,'absolute_03_within48h_measured'],private.loc[key,'relative_15_prior7d_measured'],private.loc[key,'strict_measured_scr_component']]);checks+=4
  for mode in de['mode'].unique():
   selected=post.groupby('day',sort=False).head(1) if mode=='first_per_calendar_day' else post[post.day.mod(2).eq(1)] if mode=='odd_calendar_days' else post[post.day.mod(2).eq(0)] if mode=='even_calendar_days' else post[post.day.eq(1)] if mode=='calendar_day1_only' else post[post.day.le(2)];present=int(any(v>=Fraction(3,2)*base.value for v in selected.value));row=de[de.record_key.eq(key)&de['mode'].eq(mode)].iloc[0];check([len(selected),present,int(bool(len(pos)) and not present)],[row.kept_n,row.retained_known_event,row.lost_known_event]);checks+=3
 return {'records':len(meta),'numeric_checks':checks,'strict_48h_7d_criteria':'PASS_alternative_dataframe_fraction_replay','deletion_masks':'PASS_all5_schemes_all759records'}
def other():
 low=read(R/'results/LOW_RISK_EVENTS.csv');checked=0
 for _,row in low.iterrows():
  d=read(R/'private'/f'{row.context}_DECISIONS.csv');pp=d.pbase if row.role=='base' else d.pupgrade;mask=pp<row.threshold;check([mask.sum(),d.loc[mask,'y'].sum()],[row.low_risk_n,row.low_risk_events]);n=mask.sum();a=d.loc[mask,'y'].sum();z=1.959963984540054;phat=a/n;ctr=(phat+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n);check([ctr-half,ctr+half],[row.wilson_lo,row.wilson_hi]);checked+=4
 d=read(R/'results/DAILY_POTENTIAL_DEMAND.csv');s=read(R/'results/DAILY_SUMMARY.csv');dd=0
 for _,row in s.iterrows():
  q=d[d.context.eq(row.context)&d.center.eq(row.center)];q=q[q.operations>0] if row.denominator=='active_operation_days' else q;v=q[row.metric];check([len(q),v.sum(),v.mean(),v.quantile(.95),v.max()],[row.days,row.total,row['mean'],row.p95,row['max']]);dd+=5
 # Recorded interval measured at the fixed landmark, never patient-visible prediction availability.
 source=read(O/'benchmark_and_leadtime_20261024_v1/PRIVATE_LEADTIME_JOIN.csv');ev=source[source.y_scr.eq(1)];hours=(ev.first_observed_criterion_time-ev.opstart_time)/60;check(hours,ev.scheduled_landmark_to_recorded_criterion_hours);lead={'all_recorded_events':len(ev),'median_hours':float(hours.median()),'Q1_hours':float(hours.quantile(.25)),'Q3_hours':float(hours.quantile(.75)),'prediction_release_timestamp_present':False,'label':'fixed surgery-start landmark to first recorded SCr-criterion interval'}
 (R/'audit/INSPIRE_RECORD_INTERVAL_RECHECK.json').write_text(json.dumps(lead,indent=2))
 # Verify all hard-outcome point discriminations using an alternate ranks implementation on owning source.
 registry=read(R/'private/REGISTRY_OUTCOMES.csv').set_index('excel_row');stats=read(R/'results/HARD_OUTCOME_DISCRIMINATION.csv');hc=0
 for _,row in stats.iterrows():
  if not row.population_model.startswith('Chinese_'):continue
  _,line,k=row.population_model.split('_');branch='compact' if k=='9' else 'full';pr=read(next((O/f'formal_evaluation_20261023_v1/{line}_preop/private_predictions').glob(f'preop_{branch}_champion_*/PREDICTIONS.csv'))).set_index('excel_row');col={'recorded90d_death':'X','recorded_RRT':'U','recorded30d_reoperation':'V','recorded30d_readmission':'W'}[row.outcome];yy=registry.loc[pr.index,col];good=yy.notna();yy=yy[good].to_numpy();pp=pr.loc[good,'prob_calibrated'].to_numpy();ranks=rankdata(pp);a=yy.sum();b=len(yy)-a;auc=(ranks[yy==1].sum()-a*(a+1)/2)/(a*b);check([len(yy),a,auc],[row.N,row.events,row.AUROC]);hc+=3
 return {'low_risk_point_and_Wilson_checks':checked,'daily_table_summary_scalar_checks':dd,'hard_outcome_rank_point_checks':hc,'INSPIRE_leadtime':'PASS_recorded_interval_only'}
def main():
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:contexts=list(pool.map(verify_context,['geography_preop','geography_paired','temporal_preop','temporal_paired']))
 (R/'audit/INDEPENDENT_CONTEXT_CHECKPOINT.json').write_text(json.dumps(contexts,indent=2));mimic=verify_mimic();otherproof=other();out={'status':'PASS_INDEPENDENT_NUMERIC_IMPLEMENTATIONS','contexts':contexts,'MIMIC':mimic,'other':otherproof,'seconds':time.time()-start,'same_reviewer_alternative_algorithms_not_independent_human_review':True};(R/'audit/INDEPENDENT_NUMERIC_VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
