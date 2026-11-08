from pathlib import Path
import os,sys,json,hashlib,datetime,time,resource,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,pandas as pd
from fractions import Fraction
from scipy.special import logit,expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
P=Path('/home/lei/codex-workspaces/aki-prediction-manuscript');C=P/'correction_runs/full_data_20261025_postslot';O=C/'outputs';R=Path(__file__).resolve().parents[1];M=O/'novelty_strengthening_20261101_v1/mimic_calendar_v2';H={};start=time.time();B=2000;SEED=20261108

def sha(p):
 p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();H[str(p)]={'sha256':h,'bytes':p.stat().st_size,'resolved':str(p.resolve())};return h
def js(p):sha(p);return json.loads(Path(p).read_text())
def read(p,**kw):sha(p);return pd.read_csv(p,float_precision='round_trip',**kw)
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')
def table(n,d):pd.DataFrame(d).to_csv(R/n,index=False)
def ci(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return list(map(float,np.quantile(v,[.025,.975]))) if len(v) else [None,None]
def weights(g,seed):
 un,inv=np.unique(g,return_inverse=True);a=np.random.default_rng(seed).multinomial(len(un),np.full(len(un),1/len(un)),size=B).astype(np.int16)[:,inv];return np.vstack([np.ones((1,len(g)),np.int16),a])
def aucboot(y,p,w):
 idx=np.argsort(p,kind='stable');scores=p[idx];first=np.r_[0,np.flatnonzero(np.diff(scores)!=0)+1];a=np.add.reduceat(w[:,idx]*y[idx],first,axis=1);b=np.add.reduceat(w[:,idx]*(1-y[idx]),first,axis=1);num=np.sum(a*(np.cumsum(b,axis=1)-.5*b),axis=1);den=(w@y)*(w@(1-y));return np.divide(num,den,out=np.full(len(w),np.nan),where=den>0)

def mimic():
 iq=js(M/'INPUT_QUALIFICATION.json');assert (iq['n'],iq['events'])==(759,59)
 for fn,h in iq['files'].items():assert sha(M/fn)==h
 meta=read(M/'PRIVATE_METADATA.csv',dtype=str).set_index('record_key',drop=False);labs=read(M/'PRIVATE_SELECTED_LAB_EVENTS.csv',dtype=str);contract=js(O/'novelty_strengthening_20261101_v1/qualification/MIMIC_LAB_AND_ENDPOINT_PRELOCK.json');crid={str(x['itemid']) for x in contract['selected_dictionary_entries'] if x['label']=='Creatinine'}
 by={}
 for r in labs.loc[labs.itemid.isin(crid)].to_dict('records'):
  try:
   unit=r['valueuom'].strip().lower().replace('µ','u').replace('μ','u');v=Fraction(r['valuenum']);v=v if unit=='mg/dl' else v/Fraction('88.4') if unit=='umol/l' else None
   if v is None or v<=0:continue
   ct=datetime.datetime.fromisoformat(r['charttime']); st=datetime.datetime.fromisoformat(r['storetime']) if isinstance(r['storetime'],str) else None
  except (ValueError,AttributeError,TypeError):continue
  by.setdefault(r['subject_id'],[]).append((ct,st,v,r['labevent_id']))
 rows=[];postby={};daily=[];deleted=[]
 for key,m in meta.iterrows():
  cut=datetime.datetime.fromisoformat(m.procedure_date);alllab=sorted(by[m.subject_id],key=lambda x:(x[0],str(x[1]),x[3]));base=[x for x in alllab if cut-datetime.timedelta(days=30)<=x[0]<cut and x[1] and x[1]<cut];base=max(base,key=lambda x:(x[0],x[1],int(x[3])));posts=[x for x in alllab if cut+datetime.timedelta(days=1)<=x[0]<cut+datetime.timedelta(days=8)];assert len(posts)==int(m.post_Cr_n)
  ratio=[x for x in posts if x[2]>=Fraction(3,2)*base[2]];assert int(bool(ratio))==int(m.y_scr_calendar)
  if ratio:assert ratio[0][0]==datetime.datetime.fromisoformat(m.first_recorded_positive_charttime)
  absolute=[];relative=[];prevalent=[]
  for x in alllab:
   prior48=[v for v in alllab if x[0]-datetime.timedelta(hours=48)<=v[0]<x[0]];prior7=[v for v in alllab if x[0]-datetime.timedelta(days=7)<=v[0]<x[0]]
   a=any(x[2]-v[2]>=Fraction(3,10) for v in prior48);b=any(x[2]>=Fraction(3,2)*v[2] for v in prior7)
   if x[0]<cut and x[0]>=cut-datetime.timedelta(days=2) and (a or b):prevalent.append(x)
   if x in posts:
    if a:absolute.append(x)
    if b:relative.append(x)
  strict=sorted(absolute+relative,key=lambda x:x[0]);storelag=[(x[1]-x[0]).total_seconds()/3600 for x in posts if x[1]]
  rows.append({'record_key':key,'group_hash':m.group_hash,'y_calendar_ratio':int(bool(ratio)),'post_n':len(posts),'baseline_age_hours_at_calendar_t0':(cut-base[0]).total_seconds()/3600,'baseline_stored_before_t0':base[1]<cut,'first_ratio_chart':ratio[0][0].isoformat() if ratio else None,'first_ratio_store':ratio[0][1].isoformat() if ratio and ratio[0][1] else None,'absolute_03_within48h_measured':int(bool(absolute)),'relative_15_prior7d_measured':int(bool(relative)),'strict_measured_scr_component':int(bool(strict)),'first_strict_chart':strict[0][0].isoformat() if strict else None,'pre_day0_measured_component_in_prior48h':int(bool(prevalent)),'post_store_missing_n':sum(x[1] is None for x in posts),'post_negative_store_delay_n':sum(h<0 for h in storelag),'median_store_minus_chart_hours':float(np.median(storelag)) if storelag else None,'surgical_start_end_unknown':True,'urine_criterion_unknown':True})
  postby[key]=(base,posts,ratio)
  for day in range(1,8):
   dd=[x for x in posts if (x[0]-cut).days==day];daily.append({'record_key':key,'day':day,'measured_n':len(dd),'measured':int(bool(dd)),'calendar_ratio_positive':int(any(x[2]>=Fraction(3,2)*base[2] for x in dd))})
  for mode in ['first_per_calendar_day','odd_calendar_days','even_calendar_days','calendar_day1_only','calendar_days1_2_only']:
   keep=[];seen=set()
   for x in posts:
    day=(x[0]-cut).days
    ok=(day not in seen) if mode=='first_per_calendar_day' else day%2==1 if mode=='odd_calendar_days' else day%2==0 if mode=='even_calendar_days' else day==1 if mode=='calendar_day1_only' else day<=2
    if ok:keep.append(x)
    seen.add(day)
   pos=[x for x in keep if x[2]>=Fraction(3,2)*base[2]]
   deleted.append({'record_key':key,'group_hash':m.group_hash,'mode':mode,'original_n':len(posts),'kept_n':len(keep),'zero_remaining_measurements':int(not keep),'original_y':int(bool(ratio)),'retained_known_event':int(bool(pos)),'lost_known_event':int(bool(ratio) and not pos),'retained_detection_delay_hours':(pos[0][0]-ratio[0][0]).total_seconds()/3600 if pos and ratio else None})
 rr=pd.DataFrame(rows).set_index('record_key');de=pd.DataFrame(deleted);da=pd.DataFrame(daily);w=weights(meta.group_hash.to_numpy(),SEED+21);n=len(meta);summary=[];strata={'all_observed_records':np.ones(n,bool)}
 lockacc=js(O/'common6_model_lock_20261024_v1/STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json');low=[];intensity=[]
 for ref in lockacc['models']:
  assert sha(ref['path'])==ref['sha256'];lock=js(ref['path']);line=lock['identity']['line'];pred=read(M/line/'PRIVATE_PREDICTIONS.csv',dtype={'record_key':str}).set_index('record_key').loc[meta.index];rec=js(M/line/'PREDICTION_ACCEPTANCE.json');assert sha(M/line/'PRIVATE_PREDICTIONS.csv')==rec['prediction_sha256'];p=pred.prob_calibrated.to_numpy();y=rr.y_calendar_ratio.to_numpy();threshold=lock['training_thresholds']['training_sensitivity80'];lo=p<threshold;strata[line+'_low_risk6']=lo
  dn=w@lo;ev=w@(lo*y);rate=np.divide(ev,dn,out=np.full(len(ev),np.nan),where=dn>0);wil=proportion_confint(int(np.sum(lo*y)),int(lo.sum()),method='wilson') if lo.sum() else [None,None]
  low.append({'line':line,'population':'MIMIC_current_calendar759','inputs':6,'N':n,'events':int(y.sum()),'threshold':threshold,'threshold_origin':'source training_sensitivity80','low_risk_n':int(lo.sum()),'low_risk_events':int((lo*y).sum()),'low_risk_rate':float(np.sum(lo*y)/lo.sum()) if lo.sum() else None,'wilson_lo':wil[0],'wilson_hi':wil[1],'patient_lo':ci(rate[1:])[0],'patient_hi':ci(rate[1:])[1],'NPV_not_stopping_safety':True})
  for category in ['1','2to3','4to6','7plus']:
   c=rr.post_n.to_numpy();mask=c==1 if category=='1' else (c>=2)&(c<=3) if category=='2to3' else (c>=4)&(c<=6) if category=='4to6' else c>=7
   if not mask.any():continue
   yy=y[mask];pp=p[mask];intensity.append({'line':line,'observed_post_Cr_n':category,'N':int(mask.sum()),'events':int(yy.sum()),'rate':float(yy.mean()),'mean_predicted_risk':float(pp.mean()),'Brier':float(np.mean((yy-pp)**2)),'AUROC':float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,'meaning':'observation-associated subgroup; not causal detection effect'})
 for label,mask in strata.items():
  for mode,d in de.groupby('mode',sort=True):
   d=d.set_index('record_key').loc[meta.index];sub=d.loc[mask];ww=w[:,mask];loss=ww@sub.lost_known_event.to_numpy();orig=ww@sub.original_y.to_numpy();kept=ww@sub.kept_n.to_numpy();total=ww@sub.original_n.to_numpy();frac=np.divide(loss,orig,out=np.full(len(loss),np.nan),where=orig>0)
   summary.append({'population':label,'mode':mode,'N':int(mask.sum()),'original_known_events':int(sub.original_y.sum()),'retained_known_events':int(sub.retained_known_event.sum()),'lost_known_events_lower_bound':int(sub.lost_known_event.sum()),'lost_known_fraction':float(sub.lost_known_event.sum()/sub.original_y.sum()) if sub.original_y.sum() else None,'lost_known_fraction_lo':ci(frac[1:])[0],'lost_known_fraction_hi':ci(frac[1:])[1],'original_measurements':int(sub.original_n.sum()),'remaining_measurements':int(sub.kept_n.sum()),'deleted_measurements':int(sub.original_n.sum()-sub.kept_n.sum()),'deleted_fraction':float(1-sub.kept_n.sum()/sub.original_n.sum()) if sub.original_n.sum() else None,'deleted_fraction_lo':ci((1-kept[1:]/np.maximum(total[1:],1)))[0],'deleted_fraction_hi':ci((1-kept[1:]/np.maximum(total[1:],1)))[1],'zero_remaining_records':int(sub.zero_remaining_measurements.sum()),'retained_detection_delay_hours_median':float(sub.retained_detection_delay_hours.dropna().median()) if sub.retained_detection_delay_hours.notna().any() else None,'unobserved_counterfactual_events':'not identifiable','stopping_monitoring_safe':False})
 dayagg=da.groupby('day').agg(records=('record_key','size'),measured_records=('measured','sum'),measurements=('measured_n','sum'),known_ratio_positive_records=('calendar_ratio_positive','sum')).reset_index()
 aud={'population':n,'frozen_calendar_events':int(rr.y_calendar_ratio.sum()),'absolute_03_within48h_measured':int(rr.absolute_03_within48h_measured.sum()),'relative15_prior7d_measured':int(rr.relative_15_prior7d_measured.sum()),'strict_measured_scr_component':int(rr.strict_measured_scr_component.sum()),'strict_and_calendar':int((rr.strict_measured_scr_component.astype(bool)&rr.y_calendar_ratio.astype(bool)).sum()),'strict_outside_calendar':int((rr.strict_measured_scr_component.astype(bool)&~rr.y_calendar_ratio.astype(bool)).sum()),'calendar_without_strict':int((~rr.strict_measured_scr_component.astype(bool)&rr.y_calendar_ratio.astype(bool)).sum()),'baseline_more7days_before_procedure_day0':int((rr.baseline_age_hours_at_calendar_t0>168).sum()),'pre_day0_component_known_prior48h':int(rr.pre_day0_measured_component_in_prior48h.sum()),'post_store_missing_n':int(rr.post_store_missing_n.sum()),'post_negative_store_delay_n':int(rr.post_negative_store_delay_n.sum()),'true_warning_leadtime_identified':False,'reason':'Procedure billing date only, no incision/end clock; fixed6 inputs not9/27/38, release timestamps incomplete for all fields; no urine output validation; windows differ, preserved original labels.'}
 save('audit/MIMIC_KDIGO_CLOCK_AUDIT.json',aud);rr.to_csv(R/'private/MIMIC_CLOCK_REPLAY.csv');de.to_csv(R/'private/MIMIC_MEASUREMENT_DELETION.csv',index=False);table('results/MIMIC_LOW_RISK_EVENTS.csv',low);table('results/MIMIC_OBSERVATION_INTENSITY.csv',intensity);table('results/MIMIC_DELETION_SENSITIVITY.csv',summary);table('results/MIMIC_DAILY_OBSERVATION.csv',dayagg)
 print('MIMIC_OBSERVATION_DONE',aud,flush=True)
 return aud

def outcomes():
 raw=P/'raw/inter3.xlsx';sha(raw)
 cols=['O','P','U','V','W','X'];d=pd.read_excel(raw,sheet_name='Sheet1',usecols=','.join(cols),dtype=object,engine='openpyxl');d.columns=cols;d.index=np.arange(2,len(d)+2);d.index.name='excel_row'
 overlay=read(P/'outputs/full_data_audit_20261025_v1/candidate_release/SOURCE_VALUE_OVERLAY.csv',dtype=str);amended=js(O/'corrected_support_sources/CONFIRMED_ADDITIONAL_DATE_RULES.json')
 for r in overlay.to_dict('records'):
  if r['sheet']!='Sheet1':continue
  col=''.join(x for x in r['cell'] if x.isalpha());row=int(''.join(x for x in r['cell'] if x.isdigit()))
  if col in cols:d.loc[row,col]=r['analysis_value']
 for row,r in amended.items():
  if r.get('duration_amended'):
   for col in ['O','P']:
    if col in r.get('durations',{}):d.loc[int(row),col]=r['durations'][col]
 for col in cols:d[col]=pd.to_numeric(d[col],errors='coerce')
 table('audit/REGISTRY_OUTCOME_SCHEMA.csv',[{'column':col,'meaning':meaning,'recorded_n':int(d[col].notna().sum()),'missing_n':int(d[col].isna().sum()),'negative_n':int((d[col]<0).sum()),'nonbinary_n':int((d[col].notna()&~d[col].isin([0,1])).sum()) if col not in ['O','P'] else None} for col,meaning in zip(cols,['recorded_total_LOS_days','recorded_postoperative_LOS_days','recorded_RRT_flag','recorded30d_reoperation_flag','recorded30d_readmission_flag','recorded90d_death_flag'])])
 meta=read(O/'v4_development_inputs_20261018/preop_metadata.csv').set_index('excel_row');clin=read(P/'outputs/full_data_audit_20261025_v1/candidate_release/clinical/preop_cohort.csv').set_index('excel_row');np.testing.assert_array_equal(meta.loc[clin.index,'y'],clin.y)
 # RRT analyzed with explicit incorporation/time ambiguity; use unchanged accepted source flag, not missing=>0.
 d.loc[clin.index,'U']=clin.acute_postop_RRT_confirmed.astype(int)
 table('private/REGISTRY_OUTCOMES.csv',d.reset_index().to_dict('records'))
 stats=[];assoc=[];los=[];sources=[]
 def evaluate(label,g,pp,xx,yy,cut=None):
  g=np.asarray(g);pp=np.asarray(pp); yy=np.asarray(yy,float);valid=np.isfinite(yy)&np.isfinite(pp);ev=yy[valid];pr=pp[valid];gr=g[valid];N=len(ev)
  if not N:return
  w=weights(gr,SEED+42);is_binary=xx not in ['total_LOS_days','postoperative_LOS_days','operation_end_to_discharge_days']
  if is_binary:
   assert set(np.unique(ev)).issubset({0,1});E=int(ev.sum());au=aucboot(ev,pr,w);e=w@ev;den=w.sum(1);rate=e/den;valid_auc=int(np.isfinite(au[1:]).sum())
   stats.append({'population_model':label,'outcome':xx,'N':N,'missing_or_ineligible_n':int(len(yy)-N),'events':E,'event_rate':float(ev.mean()),'event_rate_lo':ci(rate[1:])[0],'event_rate_hi':ci(rate[1:])[1],'AUROC':float(au[0]) if np.isfinite(au[0]) else None,'AUROC_lo':ci(au[1:])[0],'AUROC_hi':ci(au[1:])[1],'valid_AUROC_replicates':valid_auc,'model_probability_target':'AKI, not this hard outcome; discrimination/association only','implementation_benefit':False})
   if E>=5 and N-E>=5:
    predictor=np.log2(np.clip(pr,1e-8,1));X=sm.add_constant(predictor)
    try:
     fit=sm.GLM(ev,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':gr});beta=float(fit.params[1]);ci_beta=np.asarray(fit.conf_int())[1];oo=np.exp(np.r_[beta,ci_beta]);ok=bool(fit.converged) and np.isfinite(oo).all()
     assoc.append({'population_model':label,'outcome':xx,'N':N,'events':E,'effect':'unadjusted odds ratio per doubling frozen predicted AKI risk','OR':float(oo[0]) if ok else None,'lo':float(oo[1]) if ok else None,'hi':float(oo[2]) if ok else None,'status':'association_only_small_event_Wald_cluster_CI' if ok else 'not_estimable','confounding_control':'none; prognostic descriptive association, not treatment effect'})
    except Exception as exc:assoc.append({'population_model':label,'outcome':xx,'N':N,'events':E,'status':'not_estimable_'+type(exc).__name__})
   else:assoc.append({'population_model':label,'outcome':xx,'N':N,'events':E,'status':'too_few_events_for_regression_no_advanced_substitute'})
  else:
   good=ev>=0;ev=ev[good];pr=pr[good];gr=gr[good];w=weights(gr,SEED+43);rho=float(spearmanr(pr,ev).statistic) if np.unique(ev).size>1 and np.unique(pr).size>1 else None;rr=[];med=[]
   for ww in w[1:]:
    idx=np.repeat(np.arange(len(ev)),ww);med.append(float(np.median(ev[idx]))); rr.append(float(spearmanr(pr[idx],ev[idx]).statistic) if rho is not None else np.nan)
   los.append({'population_model':label,'outcome':xx,'N':len(ev),'missing_or_negative_n':len(yy)-len(ev),'median_days':float(np.median(ev)),'Q1_days':float(np.quantile(ev,.25)),'Q3_days':float(np.quantile(ev,.75)),'median_lo':ci(med)[0],'median_hi':ci(med)[1],'Spearman_rho':rho,'rho_lo':ci(rr)[0],'rho_hi':ci(rr)[1],'death_competing_risk':'not modelled; shorter death stays may distort LOS','implementation_benefit':False})
 for line in ['geography','temporal']:
  for stage in ['compact','full']:
   files=list((O/f'formal_evaluation_20261023_v1/{line}_preop/private_predictions').glob(f'preop_{stage}_champion_*/PREDICTIONS.csv'));assert len(files)==1;pr=read(files[0]).set_index('excel_row');xx=d.loc[pr.index];yy=pr.y.to_numpy();assert set(pr.index)==set(meta.index[meta[line+'_partition_v2'].eq('validation')]);label=f'Chinese_{line}_{9 if stage=="compact" else 27}'
   for col,name in [('X','recorded90d_death'),('U','recorded_RRT'),('V','recorded30d_reoperation'),('W','recorded30d_readmission'),('O','total_LOS_days'),('P','postoperative_LOS_days')]:evaluate(label,pr.group_hash,pr.prob_calibrated,name,xx[col])
   sources.append({'population_model':label,'N':len(pr),'source_sheet':'Sheet1+accepted_overlay+date_duration_confirmations; RRT from accepted confirmed acute-postop flag','90d_followup_dates_available':False,'RRT_start_stop_available':False,'implementation_exposure_available':False,'RRT_incorporated_into_AKI_endpoint':True})
 # Public independent cohort6 scores, exact operation clock, outcome timestamps available.
 fp=read(O/'public_eligibility_refinement_20261022_v1/PRIVATE_FIRST_OPERATION_PROFILE.csv').set_index('op_id');mm=read(O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_METADATA.csv').set_index('record_key');j=read(O/'benchmark_and_leadtime_20261024_v1/PRIVATE_LEADTIME_JOIN.csv').set_index('record_key');assert len(j)==6043
 for line in ['geography','temporal']:
  pp=read(O/f'common6_public_evaluation_20261024_v1/{line}/PRIVATE_PREDICTIONS.csv').set_index('record_key');m=mm.loc[pp.index];prof=fp.loc[m.op_id];assert len(pp)==6043;np.testing.assert_array_equal(pp.y_scr,m.y_scr)
  death=prof.inhosp_death_time.notna().to_numpy();validdeath=((prof.inhosp_death_time>=prof.opend_time)&(prof.inhosp_death_time<=prof.discharge_time)).to_numpy();bad=death&~validdeath; raw_death=death.copy(); death=death&validdeath
  label='INSPIRE_'+line+'_6';evaluate(label,pp.group_hash,pp.prob_calibrated,'recorded_inhospital_death',death.astype(int));evaluate(label,pp.group_hash,pp.prob_calibrated,'recorded_postwindow_RRT',pd.to_numeric(m.post_window_RRT_positive).to_numpy());length=(prof.discharge_time-prof.opend_time).to_numpy()/1440;evaluate(label,pp.group_hash,pp.prob_calibrated,'operation_end_to_discharge_days',length)
  sources.append({'population_model':label,'N':len(pp),'source_sheet':'locked INSPIRE first-operation profile6043','in_hospital_death_recorded_n':int(death.sum()),'death_outside_index_admission_n':int(bad.sum()),'all_recorded_inhospital_death_timestamps':int(raw_death.sum()),'exact_operation_end_available':True,'implementation_exposure_available':False,'selection_on_Cr_observation':True,'blank_death_meaning':'no recorded index-admission death timestamp; later institutional deaths retained separately, external followup incomplete','RRT_prior_positive_n':int(pd.to_numeric(m.recorded_prior_RRT_positive).sum())})
 table('results/HARD_OUTCOME_DISCRIMINATION.csv',stats);table('results/HARD_OUTCOME_ASSOCIATIONS.csv',assoc);table('results/LOS_ASSOCIATIONS.csv',los);table('audit/HARD_OUTCOME_SOURCE_AUDIT.csv',sources)
 save('audit/CAUSAL_IDENTIFIABILITY.json',{'implementation_treatment_recorded':False,'actual_action_and_stop_times':False,'common_implementation_time_zero':False,'indications_and_rescue_rules':False,'time_varying_confounding_observed_for_action':False,'positivity_for_implementation_treatment_assessable':False,'implementation_effect_estimated':False,'valid_answer':'frozen AKI-score prognostic associations; no clinical implementation effect; no IV invented'})
 print('OUTCOME_ANALYSES_DONE',flush=True)

def main():
 lock=js(R/'PROTOCOL_LOCK.json');assert sha(R/'REQUEST.json')==lock['request_sha256'];assert sha(O/'v4_development_inputs_20261018/SOURCE_LOCK.json')==lock['source_sha256']
 
 if '--outcomes-only' not in sys.argv:mimic()
 else:
  iq=js(M/'INPUT_QUALIFICATION.json')
  for fn,h in iq['files'].items():assert sha(M/fn)==h
 outcomes();sha(Path(__file__));save('audit/OBSERVATION_OUTCOME_INPUT_HASHES.json',H);save('audit/OBSERVATION_OUTCOME_RUN.json',{'status':'COMPUTED_REQUIRES_INDEPENDENT_REVIEW','B':B,'seed':SEED,'seconds':time.time()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'pid':os.getpid(),'inputs_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in H.items())});print('ALL_DONE',time.time()-start,flush=True)
if __name__=='__main__':main()
