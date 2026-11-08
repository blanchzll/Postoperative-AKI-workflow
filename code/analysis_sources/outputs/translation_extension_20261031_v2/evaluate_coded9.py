"""Frozen China9 models on explicitly zero-coded preoperative released histories; sensitivity only."""
from pathlib import Path
import os,sys,json,hashlib,fcntl,time,traceback,resource,gc
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');O=R/'outputs';T=O/'translation_extension_20261031_v2'
sys.path.insert(0,str(R/'work/model_lock_20261022_v1'));import runtime as native
sys.path.insert(0,str(R/'work/formal_evaluation_20261023_v1'));import evaluation_stats as stats
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):native.atomic(p,x)
def verify_pin():
 lock=read(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json')
 for p,h in lock['source_files_sha256'].items():assert sha(p)==h,p
 for it in lock['source_models']:
  assert sha(it['model_lock'])==it['model_lock_sha256']
  for p,h in it['components'].items():assert sha(p)==h,p
 return lock
def main():
 start=time.time();guard=(T/'RUNNER.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inputs=read(T/'HISTORY_PROXY_INPUT_ACCEPTANCE.json');assert inputs['nine_code_proxy_sensitivity_allowed'] and not inputs['full_nine_input_semantic_qualification']
 for n,h in inputs['files'].items():assert sha(T/n)==h,n
 pins=verify_pin();pre=O/'public_eligibility_refinement_20261022_v1'
 x=pd.read_csv(T/'PRIVATE_CODED_NINE_FEATURES.csv',dtype={k:'string' for k in ['record_key','group_hash','Sex','Diabetes','Hypertension','HeartDisease']},float_precision='round_trip').set_index('record_key',drop=False).sort_index()
 m=pd.read_csv(pre/'PRIVATE_COMMON6_METADATA.csv',dtype={'record_key':str,'subject_id':str,'op_id':str},float_precision='round_trip').set_index('record_key',drop=False).loc[x.index]
 assert len(x)==6043 and int(m.y_scr.sum())==253 and x.group_hash.is_unique
 six=pd.read_csv(pre/'PRIVATE_COMMON6_FEATURES.csv',dtype={k:'string' for k in ['record_key','group_hash','Sex']},float_precision='round_trip').set_index('record_key',drop=False).loc[x.index]
 for c in six:assert x[c].equals(six[c]),c
 codes=['Diabetes','Hypertension','HeartDisease']
 assert x[codes].notna().all().all() and all(x[c].isin(['0','1']).all() for c in codes)
 original=O/'common6_public_evaluation_20261024_v1'
 stage=read(original/'STAGE_PUBLIC_ACCEPTANCE.json')
 models={it['line']:read(it['model_lock']) for it in pins['source_models']}
 scorelock={'status':'LOCKED_BEFORE_ANY_NEW_PUBLIC9_PREDICTION','history_prelock_sha256':sha(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json'),'input_acceptance_sha256':sha(T/'HISTORY_PROXY_INPUT_ACCEPTANCE.json'),'input_feature_sha256':sha(T/'PRIVATE_CODED_NINE_FEATURES.csv'),'code_sha256':sha(__file__),'native_runtime_sha256':sha(R/'work/model_lock_20261022_v1/runtime.py'),'evaluation_code_sha256':sha(R/'work/formal_evaluation_20261023_v1/evaluation_stats.py'),'native_metric_code_sha256':sha(R/'work/final_evaluation_engine_20261022_v1/fixed_prediction_metrics.py'),'source_models':pins['source_models'],'six_existing_prediction_sha256':{line:sha(original/line/'PRIVATE_PREDICTIONS.csv') for line in ['geography','temporal']},'n':6043,'events':253,'no_target_recalibration_selection_or_new_fit':True,'scope':'Coded history proxy sensitivity, redaction and ascertainment differences retained; original6RF remains primary public evidence.'}
 p=T/'PRE_SCORE_LOCK.json'
 if p.exists():assert read(p)==scorelock
 else:save(p,scorelock)
 y=m.y_scr.to_numpy(int);g=x.group_hash.astype(str).to_numpy();r=x.record_key.astype(str).to_numpy();keys=(x.group_hash.astype(str)+'|'+x.record_key.astype(str)).tolist()
 predictions={};newcuts={}
 for line in ['geography','temporal']:
  old=pd.read_csv(original/line/'PRIVATE_PREDICTIONS.csv',dtype={'record_key':str,'group_hash':str},float_precision='round_trip').set_index('record_key',drop=False).loc[x.index]
  assert old.group_hash.tolist()==list(g) and old.y_scr.tolist()==list(y)
  for mode in ['raw','calibrated']:predictions[line+'__six_RF__'+mode]=old['prob_'+mode].to_numpy()
  manifest=models[line];name=line+'__coded9_'+manifest['family'];d=T/line;d.mkdir(mode=0o700,exist_ok=True)
  if (d/'PREDICTION_ACCEPTANCE.json').exists():
   a=read(d/'PREDICTION_ACCEPTANCE.json');assert a['pre_score_lock_sha256']==sha(p) and a['prediction_sha256']==sha(d/'PRIVATE_PREDICTIONS.csv')
   z=pd.read_csv(d/'PRIVATE_PREDICTIONS.csv',dtype={'record_key':str,'group_hash':str},float_precision='round_trip');assert z.record_key.tolist()==list(r) and z.y_scr.tolist()==list(y)
   raw,cal=z.prob_raw.to_numpy(),z.prob_calibrated.to_numpy()
  else:
   fields=manifest['input_columns'];assert set(fields)==set(six.columns)-{'record_key','group_hash'}|set(codes)
   raw,cal=native.route_prediction(manifest,x[fields],keys)
   # Numerical qualification depends only on feature missingness, never outcomes.
   ix=list(np.flatnonzero(x[fields].isna().any(axis=1))[:2])+list(np.flatnonzero(~x[fields].isna().any(axis=1))[:2])
   ix=list(dict.fromkeys(ix));probe=x[fields].iloc[ix];pk=[keys[j] for j in ix]
   a,b=native.route_prediction(manifest,probe,pk);ar,br=native.route_prediction(manifest,probe.iloc[::-1],pk[::-1])
   singles=[native.route_prediction(manifest,probe.iloc[j:j+1],pk[j:j+1]) for j in range(len(probe))]
   for expected,actual in [(raw[ix],a),(cal[ix],b),(a,ar[::-1]),(b,br[::-1]),(a,np.concatenate([v[0] for v in singles])),(b,np.concatenate([v[1] for v in singles]))]:np.testing.assert_allclose(expected,actual,atol=1e-12,rtol=0)
   z=pd.DataFrame({'record_key':r,'group_hash':g,'y_scr':y,'prob_raw':raw,'prob_calibrated':cal});z.to_csv(d/'PRIVATE_PREDICTIONS.csv',index=False)
   save(d/'PREDICTION_ACCEPTANCE.json',{'status':'PASS_FROZEN_NINE_INPUT_CODE_PROXY_PREDICTIONS','pre_score_lock_sha256':sha(p),'prediction_sha256':sha(d/'PRIVATE_PREDICTIONS.csv'),'n':6043,'events':253,'new_predictive_fits':0,'source_only_preprocessing_and_calibration':True,'batch_singleton_reorder_parity_atol':1e-12,'full_nine_input_semantic_qualification':False})
  for mode,v in [('raw',raw),('calibrated',cal)]:predictions[name+'__'+mode]=v
  newcuts[line]={'name':name+'__calibrated','thresholds':manifest['training_thresholds']}
  print('SCORED_FROZEN_CODED9',line,manifest['family'],flush=True)
 comps=[]
 for line in ['geography','temporal']:
  fam=models[line]['family']
  for mode in ['raw','calibrated']:comps.append((line+'__coded9_'+fam+'__'+mode,line+'__six_RF__'+mode))
 def progress(k):save(T/'PROGRESS.json',{'phase':'paired_fixed_model_public_metric_bootstrap','completed':k,'total':2000,'pid':os.getpid()})
 p=T/'PUBLIC_EVALUATION.json'
 if p.exists():result=read(p);assert read(T/'PUBLIC_EVALUATION_ACCEPTANCE.json')['sha256']==sha(p)
 else:
  result=stats.evaluate(y,predictions,g,r,['INSPIRE']*len(y),{},comps,replicates=2000,seed=20261017,progress=progress)
  save(p,result);save(T/'PUBLIC_EVALUATION_ACCEPTANCE.json',{'sha256':sha(p),'pre_score_lock_sha256':sha(T/'PRE_SCORE_LOCK.json')})
 oldev=read(original/'EVALUATION.json');assert result['receipt']['resample_weight_sha256']==oldev['receipt']['resample_weight_sha256']
 for line in ['geography','temporal']:
  for mode in ['raw','calibrated']:
   a=result['point'][line+'__six_RF__'+mode];b=oldev['point'][line+'__'+mode]
   for metric in ['AUROC','AP','Brier','capacity10_captured_n','capacity20_captured_n','observed_expected']:assert abs(a[metric]-b[metric])<1e-12
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(T/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(T/'INTERVALS.csv',index=False);pd.DataFrame(result['paired_differences']).to_csv(T/'PAIRED_DIFFERENCES_VS_SIX_RF.csv',index=False)
 # New9 source-threshold transfers, same patient-group draws; no target-selected cut.
 yy,gg,rr,cc,strata,risks=stats._design(y,g,r,['INSPIRE']*len(y),predictions)
 configs={(line,rule):float(v) for line,z in newcuts.items() for rule,v in z['thresholds'].items() if rule.startswith('training_')}
 point={k:stats.counts(yy,risks[newcuts[k[0]]['name']].p>=v) for k,v in configs.items()}
 fields=[k for k in next(iter(point.values())) if k not in ['n','events']]
 values={key:{f:[] for f in fields} for key in configs};rng=np.random.default_rng(20261017);proof=hashlib.sha256()
 for rep in range(2000):
  w=stats.group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes());idx=np.repeat(np.arange(len(yy)),w)
  for key,cut in configs.items():
   q=stats.counts(yy[idx],risks[newcuts[key[0]]['name']].p[idx]>=cut)
   for f in fields:
    if q[f] is not None:values[key][f].append(float(q[f]))
 assert proof.hexdigest()==result['receipt']['resample_weight_sha256']
 rows=[]
 for key,cols in values.items():
  for f,v in cols.items():
   ci=np.quantile(v,[.025,.975],method='linear').tolist() if v else [None,None]
   rows.append({'source_line':key[0],'model':newcuts[key[0]]['name'],'rule':key[1],'threshold':configs[key],'metric':f,'estimate':point[key][f],'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(v)})
 pd.DataFrame(rows).to_csv(T/'NINE_SOURCE_THRESHOLD_BURDEN.csv',index=False)
 curves=[]
 for nm,risk in risks.items():
  cum=np.r_[0,np.cumsum(risk.y[risk.order]).astype(int)]
  for k,tp in enumerate(cum):curves.append({'model':nm,'reviewed_n':k,'captured_n':int(tp),'reviewed_per1000':1000*k/len(yy),'captured_per1000':1000*tp/len(yy),'sensitivity':tp/int(yy.sum())})
 pd.DataFrame(curves).to_csv(T/'FULL_CAPTURE_REVIEW_CURVE.csv',index=False)
 verify_pin()
 save(T/'COMPUTATION_ACCEPTANCE.json',{'status':'PASS_CODED9_PROXY_SENSITIVITY_COMPUTATION_PENDING_INDEPENDENT_REVIEW','n':6043,'events':253,'views':8,'new_model_fits_or_target_updates':0,'paired_contrasts':len(comps),'bootstrap_repetitions':2000,'resampling_sha256':proof.hexdigest(),'pre_score_lock_sha256':sha(T/'PRE_SCORE_LOCK.json'),'code_sha256':sha(__file__),'source_history_proxy_not_full_measurement_equivalence':True,'files':{n:sha(T/n) for n in ['POINT_ESTIMATES.csv','INTERVALS.csv','PAIRED_DIFFERENCES_VS_SIX_RF.csv','NINE_SOURCE_THRESHOLD_BURDEN.csv','FULL_CAPTURE_REVIEW_CURVE.csv','PUBLIC_EVALUATION.json','PUBLIC_EVALUATION_ACCEPTANCE.json']},'elapsed_seconds':time.time()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
 guard.close()
if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):main()
 except Exception:
  save(T/'COMPUTE_FAILURE.json',{'traceback':traceback.format_exc(),'pid':os.getpid()});raise
