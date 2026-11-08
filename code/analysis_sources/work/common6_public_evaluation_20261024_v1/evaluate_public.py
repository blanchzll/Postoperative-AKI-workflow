"""Fixed source common6 pipelines on the prespecified INSPIRE SCr cohort; no model fitting."""
from pathlib import Path
import sys,os,json,time,fcntl,hashlib,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/common6_public_evaluation_20261024_v1'
LOCK=ROOT/'outputs/common6_model_lock_20261024_v1'
PUBLIC=ROOT/'outputs/public_eligibility_refinement_20261022_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
import evaluation_stats as stats
sys.path.insert(0,str(ROOT/'work/common6_model_lock_20261024_v1'))
import runtime
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
SIX=['Age','PreopCr','PreopBUN','PreopAlb','PreopHb','Sex']
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def source():
 contract=base.read(PUBLIC/'COMMON6_INPUT_CONTRACT.json')
 assert set(contract['features'])==set(SIX)
 for f,h in contract['files'].items():assert base.sha(PUBLIC/f)==h
 x=pd.read_csv(PUBLIC/'PRIVATE_COMMON6_FEATURES.csv',dtype={'record_key':str,'group_hash':str,'Sex':'string'},float_precision='round_trip')
 m=pd.read_csv(PUBLIC/'PRIVATE_COMMON6_METADATA.csv',dtype={'record_key':str,'subject_id':str,'op_id':str},float_precision='round_trip')
 assert x.record_key.is_unique and m.record_key.is_unique and set(x.record_key)==set(m.record_key)
 x=x.set_index('record_key',drop=False).sort_index();m=m.set_index('record_key',drop=False).loc[x.index]
 assert len(m)==6043 and int(m.y_scr.sum())==253
 assert m.y_scr.isin([0,1]).all() and m.stage_scr.isin([0,1,2,3]).all()
 assert np.array_equal(m.y_scr.to_numpy(),(m.stage_scr.to_numpy()>0).astype(int))
 assert x[SIX].notna().any().all() and x.Sex.dropna().isin(['0','1']).all()
 assert set(x.columns)==set(SIX)|{'record_key','group_hash'}
 assert not x.group_hash.isna().any()
 pairs=pd.DataFrame({'subject':m.subject_id.to_numpy(),'group':x.group_hash.to_numpy()})
 assert pairs.groupby('subject').group.nunique().max()==1 and pairs.groupby('group').subject.nunique().max()==1
 for v in SIX:
  assert int(x[v].isna().sum())==contract['ordinary_missing_n'][v]
  if v!='Sex':assert np.isfinite(x[v].dropna().to_numpy()).all()
 return x,m
def locks():
 rec=base.read(LOCK/'STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json')
 assert rec['status']=='PASS_two_common6_source_model_locks' and len(rec['models'])==2
 assert base.sha(LOCK/'CODE_LOCK.json')==rec['code_lock_sha256']
 for name,h in base.read(LOCK/'CODE_LOCK.json').items():assert base.sha(ROOT/'work/common6_model_lock_20261024_v1'/name)==h
 for name,h in base.read(LOCK/'DEPENDENCIES.json').items():assert base.sha(name)==h
 result={}
 for it in rec['models']:
  assert base.sha(it['path'])==it['sha256']
  m=base.read(it['path']);line=m['identity']['line']
  assert line in ['geography','temporal'] and line not in result
  assert set(m['input_columns'])==set(SIX) and m['actual_input_count']==6
  for cp in list(m['base_components'].values())+[m['calibrator']]+([m['meta']] if m['meta'] else []):
   assert base.sha(cp['path'])==cp['sha256']
  assert m['identity']['role']=='champion' and m['contains_training_information_keep_private']
  result[line]=m
 return rec,result
def threshold_intervals(y,pred,g,r,c,cuts,replicates=2000):
 yy,gg,rr,cc,strata,models=stats._design(y,g,r,c,pred)
 configs={(line,name):float(value) for line,ts in cuts.items() for name,value in ts.items() if name.startswith('training_')}
 point={key:stats.counts(yy,models[key[0]+'__calibrated'].p>=v) for key,v in configs.items()}
 fields=[f for f in next(iter(point.values())) if f not in ['n','events']]
 vals={key:{f:[] for f in fields} for key in configs};rng=np.random.default_rng(20261017);proof=hashlib.sha256()
 for rep in range(replicates):
  w=stats.group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes());idx=np.repeat(np.arange(len(yy)),w)
  for key,t in configs.items():
   z=stats.counts(yy[idx],models[key[0]+'__calibrated'].p[idx]>=t)
   for f in fields:
    v=z[f]
    if v is not None and np.isfinite(v):vals[key][f].append(float(v))
 rows=[]
 for key in configs:
  for f,v in vals[key].items():
   ci=np.quantile(v,[.025,.975]).tolist() if v else [None,None]
   rows.append({'source_line':key[0],'rule':key[1],'threshold':configs[key],'metric':f,'estimate':point[key][f],
    'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(v),'requested_replicates':replicates})
 return rows,proof.hexdigest()
def execute():
 check();rec,models=locks();x,m=source()
 # Freeze exact model hashes and endpoint inputs before any public inference.
 frozen={'model_lock_receipt_sha256':base.sha(LOCK/'STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json'),
  'models':rec['models'],'public_contract_sha256':base.sha(PUBLIC/'COMMON6_INPUT_CONTRACT.json'),
  'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'cohort_n':len(m),'events':int(m.y_scr.sum()),
  'no_target_recalibration_or_model_selection':True}
 target=OUT/'PRE_SCORE_LOCK.json'
 if target.exists():assert base.read(target)==frozen
 else:base.atomic(target,frozen)
 g=x.group_hash.astype(str).to_numpy();r=x.record_key.astype(str).to_numpy();keys=(x.group_hash+'|'+x.record_key).tolist()
 y=m.y_scr.to_numpy(dtype=int);centers=['INSPIRE']*len(m);pred={}
 for line,model in models.items():
  d=OUT/line;d.mkdir(exist_ok=True);done=d/'PREDICTION_ACCEPTANCE.json'
  if done.exists():
   old=base.read(done);assert old['pre_score_lock_sha256']==base.sha(target)
   assert old['prediction_sha256']==base.sha(d/'PRIVATE_PREDICTIONS.csv')
   z=pd.read_csv(d/'PRIVATE_PREDICTIONS.csv',dtype={'record_key':str,'group_hash':str},float_precision='round_trip')
   assert z.record_key.tolist()==list(r) and z.group_hash.tolist()==list(g) and z.y_scr.tolist()==list(y)
   raw,cal=z.prob_raw.to_numpy(),z.prob_calibrated.to_numpy()
  else:
   raw,cal=runtime.route_prediction(model,x[SIX],keys)
   probe=list(np.flatnonzero(x[SIX].isna().any(axis=1))[:4])+list(np.flatnonzero(~x[SIX].isna().any(axis=1))[:4])
   probe=list(dict.fromkeys(probe));px=x[SIX].iloc[probe];pk=[keys[i] for i in probe]
   a,b=runtime.route_prediction(model,px,pk);ar,br=runtime.route_prediction(model,px.iloc[::-1],pk[::-1])
   singles=[runtime.route_prediction(model,px.iloc[i:i+1],pk[i:i+1]) for i in range(len(probe))]
   for expected,actual in [(raw[probe],a),(cal[probe],b),(a,ar[::-1]),(b,br[::-1]),
       (a,np.concatenate([v[0] for v in singles])),(b,np.concatenate([v[1] for v in singles]))]:
    np.testing.assert_allclose(expected,actual,atol=1e-12,rtol=1e-12)
   z=pd.DataFrame({'record_key':r,'group_hash':g,'y_scr':y,'prob_raw':raw,'prob_calibrated':cal})
   z.to_csv(d/'PRIVATE_PREDICTIONS.csv',index=False)
   base.atomic(done,{'status':'PASS_locked_source_common6_prediction','pre_score_lock_sha256':base.sha(target),
    'prediction_sha256':base.sha(d/'PRIVATE_PREDICTIONS.csv'),'source_only_imputer':True,'public_probe_parity':True})
  pred[line+'__raw']=raw;pred[line+'__calibrated']=cal
 def progress(n):base.atomic(OUT/'status.json',{'state':'public_fixed_model_bootstrap','completed':n,'total':2000,'time_UTC':base.stamp(),'pid':os.getpid()})
 ev=OUT/'EVALUATION.json'
 if ev.exists():
  er=base.read(OUT/'EVALUATION_ACCEPTANCE.json');assert er['sha256']==base.sha(ev) and er['pre_score_lock_sha256']==base.sha(target)
  result=base.read(ev)
 else:
  result=stats.evaluate(y,pred,g,r,centers,{},[],replicates=2000,seed=20261017,progress=progress)
  base.atomic(ev,result);base.atomic(OUT/'EVALUATION_ACCEPTANCE.json',{'sha256':base.sha(ev),'pre_score_lock_sha256':base.sha(target)})
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(OUT/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(OUT/'INTERVALS.csv',index=False)
 burden,proof=threshold_intervals(y,pred,g,r,centers,{k:v['training_thresholds'] for k,v in models.items()})
 assert proof==result['receipt']['resample_weight_sha256']
 pd.DataFrame(burden).to_csv(OUT/'SOURCE_THRESHOLD_BURDEN.csv',index=False)
 curves=[];severe=[]
 for name,p in pred.items():
  bins=pd.qcut(pd.Series(p),10,duplicates='drop')
  for label,idx in pd.Series(np.arange(len(p))).groupby(bins,observed=True):
   ii=idx.to_numpy();curves.append({'model':name,'bin':str(label),'n':len(ii),'events':int(y[ii].sum()),'mean_predicted':float(p[ii].mean()),'observed':float(y[ii].mean())})
  for f in [.1,.2]:
   flags=stats.FixedRisk(y,p,g,r).review_counts(f).astype(bool)
   severe.append({'model':name,'fraction':f,'severe_events':int((m.stage_scr>=2).sum()),'severe_captured':int(((m.stage_scr>=2).to_numpy()&flags).sum()),'reviewed':int(flags.sum())})
 pd.DataFrame(curves).to_csv(OUT/'CALIBRATION_CURVES.csv',index=False)
 pd.DataFrame(severe).to_csv(OUT/'SEVERE_CAPTURE_DESCRIPTIVE.csv',index=False)
 check();locks()
 base.atomic(OUT/'STAGE_PUBLIC_ACCEPTANCE.json',{'status':'PASS_INSPIRE_locked_common6_fixed_model_evaluation',
  'n':6043,'events':253,'replicates':2000,'source_line_models':list(models),'pre_score_lock_sha256':base.sha(target),
  'files':{str(p.relative_to(OUT)):base.sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.csv','.json'] and p.name not in ['status.json','LAUNCH.json','STAGE_PUBLIC_ACCEPTANCE.json']},
  'endpoint':'recorded postoperative serum creatinine AKI; not identical to Chinese clinical 7-day AKI',
  'original_9_27_38_or_selective_strategies_internationally_validated':False,'public_domain_calibration_fitted':False,
  'performance_intervals_conditional_on_locked_models':True,'entire_research_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_common6_public_evaluation','time_UTC':base.stamp(),'owned_workers':0})
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_public_evaluation_preparation_only'
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 while not (LOCK/'STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json').exists():
  check()
  if (LOCK/'FAILURE.json').exists():raise RuntimeError('Common6 source model lock failed')
  base.atomic(OUT/'status.json',{'state':'waiting_both_common6_model_locks','pid':os.getpid(),'time_UTC':base.stamp(),'public_scores_accessed':False})
  time.sleep(30)
 upstream=(LOCK/'SUPERVISOR.lock').open('a')
 while True:
  try:fcntl.flock(upstream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 while base.mem()<6*1024**3:time.sleep(30)
 if not (OUT/'STAGE_PUBLIC_ACCEPTANCE.json').exists():
  with threadpool_limits(limits=1):execute()
 upstream.close();guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc()});raise
