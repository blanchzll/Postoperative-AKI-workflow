"""Fixed-choice secondary update with nested construction of every stacked OOF function."""
from pathlib import Path
import sys,os,json,fcntl,time,gc,resource,warnings
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pod1_corrected_fixed_update_20261024_v1'
sys.path.insert(0,str(CODE));from cache_identity import canonical_sha256 as digest
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'));import common as base
os.environ['AKI_SCOPE_OUT']=str(ROOT/'outputs/ensemble_calibration_20261020_v1')
rt=base.module('pod1_fixed_source_runtime',ROOT/'work/model_lock_20261022_v1/runtime.py')
candidates=base.module('pod1_fixed_original_candidates',ROOT/'work/preop_temporal_python_20261017/model_candidates.py')
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
MEMBERS=['LR','RF','XGB'];THREADS=1;DATA=None;SPECS=None;GLOBAL=None;SORTKEY=None;KEYS=None
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
 for f,h in base.read(OUT/'INPUT_ACCEPTANCE.json')['files'].items():assert base.sha(OUT/f)==h
def init():
 global DATA,SPECS,GLOBAL,SORTKEY,KEYS
 if DATA is not None:return
 SPECS=base.read(OUT/'FEATURE_SPECS.json');cats=SPECS['matched_pod1']['cat']
 DATA=pd.read_csv(OUT/'PRIVATE_CURRENT_POD1_COHORT.csv',dtype={'group_hash':str,**{c:'string' for c in cats}},float_precision='round_trip').set_index('excel_row',drop=False)
 SORTKEY={int(r):(str(g),str(r)) for r,g in zip(DATA.excel_row,DATA.group_hash)}
 KEYS={r:g+'|'+s for r,(g,s) in SORTKEY.items()}
 GLOBAL={'cohort':base.sha(OUT/'PRIVATE_CURRENT_POD1_COHORT.csv'),'code':base.sha(OUT/'CODE_LOCK.json')}
def canonical(rows):
 return sorted([int(r) for r in rows],key=lambda r:SORTKEY[r])
def frame(rows,sp):return DATA.loc[rows,sp['num']+sp['cat']].reset_index(drop=True)
def metadata(rows):return DATA.loc[rows,['excel_row','group_hash','center','surgery_date','y','clinical_y']].reset_index(drop=True)
def keys(rows):return [KEYS[int(r)] for r in rows]
def folds(rows):
 m=metadata(rows);assert m.y.nunique()==2
 out=[]
 for tr,va in StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(np.zeros(len(m)),m.y,m.group_hash):
  a=[rows[i] for i in tr];b=[rows[i] for i in va]
  assert set(m.iloc[tr].group_hash).isdisjoint(m.iloc[va].group_hash)
  assert m.iloc[tr].y.nunique()==2
  out.append((a,b))
 assert sorted(r for _,v in out for r in v)==sorted(rows)
 return out
def safe_observed(raw,filled,sp):
 assert list(filled)==sp['num']+sp['cat'] and not filled.isna().any().any()
 for c in filled:
  mask=raw[c].notna()
  if c in sp['num']:np.testing.assert_array_equal(raw.loc[mask,c],filled.loc[mask,c])
  else:assert raw.loc[mask,c].tolist()==filled.loc[mask,c].tolist()
def impute(task,tr,va):
 sp=SPECS[task['branch']];method=base.read(OUT/'EXECUTION_CONTRACT.json')['imputation'][task['line']]
 sig={**GLOBAL,'train':tr,'valid':va,'spec':sp,'method':method}
 dest=OUT/'imputation_cache'/digest(sig);dest.mkdir(parents=True,exist_ok=True)
 with (dest/'FIT.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not (dest/'COMPLETE.json').exists():
   raw=frame(tr,sp);valid=frame(va,sp)
   imp=rt.StableMixedImputer(sp,method,seed=42,iterations=5,trees=200,neighbors=5,threads=1)
   tx=imp.fit_transform(raw,keys(tr));vx=imp.transform(valid,keys(va))
   safe_observed(raw,tx,sp);safe_observed(valid,vx,sp)
   joblib.dump({'train':tx,'valid':vx,'imputer':imp,'spec':sp},dest/'PRIVATE_STATE.joblib',compress=3)
   base.atomic(dest/'COMPLETE.json',{'signature':sig,'files':{'PRIVATE_STATE.joblib':base.sha(dest/'PRIVATE_STATE.joblib')},'training_only':True})
  r=base.read(dest/'COMPLETE.json');assert r['signature']==sig
  for f,h in r['files'].items():assert base.sha(dest/f)==h
  return joblib.load(dest/'PRIVATE_STATE.joblib'),str(dest/'PRIVATE_STATE.joblib')
def encoder(sp,est):
 return Pipeline([('encode',ColumnTransformer([('numeric',StandardScaler(),sp['num']),
  ('categorical',OneHotEncoder(categories=[sp['cat_levels'][c] for c in sp['cat']],handle_unknown='ignore',sparse_output=False),sp['cat'])])),
  ('constant',VarianceThreshold(0)),('model',est)])
def meta_estimator():return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=2000,random_state=42))])
def logit(p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def raw_predict(task,family,train,valid,save_model=False):
 tr=canonical(train);va=canonical(valid)
 assert set(DATA.loc[tr,'group_hash']).isdisjoint(DATA.loc[va,'group_hash'])
 sig={**GLOBAL,'line':task['line'],'branch':task['branch'],'family':family,'parameters':task['parameters'],
  'train':tr,'valid':va,'save_model':bool(save_model),'GPU_XGB':'original_cuda_hist'}
 dest=OUT/'raw_cache'/digest(sig);dest.mkdir(parents=True,exist_ok=True)
 with (dest/'FIT.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not (dest/'COMPLETE.json').exists():
   refs={};bundle=None
   if family=='Stacking':
    oof=np.full((len(tr),3),np.nan);pos={r:i for i,r in enumerate(tr)}
    for a,b in folds(tr):
     for j,fam in enumerate(MEMBERS):
      pred,receipt=raw_predict(task,fam,a,b);oof[[pos[r] for r in b],j]=pred;refs[receipt]=base.sha(receipt)
    assert np.isfinite(oof).all()
    model=meta_estimator()
    with threadpool_limits(limits=1):model.fit(oof,DATA.loc[tr,'y'].to_numpy())
    pp=[];member_paths=[]
    for fam in MEMBERS:
     pred,receipt=raw_predict(task,fam,tr,va,save_model);pp.append(pred);refs[receipt]=base.sha(receipt)
     if save_model:member_paths.append(str(Path(receipt).parent/'PRIVATE_MODEL.joblib'))
    with threadpool_limits(limits=1):p=model.predict_proba(np.column_stack(pp))[:,1]
    if save_model:bundle={'kind':'Stacking','model':model,'members':member_paths}
   else:
    z,imp_path=impute(task,tr,va);sp=z['spec']
    est=dict(candidates.model_candidates(42,THREADS,True)[family])[task['parameters'][family]]
    model=encoder(sp,est)
    with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):
     warnings.simplefilter('always');model.fit(z['train'],DATA.loc[tr,'y'].to_numpy());p=model.predict_proba(z['valid'])[:,1]
    if family=='XGB':
     device=json.loads(model.named_steps['model'].get_booster().save_config())['learner']['generic_param']['device']
     assert device.startswith('cuda'),device
    refs[str(Path(imp_path).parent/'COMPLETE.json')]=base.sha(Path(imp_path).parent/'COMPLETE.json')
    if save_model:bundle={'kind':family,'model':model,'imputer_state':imp_path,'columns':sp['num']+sp['cat']}
    del z;gc.collect()
   assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
   np.save(dest/'raw.npy',np.asarray(p,dtype=float))
   files={'raw.npy':base.sha(dest/'raw.npy')}
   if save_model:joblib.dump(bundle,dest/'PRIVATE_MODEL.joblib',compress=3);files['PRIVATE_MODEL.joblib']=base.sha(dest/'PRIVATE_MODEL.joblib')
   base.atomic(dest/'COMPLETE.json',{'signature':sig,'files':files,'upstream_receipts':refs,'scored_groups_excluded_from_upstream_fit':True})
  r=base.read(dest/'COMPLETE.json');assert r['signature']==sig
  for f,h in r['files'].items():assert base.sha(dest/f)==h
  raw=np.load(dest/'raw.npy')
 order={r:i for i,r in enumerate(va)}
 return raw[[order[int(r)] for r in valid]],str(dest/'COMPLETE.json')
def predict_bundle(path,rows):
 b=joblib.load(path)
 if b['kind']=='Stacking':x=np.column_stack([predict_bundle(q,rows) for q in b['members']])
 else:
  z=joblib.load(b['imputer_state']);x=z['imputer'].transform(DATA.loc[rows,b['columns']].reset_index(drop=True),keys(rows))
 with threadpool_limits(limits=1):return b['model'].predict_proba(x)[:,1]
def run(index,threads):
 global THREADS
 THREADS=int(threads);check();init();task=base.read(OUT/'TASKS.json')[int(index)]
 dest=OUT/'tasks'/str(index).zfill(3);dest.mkdir(parents=True,exist_ok=True)
 signature={'task':task,'code':base.sha(OUT/'CODE_LOCK.json'),'inputs':base.sha(OUT/'INPUT_ACCEPTANCE.json')}
 with (dest/'TASK.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX)
  if (dest/'COMPLETE.json').exists():
   r=base.read(dest/'COMPLETE.json');assert r['signature']==signature
   for f,h in r['files'].items():assert base.sha(dest/f)==h
   return
  tr=canonical(task['train_rows']);va=canonical(task['valid_rows']);pos={r:i for i,r in enumerate(tr)}
  oof=np.full(len(tr),np.nan);refs={}
  for a,b in folds(tr):
   pred,r=raw_predict(task,task['family'],a,b)
   oof[[pos[r] for r in b]]=pred;refs[r]=base.sha(r)
  assert np.isfinite(oof).all()
  cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=42)
  with threadpool_limits(limits=1):cal.fit(logit(oof),DATA.loc[tr,'y'].to_numpy())
  raw,r=raw_predict(task,task['family'],tr,va,task['role']=='final');refs[r]=base.sha(r)
  with threadpool_limits(limits=1):
   prob=cal.predict_proba(logit(raw))[:,1];traincal=cal.predict_proba(logit(oof))[:,1]
  _,tpr,cut=roc_curve(DATA.loc[tr,'y'].to_numpy(),traincal,drop_intermediate=False)
  threshold=float(np.max(cut[np.isfinite(cut)&(tpr>=.8)]))
  cuts={'training_sensitivity80':threshold,'training_calibrated_q80':float(np.quantile(traincal,.8))}
  out=metadata(va);out['prob_raw']=raw;out['prob_calibrated']=prob
  out.to_csv(dest/'PRIVATE_PREDICTIONS.csv',index=False)
  train=metadata(tr);train['prob_raw_OOF']=oof;train['prob_calibrated_training']=traincal
  train.to_csv(dest/'PRIVATE_TRAINING_OOF.csv',index=False)
  joblib.dump(cal,dest/'calibrator.joblib',compress=3)
  base.atomic(dest/'THRESHOLDS.json',cuts)
  diagnostic={}
  if task['role']=='final':
   bundle=Path(r).parent/'PRIVATE_MODEL.joblib'
   full=predict_bundle(bundle,va);np.testing.assert_allclose(full,raw,rtol=1e-12,atol=1e-12)
   probe=va[:3];pp=predict_bundle(bundle,probe);rev=predict_bundle(bundle,probe[::-1])[::-1]
   solo=np.concatenate([predict_bundle(bundle,[row]) for row in probe])
   err=float(max(np.max(abs(q-raw[:len(probe)])) for q in [pp,rev,solo]))
   diagnostic={'same_batch_reload_pass':True,'max_single_batch_abs':err,'strict_1e12_pass':err<=1e-12,
    'not_deployment_qualification':err>1e-12,'bundle_path':str(bundle)}
  files=['PRIVATE_PREDICTIONS.csv','PRIVATE_TRAINING_OOF.csv','calibrator.joblib','THRESHOLDS.json']
  base.atomic(dest/'COMPLETE.json',{'status':'PASS_fixed_choice_POD1_context_fitted','signature':signature,
   'files':{f:base.sha(dest/f) for f in files},'raw_fit_receipts':refs,'diagnostic':diagnostic,
   'training_only_nested_stack_rebuilt':True,'threads':THREADS,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   'not_unbiased_full_selection_internal_evaluation':True,'heldout_scores_not_evaluated':True})
  print('POD1_CONTEXT_DONE',index,task['line'],task['branch'],task['family'],task['role'],flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):run(int(sys.argv[1]),int(sys.argv[2]))
