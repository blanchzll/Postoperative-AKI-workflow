"""Original-runtime full-development fitting and private inference; no final-evaluation loader."""
from pathlib import Path
import os,sys,json,hashlib,fcntl,warnings,gc
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/common6_model_lock_20261024_v1'
SOURCE=ROOT/'outputs/common6_development_20261024_v1/inputs/geography'
BASE=ROOT/'outputs/formal_development_20261018_v1'
D=ROOT/'outputs/ensemble_continuation_20261018_v1'
E=ROOT/'outputs/common6_internal_assembly_20261024_v1'
sys.path.insert(0,str(ROOT/'work/ensemble_calibration_20261020_v1'))
from model_engine import encoder,one_fit
from preprocess import StableMixedImputer
from ensemble_math import calibrator,apply_calibration,rank_families
from core import atomic,sha,canonical,digest,COMPACT,FAMILIES
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

def checked(p,files=True):
 p=Path(p);d=json.loads(p.read_text())
 if files:
  for name,h in d.get('files',{}).items():assert sha(p.parent/name)==h,(p,name)
 return d
def route_id(t):return '/'.join(t[x] for x in ['line','cohort','stage','branch'])
def base_id(t):return route_id(t)+'/'+t['family']
def source_spec(stage,columns=None):
 s=json.loads((SOURCE/'FEATURE_SPECS.json').read_text())[stage]
 if columns is None:return s
 assert set(columns)<=set(s['num']+s['cat'])
 return {'num':[c for c in s['num'] if c in columns],
         'cat':[c for c in s['cat'] if c in columns],
         'cat_levels':{c:s['cat_levels'][c] for c in s['cat'] if c in columns}}
def data(t,sp=None):
 sp=sp or source_spec(t['stage'])
 m=pd.read_csv(SOURCE/(t['cohort']+'_metadata.csv'),dtype={'group_hash':str}).set_index('excel_row',drop=False)
 rows=t['train_rows'];m=m.loc[rows]
 assert m.index.is_unique and m[t['line']+'_partition_v2'].eq('development').all()
 assert m.y.nunique()==2
 x=pd.read_csv(SOURCE/(t['cohort']+'_'+t['stage']+'_features.csv'),
   dtype={c:'string' for c in sp['cat']},float_precision='round_trip').set_index('excel_row')
 return x.loc[rows,sp['num']+sp['cat']],m
def keys(m):return (m.group_hash.astype(str)+'|'+m.excel_row.astype(str)).tolist()
def complete(d,sig):
 p=Path(d)/'COMPLETE.json'
 if not p.exists():return False
 assert checked(p)['signature']==sig,'Changed scope/code: '+str(d)
 return True
def impute_final(t,columns=None):
 sp=source_spec(t['stage'],columns)
 signature={'input':sha(SOURCE/(t['cohort']+'_'+t['stage']+'_features.csv')),
  'rows':t['train_rows'],'spec':sp,'method':t['method'],'seed':42,'iterations':5,'trees':200,'neighbors':5,
  'runtime_code':sha(__file__),'imputer_sources':{f:sha(ROOT/'work'/f) for f in [
   'v2_stable_imputation_20261017/stable_imputer.py','python_six_outcomes_20261017/mixed_imputer.py',
   'python_six_outcomes_20261017/donor_helpers.py']}}
 dest=OUT/'cache'/digest(signature)[:24];dest.mkdir(parents=True,exist_ok=True)
 with (dest/'task.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not complete(dest,signature):
   x,m=data(t,sp)
   with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):
    warnings.simplefilter('always')
    imp=StableMixedImputer(sp,t['method'],seed=42,iterations=5,trees=200,neighbors=5,threads=1)
    filled=imp.fit_transform(x,keys(m))
   assert list(filled)==sp['num']+sp['cat'] and not filled.isna().any().any()
   for col in filled:
    observed=x[col].notna()
    if col in sp['num']:np.testing.assert_array_equal(x.loc[observed,col],filled.loc[observed,col])
    else:assert x.loc[observed,col].tolist()==filled.loc[observed,col].tolist()
   joblib.dump(imp,dest/'imputer.joblib',compress=3)
   filled.to_csv(dest/'train.csv',index_label='excel_row')
   atomic(dest/'COMPLETE.json',{'signature':signature,
    'files':{f:sha(dest/f) for f in ['imputer.joblib','train.csv']},
    'columns':list(filled),'all_missing_training_fields':imp.all_missing_,
    'warnings':sorted(set(str(w.message)[:500] for w in ww)),
    'observed_values_preserved':True,'no_outcome_in_imputer':True,'training_only':True})
 return dest,sp
def read_filled(dest,sp):
 return pd.read_csv(dest/'train.csv',dtype={c:'string' for c in sp['cat']},float_precision='round_trip').set_index('excel_row')
def rfe_path(tx,y,sp):
 cols=list(tx);paths={}
 if len(cols)<3:raise ValueError('RF-RFE expects at least three original variables')
 with threadpool_limits(limits=1):
  while True:
   paths[str(len(cols))]=list(cols)
   if len(cols)==3:break
   nums=[c for c in sp['num'] if c in cols];cats=[c for c in sp['cat'] if c in cols]
   enc=ColumnTransformer([('num',StandardScaler(),nums),
     ('cat',OneHotEncoder(categories=[sp['cat_levels'][c] for c in cats],handle_unknown='ignore',sparse_output=False),cats)])
   xx=enc.fit_transform(tx[cols]);keep=VarianceThreshold().fit(xx)
   model=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features='sqrt',random_state=42,n_jobs=1)
   model.fit(keep.transform(xx),y)
   owner=np.asarray(nums+[c for c in cats for _ in sp['cat_levels'][c]])[keep.get_support()]
   importance={c:float(model.feature_importances_[owner==c].sum()) for c in cols}
   cols.remove(min(cols,key=lambda c:(importance[c],c)))
 return paths

def selected_columns(t,k):
 full,sp=impute_final(t);dest=full/'RFE';dest.mkdir(exist_ok=True)
 _,m=data(t,sp)
 signature={'imputation_receipt':sha(full/'COMPLETE.json'),'training_y':digest(m.y.tolist()),
  'training_groups':digest(m.group_hash.tolist()),'source_frozen_algorithm':sha(ROOT/'work/ensemble_calibration_20261020_v1/preprocess.py'),
  'rf':{'trees':200,'leaf':5,'max_features':'sqrt','seed':42}}
 with (dest/'selector.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not complete(dest,signature):
   tx=read_filled(full,sp);paths=rfe_path(tx,m.y,sp)
   atomic(dest/'path.json',{'features_by_count':paths,'training_only':True})
   atomic(dest/'COMPLETE.json',{'signature':signature,'files':{'path.json':sha(dest/'path.json')}})
 return json.loads((dest/'path.json').read_text())['features_by_count'][str(k)]
def predict_raw(bundle,frame,case_keys):
 columns=bundle['input_columns']
 if len(frame)!=len(case_keys):raise ValueError('Case identity length mismatch')
 filled=bundle['imputer'].transform(frame.loc[:,columns],case_keys)
 p=bundle['pipeline'].predict_proba(filled)[:,1]
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid risk')
 return p
def require_e():
 rec=checked(E/'STAGE_INTERNAL_ACCEPTANCE.json',False)
 assert rec['status']=='PASS_common6_strict_internal_evaluation'
 assert rec['candidate_fold_metrics']==110 and rec['procedure_fold_metrics']==10
 assert sha(E/'calibrated_internal/ACCEPTANCE.json')==rec['detail_sha256']
 sp=source_spec('preop')
 assert set(sp['num']+sp['cat'])=={'Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb'}
 return rec
def base_signature(t):
 return {'task':t,'code':sha(__file__),'frozen_model_engine':sha(ROOT/'work/ensemble_calibration_20261020_v1/model_engine.py')}
def fit_base(t):
 require_e()
 assert sha(SOURCE/'SOURCE_LOCK.json')==t['input_source_lock_sha256']
 assert sha(SOURCE/(t['cohort']+'_'+t['stage']+'_features.csv'))==t['feature_csv_sha256']
 assert sha(SOURCE/(t['cohort']+'_metadata.csv'))==t['metadata_sha256']
 dest=OUT/'base_fits'/base_id(t);dest.mkdir(parents=True,exist_ok=True)
 sig=base_signature(t)
 if complete(dest,sig):return
 columns=COMPACT if t['branch']=='compact' else (
  selected_columns(t,int(t['params']['selector_k'])) if t['branch']=='selected' else None)
 cache,sp=impute_final(t,columns);train=read_filled(cache,sp);raw,m=data(t,sp)
 assert train.index.tolist()==m.index.tolist()
 probe=raw.loc[raw.isna().any(axis=1)].head(4)
 if len(probe)<4:probe=raw.head(4)
 pm=m.loc[probe.index];imp=joblib.load(cache/'imputer.joblib')
 probe_filled=imp.transform(probe,keys(pm))
 pipe,pr,notes,fallback=one_fit(train,m,probe_filled,sp,t['family'],t['params'],1)
 bundle={'imputer':imp,'pipeline':pipe,'input_spec':sp,'input_columns':list(train),
  'method':t['method'],'params':t['params'],'family':t['family'],
  'raw_internal_model_only':False,'calibrator':None,'training_groups_sha256':digest(m.group_hash.tolist()),
  'case_key_required_for_stochastic_completion':True,'imputer_sha256':sha(cache/'imputer.joblib')}
 joblib.dump(bundle,dest/'raw_bundle.joblib',compress=3)
 loaded=joblib.load(dest/'raw_bundle.joblib')
 with threadpool_limits(limits=1):
  batch=predict_raw(loaded,probe,keys(pm))
  reverse=predict_raw(loaded,probe.iloc[::-1],keys(pm.iloc[::-1]))[::-1]
  solo=np.concatenate([predict_raw(loaded,probe.iloc[i:i+1],keys(pm.iloc[i:i+1])) for i in range(len(probe))])
 for p in [batch,reverse,solo]:np.testing.assert_allclose(pr,p,atol=1e-12,rtol=1e-12)
 atomic(dest/'COMPLETE.json',{'signature':sig,'files':{'raw_bundle.joblib':sha(dest/'raw_bundle.joblib')},
  'cache_receipt':str(cache/'COMPLETE.json'),'cache_receipt_sha256':sha(cache/'COMPLETE.json'),
  'input_columns':list(train),'n_train':len(m),'training_events':int(m.y.sum()),
  'warnings':notes,'spline_fallbacks':fallback,'single_batch_reorder_reload_verified':True,
  'final_hospital_time_performance_accessed':False})
def thresholds(y,p):
 y=np.asarray(y);p=np.asarray(p,dtype=float)
 if set(y)-{0,1} or not y.sum() or len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid training threshold data')
 positive=np.sort(p[y==1])[::-1];at=int(np.ceil(.8*len(positive)))-1
 return {'training_sensitivity80':float(positive[at]),
  'training_capacity10':float(np.quantile(p,.9,method='higher')),
  'training_capacity20':float(np.quantile(p,.8,method='higher')),
  'alert_comparison':'>=','ties_included':True,
  'source':'Honest training raw OOF transformed by final training-fitted calibrator; not an unbiased calibrated training metric.'}
def route_prediction(manifest,frame,case_keys):
 preds={};filled_cache={}
 for f,rec in manifest['base_components'].items():
  p=Path(rec['path']);assert sha(p)==rec['sha256']
  bundle=joblib.load(p)
  cache_key=bundle.get('imputer_sha256',rec['sha256'])
  if cache_key not in filled_cache:
   filled_cache[cache_key]=bundle['imputer'].transform(frame.loc[:,bundle['input_columns']],case_keys)
  preds[f]=bundle['pipeline'].predict_proba(filled_cache[cache_key])[:,1]
 if manifest['family'] in FAMILIES:raw=preds[manifest['family']]
 elif manifest['family']=='Voting':raw=np.column_stack([preds[f] for f in manifest['members']]).mean(axis=1)
 else:
  p=Path(manifest['meta']['path']);assert sha(p)==manifest['meta']['sha256']
  xx=pd.DataFrame({f:preds[f] for f in manifest['members']})
  raw=joblib.load(p).predict_proba(xx)[:,1]
 p=Path(manifest['calibrator']['path']);assert sha(p)==manifest['calibrator']['sha256']
 return raw,apply_calibration(joblib.load(p),np.asarray(raw,dtype=np.float64))
