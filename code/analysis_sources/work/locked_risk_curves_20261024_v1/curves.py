"""Development-reference risk curves for frozen V1; prediction only."""
from pathlib import Path
import os,sys,json,hashlib,time,resource
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/locked_risk_curves_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
sys.path.insert(0,str(ROOT/'work/model_lock_20261022_v1'))
import runtime as rt
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def curve_stats(raw,cal):
 assert np.isfinite(raw).all() and np.isfinite(cal).all()
 return {'mean_raw_risk':float(np.mean(raw)),'mean_calibrated_risk':float(np.mean(cal)),
 'reference_risk_p10':float(np.quantile(cal,.1)),'reference_risk_median':float(np.median(cal)),
 'reference_risk_p90':float(np.quantile(cal,.9))}
class Predictor:
 def __init__(self,manifest):
  self.m=manifest;self.b={}
  for k,v in manifest['base_components'].items():
   assert base.sha(v['path'])==v['sha256'];self.b[k]=joblib.load(v['path'])
  p=manifest['calibrator'];assert base.sha(p['path'])==p['sha256'];self.cal=joblib.load(p['path'])
  self.meta=None
  if manifest.get('meta'):
   p=manifest['meta'];assert base.sha(p['path'])==p['sha256'];self.meta=joblib.load(p['path'])
 def __call__(self,x,keys):
  filled={};pred={}
  for k,b in self.b.items():
   sig=b['imputer_sha256']
   if sig not in filled:filled[sig]=b['imputer'].transform(x[b['input_columns']],keys)
   pred[k]=b['pipeline'].predict_proba(filled[sig])[:,1]
  if self.m['family'] in rt.FAMILIES:raw=pred[self.m['family']]
  elif self.m['family']=='Voting':raw=np.column_stack([pred[k] for k in self.m['members']]).mean(axis=1)
  else:raw=self.meta.predict_proba(pd.DataFrame({k:pred[k] for k in self.m['members']}))[:,1]
  raw=np.asarray(raw,dtype=np.float64)
  return raw,rt.apply_calibration(self.cal,raw)
def inputs(route):
 rec=base.read(OUT/'ROUTES.json')[route];assert base.sha(rec['path'])==rec['sha256']
 m=base.read(rec['path']);t=m['identity'];sp=rt.source_spec(t['stage'],m['input_columns'])
 meta=pd.read_csv(base.SOURCE/(t['cohort']+'_metadata.csv'),dtype={'group_hash':str})
 meta=meta[meta[t['line']+'_partition_v2'].eq('development')].set_index('excel_row',drop=False)
 assert len(meta)==m['training_n']
 x=pd.read_csv(base.SOURCE/(t['cohort']+'_'+t['stage']+'_features.csv'),dtype={c:'string' for c in sp['cat']},
  float_precision='round_trip').set_index('excel_row').loc[meta.index,m['input_columns']]
 contract=base.read(OUT/'CONTRACT.json')
 groups=sorted(meta.group_hash.unique(),key=lambda g:hashlib.sha256((str(contract['seed'])+'|'+str(g)).encode()).hexdigest())[:contract['reference_groups']]
 reference=meta[meta.group_hash.isin(groups)].reset_index(drop=True).sort_values(['group_hash','excel_row']).set_index('excel_row',drop=False)
 # Avoid index/column name ambiguity without changing record identity.
 return m,sp,meta,x,reference,contract
def execute(route):
 check();dest=OUT/'routes'/route;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():
  d=base.read(dest/'COMPLETE.json')
  for f,h in d['files'].items():assert base.sha(dest/f)==h
  return
 m,sp,meta,x,reference,c=inputs(route)
 pred=Predictor(m);ref=x.loc[reference.index].copy();keys=rt.keys(reference)
 features=[f for f in c['clinical_features'] if f in sp['num']]
 signature={'manifest':base.read(OUT/'ROUTES.json')[route],'reference_rows':reference.excel_row.tolist(),
 'reference_groups':int(reference.group_hash.nunique()),'code_lock':base.sha(OUT/'CODE_LOCK.json'),'contract':c}
 base.atomic(dest/'EXPLANATION_LOCK.json',signature)
 started=time.monotonic();probe=ref.head(4);pk=keys[:4]
 orig=rt.route_prediction(m,probe,pk);cached=pred(probe,pk)
 for a,b in zip(orig,cached):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
 # Record strict probe parity; do not falsely certify new deployment versions.
 single=np.asarray([pred(probe.iloc[i:i+1],[pk[i]])[1][0] for i in range(len(probe))])
 maxdiff=float(np.max(np.abs(single-cached[1])))
 base.atomic(dest/'INFERENCE_ADAPTER_QUALIFICATION.json',{'same_batch_original_adapter_agreement':True,
  'single_batch_max_abs':maxdiff,'single_batch_strict_1e12':bool(np.allclose(single,cached[1],atol=1e-12,rtol=1e-12)),
  'not_a_new_deployment_qualification':True})
 baseline=pred(ref,keys)
 base.atomic(dest/'REFERENCE_BASELINE.json',{'n_rows':len(ref),'n_patient_groups':int(reference.group_hash.nunique()),**curve_stats(*baseline)})
 for f in features:
  p=dest/(f+'.csv')
  if (dest/(f+'.COMPLETE.json')).exists():
   assert base.sha(p)==base.read(dest/(f+'.COMPLETE.json'))['sha256'];continue
  a=pd.to_numeric(x[f],errors='coerce').dropna().to_numpy()
  grid=np.unique(np.quantile(a,np.linspace(c['quantile_min'],c['quantile_max'],c['grid_points']))) if len(a) else []
  rows=[]
  for value in grid:
   altered=ref.copy();altered[f]=float(value)
   raw,cal=pred(altered,keys)
   rows.append({'feature':f,'value':float(value),'reference_rows':len(ref),'reference_groups':reference.group_hash.nunique(),
    'development_observed_n':len(a),'development_missing_n':int(x[f].isna().sum()),
    'development_fraction_leq_value':float(np.mean(a<=value)),**curve_stats(raw,cal)})
  pd.DataFrame(rows).to_csv(p,index=False)
  base.atomic(dest/(f+'.COMPLETE.json'),{'sha256':base.sha(p),'grid_count':len(rows),'feature':f})
  base.atomic(dest/'progress.json',{'completed_feature':f,'completed_features':len(list(dest.glob('*.csv'))),
    'planned_features':len(features),'elapsed_seconds':time.monotonic()-started,'pid':os.getpid(),'time_UTC':base.stamp()})
 files=[p for p in dest.iterdir() if p.name!='COMPLETE.json' and p.is_file()]
 base.atomic(dest/'COMPLETE.json',{'status':'PASS_locked_development_reference_risk_curves','route':route,'features':features,
  'files':{p.name:base.sha(p) for p in files},'elapsed_seconds':time.monotonic()-started,
  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'heldout_used':False,
  'models_refitted':False,'causal_interpretation':False,'SHAP_completed':False})
if __name__=='__main__':
 with threadpool_limits(limits=1):execute(sys.argv[1])
