"""Same-patient comparison of prespecified altered refits and matched fixed-flow controls."""
from pathlib import Path
import os,sys,json,time,fcntl,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/fixed_flow_sensitivity_evaluation_current_20261030_r1';FITS=ROOT/'outputs/fixed_flow_sensitivity_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
import evaluation_stats as stats
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
HIGH=set(pd.read_csv(ROOT/'outputs/cohort_v2_rrt_days_20261017/UNVERIFIED_HIGH_BASELINE_SCR.csv').excel_row.astype(int))
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def select_mask(meta,scenario):
 return ~meta.excel_row.isin(HIGH).to_numpy() if scenario=='exclude_high_baseline_Cr' else np.ones(len(meta),dtype=bool)
def ready():
 a=base.read(FITS/'STAGE_REFIT_ACCEPTANCE.json');assert a['status']=='PASS_thirty_fixed_flow_scenario_routes_fitted'
 assert len(a['receipts'])==30
 for p,h in a['receipts'].items():assert base.sha(p)==h
 return a
def load(route,scenario):
 rec=base.read(FITS/'routes'/route/scenario/'COMPLETE.json');d=Path(rec['fit_cache'])
 assert base.sha(d/'COMPLETE.json')==rec['fit_receipt_sha256']
 q=base.read(d/'COMPLETE.json')
 for f,h in q['files'].items():assert base.sha(d/f)==h
 assert base.sha(d/'full/COMPLETE.json')==q['full_receipt']
 for k,h in q['fold_receipts'].items():assert base.sha(d/('fold_'+k)/'COMPLETE.json')==h
 x=pd.read_csv(d/'PRIVATE_HELDOUT_PREDICTIONS.csv',dtype={'group_hash':str},float_precision='round_trip')
 assert x.excel_row.is_unique
 return x,base.read(d/'TRAINING_THRESHOLDS.json'),str(d/'COMPLETE.json')
def execute(line,cohort,scenario):
 check();a=ready();routes=base.read(FITS/'ROUTES.json')
 chosen={name:ref for name,ref in routes.items() if ref['identity']['line']==line and ref['identity']['cohort']==cohort}
 preds={};raws={};cuts={};sources={};meta=None
 for route in chosen:
  for sc in ['matched_control',scenario]:
   d,t,receipt=load(route,sc);sources[receipt]=base.sha(receipt)
   keep=select_mask(d,scenario);d=d.loc[keep].reset_index(drop=True)
   m=d.drop(columns=['prob_raw','prob_calibrated'])
   if meta is None:meta=m
   else:assert meta.equals(m)
   key=route+'__'+sc;preds[key]=d.prob_calibrated.to_numpy();raws[key]=d.prob_raw.to_numpy();cuts[key]=t
 comparisons=[(name+'__'+scenario,name+'__matched_control') for name in chosen]
 for sc in ['matched_control',scenario]:
  if cohort=='paired':
   comparisons.append((line+'_paired_endop_full__'+sc,line+'_paired_preop_full__'+sc))
  else:
   comparisons.extend([(line+'_preop_preop_full__'+sc,line+'_preop_preop_compact__'+sc),
    (line+'_preop_preop_selected__'+sc,line+'_preop_preop_full__'+sc)])
 dest=OUT/(line+'_'+cohort)/scenario;dest.mkdir(parents=True,exist_ok=True)
 sig={'fit_acceptance_sha256':base.sha(FITS/'STAGE_REFIT_ACCEPTANCE.json'),'source_receipts':sources,
  'line':line,'cohort':cohort,'scenario':scenario,'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')}
 if (dest/'COMPLETE.json').exists():
  old=base.read(dest/'COMPLETE.json');assert old['signature']==sig
  for f,h in old['files'].items():assert base.sha(dest/f)==h
  return
 y=meta.y.to_numpy();g=meta.group_hash.astype(str).tolist();r=meta.excel_row.astype(str).tolist();centers=meta.center.astype(str).tolist()
 base.atomic(dest/'EVALUATION_LOCK.json',{'signature':sig,'n':len(meta),'events':int(y.sum()),
  'same_retained_patients_for_matched_control_and_altered_refit':True,'original_V1_not_replaced':True,
  'calibration_regime':'Fixed chosen family/member flow, matched across refits; differs from V1 family-selection calibration',
  'selective_policies_not_transferred_to_changed_probabilities':True})
 def progress(n):base.atomic(OUT/'status.json',{'state':'evaluating_fixed_flow_sensitivities','task':line+'_'+cohort+'/'+scenario,'completed_bootstrap':n,'total_bootstrap':2000,'pid':os.getpid(),'time_UTC':base.stamp()})
 result=stats.evaluate(y,preds,g,r,centers,{},comparisons,replicates=2000,seed=20261017,progress=progress)
 base.atomic(dest/'EVALUATION.json',result)
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(dest/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(dest/'INTERVALS.csv',index=False)
 pd.DataFrame(result['paired_differences']).to_csv(dest/'PAIRED_DIFFERENCES.csv',index=False)
 base.atomic(dest/'RAW_POINT_ESTIMATES.json',{k:stats.summary(y,p,g,r) for k,p in raws.items()})
 rows=[]
 for k,p in preds.items():
  for rule,t in cuts[k].items():
   if rule.startswith('training_') and isinstance(t,(float,int)):rows.append({'model':k,'rule':rule,'threshold':t,**stats.counts(y,p>=t)})
 pd.DataFrame(rows).to_csv(dest/'TRAINING_THRESHOLD_COUNTS.csv',index=False)
 check()
 files=['EVALUATION_LOCK.json','EVALUATION.json','POINT_ESTIMATES.csv','INTERVALS.csv','PAIRED_DIFFERENCES.csv','RAW_POINT_ESTIMATES.json','TRAINING_THRESHOLD_COUNTS.csv']
 base.atomic(dest/'COMPLETE.json',{'status':'PASS_matched_fixed_flow_sensitivity_evaluation','signature':sig,'files':{f:base.sha(dest/f) for f in files},'n':len(meta),'events':int(y.sum())})

if __name__=='__main__':
 raise RuntimeError('Use the resource-guarded run_correction.py --run entry in its owned finite unit')
