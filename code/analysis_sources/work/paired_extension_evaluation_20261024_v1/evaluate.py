"""Fixed-choice paired cohort expansion: within-cohort and common-reference comparisons."""
from pathlib import Path
import os,sys,json,time,fcntl,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/paired_extension_evaluation_20261024_v1';FITS=ROOT/'outputs/paired_extension_clinical_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'));import common as base
import evaluation_stats as stats
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def load(route,scenario):
 d=FITS/'fits'/route/scenario;r=base.read(d/'COMPLETE.json')
 for f,h in r['files'].items():assert base.sha(d/f)==h
 assert base.sha(d/'full/COMPLETE.json')==r['full_receipt']
 for f,h in base.read(d/'full/COMPLETE.json')['files'].items():assert base.sha(d/'full'/f)==h
 for k,h in r['fold_receipts'].items():assert base.sha(d/('fold_'+k)/'COMPLETE.json')==h
 z=pd.read_csv(d/'PRIVATE_HELDOUT_PREDICTIONS.csv',dtype={'group_hash':str},float_precision='round_trip')
 assert z.excel_row.is_unique
 return z,base.read(d/'TRAINING_THRESHOLDS.json'),str(d/'COMPLETE.json')
def align_to_reference(frame,reference):
 assert frame.excel_row.is_unique and reference.excel_row.is_unique
 assert set(reference.excel_row)<=set(frame.excel_row)
 z=frame.set_index('excel_row',drop=False).loc[reference.excel_row].reset_index(drop=True)
 cols=[c for c in z if c not in ['prob_raw','prob_calibrated']]
 assert z[cols].equals(reference[cols])
 return z
def task_list():
 return [(line,sc,scope) for line in ['geography','temporal'] for sc in ['reference','expanded_h4','full_source']
  for scope in (['cohort'] if sc=='reference' else ['cohort','original_reference'])]
def execute(line,scenario,scope):
 check();sources={};preds={};raws={};cuts={};meta=None
 reference={}
 if scope=='original_reference':
  for stage in ['preop','endop']:
   route=line+'_paired_'+stage+'_full';z,c,receipt=load(route,'reference')
   reference[stage]=z;sources[receipt]=base.sha(receipt)
 records=[]
 for stage in ['preop','endop']:
  route=line+'_paired_'+stage+'_full';z,c,receipt=load(route,scenario);sources[receipt]=base.sha(receipt)
  if scope=='original_reference':
   z=align_to_reference(z,reference[stage])
   zr,cr,rr=load(route,'reference');records.append((line+'__reference__'+stage,zr,cr))
  records.append((line+'__'+scenario+'__'+stage,z,c))
 for key,z,c in records:
  m=z.drop(columns=['prob_raw','prob_calibrated'])
  if meta is None:meta=m
  else:assert meta.equals(m)
  preds[key]=z.prob_calibrated.to_numpy();raws[key]=z.prob_raw.to_numpy();cuts[key]=c
 y=meta.y.to_numpy();g=meta.group_hash.astype(str).tolist();rows=meta.excel_row.astype(str).tolist();centers=meta.center.astype(str).tolist()
 comparisons=[(line+'__'+scenario+'__endop',line+'__'+scenario+'__preop')]
 if scope=='original_reference':
  comparisons += [(line+'__reference__endop',line+'__reference__preop')]
  comparisons += [(line+'__'+scenario+'__'+stage,line+'__reference__'+stage) for stage in ['preop','endop']]
 d=OUT/line/scenario/scope;d.mkdir(parents=True,exist_ok=True)
 sig={'line':line,'scenario':scenario,'evaluation_population':scope,'source_receipts':sources,
  'fit_acceptance_sha256':base.sha(FITS/'STAGE_EXTENSION_FIT_ACCEPTANCE.json'),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')}
 if (d/'COMPLETE.json').exists():
  r=base.read(d/'COMPLETE.json');assert r['signature']==sig
  for f,h in r['files'].items():assert base.sha(d/f)==h
  return
 base.atomic(d/'PRE_SCORE_LOCK.json',{'signature':sig,'n':len(meta),'events':int(y.sum()),
  'same_patients_outcomes_order_for_all_models':True,'no_V1_selective_policy_transferred':True,
  'clinical_population_has_imputed_missing_intraoperative_blocks':scenario!='reference'})
 def progress(n):base.atomic(OUT/'status.json',{'state':'evaluating_cohort_expansion','task':'/'.join([line,scenario,scope]),
  'completed_bootstrap':n,'total_bootstrap':2000,'pid':os.getpid(),'time_UTC':base.stamp()})
 result=stats.evaluate(y,preds,g,rows,centers,{},comparisons,replicates=2000,seed=20261017,progress=progress)
 base.atomic(d/'EVALUATION.json',result)
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(d/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(d/'INTERVALS.csv',index=False)
 pd.DataFrame(result['paired_differences']).to_csv(d/'PAIRED_DIFFERENCES.csv',index=False)
 base.atomic(d/'RAW_POINT_ESTIMATES.json',{k:stats.summary(y,p,g,rows) for k,p in raws.items()})
 threshold_rows=[]
 for k,p in preds.items():
  for rule,t in cuts[k].items():
   if rule.startswith('training_') and isinstance(t,(float,int)):
    threshold_rows.append({'model':k,'rule':rule,'threshold':t,**stats.counts(y,p>=t)})
 pd.DataFrame(threshold_rows).to_csv(d/'TRAINING_THRESHOLD_COUNTS.csv',index=False)
 check();files=['PRE_SCORE_LOCK.json','EVALUATION.json','POINT_ESTIMATES.csv','INTERVALS.csv','PAIRED_DIFFERENCES.csv','RAW_POINT_ESTIMATES.json','TRAINING_THRESHOLD_COUNTS.csv']
 base.atomic(d/'COMPLETE.json',{'status':'PASS_fixed_cohort_expansion_evaluation','signature':sig,
  'files':{f:base.sha(d/f) for f in files},'n':len(meta),'events':int(y.sum()),
  'fixed_choice_conditional_sensitivity_not_full_redevelopment':True,'not_hospital1_true_information_validation':True})
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_cohort_reference_alignment_and_paired_stats_qualification'
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],
  'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 while not (FITS/'STAGE_EXTENSION_FIT_ACCEPTANCE.json').exists():
  if (FITS/'FAILURE.json').exists():raise RuntimeError('Upstream conditional cohort fit failure; diagnose')
  base.atomic(OUT/'status.json',{'state':'waiting_cohort_expansion_fits','pid':os.getpid(),'time_UTC':base.stamp()});time.sleep(30)
 upstream=(FITS/'SUPERVISOR.lock').open('a')
 while True:
  try:fcntl.flock(upstream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 a=base.read(FITS/'STAGE_EXTENSION_FIT_ACCEPTANCE.json');assert a['status']=='PASS_twelve_fixed_cohort_expansion_flows_fitted'
 assert len(a['receipts'])==12
 for p,h in a['receipts'].items():assert base.sha(p)==h
 for j in task_list():
  while base.mem()<6*1024**3:time.sleep(30)
  execute(*j)
 receipts=list(OUT.glob('*/*/*/COMPLETE.json'));assert len(receipts)==10
 base.atomic(OUT/'STAGE_EXTENSION_EVALUATION_ACCEPTANCE.json',{'status':'PASS_ten_fixed_cohort_and_common_reference_evaluations',
  'receipts':{str(p):base.sha(p) for p in receipts},'replicates_per_task':2000,
  'no_V1_models_or_policies_replaced':True,'hospital1_measured_endoperative_information_not_validated':True,'whole_project_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_cohort_expansion_evaluation','completed':10,'total':10,'owned_workers':0,'time_UTC':base.stamp()})
 upstream.close();guard.close()
if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc()});raise
