"""Fixed-prediction outcome-uncertainty scenarios; no model retraining or clinical relabelling."""
from pathlib import Path
import os,sys,time,fcntl,json,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/outcome_sensitivity_20261024_v1'
CAND=ROOT/'outputs/cohort_v2_rrt_days_20261017/REMAINING_NUMERIC_LABEL_REVIEW_CANDIDATES.csv'
sys.path.insert(0,str(ROOT/'work/decision_workload_20261024_v1'))
import clinical_adapter as adapter
base=adapter.base
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import evaluation_stats as stats
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
SCENARIOS=['exclude_relative_SCr_additions','recorded_clinical_RRT_only']
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def candidates():
 d=pd.read_csv(CAND,dtype={'group_hash':str},float_precision='round_trip')
 assert len(d)==34 and d.excel_row.is_unique and d.group_hash.is_unique
 assert d.y.eq(1).all() and d.clinical_y_before_rrt_revision.eq(0).all() and not d.acute_postop_RRT_confirmed.eq(True).any()
 assert (d.pod1_relative_AKI.eq(True)|d.pod23_relative_AKI.eq(True)).all()
 return d
def transform(meta,pred,ids,scenario):
 if scenario not in SCENARIOS:raise ValueError('Unknown prespecified scenario')
 mask=meta.excel_row.isin(ids).to_numpy()
 assert meta.loc[mask,'y'].eq(1).all()
 keep=~mask if scenario=='exclude_relative_SCr_additions' else np.ones(len(meta),dtype=bool)
 m=meta.loc[keep].copy().reset_index(drop=True)
 if scenario=='recorded_clinical_RRT_only':m.loc[m.excel_row.isin(ids),'y']=0
 pr={k:np.asarray(v)[keep].copy() for k,v in pred.items()}
 assert len(m)>0 and m.excel_row.is_unique
 return m,pr,mask
def execute(line,cohort,scenario):
 check();d=OUT/(line+'_'+cohort)/scenario;d.mkdir(parents=True,exist_ok=True)
 old,pred,thresholds,pol,eligible,comparisons,refs=adapter.load(line,cohort)
 cand=candidates();ids=set(cand.excel_row)
 shared=old.excel_row.isin(ids)
 pairs=old.loc[shared].set_index('excel_row').group_hash
 assert pairs.equals(cand.set_index('excel_row').loc[pairs.index,'group_hash'])
 m,pr,affected=transform(old,pred,ids,scenario)
 signature={'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'candidate_sha256':base.sha(CAND),
  'original_evaluation_sha256':base.sha(base.OUT/(line+'_'+cohort)/'EVALUATION.json'),
  'line':line,'cohort':cohort,'scenario':scenario,'prediction_refs':refs}
 if (d/'COMPLETE.json').exists():
  rec=base.read(d/'COMPLETE.json');assert rec['signature']==signature
  for f,h in rec['files'].items():assert base.sha(d/f)==h
  return
 if cohort=='preop':
  selected=[k for k in pr if k.startswith('preop_selected_champion_')]
  full=[k for k in pr if k.startswith('preop_full_champion_')]
  assert len(selected)==len(full)==1
  comparisons=list(dict.fromkeys(comparisons+[(selected[0],full[0])]))
 y=m.y.to_numpy();g=m.group_hash.astype(str).to_numpy();r=m.excel_row.astype(str).to_numpy();centers=m.center.astype(str).to_numpy()
 base.atomic(d/'SCENARIO_LOCK.json',{'signature':signature,'original_n':len(old),'original_events':int(old.y.sum()),
  'affected_heldout_n':int(affected.sum()),'scenario_n':len(m),'scenario_events':int(m.y.sum()),
  'training_labels_and_models_unchanged':True,'not_a_clinical_adjudication':True,
  'interpretation':'Evaluation-definition sensitivity conditional on original fitted models; not full redevelopment with changed training outcomes.',
  'capacity_policy_reallocated_in_retained_cohort_and_every_resample':True})
 def progress(n):base.atomic(OUT/'status.json',{'state':'running_outcome_sensitivity','task':line+'_'+cohort+'/'+scenario,'bootstrap_completed':n,'bootstrap_total':2000,'pid':os.getpid(),'time_UTC':base.stamp()})
 result=stats.evaluate(y,pr,g,r,centers,pol,comparisons,replicates=2000,seed=20261017,progress=progress)
 base.atomic(d/'EVALUATION.json',result)
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(d/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(d/'INTERVALS.csv',index=False)
 pd.DataFrame(result['paired_differences']).to_csv(d/'PAIRED_DIFFERENCES.csv',index=False)
 allpr=dict(pr)
 for nm,it in pol.items():_,allpr[nm],_=stats.route(it,pr[it['base']],pr[it['upgrade']],g,r)
 rows=[]
 for nm,p in allpr.items():
  for rule,t in thresholds[nm].items():
   if rule.startswith('training_') and isinstance(t,(int,float)):
    rows.append({'model':nm,'rule':rule,'fixed_training_threshold':t,**stats.counts(y,p>=t)})
 pd.DataFrame(rows).to_csv(d/'FIXED_THRESHOLD_POINT_COUNTS.csv',index=False)
 check()
 files=['SCENARIO_LOCK.json','EVALUATION.json','POINT_ESTIMATES.csv','INTERVALS.csv','PAIRED_DIFFERENCES.csv','FIXED_THRESHOLD_POINT_COUNTS.csv']
 base.atomic(d/'COMPLETE.json',{'status':'PASS_fixed_prediction_outcome_scenario','signature':signature,
  'files':{f:base.sha(d/f) for f in files},'n':len(m),'events':int(m.y.sum()),'affected_n':int(affected.sum()),
  'new_diagnostic_measurements_or_onset_imputed':False,'model_refit':False})
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check();assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_outcome_sensitivity_source_and_transform'
 base.gate()
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 completed=[]
 for line in ['geography','temporal']:
  for cohort in ['preop','paired']:
   for scenario in SCENARIOS:
    while base.mem()<6*1024**3:time.sleep(30)
    execute(line,cohort,scenario)
    completed.append(str(OUT/(line+'_'+cohort)/scenario/'COMPLETE.json'))
 check()
 base.atomic(OUT/'STAGE_OUTCOME_SENSITIVITY_ACCEPTANCE.json',{'status':'PASS_eight_fixed_prediction_outcome_scenarios',
  'receipts':{f:base.sha(f) for f in completed},'clinical_source_labels_unchanged':True,
  'refitting_sensitivities_not_completed_by_this_stage':True,'whole_project_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_outcome_evaluation_sensitivity','time_UTC':base.stamp(),'completed':8,'total':8,'owned_workers':0})
 guard.close()
if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc()});raise
