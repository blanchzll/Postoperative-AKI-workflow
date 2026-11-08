"""Fixed prior choices: final POD1 secondary evaluation, same patients within comparisons."""
from pathlib import Path
import sys,os,time,json,fcntl,subprocess,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pod1_corrected_evaluation_20261024_v1';FIT=ROOT/'outputs/pod1_corrected_fixed_update_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'));import common as base
import evaluation_stats as stats
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def load(i):
 d=FIT/'tasks'/str(i).zfill(3);r=base.read(d/'COMPLETE.json')
 for f,h in r['files'].items():assert base.sha(d/f)==h
 for f,h in r['raw_fit_receipts'].items():assert base.sha(f)==h
 return pd.read_csv(d/'PRIVATE_PREDICTIONS.csv',dtype={'group_hash':str},float_precision='round_trip'),base.read(d/'THRESHOLDS.json'),str(d/'COMPLETE.json')
def context(line,population):
 tasks=base.read(FIT/'TASKS.json')
 branches=['all_preop','all_pod1'] if population=='all' else ['matched_preop','matched_endop','matched_pod1']
 indices=[i for i,t in enumerate(tasks) if t['line']==line and t['role']=='final' and t['branch'] in branches]
 pred={};raw={};refs={};thresholds={};meta=None
 for i in indices:
  z,c,p=load(i);t=tasks[i];key=t['branch']+'__'+t['family']
  m=z.drop(columns=['prob_raw','prob_calibrated'])
  if meta is None:meta=m
  else:assert meta.equals(m),(line,population,key)
  pred[key]=z.prob_calibrated.to_numpy();raw[key]=z.prob_raw.to_numpy();thresholds[key]=c;refs[p]=base.sha(p)
 assert not meta.excel_row.duplicated().any()
 comparisons=[('all_pod1__'+f,'all_preop__'+f) for f in ['LR','RF','XGB','Stacking']] if population=='all' else [
  ('matched_endop__LR','matched_preop__LR'),('matched_pod1__LR','matched_endop__LR'),('matched_pod1__LR','matched_preop__LR')]
 dest=OUT/(line+'_'+population);dest.mkdir(exist_ok=True)
 sig={'line':line,'population':population,'model_locks_sha256':base.sha(FIT/'MODEL_LOCKS.json'),'fit_acceptance_sha256':base.sha(FIT/'STAGE_POD1_FIT_ACCEPTANCE.json'),
  'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'sources':refs}
 if (dest/'COMPLETE.json').exists():
  r=base.read(dest/'COMPLETE.json');assert r['signature']==sig
  for f,h in r['files'].items():assert base.sha(dest/f)==h
  return
 base.atomic(dest/'PRE_SCORE_LOCK.json',{'signature':sig,'primary':'Prespecified LR POD1 minus preoperative Brier; all other families supporting only','no_heldout_model_selection':True,
  'conditional_observed_POD1_to_POD23_SCr_endpoint':True,'not_7day_clinical_AKI_or_exact_24hour_event_free':True,'n':len(meta),'events':int(meta.y.sum())})
 y=meta.y.to_numpy();g=meta.group_hash.astype(str).tolist();rows=meta.excel_row.astype(str).tolist();centers=meta.center.astype(str).tolist()
 def progress(n):base.atomic(dest/'progress.json',{'completed_bootstraps':n,'total':2000,'pid':os.getpid(),'time_UTC':base.stamp()})
 result=stats.evaluate(y,pred,g,rows,centers,{},comparisons,replicates=2000,seed=20261017,progress=progress)
 base.atomic(dest/'EVALUATION.json',result)
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(dest/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(dest/'INTERVALS.csv',index=False)
 pd.DataFrame(result['paired_differences']).to_csv(dest/'PAIRED_DIFFERENCES.csv',index=False)
 base.atomic(dest/'RAW_POINT_ESTIMATES.json',{k:stats.summary(y,p,g,rows) for k,p in raw.items()})
 fixed=[]
 for k,p in pred.items():
  for rule,t in thresholds[k].items():fixed.append({'model':k,'rule':rule,'threshold':t,**stats.counts(y,p>=t)})
 pd.DataFrame(fixed).to_csv(dest/'TRAINING_THRESHOLD_COUNTS.csv',index=False)
 # Conditional fixed-parameter development folds; not unbiased full-selection validation.
 internal=[]
 for branch in branches:
  for family in sorted(set(tasks[i]['family'] for i in indices if tasks[i]['branch']==branch)):
   kk=branch+'__'+family;parts=[]
   for i,t in enumerate(tasks):
    if t['line']==line and t['branch']==branch and t['family']==family and t['role']=='conditional_internal':
     z,c,p=load(i);parts.append(z)
   z=pd.concat(parts,ignore_index=True);assert z.excel_row.is_unique
   internal.append({'model':kk,'conditional_on_prior_fixed_choices':True,**stats.summary(z.y.to_numpy(),z.prob_calibrated.to_numpy(),z.group_hash.astype(str).tolist(),z.excel_row.astype(str).tolist())})
 pd.DataFrame(internal).to_csv(dest/'CONDITIONAL_FIXED_CHOICE_INTERNAL_POINTS.csv',index=False)
 files=[p.name for p in dest.iterdir() if p.suffix in ['.csv','.json'] and p.name not in ['COMPLETE.json','progress.json']]
 base.atomic(dest/'COMPLETE.json',{'status':'PASS_current_input_POD1_fixed_choice_secondary_evaluation','signature':sig,
  'files':{f:base.sha(dest/f) for f in files},'n':len(meta),'events':int(y.sum()),'replicates':2000,
  'no_heldout_winner_selection':True,'not_new_independent_confirmation':True,'whole_project_complete':False})
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_POD1_evaluation_source_alignment_and_original_statistics_reuse'
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 while True:
  if (FIT/'FAILURE.json').exists():raise RuntimeError('POD1 upstream failed; inspect before continuing')
  o=base.read(FIT/'LAUNCH.json') if (FIT/'LAUNCH.json').exists() else None
  p=Path('/proc',str(o['pid']),'stat') if o else None
  alive=p is not None and p.exists() and p.read_text().split()[21]==str(o['start_ticks'])
  if (FIT/'STAGE_POD1_FIT_ACCEPTANCE.json').exists() and not alive:break
  base.atomic(OUT/'status.json',{'state':'waiting_POD1_fit_and_model_locks','time_UTC':base.stamp(),'owned_workers':0});time.sleep(30)
 lock=(FIT/'SUPERVISOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 waiting=[(line,pop) for line in ['geography','temporal'] for pop in ['all','matched']];active={};receipts={}
 logs=OUT/'logs';logs.mkdir(exist_ok=True)
 try:
  while waiting or active:
   check()
   for key,(p,h) in list(active.items()):
    rc=p.poll()
    if rc is None:continue
    h.close();del active[key];assert rc==0,'Evaluation worker failed: '+key
    cp=OUT/key/'COMPLETE.json';r=base.read(cp)
    for f,hh in r['files'].items():assert base.sha(cp.parent/f)==hh
    receipts[str(cp)]=base.sha(cp)
   while waiting and base.mem()>(4+len(active))*1024**3 and len(active)<len(os.sched_getaffinity(0)):
    line,pop=waiting.pop(0);key=line+'_'+pop;h=(logs/(key+'.log')).open('ab')
    p=subprocess.Popen([str(base.PYTHON),'-B',str(CODE/'evaluate.py'),line,pop],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
    active[key]=(p,h)
   base.atomic(OUT/'status.json',{'state':'running_POD1_secondary_evaluations','completed':len(receipts),'total':4,'owned_workers':{k:p.pid for k,(p,h) in active.items()},'time_UTC':base.stamp()})
   if waiting or active:time.sleep(5)
 finally:
  for p,h in active.values():p.wait();h.close()
 base.atomic(OUT/'STAGE_POD1_EVALUATION_ACCEPTANCE.json',{'status':'PASS_four_current_input_POD1_secondary_evaluations','receipts':receipts,'replicates_per_task':2000,'interpretation_pending':True,'whole_project_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_POD1_evaluation_interpretation_pending','completed':4,'total':4,'owned_workers':0,'time_UTC':base.stamp()})
 lock.close();guard.close()
if __name__=='__main__':
 try:
  with threadpool_limits(limits=1):
   if len(sys.argv)==3:context(sys.argv[1],sys.argv[2])
   else:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'no_unchanged_retry':True});raise
