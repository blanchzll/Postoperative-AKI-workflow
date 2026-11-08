"""Source-locked inference; final metadata and scoring are gated before this entrypoint."""
from common import *
import os,sys,time,gc,warnings
os.environ['AKI_SCOPE_OUT']=str(ROOT/'outputs/ensemble_calibration_20261020_v1')
source=module('aki_source_runtime_evaluation',ROOT/'work/model_lock_20261022_v1/runtime.py')
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from evaluation_stats import evaluate,summary,route,counts,calibration,FixedRisk,DCA
def name(lock):
 i=lock['identity'];return '_'.join(i[k] for k in ['stage','branch','role','family'])
def inputs(lock,validation=True):
 i=lock['identity'];cohort,line,stage=i['cohort'],i['line'],i['stage']
 assert lock['input_source_sha256']==sha(SOURCE/'SOURCE_LOCK.json')
 m=pd.read_csv(SOURCE/(cohort+'_metadata.csv'),dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False)
 component=next(iter(lock['base_components'].values()))
 rec=read(Path(component['path']).parent/'COMPLETE.json')
 trainrows=rec['signature']['task']['train_rows'];train=m.loc[trainrows]
 assert train.index.is_unique and train[line+'_partition_v2'].eq('development').all()
 assert len(train)==lock['training_n'] and source.digest(trainrows)==lock['training_rows_sha256']
 assert source.digest(sorted(train.group_hash.tolist()))==lock['training_groups_sha256']
 if validation:
  selected=m.loc[m[line+'_partition_v2'].eq('validation')].copy()
  assert set(selected.group_hash).isdisjoint(train.group_hash)
  assert (len(selected),int(selected.y.sum()))==EXPECTED[(line,cohort)]
 else:selected=train.copy()
 sp=source.source_spec(stage,lock['input_columns'])
 x=pd.read_csv(SOURCE/(cohort+'_'+stage+'_features.csv'),dtype={c:'string' for c in sp['cat']},float_precision='round_trip').set_index('excel_row')
 assert x.index.is_unique and selected.index.is_unique
 return x.loc[selected.index,lock['input_columns']],selected
def qualify_development():
 a=read(MODELS/'STAGE_MODEL_LOCK_ACCEPTANCE.json');checks=[]
 for ref in a['models']:
  assert sha(ref['path'])==ref['sha256'];lock=read(ref['path'])
  assert sha(ROOT/'work/model_lock_20261022_v1/runtime.py')==lock['inference_code_sha256']
  x,m=inputs(lock,False)
  probe=x.index[x.isna().any(axis=1)].tolist()[:2]
  probe += [j for j in x.index if j not in probe][:max(0,2-len(probe))]
  x=x.loc[probe];m=m.loc[probe]
  raw,cal=source.route_prediction(lock,x,source.keys(m))
  rr,cc=source.route_prediction(lock,x.iloc[::-1],source.keys(m.iloc[::-1]))
  np.testing.assert_allclose(raw,rr[::-1],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(cal,cc[::-1],rtol=1e-12,atol=1e-12)
  checks.append({'identity':lock['identity'],'development_only':True,'n_probe':len(probe),'reverse_parity':True})
 assert len(checks)==len(a['models']) and a['primary_routes']==10
 assert len({ref['path'] for ref in a['models']})==len(checks)
 atomic(OUT/'CLINICAL_LOADER_QUALIFICATION.json',{'status':'PASS_all_current_locked_routes_development_only','checks':checks,'final_scores_accessed':False})
def clinical_main(line,cohort):
 a,b=gate()
 assert read(OUT/'FUNCTIONAL_QUALIFICATION.json')['status']=='PASS_synthetic_metrics_calibration_policy_bootstrap'
 assert read(OUT/'CLINICAL_LOADER_QUALIFICATION.json')['status']=='PASS_all_current_locked_routes_development_only'
 dest=OUT/(line+'_'+cohort);dest.mkdir(parents=True,exist_ok=True)
 signature={'code_lock':sha(OUT/'CODE_LOCK.json'),'model_acceptance':sha(MODELS/'STAGE_MODEL_LOCK_ACCEPTANCE.json'),
 'policy_acceptance':sha(POLICIES/'STAGE_POLICY_LOCK_ACCEPTANCE.json'),'source':sha(SOURCE/'SOURCE_LOCK.json'),'line':line,'cohort':cohort}
 if (dest/'COMPLETE.json').exists():
  old=read(dest/'COMPLETE.json');assert old['signature']==signature
  for f,h in old['files'].items():assert sha(dest/f)==h
  return
 predictions={};rawpred={};locks={};refs={};primary={}
 meta=None
 for ref in a['models']:
  lock=read(ref['path']);ident=lock['identity']
  if (ident['line'],ident['cohort'])!=(line,cohort):continue
  nm=name(lock);x,m=inputs(lock,True)
  if meta is None:meta=m
  else:assert meta.equals(m),'Paired metadata differs'
  target=dest/'private_predictions'/nm;target.mkdir(parents=True,exist_ok=True)
  sig={'signature':signature,'model_lock':ref['sha256']}
  if (target/'COMPLETE.json').exists():
   old=read(target/'COMPLETE.json')
   compatible=(old['signature']==sig)
   if not compatible:
    recovery=read(OUT/'DATE_RECOVERY_COMPATIBILITY.json')
    legacy=json.loads(json.dumps(sig));legacy['signature']['code_lock']=recovery['previous_code_lock_sha256']
    compatible=(old['signature']==legacy and recovery['prediction_receipts'].get(str(target/'COMPLETE.json'))==sha(target/'COMPLETE.json'))
   assert compatible and sha(target/'PREDICTIONS.csv')==old['sha256']
   pr=pd.read_csv(target/'PREDICTIONS.csv',dtype={'group_hash':str},float_precision='round_trip')
   assert pr.excel_row.tolist()==m.excel_row.tolist() and pr.group_hash.tolist()==m.group_hash.tolist() and pr.y.tolist()==m.y.tolist()
   raw=pr.prob_raw.to_numpy();cal=pr.prob_calibrated.to_numpy()
  else:
   start=time.time()
   with warnings.catch_warnings(record=True) as notes:
    warnings.simplefilter('always')
    raw,cal=source.route_prediction(lock,x,source.keys(m))
   pr=m.reset_index(drop=True).copy();pr['prob_raw']=np.asarray(raw,dtype=np.float64);pr['prob_calibrated']=np.asarray(cal,dtype=np.float64)
   pr.to_csv(target/'PREDICTIONS.csv',index=False)
   atomic(target/'COMPLETE.json',{'signature':sig,'sha256':sha(target/'PREDICTIONS.csv'),'seconds':time.time()-start,
     'n':len(pr),'events':int(pr.y.sum()),'warnings':sorted(set(str(w.message)[:500] for w in notes)),
     'frozen_models_applied_no_refitting':True,'policy_lock_verified_before_final_prediction':True})
  predictions[nm]=np.asarray(cal);rawpred[nm]=np.asarray(raw);locks[nm]=lock;refs[str(Path(ref['path']))]=nm
  if ref['primary_route_winner']:primary[(ident['stage'],ident['branch'])]=nm
  atomic(dest/'status.json',{'state':'predicting_locked_models','models_done':len(predictions),'time_UTC':stamp()})
  del x;gc.collect()
 assert meta is not None
 policies={}
 for ref in b['policy_manifests']:
  lock=read(ref['path']);i=lock['identity']
  if (i['line'],i['cohort'])!=(line,cohort):continue
  assert sha(lock['calibrator']['path'])==lock['calibrator']['sha256']
  nm=f"policy_{i['upgrade_fraction']}_{i['mode']}_{i['kind']}"
  policies[nm]={'lock':lock,'base':refs[lock['base_model']['path']],'upgrade':refs[lock['upgrade_model']['path']],'calibrator':joblib.load(lock['calibrator']['path'])}
 assert len(policies)==6
 comparisons=[]
 base=primary[('preop','compact' if cohort=='preop' else 'full')]
 up=primary[('preop' if cohort=='preop' else 'endop','full')]
 comparisons.append((up,base))
 if cohort=='preop':comparisons.append((primary[('preop','selected')],up))
 else:
  for fam in {locks[base]['family'],locks[up]['family']}:
   pre=[n for n,l in locks.items() if l['identity']['role']=='reference' and l['identity']['stage']=='preop' and l['family']==fam]
   end=[n for n,l in locks.items() if l['identity']['role']=='reference' and l['identity']['stage']=='endop' and l['family']==fam]
   assert len(pre)==len(end)==1;comparisons.append((end[0],pre[0]))
 for nm,item in policies.items():
  comparisons.extend([(nm,base),(nm,up)])
  i=item['lock']['identity']
  if i['mode']=='capacity' and i['kind']=='selective':
   comparisons.append((nm,f"policy_{i['upgrade_fraction']}_capacity_random"))
 comparisons=list(dict.fromkeys(comparisons))
 def progress(n):atomic(dest/'status.json',{'state':'bootstrap_fixed_models_and_rerouted_policies','complete':n,'total':2000,'time_UTC':stamp()})
 recovery=read(OUT/'DATE_RECOVERY_COMPATIBILITY.json')
 cache_sha=recovery['completed_evaluations'].get(dest.name)
 if cache_sha:
  assert sha(dest/'EVALUATION.json')==cache_sha
  assert sha(CODE/'evaluation_stats.py')==recovery['unchanged_statistics_sha256']
  result=read(dest/'EVALUATION.json')
  assert result['receipt']['replicates']==2000 and result['receipt']['seed']==20261017
 else:
  result=evaluate(meta.y.to_numpy(),predictions,meta.group_hash.astype(str).tolist(),meta.excel_row.astype(str).tolist(),meta.center.astype(str).tolist(),policies,comparisons,progress=progress)
 atomic(dest/'EVALUATION.json',result)
 pd.DataFrame([{'model':k,**v} for k,v in result['point'].items()]).to_csv(dest/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['intervals']).to_csv(dest/'INTERVALS.csv',index=False)
 pd.DataFrame(result['paired_differences']).to_csv(dest/'PAIRED_DIFFERENCES.csv',index=False)
 # Raw constituent risks retained separately; do not mix raw and calibrated comparisons.
 groups=meta.group_hash.astype(str).tolist();records=meta.excel_row.astype(str).tolist();yy=meta.y.to_numpy()
 atomic(dest/'RAW_MODEL_POINT_ESTIMATES.json',{nm:summary(yy,p,groups,records) for nm,p in rawpred.items()})
 allpred=dict(predictions);upgraded={}
 for nm,item in policies.items():
  raw,cal,mask=route(item,predictions[item['base']],predictions[item['upgrade']],groups,records)
  allpred[nm]=cal;allpred[nm+'__raw_mixture']=raw;upgraded[nm]=mask
  q=meta.reset_index(drop=True).copy();q['prob_raw_mixture']=raw;q['prob_calibrated']=cal;q['upgraded']=mask
  q.to_csv(dest/(nm+'_PRIVATE.csv'),index=False)
 # Hospital and annual results inherit cohort-wide capacity allocations; no local reranking.
 rows=[];thresholds=[];curves=[]
 for nm,p in allpred.items():
  f=FixedRisk(yy,p,groups,records);global_alerts={fraction:f.review_counts(fraction).astype(bool) for fraction in [.1,.2]}
  subsets=[('overall','all',np.ones(len(meta),dtype=bool))]
  for field,vector in [('hospital',meta.center.astype(str)),('year',pd.to_datetime(meta.surgery_date,format="ISO8601",errors="raise").dt.year.astype(str))]:
   subsets.extend((field,level,np.asarray(vector==level)) for level in sorted(set(vector)))
  for field,level,mask in subsets:
   if not mask.any():continue
   sf=FixedRisk(yy[mask],p[mask],np.asarray(groups)[mask],np.asarray(records)[mask])
   score={**sf.score(),**calibration(yy[mask],p[mask])}
   for fraction,a_mask in global_alerts.items():
    rows.append({'model':nm,'stratum':field,'level':level,'review_fraction':fraction,**score,**counts(yy[mask],a_mask[mask]),
       'capacity_ranking':'whole_evaluation_cohort_inherited','calibration_small_event_caution':int(yy[mask].sum())<10})
  lock=locks.get(nm) or (policies[nm]['lock'] if nm in policies else None)
  if lock:
   for rule,t in lock['training_thresholds'].items():
    if rule.startswith('training_') and isinstance(t,(int,float)):
     thresholds.append({'model':nm,'rule':rule,'risk_threshold':t,**counts(yy,p>=t)})
  order=np.argsort(p,kind='stable')
  for k,ix in enumerate(np.array_split(order,10)):
   curves.append({'model':nm,'bin':k+1,'n':len(ix),'events':int(yy[ix].sum()),'mean_predicted':float(p[ix].mean()),'observed_rate':float(yy[ix].mean()),
      'bin_definition':'equal_count_risk_sorted_display_only_no_model_update'})
 pd.DataFrame(rows).to_csv(dest/'HOSPITAL_YEAR_DESCRIPTIVE.csv',index=False)
 pd.DataFrame(thresholds).to_csv(dest/'LOCKED_THRESHOLD_BURDEN.csv',index=False)
 pd.DataFrame(curves).to_csv(dest/'CALIBRATION_CURVES.csv',index=False)
 files=['EVALUATION.json','POINT_ESTIMATES.csv','INTERVALS.csv','PAIRED_DIFFERENCES.csv','RAW_MODEL_POINT_ESTIMATES.json','HOSPITAL_YEAR_DESCRIPTIVE.csv','LOCKED_THRESHOLD_BURDEN.csv','CALIBRATION_CURVES.csv']
 atomic(dest/'COMPLETE.json',{'status':'PASS_fixed_models_and_locked_policy_heldout_evaluation','signature':signature,
    'files':{f:sha(dest/f) for f in files},'models':len(predictions),'policy_variants':len(policies),'n':len(meta),'events':int(meta.y.sum()),
    'patient_group_disjoint_from_training':True,'hospital_year_details_descriptive_intervals_pending':True,
    'subgroup_and_sensitivity_pending':True,'no_training_or_reselection_on_evaluation':True})
 atomic(dest/'status.json',{'state':'complete_cohort_evaluation','time_UTC':stamp(),'task_owned_workers_remaining':0})
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if sys.argv[1]=='--qualify':qualify_development()
  else:clinical_main(sys.argv[1],sys.argv[2])
