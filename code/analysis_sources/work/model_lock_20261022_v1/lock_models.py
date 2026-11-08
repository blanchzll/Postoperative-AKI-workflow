"""Finite, resumable model-lock supervisor. Waits for E; never reads final evaluation scores."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,os,json,hashlib,time,fcntl,subprocess,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/model_lock_20261022_v1'
E=ROOT/'outputs/ensemble_calibration_20261020_v1'
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
OWNED_WORKERS={}
OWNED_GUARD=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def pick(rows):
 best=max(float(r['AP_mean']) for r in rows);rows=[r for r in rows if float(r['AP_mean'])>=best-1e-12]
 best=min(float(r['Brier_mean']) for r in rows);rows=[r for r in rows if float(r['Brier_mean'])<=best+1e-12]
 return min(rows,key=lambda r:json.dumps({k:r[k] for k in ['line','cohort','stage','branch','family']},sort_keys=True,separators=(',',':')))
def jobs_and_routes():
 import pandas as pd
 import runtime as r
 r.require_e();r.checked(E/'calibrated_internal/ACCEPTANCE.json')
 source=E/'calibrated_internal/CALIBRATED_ELEVEN_FAMILY_SUMMARY.csv'
 df=pd.read_csv(source);group=['line','cohort','stage','branch']
 assert len(df)==110 and df.groupby(group).size().eq(11).all()
 winners={tuple(k):pick(g.to_dict('records')) for k,g in df.groupby(group,sort=True)}
 assert len(winners)==10
 full_manifest=read(r.D/'SCOPE_MANIFEST.json')
 roots={(x['line'],x['cohort']):x for x in full_manifest['roots'] if x['context']=='final'}
 routes=[];jobs={}
 for k,w in winners.items():
  line,cohort,stage,branch=k;root=roots[(line,cohort)]
  references={'LR'}
  if cohort=='paired':
   references.update(x['family'] for kk,x in winners.items() if kk[0]==line and kk[1]=='paired')
  methodfile=r.BASE/'selection'/line/cohort/'final/METHOD_SELECTION.json'
  method=read(methodfile)['selected_method']
  for family,role in [(w['family'],'champion')]+[(f,'reference') for f in sorted(references)]:
   task={'line':line,'cohort':cohort,'stage':stage,'branch':branch,'family':family,
     'train_rows':root['train_rows'],'method':method,'method_receipt':str(methodfile),
     'method_receipt_sha256':sha(methodfile),
     'input_source_lock_sha256':sha(r.SOURCE/'SOURCE_LOCK.json'),
     'feature_csv_sha256':sha(r.SOURCE/(cohort+'_'+stage+'_features.csv')),
     'metadata_sha256':sha(r.SOURCE/(cohort+'_metadata.csv'))}
   rawdir=r.D/'ensemble_results'/line/cohort/stage/branch/'final'
   rawrec=r.checked(rawdir/'COMPLETE.json')
   oof=rawdir/'honest_base_oof.csv'
   if family in r.FAMILIES:members=[family];ensemble_params=None
   else:
    chosen=read(rawdir/'SELECTION.json')['winners'][family]
    z=pd.read_csv(oof,dtype={'group_hash':str},float_precision='round_trip')
    members=r.rank_families(z)[:chosen['k']];ensemble_params={'k':chosen['k'],'C':chosen['C']}
   for member in members:
    stask={**task,'family':member}
    sp=r.BASE/'search_v2'/line/cohort/stage/branch/'final'/member/'COMPLETE.json'
    sr=r.checked(sp);sel=sr['selected']
    assert sha(sel['prediction_file'])==sel['prediction_sha256']
    stask.update(params=sel['params'],search_receipt=str(sp),search_receipt_sha256=sha(sp))
    jid=r.base_id(stask)
    if jid in jobs:assert jobs[jid]==stask
    jobs[jid]=stask
   routes.append({**task,'role':role,'members':members,'ensemble_params':ensemble_params,
     'primary_route_winner':role=='champion','winner_internal_selection_scores':w if role=='champion' else None,
     'development_raw_oof':str(oof),'development_raw_oof_sha256':sha(oof),
     'root':root['id']})
 selection={'time_beijing':stamp(),'score_source':str(source),'score_source_sha256':sha(source),
  'selection_rule':'Calibrated candidate mean outer AP, mean Brier, canonical route/family ID; tolerance1e-12.',
  'E_acceptance_sha256':sha(E/'STAGE_E_ACCEPTANCE.json'),'route_winners':list(winners.values()),
  'routes_and_references':routes,'base_jobs':jobs,'final_evaluation_accessed':False}
 selection['overall_preop_branch_choice']={line:pick([w for k,w in winners.items() if k[0]==line and k[1]=='preop']) for line in ['geography','temporal']}
 target=OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json'
 if target.exists():
  old=read(target)
  assert {k:v for k,v in old.items() if k!='time_beijing'}=={k:v for k,v in selection.items() if k!='time_beijing'}
  return old
 atomic(target,selection)
 return selection
def route_selection_oof(t,adapter=None):
 import pandas as pd
 import runtime as r
 if adapter is None:
  import complete_e as frozen
 else:frozen=adapter
 lineage={};parts=[];choices=[]
 children=sorted([x for x in frozen.DM['scopes'] if x['parent']==t['root']],key=lambda x:x['parent_fold'])
 assert len(children)==3
 for child in children:
  pool=frozen.training_pool(child,t['stage'],[t['branch']],lineage)
  assert len(pool)==11
  winner=frozen.choose(pool)
  x,rec=frozen.esource(child,t['stage'],t['branch'],winner['family'])
  parts.append(x.assign(fold=child['parent_fold'],selected_family=winner['family']))
  lineage[str(rec)]=sha(rec)
  choices.append({'scope':child['id'],'winner':winner})
 training=pd.concat(parts,ignore_index=True)
 assert training.excel_row.is_unique and set(training.excel_row)==set(t['train_rows'])
 return training,{'training_only_child_family_choices':choices,'source_receipts':lineage,
  'scored_patients_excluded_from_child_selection_and_fitting':True}

def finish_routes(selection):
 import numpy as np,pandas as pd,joblib
 import runtime as r
 locks=[]
 for t in selection['routes_and_references']:
  d=OUT/'locked_routes'/r.route_id(t)/t['role']/t['family'];d.mkdir(parents=True,exist_ok=True)
  if (d/'ACCEPTANCE.json').exists():
   old=read(d/'MODEL_LOCK.json');acc=read(d/'ACCEPTANCE.json')
   assert sha(d/'MODEL_LOCK.json')==acc['lock_sha256']
   assert old['source_selection_sha256']==sha(OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json')
   for comp in list(old['base_components'].values())+[old['calibrator']]+([old['meta']] if old['meta'] else [])+[old['calibration_training_scores']]:
    assert sha(comp['path'])==comp['sha256']
   locks.append({'path':str(d/'MODEL_LOCK.json'),'sha256':sha(d/'MODEL_LOCK.json'),'identity':old['identity'],'primary_route_winner':old['primary_route_winner']})
   continue
  source=pd.read_csv(t['development_raw_oof'],dtype={'group_hash':str},float_precision='round_trip')
  m=pd.read_csv(r.SOURCE/(t['cohort']+'_metadata.csv'),dtype={'group_hash':str}).set_index('excel_row',drop=False)
  assert source.excel_row.is_unique and set(source.excel_row)==set(t['train_rows'])
  mm=m.loc[source.excel_row]
  assert source.group_hash.tolist()==mm.group_hash.tolist() and source.y.tolist()==mm.y.tolist()
  assert mm[t['line']+'_partition_v2'].eq('development').all()
  components={};union=set()
  for family in t['members']:
   bd=OUT/'base_fits'/('/'.join([t[k] for k in ['line','cohort','stage','branch']])+'/'+family)
   br=r.checked(bd/'COMPLETE.json');bp=bd/'raw_bundle.joblib'
   components[family]={'path':str(bp),'sha256':sha(bp)}
   union.update(br['input_columns'])
  meta=None
  if t['role']=='champion':
   training,cal_lineage=route_selection_oof(t)
   cal=r.calibrator(training);joblib.dump(cal,d/'calibrator.joblib',compress=3)
   atomic(d/'CALIBRATION_LINEAGE.json',cal_lineage)
  if t['family'] in r.FAMILIES:
   if t['role']!='champion':
    training=source[['excel_row','group_hash','y',t['family']]].rename(columns={t['family']:'prob_raw'})
    cal=r.calibrator(training);joblib.dump(cal,d/'calibrator.joblib',compress=3)
  else:
   ec=r.E/'calibrated_internal/candidate_calibration'/t['line']/t['cohort']/t['stage']/t['branch']/'final'/t['family']
   if t['role']!='champion':
    cal=joblib.load(ec/'calibrator.joblib')
    joblib.dump(cal,d/'calibrator.joblib',compress=3)
    training=pd.read_csv(ec/'PRIVATE_HONEST_CALIBRATION_OOF.csv',dtype={'group_hash':str},float_precision='round_trip')
   assert training.excel_row.is_unique and set(training.excel_row)==set(t['train_rows'])
   tm=m.loc[training.excel_row]
   assert training.group_hash.tolist()==tm.group_hash.tolist() and training.y.tolist()==tm.y.tolist()
   if t['family']=='Stacking':
    meta=r.LogisticRegression(C=t['ensemble_params']['C'],penalty='l2',solver='lbfgs',max_iter=5000,random_state=42)
    with r.threadpool_limits(limits=1):meta.fit(source[t['members']],source.y)
    joblib.dump(meta,d/'meta.joblib',compress=3)
   else:meta=None
  training['prob_calibrated']=r.apply_calibration(cal,training.prob_raw)
  cuts=r.thresholds(training.y.to_numpy(),training.prob_calibrated.to_numpy())
  training.to_csv(d/'PRIVATE_TRAINING_THRESHOLD_SCORES.csv',index=False)
  sp=r.source_spec(t['stage']);columns=[c for c in sp['num']+sp['cat'] if c in union]
  manifest={'version':'source_locked_private_pipeline_v1','identity':{k:t[k] for k in ['line','cohort','stage','branch','role','family']},
   'calibration_scope':('training-child family selection within fixed input branch' if t['role']=='champion' else 'fixed-family controlled reference'),
   'family':t['family'],'members':t['members'],'primary_route_winner':t['primary_route_winner'],
   'input_columns':columns,'actual_input_count':len(columns),'base_components':components,
   'calibrator':{'path':str(d/'calibrator.joblib'),'sha256':sha(d/'calibrator.joblib')},
   'meta':({'path':str(d/'meta.joblib'),'sha256':sha(d/'meta.joblib')} if meta is not None else None),
   'training_thresholds':cuts,'case_key_required_for_stochastic_completion':True,
   'calibration_training_scores':{'path':str(d/'PRIVATE_TRAINING_THRESHOLD_SCORES.csv'),'sha256':sha(d/'PRIVATE_TRAINING_THRESHOLD_SCORES.csv')},
   'calibration_lineage':({'path':str(d/'CALIBRATION_LINEAGE.json'),'sha256':sha(d/'CALIBRATION_LINEAGE.json')} if t['role']=='champion' else None),
   'input_spec_source':str(r.SOURCE/'FEATURE_SPECS.json'),'input_source_sha256':sha(r.SOURCE/'SOURCE_LOCK.json'),
   'training_rows_sha256':r.digest(t['train_rows']),'training_n':len(t['train_rows']),
   'training_groups_sha256':r.digest(sorted(mm.group_hash.tolist())),'model_runtime':str(PYTHON),
   'inference_code_sha256':sha(CODE/'runtime.py'),'model_and_strategy_locks_required_before_final_scoring':True,
   'source_selection_sha256':sha(OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json'),
   'contains_training_information_keep_private':True}
  # Verify complete private pipeline using only full-development training probes.
  raw,meta_rows=r.data(t,r.source_spec(t['stage'],columns))
  probe=raw.loc[raw.isna().any(axis=1)].head(4)
  if len(probe)<4:probe=raw.head(4)
  pm=meta_rows.loc[probe.index];ids=r.keys(pm)
  p,pc=r.route_prediction(manifest,probe,ids)
  rev,rc=r.route_prediction(manifest,probe.iloc[::-1],ids[::-1])
  solo=[r.route_prediction(manifest,probe.iloc[i:i+1],ids[i:i+1]) for i in range(len(probe))]
  np.testing.assert_allclose(p,rev[::-1],atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(pc,rc[::-1],atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(p,np.concatenate([v[0] for v in solo]),atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(pc,np.concatenate([v[1] for v in solo]),atol=1e-12,rtol=1e-12)
  for cut_name in ['training_sensitivity80','training_capacity10','training_capacity20']:
   cut=cuts[cut_name]
   np.testing.assert_array_equal(pc>=cut,rc[::-1]>=cut)
   np.testing.assert_array_equal(pc>=cut,np.concatenate([v[1] for v in solo])>=cut)
  atomic(d/'MODEL_LOCK.json',manifest)
  reloaded=read(d/'MODEL_LOCK.json');rp,rpc=r.route_prediction(reloaded,probe,ids)
  np.testing.assert_allclose(p,rp,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(pc,rpc,atol=1e-12,rtol=1e-12)
  atomic(d/'ACCEPTANCE.json',{'status':'PASS_private_model_route_lock','lock_sha256':sha(d/'MODEL_LOCK.json'),
   'single_batch_reorder_reload_verified':True,'thresholds_training_only':True,'final_scores_accessed':False})
  locks.append({'path':str(d/'MODEL_LOCK.json'),'sha256':sha(d/'MODEL_LOCK.json'),
   'identity':manifest['identity'],'primary_route_winner':manifest['primary_route_winner']})
  del source,training,raw;gc=__import__('gc');gc.collect()
 atomic(OUT/'STAGE_MODEL_LOCK_ACCEPTANCE.json',{'status':'PASS','time_beijing':stamp(),'models':locks,
  'primary_routes':sum(x['primary_route_winner'] for x in locks),'selection_sha256':sha(OUT/'PRIVATE_MODEL_SELECTION_AND_JOBS.json'),
  'final_model_locked':True,'selective_policy_locked':False,'final_hospital_time_scoring_permitted':False,
  'next_required':'scope-compatible selective-policy fitting, calibration and locks before final evaluation',
  'entire_research_complete':False})
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 global OWNED_GUARD
 guard=(OUT/'MODEL_LOCK_COORDINATOR.lock').open('a');OWNED_GUARD=guard
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source_lock=read(OUT/'CODE_LOCK.json')
 assert source_lock=={p.name:sha(p) for p in CODE.glob('*.py')}
 fq=read(OUT/'FUNCTIONAL_QUALIFICATION.json');assert fq['status']=='PASS_model_lock_engineering_only'
 atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'time_beijing':stamp(),'state':'waiting_for_E_acceptance',
  'source_lock_sha256':sha(OUT/'CODE_LOCK.json'),'final_scores_accessed':False})
 while not (E/'STAGE_E_ACCEPTANCE.json').exists() or read(E/'status_e.json')['state']!='complete_strict_calibration_internal_evaluation':
  assert source_lock=={p.name:sha(p) for p in CODE.glob('*.py')}
  st=read(E/'status_e.json')
  if st.get('failed') or st['state'].startswith('failed'):raise RuntimeError('E failed; model lock not started')
  atomic(OUT/'status.json',{'state':'waiting_for_E_acceptance','pid':os.getpid(),'time_beijing':stamp(),
   'parent_E_state':st['state'],'parent_E_completed_tasks':st.get('completed_tasks'),
   'models_fitted':0,'final_model_locked':False,'final_scores_accessed':False})
  time.sleep(30)
 import runtime as r
 from core import resource_snapshot,process_rss
 sys.path.insert(0,str(ROOT/'work/ensemble_speedup_20261021_v2'))
 from resource_policy import CPUShare
 sel=jobs_and_routes();jobs=sel['base_jobs'];running=OWNED_WORKERS;done=set();failed=[];cpu=CPUShare();time.sleep(2)
 for key,t in jobs.items():
  path=OUT/'base_fits'/key/'COMPLETE.json'
  if path.exists():assert r.complete(path.parent,r.base_signature(t));done.add(key)
 while len(done)<len(jobs):
  for key,v in list(running.items()):
   proc,log=v['process'],v['log']
   if proc.poll() is None:continue
   log.close()
   if proc.returncode==0:r.checked(OUT/'base_fits'/key/'COMPLETE.json');done.add(key)
   else:failed.append({'job':key,'exit':proc.returncode})
   del running[key]
  assert source_lock=={p.name:sha(p) for p in CODE.glob('*.py')}
  snap=resource_snapshot();share=cpu.sample(snap.get('affinity_and_quota_cpu',snap['effective_cpu']),[v['process'].pid for v in running.values()])
  for v in running.values():
   rss=process_rss(v['process'].pid)
   v['reservation']=max(v['reservation'],int(rss*1.25))
  growth=sum(max(0,v['reservation']-process_rss(v['process'].pid)) for v in running.values())
  budget=snap['available_bytes']-max(4*1024**3,.25*snap['available_bytes'])-growth
  free=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize
  if not failed and free>60*1024**3:
   for key,t in jobs.items():
    if key in done or key in running or len(running)>=share or budget<1024**3:continue
    fn=OUT/'requests'/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.json');atomic(fn,t)
    lp=OUT/'logs'/(fn.stem+'.log');lp.parent.mkdir(exist_ok=True)
    log=lp.open('ab');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',AKI_SCOPE_OUT=str(E),AKI_SCOPE_INPUT=str(r.SOURCE))
    proc=subprocess.Popen([str(PYTHON),'-B',str(CODE/'base_worker.py'),str(fn)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    running[key]={'process':proc,'log':log,'reservation':1024**3};budget-=1024**3
  atomic(OUT/'status.json',{'state':'fitting_final_training_components' if not failed else 'failed_draining',
   'pid':os.getpid(),'time_beijing':stamp(),'complete':len(done),'total':len(jobs),
   'running':{k:v['process'].pid for k,v in running.items()},'failed':failed,
   'final_model_locked':False,'final_scores_accessed':False,'resource':snap})
  if failed and not running:raise RuntimeError('Final component failed; checkpoints retained')
  time.sleep(5)
 while resource_snapshot()['available_bytes']<5*1024**3:
  atomic(OUT/'status.json',{'state':'waiting_memory_for_final_lock_checks','pid':os.getpid(),'time_beijing':stamp(),'final_model_locked':False})
  time.sleep(30)
 finish_routes(sel)
 atomic(OUT/'status.json',{'state':'complete_model_locks_policy_pending','time_beijing':stamp(),
   'task_owned_workers_remaining':0,'final_model_locked':True,'selective_policy_locked':False,
   'final_scores_accessed':False,'entire_research_complete':False})
if __name__=='__main__':
 try:main()
 except Exception:
  # Retain the exclusive supervisor lock while owned workers finish naturally.
  for v in list(OWNED_WORKERS.values()):
   v['process'].wait();v['log'].close()
  atomic(OUT/'FAILURE.json',{'time_beijing':stamp(),'traceback':traceback.format_exc()})
  raise
