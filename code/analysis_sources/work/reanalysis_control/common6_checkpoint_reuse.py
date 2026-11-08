"""Read-only current-source common6 checkpoint qualification and receipt-reference repair."""
from pathlib import Path
import collections,csv,datetime,fcntl,hashlib,json,os,shutil,subprocess
R=Path(__file__).resolve().parents[2]
O=R/'outputs/common6_development_20261024_v1'
C=R/'outputs/reanalysis_control/common6_reference_repair_20261028_v1'
C.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
cache={}
def sha(p):
 p=Path(p);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
def save(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def check(p,h):
 assert Path(p).is_file(),str(p)
 assert sha(p)==h,('hash mismatch',str(p))
def resolved_refs(rec,jobs):
 refs={}
 for ident,h in rec['receipts'].items():
  if ident in jobs:p=Path(jobs[ident]['dest'])/'COMPLETE.json'
  else:
   p=Path(ident)
   assert str(p) in {str(Path(j['dest'])/'COMPLETE.json') for j in jobs.values()},('unknown receipt reference',ident)
  check(p,h);refs[str(p)]=h
 assert len(refs)==len(rec['receipts'])
 return refs
def verify():
 lock=(O/'SUPERVISOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(R/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json')=='afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3'
 code=read(O/'CODE_LOCK.json')
 for f,h in code.items():check(R/'work/common6_development_20261024_v1'/f,h)
 for f,h in read(O/'DEPENDENCIES.json').items():check(f,h)
 proof={};alljobs={};changes=[];summary={}
 for line,counts in [('geography',(2551,133)),('temporal',(2024,139))]:
  inp=O/'inputs'/line;il=read(inp/'SOURCE_LOCK.json')
  assert il['source_parent_sha256']==sha(R/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json')
  assert il['features']==['Age','PreopCr','PreopBUN','PreopAlb','PreopHb','Sex']
  for f,h in il['input_sha256'].items():check(inp/f,h)
  meta=list(csv.DictReader((inp/'preop_metadata.csv').open()))
  assert (len(meta),sum(int(float(m['y'])) for m in meta))==counts
  assert all(m[line+'_partition_v2']=='development' for m in meta)
  mg={int(m['excel_row']):m['group_hash'] for m in meta};assert len(mg)==len(meta)
  spec=read(inp/'FEATURE_SPECS.json')['preop'];assert spec['num']+spec['cat']==il['features']
  for phase in ['A','B']:
   mp=O/'runs'/line/(phase+'_TASKS.json');jobs=read(mp);proof[str(mp)]=sha(mp)
   for j in jobs:
    assert j['line']==line and str(j['dest']).startswith(str(O/'runs'/line)+'/')
    assert j['id'] not in alljobs;alljobs[j['id']]=j
    p=Path(j['dest'])/'COMPLETE.json';rec=read(p);task=j['task'];sig=rec['signature']
    request=O/'requests'/(j['id']+'.json');assert read(request)==task;proof[str(request)]=sha(request)
    if j['kind']=='impute':
     expected={'engine':'stable_v2_original_sklearn1.5.2','input_sha256':sha(inp/'preop_preop_features.csv'),'spec':spec,
      'train_rows':task['train_rows'],'assessment_rows':task['assessment_rows'],'method':task['method'],
      'seed':42,'iterations':5,'trees':200,'neighbors':5,
      'imputer_code':{f:sha(R/'work'/f) for f in ['v2_stable_imputation_20261017/stable_imputer.py','python_six_outcomes_20261017/mixed_imputer.py','python_six_outcomes_20261017/donor_helpers.py']}}
     assert sig==expected,('imputation signature',j['id'])
     tr,va=task['train_rows'],task['assessment_rows']
     assert set(tr)<=set(mg) and set(va)<=set(mg) and not {mg[i] for i in tr}&{mg[i] for i in va}
     assert rec['no_outcome_in_imputer'] and rec['no_final_evaluation_used'] and rec['observed_values_unchanged']
    elif j['kind']=='coarse':
     expected={'task':task,'code':sha(R/'work/common6_development_20261024_v1/engine/coarse.py'),
      'factory':sha(R/'work/v4_development_pipeline_20261018/model_candidates.py'),'input_lock':sha(inp/'SOURCE_LOCK.json')}
     assert sig==expected,('coarse signature',j['id'])
    else:
     assert j['kind']=='search' and sig['task']==task and rec['no_final_evaluation_used']
     expected_hashes={'plan_sha256':R/'outputs/research_plan_revision_20261018_094007/PROTOCOL.json',
      'search_space_sha256':R/'outputs/research_plan_revision_20261018_094007/PARAMETER_SEARCH_SPACE.json',
      'controller_sha256':R/'work/common6_development_20261024_v1/engine/search_worker.py',
      'engine_sha256':R/'work/common6_development_20261024_v1/engine/model_engine.py',
      'source_input_lock_sha256':inp/'SOURCE_LOCK.json','core_sha256':R/'work/common6_development_20261024_v1/engine/core.py'}
     for k,v in expected_hashes.items():assert sig[k]==sha(v),(j['id'],k)
     assert sig['original_model_runtime']==str(R/'envs/prediction_gpu/bin/python')
     trials=read(p.parent/'TRIALS.json')
     assert trials['selected']==rec['selected'] and len(trials['trials'])==rec['unique_trials']
     for t in trials['trials']:
      pred=Path(t['prediction_file']);assert str(pred).startswith(str(p.parent)+'/');check(pred,t['prediction_sha256']);proof[str(pred)]=sha(pred)
    for f,h in rec['files'].items():check(p.parent/f,h);proof[str(p.parent/f)]=h
    proof[str(p)]=sha(p)
   print('verified',line,phase,len(jobs),flush=True)
 for phase,n in [('A',864),('B',108)]:
  path=O/(phase+'_ACCEPTANCE.json');rec=read(path)
  jobs={k:j for k,j in alljobs.items() if (j['kind']=='search')==(phase=='B')}
  assert len(jobs)==rec['task_count']==n and rec['status']=='PASS_common6_'+phase+'_only'
  assert rec['code_lock_sha256']==sha(O/'CODE_LOCK.json')
  refs=resolved_refs(rec,jobs);new=dict(rec,receipts=refs);changes.append((path,rec,new))
  summary[phase]={'tasks':n,'complete':len(refs)}
  pilot=O/(phase+'_PILOT_ACCEPTANCE.json');pr=read(pilot)
  changes.append((pilot,pr,dict(pr,receipts=resolved_refs(pr,jobs))))
 assert len(alljobs)==972
 for path,old,new in changes:
  backup=C/(path.name+'.before.json')
  if not backup.exists():shutil.copy2(path,backup);assert sha(backup)==sha(path)
  if old!=new:save(path,new)
  proof[str(path)]=sha(path)
 for f in ['CODE_LOCK.json','DEPENDENCIES.json','QUALIFICATION.json']:proof[str(O/f)]=sha(O/f)
 for line in ['geography','temporal']:
  for p in (O/'inputs'/line).iterdir():
   if p.is_file():proof[str(p)]=sha(p)
 report={'status':'PASS_current_source_common6_972_checkpoints_verified_without_refitting','source_lock_sha256':sha(R/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json'),
  'counts':summary,'files':proof,'numeric_code_changed':False,'fits_rerun':0,'foreign_fits_imported':0,
  'repair':'Resolve task IDs through existing current-run A_TASKS/B_TASKS, preserving target hashes, statuses and all numeric payloads.',
  'scientific_review_performed':False,'public_performance_accessed':False,'time_Asia_Shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
 save(O/'CHECKPOINT_REUSE_SEAL.json',report)
 save(C/'CHECKPOINT_AND_REFERENCE_ACCEPTANCE.json',{k:v for k,v in report.items() if k!='files'}|{'reuse_seal_sha256':sha(O/'CHECKPOINT_REUSE_SEAL.json'),'verified_files':len(proof)})
 print('PASS verified',len(alljobs),'current-source tasks; reference-only repair; no fits',flush=True)
 return report
if __name__=='__main__':verify()
