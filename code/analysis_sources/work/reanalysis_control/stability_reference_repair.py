"""Verify completed current-source stability payloads and correct receipt path references only."""
from pathlib import Path
import collections,csv,datetime,fcntl,hashlib,json,os,shutil,time
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/conditional_stability_20261024_v1'
REPAIR=ROOT/'outputs/reanalysis_control/stability_reference_repair_20261028_v1'
REPAIR.mkdir(mode=0o700,parents=True,exist_ok=True)
def read(p): return json.loads(p.read_text())
cache={}
def sha(p):
 p=Path(p); st=p.stat(); key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
  cache[key]=h.hexdigest()
 return cache[key]
def save(p,v):
 t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:
  json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def check(p,h):
 assert p.is_file(),str(p)
 assert sha(p)==h,('hash mismatch',str(p))
def run():
 owner=(REPAIR/'OWNER.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 native=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(native,fcntl.LOCK_EX|fcntl.LOCK_NB)
 acceptance=OUT/'STAGE_STABILITY_ACCEPTANCE.json'
 old=read(acceptance);old_sha=sha(acceptance)
 assert old['status']=='PASS_200_planned_conditional_bootstraps_per_10_routes'
 assert old['owned_workers_remaining']==0
 routes=read(OUT/'ROUTES.json');assert len(routes)==10
 fit_h=sha(OUT/'FIT_IMPLEMENTATION_CODE_LOCK.json')
 code_h=sha(OUT/'CODE_LOCK.json'); dep_h=sha(OUT/'DEPENDENCIES.json')
 check(OUT/'CODE_LOCK.json',old['code_lock_sha256'])
 check(OUT/'ROUTE_STABILITY_SUMMARY.csv',old['summary_sha256'])
 check(OUT/'CONDITIONAL_STABILITY_REPORT_ZH.md',old['report_sha256'])
 for name,h in read(OUT/'CODE_LOCK.json').items():
  check(ROOT/'work/conditional_stability_20261024_v1'/name,h)
 for name,h in read(OUT/'FIT_IMPLEMENTATION_CODE_LOCK.json').items():
  p=ROOT/'work/model_lock_20261022_v1/runtime.py' if name=='model_runtime.py' else ROOT/'work/conditional_stability_20261024_v1'/name
  check(p,h)
 for path,h in read(OUT/'DEPENDENCIES.json').items():check(Path(path),h)
 for route,ref in routes.items():
  assert str(ref['path']).startswith(str(ROOT)+'/outputs/model_lock_20261022_v1/locked_routes/')
  check(Path(ref['path']),ref['sha256'])
 expected={str(OUT/'replicates'/route/f'{rep:03d}'/'COMPLETE.json') for route in routes for rep in range(200)}
 assert {str(p) for p in (OUT/'replicates').glob('*/*/COMPLETE.json')}==expected
 newrefs={};rows=[];byroute={}
 for route,ref in sorted(routes.items()):
  states=collections.Counter()
  for rep in range(200):
   d=OUT/'replicates'/route/f'{rep:03d}';p=d/'COMPLETE.json';h=sha(p)
   assert old['receipts'].get(route+'/'+str(rep),old['receipts'].get(str(p)))==h,('unmatched prior reference',route,rep)
   c=read(p)
   assert c['signature']=={'route_lock':ref['sha256'],'code_lock':fit_h,'dependencies':dep_h,'replicate':rep,'bootstrap_seed':20261023},str(p)
   for f,fh in c['files'].items():check(d/f,fh)
   if c['status']=='PASS_conditional_fixed_flow_refit':
    assert c['execution_code_lock_sha256']==code_h
    assert set(c['fold_receipts'])=={'0','1','2'}
    child_paths=[(d/f'fold_{f}'/'COMPLETE.json',fh) for f,fh in c['fold_receipts'].items()]
    child_paths.append((d/'full/COMPLETE.json',c['full_receipt_sha256']))
    for child,ch in child_paths:
     check(child,ch)
     for f,fh in read(child)['files'].items():check(child.parent/f,fh)
   else:
    assert c['status']=='INVALID_planned_sample_retained_no_replacement' and c.get('reason')
   states[c['status']]+=1;newrefs[str(p)]=h
   rows.append({'route':route,'replicate':rep,'status':c['status'],'receipt_sha256':h})
  byroute[route]=dict(states)
  print('verified_route',route,dict(states),flush=True)
 summary=list(csv.DictReader((OUT/'ROUTE_STABILITY_SUMMARY.csv').open()))
 assert len(summary)==10
 for row in summary:
  states=byroute[row['route']]
  assert int(row['planned'])==200
  assert int(row['valid'])==states.get('PASS_conditional_fixed_flow_refit',0)
  assert int(row['invalid'])==states.get('INVALID_planned_sample_retained_no_replacement',0)
 assert len(old['receipts'])==len(newrefs)==2000
 backup=REPAIR/'STAGE_STABILITY_ACCEPTANCE.before.json'
 if backup.exists(): assert sha(backup)==old_sha or old['receipts']==newrefs
 else:shutil.copy2(acceptance,backup);check(backup,old_sha)
 new=dict(old);new['receipts']=newrefs
 assert {k:v for k,v in new.items() if k!='receipts'}=={k:v for k,v in old.items() if k!='receipts'}
 rowpath=REPAIR/'VERIFIED_REPLICATES.json';save(rowpath,rows)
 verification={
  'status':'PASS_current_source_payloads_verified_and_logical_references_resolved',
  'time_Asia_Shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  'source_lock_sha256':sha(ROOT/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json'),
  'planned_replicates':2000,'routes':10,'replicates_by_status':byroute,
  'verified_unique_files':len(cache),'prior_acceptance_digest':sha(backup),
  'preserved_prior_copy':str(backup),'verified_replicates_digest':sha(rowpath),
  'code_lock_sha256':code_h,'fit_implementation_lock_sha256':fit_h,'dependencies_sha256':dep_h,
  'numeric_code_changed':False,'fits_rerun':0,'model_prediction_payloads_changed':False,
  'pass_status_string_changed':False,'scientific_review_performed':False,
  'change':'Map each route/replicate logical key to the verified existing absolute COMPLETE.json path; preserve all target hashes and other acceptance fields.',
  'residual_boundary':'Conditional fixed-choice development-reference stability; retain invalid draws and existing probe-parity diagnostic limitations; not unbiased external performance or deployment qualification.'
 }
 save(REPAIR/'VERIFICATION_BEFORE_WRITE.json',verification)
 assert sha(acceptance)==old_sha
 save(acceptance,new)
 assert read(acceptance)==new
 verification['repaired_acceptance_sha256']=sha(acceptance)
 save(REPAIR/'REPAIR_ACCEPTANCE.json',verification)
 print(json.dumps({k:verification[k] for k in ['status','planned_replicates','routes','verified_unique_files','fits_rerun','repaired_acceptance_sha256']},ensure_ascii=False))
if __name__=='__main__':run()
