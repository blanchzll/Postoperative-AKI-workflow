"""Necessary selective-policy upstream searches/fits; immutable E science workers."""
from pathlib import Path
import os,sys,json,time,fcntl,traceback,argparse,datetime
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/selective_upstream_20261023_v1'
HERE=Path(__file__).resolve().parent
E_CODE=ROOT/'work/ensemble_calibration_20261020_v1'
AUDIT=ROOT/'outputs/selective_search_20261023_v1'
os.environ['AKI_SCOPE_OUT']=str(OUT)
os.environ['AKI_SCOPE_INPUT']=str(ROOT/'outputs/v4_development_inputs_20261018')
sys.path.insert(0,str(E_CODE))
import coordinator_e as frozen
from core import atomic,sha,digest,ready,FAMILIES,METHODS,resource_snapshot,process_rss
sys.path.insert(0,str(ROOT/'work/ensemble_speedup_20261021_v2'))
from resource_policy import CPUShare,reservation,historical_envelopes,key,G,M
RUNNING={}
def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def routes(cohort):
 return [('preop','full'),('preop','compact')] if cohort=='preop' else [('preop','full'),('endop','full')]
def check_sources():
 assert read(AUDIT/'CODE_LOCK.json')=={p.name:sha(p) for p in HERE.glob('*.py')},'Coordinator source changed'
 assert read(OUT/'E_FROZEN_CODE_LOCK.json')=={p.name:sha(p) for p in E_CODE.glob('*.py')},'E source changed'
 assert read(OUT/'CODE_LOCK.json')=={p.name:sha(p) for p in (ROOT/'work/selective_upstream_20261023_v1').glob('*.py')},'Prepared-scope source changed'
 q=read(AUDIT/'FUNCTIONAL_QUALIFICATION.json')
 assert q['scope_manifest_sha256']==sha(OUT/'SCOPE_MANIFEST.json')
 assert q['source_input_lock_sha256']==sha(ROOT/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json')
def build(man):
 by={s['id']:s for s in man['scopes']}
 roots=sorted(man['roots'],key=lambda r:(r['line'],r['cohort'],r['id']))
 order={r['id']:i for i,r in enumerate(roots)}
 new=[s for s in man['scopes'] if s['level']==4]
 jobs=[];bank={}
 for s in new:
  rid=by[s['parent']]['parent'];idx=order[rid];sid=s['id']
  for stage,branch in routes(s['cohort']):
   for family in FAMILIES:
    t={'line':s['line'],'cohort':s['cohort'],'stage':stage,'branch':branch,'context':'outer_0','family':family,'scope_id':sid,'scope_sha256':sha(OUT/'scopes'/sid/'SCOPE.json')}
    h=digest(t)[:24]
    base=Path(s['output'])
    search=base/'search_v2'/s['line']/s['cohort']/stage/branch/'outer_0'/family
    fit=base/'outer_raw'/s['line']/s['cohort']/stage/branch/'outer_0'/family
    priority=[idx,1,stage,branch,FAMILIES.index(family),sid]
    jobs.append({'id':'search_'+h,'kind':'search','scope':s,'task':t,'dest':str(search/'COMPLETE.json'),'depends':[],'priority':priority})
    jobs.append({'id':'fit_'+h,'kind':'fit','scope':s,'task':t,'dest':str(fit/'COMPLETE.json'),'depends':['search_'+h],'priority':[idx,0,stage,branch,FAMILIES.index(family),sid]})
    bank[(sid,stage,branch,family)]='fit_'+h
 for r in roots:
  ss=[s for s in new if by[s['parent']]['parent']==r['id']]
  assert len(ss)==9
  for stage,branch in routes(r['cohort']):
   deps=[bank[(s['id'],stage,branch,f)] for s in ss for f in FAMILIES]
   jid='ensemble_'+digest({'root':r['id'],'stage':stage,'branch':branch})[:24]
   dest=OUT/'ensemble_results'/r['line']/r['cohort']/stage/branch/r['id']
   jobs.append({'id':jid,'kind':'ensemble','task':{'root':r,'stage':stage,'branch':branch},'dest':str(dest/'COMPLETE.json'),'depends':deps,'priority':[order[r['id']],-1,stage,branch]})
 assert len(new)==324 and len(roots)==36 and len(jobs)==11736
 assert len({j['id'] for j in jobs})==len(jobs)
 return sorted(jobs,key=lambda j:j['priority'])
def verify_coarse(man,deep=False):
 rec=read(OUT/'STAGE_F_COARSE_ACCEPTANCE.json')
 assert rec['status']=='PASS_new_scope_coarse_selection_only' and rec['completed']==324
 assert rec['scope_manifest_sha256']==sha(OUT/'SCOPE_MANIFEST.json')
 assert rec['task_owned_workers_remaining']==0
 for p,h in rec['receipts_sha256'].items():assert sha(p)==h
 count=0
 for s in man['scopes']:
  if s['level']!=4:continue
  sig=sha(OUT/'scopes'/s['id']/'SCOPE.json')
  r=read(Path(s['output'])/'COARSE_ACCEPTANCE.json')
  assert r['source_scope_sha256']==sig and r['status']=='PASS'
  mp=Path(s['output'])/'selection'/s['line']/s['cohort']/'outer_0/METHOD_SELECTION.json'
  assert read(mp)['selected_method']==r['selected_method']
  if deep:
   for method in METHODS:
    for family in FAMILIES:
     dest=Path(s['output'])/'coarse'/s['line']/s['cohort']/'outer_0'/method/family
     assert ready(dest)
     q=read(dest/'COMPLETE.json')['signature']['task']
     assert q['scope_id']==s['id'] and q['scope_sha256']==sig
     count+=1
 return count
def qualify():
 man=read(OUT/'SCOPE_MANIFEST.json');jobs=build(man)
 n=verify_coarse(man,deep=True)
 assert n==17496
 ids={j['id'] for j in jobs}
 assert all(set(j['depends'])<=ids for j in jobs)
 assert sum(j['kind']=='search' for j in jobs)==5832
 assert sum(j['kind']=='fit' for j in jobs)==5832
 assert sum(j['kind']=='ensemble' for j in jobs)==72
 assert all(j.get('task',{}).get('branch')!='selected' for j in jobs)
 # Confirm inherited exact task signatures and checked original runtime for each new input scope.
 scopes={s['id']:s for s in man['scopes']}
 for j in jobs:
  if j['kind']=='ensemble':
   root=j['task']['root']
   assert len(j['depends'])==81
   assert len(root['children'])==3
   # All pre-existing third-level route/family fits required by this root are accepted.
   sources=[scopes[k] for k in root['children']]
   root_source={'output':root['base_output'],'line':root['line'],'cohort':root['cohort']}
   for z in sources+[root_source]:
    for f in FAMILIES:
     d=Path(z['output'])/'outer_raw'/z['line']/z['cohort']/j['task']['stage']/j['task']['branch']/'outer_0'/f
     assert ready(d),str(d)
  else:
   assert j['task']['context']=='outer_0'
   assert j['task']['scope_sha256']==sha(OUT/'scopes'/j['scope']['id']/'SCOPE.json')
   assert j['dest'].startswith(j['scope']['output']+'/')
 # Metadata verifies actual groups; no final hospital/time cases may enter any new fit.
 import pandas as pd
 metas={c:pd.read_csv(ROOT/'outputs/v4_development_inputs_20261018'/(c+'_metadata.csv'),dtype={'group_hash':str}).set_index('excel_row') for c in ['preop','paired']}
 for s in scopes.values():
  m=metas[s['cohort']];tr=m.loc[s['train_rows']];va=m.loc[s['assessment_rows']]
  assert set(tr.group_hash).isdisjoint(va.group_hash)
  assert tr[s['line']+'_partition_v2'].eq('development').all() and va[s['line']+'_partition_v2'].eq('development').all()
  if s['level']==4:
   parent=scopes[s['parent']]
   forbidden=set(m.loc[parent['assessment_rows'],'group_hash'])
   assert forbidden.isdisjoint(tr.group_hash) and forbidden.isdisjoint(va.group_hash)
 compact=[j for j in jobs if j.get('task',{}).get('branch')=='compact']
 assert compact and all(j['task']['stage']=='preop' and j['task'].get('cohort',j['task'].get('root',{}).get('cohort'))=='preop' for j in compact)
 # Immutable workers use separate compact imputers; selected routes and external data are absent.
 rec={'status':'PASS_dependency_and_source_qualification_not_new_model_fit','time_UTC':stamp(),'scope_manifest_sha256':sha(OUT/'SCOPE_MANIFEST.json'),'source_input_lock_sha256':sha(ROOT/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json'),'coarse_receipts_verified':324,'coarse_family_result_files_verified':n,'tasks':11736,'searches':5832,'fits':5832,'ensemble_tasks':72,'no_selected_route_reopened':True,'development_only_real_groups':True,'existing_third_and_second_dependencies_accepted':True,'frozen_science_workers_reused':True,'pilot_first_full_search_required':True,'policy_locked':False,'final_scores_accessed':False}
 atomic(AUDIT/'FUNCTIONAL_QUALIFICATION.json',rec)
 atomic(AUDIT/'JOB_MANIFEST.json',{'total':len(jobs),'jobs':[{k:v for k,v in j.items() if k not in ['scope','priority']}|({'scope_id':j['scope']['id']} if 'scope' in j else {}) for j in jobs]})
 print(json.dumps({k:v for k,v in rec.items() if not isinstance(v,(dict,list))}),flush=True)
def main():
 guards=[]
 for fn in [OUT/'COARSE_COORDINATOR.lock',AUDIT/'SEARCH_COORDINATOR.lock']:
  f=fn.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.append(f)
 check_sources()
 assert read(AUDIT/'FUNCTIONAL_QUALIFICATION.json')['status']=='PASS_dependency_and_source_qualification_not_new_model_fit'
 man=read(OUT/'SCOPE_MANIFEST.json');verify_coarse(man)
 jobs=build(man)
 for d in ['logs','requests']:(OUT/d).mkdir(exist_ok=True)
 done={j['id'] for j in jobs if frozen.good(j)}
 atomic(AUDIT/'REUSED_AT_LAUNCH.json',{'verified_existing_tasks':len(done),'existing_completed_results_not_refitted':True,'completed_ids_sha256':digest(sorted(done))})
 atomic(AUDIT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':stamp(),'code_lock_sha256':sha(AUDIT/'CODE_LOCK.json'),'tasks':len(jobs),'original_runtime':str(frozen.PYTHON),'search_runtime':str(frozen.SHARED)})
 peaks={} # New-source run: start with conservative reservations, then learn actual RSS.
 cpu=CPUShare();time.sleep(2);failed=[];last=0
 pilot=next(j['id'] for j in jobs if j['kind']=='search')
 while len(done)<len(jobs):
  for jid,v in list(RUNNING.items()):
   p=v['process']
   if p.poll() is None:continue
   v['log'].close();j=v['job']
   if p.returncode==0 and frozen.good(j):
    done.add(jid);r=read(j['dest'])
    peak=max(r.get('peak_model_rss_bytes',r.get('peak_rss_bytes',0)),v.get('observed_rss',0))
    peaks[key(j)]=max(peaks.get(key(j),0),peak)
    if jid==pilot:atomic(AUDIT/'PILOT_SEARCH_ACCEPTANCE.json',{'status':'PASS_first_original_runtime_full_search','task_id':jid,'receipt_sha256':sha(j['dest']),'time_UTC':stamp(),'complete_search_only':True})
   else:failed.append({'id':jid,'exitcode':p.returncode,'log':str(v['logpath'])})
   del RUNNING[jid]
  check_sources();snap=resource_snapshot()
  share=cpu.sample(snap.get('affinity_and_quota_cpu',snap['effective_cpu']),[v['process'].pid for v in RUNNING.values()])
  rss={jid:process_rss(v['process'].pid) for jid,v in RUNNING.items()}
  for jid,v in RUNNING.items():
   v['observed_rss']=max(v.get('observed_rss',0),rss[jid])
   v['reservation']=max(v['reservation'],int(rss[jid]*1.6+64*M))
  growth=sum(max(0,v['reservation']-rss[jid]) for jid,v in RUNNING.items())
  budget=snap['available_bytes']-max(4*G,.25*snap['available_bytes'])-growth
  slots=max(0,share-len(RUNNING))
  stat=os.statvfs(OUT);free=stat.f_bavail*stat.f_frsize
  if not failed and free>60*G:
   for j in jobs:
    jid=j['id']
    if pilot not in done and jid!=pilot:continue
    if jid in done or jid in RUNNING or not set(j['depends'])<=done:continue
    need=reservation(j,peaks)
    if budget<need or slots<1:continue
    v=frozen.launch(j,1);v['reservation']=need;RUNNING[jid]=v;budget-=need;slots-=1
  if time.time()-last>=15 or failed:
   counts={k:{'complete':sum(j['id'] in done and j['kind']==k for j in jobs),'total':sum(j['kind']==k for j in jobs)} for k in ['search','fit','ensemble']}
   state='failed_draining_owned_workers' if failed else ('running_necessary_upstream_search_fit' if RUNNING else ('waiting_disk' if free<=60*G else 'waiting_memory'))
   atomic(AUDIT/'status.json',{'state':state,'pid':os.getpid(),'time_UTC':stamp(),'complete':len(done),'total':len(jobs),'counts':counts,'running':[{'id':jid,'pid':v['process'].pid,'reservation_bytes':v['reservation']} for jid,v in RUNNING.items()],'failed':failed,'resources':snap,'own_CPU_budget':share,'other_CPU_cores':cpu.other_cores,'growth_reserved_bytes':growth,'disk_free_bytes':free,'pilot_accepted':pilot in done,'policy_locked':False,'final_scores_accessed':False,'entire_research_complete':False})
   last=time.time()
  if failed and not RUNNING:raise RuntimeError('Task failed; drained owned workers, checkpoints retained. Diagnose before retry.')
  time.sleep(5)
 check_sources()
 receipts={j['dest']:sha(j['dest']) for j in jobs}
 atomic(AUDIT/'STAGE_F_SEARCH_FIT_ACCEPTANCE.json',{'status':'PASS_necessary_upstream_search_fit_ensemble_only','tasks':len(jobs),'receipts_sha256':receipts,'source_manifest_sha256':sha(OUT/'SCOPE_MANIFEST.json'),'task_owned_workers_remaining':0,'policy_locked':False,'next_required':'first-child constituent calibration, honest gate predictions, whole-policy calibration and two policy locks before final scores','final_scores_accessed':False,'entire_research_complete':False})
 atomic(AUDIT/'status.json',{'state':'complete_upstream_search_fit_ensemble_only','complete':len(done),'total':len(jobs),'task_owned_workers_remaining':0,'policy_locked':False,'final_scores_accessed':False,'entire_research_complete':False})
 for f in guards:f.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--qualify',action='store_true');a=ap.parse_args()
 try:
  if a.qualify:qualify()
  else:main()
 except Exception:
  for v in list(RUNNING.values()):v['process'].wait();v['log'].close()
  atomic(AUDIT/'FAILURE.json',{'time_UTC':stamp(),'phase':'qualification' if a.qualify else 'dispatch','traceback':traceback.format_exc()});raise
