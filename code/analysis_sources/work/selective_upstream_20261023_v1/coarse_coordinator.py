"""Finite task-owned selection batch for newly necessary selective-policy scopes."""
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,hashlib,traceback,datetime
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/selective_upstream_20261023_v1'
CODE=Path(__file__).resolve().parent
E_CODE=ROOT/'work/ensemble_calibration_20261020_v1'
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
os.environ['AKI_SCOPE_OUT']=str(OUT)
os.environ['AKI_SCOPE_INPUT']=str(ROOT/'outputs/v4_development_inputs_20261018')
sys.path.insert(0,str(E_CODE))
from core import atomic,sha,resource_snapshot,process_rss
sys.path.insert(0,str(ROOT/'work/ensemble_speedup_20261021_v2'))
from resource_policy import CPUShare
RUNNING={};GUARD=None
def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 global GUARD
 GUARD=(OUT/'COARSE_COORDINATOR.lock').open('a');fcntl.flock(GUARD,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(OUT/'BOUNDARY_QUALIFICATION.json')['status']=='PASS_real_development_partition_preparation_only'
 manifest=read(OUT/'SCOPE_MANIFEST.json')
 assert sha(OUT/'SCOPE_MANIFEST.json')==read(OUT/'BOUNDARY_QUALIFICATION.json')['scope_manifest_sha256']
 lock=read(OUT/'CODE_LOCK.json')
 def check():
  assert lock=={p.name:sha(p) for p in CODE.glob('*.py')}
  assert read(OUT/'E_FROZEN_CODE_LOCK.json')=={p.name:sha(p) for p in E_CODE.glob('*.py')}
 check()
 tasks={s['id']:s for s in manifest['scopes'] if s['level']==4};assert len(tasks)==324
 def accepted(s):
  p=Path(s['output'])/'COARSE_ACCEPTANCE.json'
  if not p.exists():return False
  r=read(p);assert r['status']=='PASS' and r['source_scope_sha256']==sha(OUT/'scopes'/s['id']/'SCOPE.json')
  return True
 done={k for k,s in tasks.items() if accepted(s)};failed=[];cpu=CPUShare();time.sleep(2)
 (OUT/'logs').mkdir(exist_ok=True)
 atomic(OUT/'LAUNCH_COARSE.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':stamp(),'code_lock_sha256':sha(OUT/'CODE_LOCK.json'),'phase':'necessary_new_scope_six_imputation_nine_base_coarse_selection','tasks':324})
 while len(done)<len(tasks):
  for k,v in list(RUNNING.items()):
   p=v['process']
   if p.poll() is None:continue
   v['log'].close()
   if p.returncode==0 and accepted(tasks[k]):done.add(k)
   else:failed.append({'scope':k,'exitcode':p.returncode})
   del RUNNING[k]
  check();snap=resource_snapshot();share=cpu.sample(snap.get('affinity_and_quota_cpu',snap['effective_cpu']),[v['process'].pid for v in RUNNING.values()])
  for v in RUNNING.values():v['reservation']=max(v['reservation'],int(process_rss(v['process'].pid)*1.6+64*1024**2))
  growth=sum(max(0,v['reservation']-process_rss(v['process'].pid)) for v in RUNNING.values())
  budget=snap['available_bytes']-max(4*1024**3,.25*snap['available_bytes'])-growth
  free=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize
  if not failed and free>60*1024**3:
   for k,s in tasks.items():
    if not done and RUNNING:break
    if k in done or k in RUNNING or len(RUNNING)>=share or budget<1024**3:continue
    env=os.environ.copy();env.update(AKI_SCOPE_OUT=s['output'],AKI_SCOPE_INPUT=s['input'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    log=(OUT/'logs'/(k+'.log')).open('ab')
    p=subprocess.Popen([str(PYTHON),'-B',str(E_CODE/'scope_worker.py'),'coarse',str(OUT/'scopes'/k/'SCOPE.json'),'--threads','1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    RUNNING[k]={'process':p,'log':log,'reservation':1024**3};budget-=1024**3
  atomic(OUT/'status_coarse.json',{'state':'running_new_scope_coarse_selection' if not failed else 'failed_draining_owned_workers','pid':os.getpid(),'time_UTC':stamp(),'complete':len(done),'total':len(tasks),'running':{k:v['process'].pid for k,v in RUNNING.items()},'failed':failed,'available_RAM_bytes':snap['available_bytes'],'CPU_budget':share,'growth_reserved_bytes':growth,'disk_free_bytes':free,'policy_locked':False,'final_scores_accessed':False,'entire_research_complete':False})
  if failed and not RUNNING:raise RuntimeError('Owned coarse selection failed; logs and checkpoints retained.')
  time.sleep(5)
 receipts={str(Path(s['output'])/'COARSE_ACCEPTANCE.json'):sha(Path(s['output'])/'COARSE_ACCEPTANCE.json') for s in tasks.values()}
 atomic(OUT/'STAGE_F_COARSE_ACCEPTANCE.json',{'status':'PASS_new_scope_coarse_selection_only','completed':len(done),'receipts_sha256':receipts,'scope_manifest_sha256':sha(OUT/'SCOPE_MANIFEST.json'),'task_owned_workers_remaining':0,'policy_locked':False,'next_required':'Only needed route searches/fits, second-layer ensembles, child calibration and whole-policy calibration/locks.','final_scores_accessed':False,'entire_research_complete':False})
 atomic(OUT/'status_coarse.json',{'state':'complete_coarse_selection_only','complete':len(done),'total':len(tasks),'task_owned_workers_remaining':0,'policy_locked':False,'entire_research_complete':False})
if __name__=='__main__':
 try:main()
 except Exception:
  for v in list(RUNNING.values()):v['process'].wait();v['log'].close()
  atomic(OUT/'COARSE_FAILURE.json',{'time_UTC':stamp(),'traceback':traceback.format_exc()});raise
