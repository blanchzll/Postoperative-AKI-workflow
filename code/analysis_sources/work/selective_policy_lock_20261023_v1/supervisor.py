"""Finite wait-and-run owner for the accepted selective policy assembly stage."""
from pathlib import Path
import os,json,time,fcntl,subprocess,traceback,datetime,hashlib
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/selective_policy_lock_20261023_v1'
SEARCH=ROOT/'outputs/selective_search_20261023_v1'
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def mem():
 n=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 for row in Path('/proc/self/cgroup').read_text().splitlines():
  _,ctrl,rel=row.split(':',2)
  if not ctrl:
   base=Path('/sys/fs/cgroup')/rel.lstrip('/')
   for p in [base,*base.parents]:
    lim=p/'memory.max';used=p/'memory.current'
    if lim.exists() and used.exists() and lim.read_text().strip()!='max':n=min(n,max(0,int(lim.read_text())-int(used.read_text())))
 return n
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 expected=read(OUT/'CODE_LOCK.json')
 assert read(OUT/'FUNCTIONAL_QUALIFICATION.json')['source_code_lock_sha256']==sha(OUT/'CODE_LOCK.json')
 atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':stamp(),'source_code_lock_sha256':sha(OUT/'CODE_LOCK.json')})
 while not (SEARCH/'STAGE_F_SEARCH_FIT_ACCEPTANCE.json').exists():
  assert expected=={p.name:sha(p) for p in CODE.glob('*.py')}
  state='waiting_existing_search_fit_batch'
  if (SEARCH/'FAILURE.json').exists():raise RuntimeError('Upstream batch failed; do not duplicate or bypass its owner.')
  atomic(OUT/'status.json',{'state':state,'pid':os.getpid(),'time_UTC':stamp(),'owned_fit_workers':0,'final_scores_accessed':False,'policy_locked':False})
  time.sleep(30)
 assert read(SEARCH/'STAGE_F_SEARCH_FIT_ACCEPTANCE.json')['status']=='PASS_necessary_upstream_search_fit_ensemble_only'
 prior=(SEARCH/'SEARCH_COORDINATOR.lock').open('a')
 while True:
  try:fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 while mem()<5*1024**3:
  atomic(OUT/'status.json',{'state':'waiting_memory_after_upstream_exit','pid':os.getpid(),'time_UTC':stamp(),'owned_fit_workers':0,'policy_locked':False});time.sleep(30)
 assert expected=={p.name:sha(p) for p in CODE.glob('*.py')}
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for name in ['build_policy.py','qualify_clinical.py']:
  if name=='build_policy.py' and (OUT/'POLICY_FITTING_ACCEPTANCE.json').exists():continue
  if name=='qualify_clinical.py' and (OUT/'STAGE_POLICY_LOCK_ACCEPTANCE.json').exists():continue
  log=(OUT/(name+'.log')).open('ab')
  p=subprocess.Popen([str(PYTHON),'-B',str(CODE/name)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  atomic(OUT/'status.json',{'state':name,'pid':os.getpid(),'worker_pid':p.pid,'time_UTC':stamp(),'policy_locked':False,'final_scores_accessed':False})
  rc=p.wait();log.close()
  if rc:raise RuntimeError(name+' failed; logs/checkpoints retained, no automatic unchanged retry')
  assert expected=={p.name:sha(p) for p in CODE.glob('*.py')}
 assert read(OUT/'STAGE_POLICY_LOCK_ACCEPTANCE.json')['status']=='PASS_two_selective_strategies_and_prespecified_variants_locked'
 atomic(OUT/'status.json',{'state':'complete_policy_locks_final_evaluation_pending','time_UTC':stamp(),'task_owned_workers_remaining':0,'policy_locked':True,'final_scores_accessed':False,'entire_research_complete':False})
 prior.close();guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  atomic(OUT/'FAILURE.json',{'time_UTC':stamp(),'traceback':traceback.format_exc()});raise
