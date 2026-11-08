"""Finite launcher: qualify with RAM admission, then queue model locking behind accepted E."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,sys,json,hashlib,time,fcntl,subprocess,traceback
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/model_lock_20261022_v1'
E=ROOT/'outputs/ensemble_calibration_20261020_v1';PYTHON=ROOT/'envs/prediction_gpu/bin/python'
sys.path.insert(0,str(ROOT/'work/ensemble_calibration_20261020_v1'))
from core import resource_snapshot
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
 q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def main():
 guard=(OUT/'PREPARE_LAUNCH.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'LAUNCH.json').exists():
  old=read(OUT/'LAUNCH.json')
  try:live=Path('/proc')/str(old['pid'])/'cmdline';active=b'work/model_lock_20261022_v1/lock_models.py' in live.read_bytes()
  except FileNotFoundError:active=False
  if active:raise RuntimeError('Existing model-lock supervisor owns continuation; do not duplicate')
 lock={p.name:sha(p) for p in CODE.glob('*.py')}
 if (OUT/'CODE_LOCK.json').exists():assert read(OUT/'CODE_LOCK.json')==lock
 else:atomic(OUT/'CODE_LOCK.json',lock)
 atomic(OUT/'PREPARATION_LAUNCH.json',{'pid':os.getpid(),'time_beijing':stamp(),'code_lock_sha256':sha(OUT/'CODE_LOCK.json')})
 while True:
  snap=resource_snapshot()
  live=read(ROOT/'outputs/ensemble_speedup_20261021_v2/live_resources.json')
  status=read(E/'status_e.json')
  growth=live.get('growth_reserved_bytes',0) if status['state']=='running' else 0
  free=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize
  budget=snap['available_bytes']-max(4*1024**3,snap['available_bytes']*.25)-growth
  if budget>=512*1024**2 and free>60*1024**3:break
  atomic(OUT/'PREPARATION_STATUS.json',{'state':'waiting_resources_for_engineering_qualification',
   'pid':os.getpid(),'time_beijing':stamp(),'available_memory':snap['available_bytes'],
   'other_owned_E_growth_reserved':growth,'qualification_reserve':512*1024**2,
   'E_running_unmodified':True,'clinical_models_fitted':False})
  time.sleep(30)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
   AKI_SCOPE_OUT=str(E),AKI_SCOPE_INPUT=str(ROOT/'outputs/v4_development_inputs_20261018'))
 with (OUT/'QUALIFICATION.log').open('ab') as f:
  p=subprocess.run([str(PYTHON),'-B',str(CODE/'qualify_worker.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('Model-lock engineering qualification failed; do not launch')
 q=read(OUT/'FUNCTIONAL_QUALIFICATION.json')
 assert q['status']=='PASS_model_lock_engineering_only' and q['code_sha256']==lock
 log=(OUT/'MODEL_LOCK_COORDINATOR.log').open('ab')
 proc=subprocess.Popen([str(PYTHON),'-B',str(CODE/'lock_models.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 log.close()
 atomic(OUT/'MODEL_LOCK_DISPATCH.json',{'state':'dispatched_waiting_for_E','pid':proc.pid,
  'time_beijing':stamp(),'code_lock_sha256':sha(OUT/'CODE_LOCK.json'),
  'qualification_sha256':sha(OUT/'FUNCTIONAL_QUALIFICATION.json'),'existing_E_not_restarted':True})
 atomic(OUT/'PREPARATION_STATUS.json',{'state':'qualified_and_model_lock_dispatched','pid':os.getpid(),
  'time_beijing':stamp(),'model_lock_pid':proc.pid,'clinical_models_fitted':False})
if __name__=='__main__':
 try:main()
 except Exception:
  atomic(OUT/'PREPARATION_FAILURE.json',{'time_beijing':stamp(),'traceback':traceback.format_exc()})
  raise
