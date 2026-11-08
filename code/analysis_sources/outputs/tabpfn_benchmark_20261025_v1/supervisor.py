from pathlib import Path
import os,sys,subprocess,time,json,signal,fcntl,hashlib,datetime
OUT=Path(__file__).resolve().parent
PY='/home/lei/codex-workspaces/shared-python-envs/envs/tabular-foundation-py312-torch260-cu118-tabpfn221-v1/bin/python'
routes=['geography|paired|preop|full','geography|paired|endop|full','temporal|paired|preop|full','temporal|paired|endop|full','geography|preop|preop|compact','geography|preop|preop|full','temporal|preop|preop|compact','temporal|preop|preop|full','temporal|paired|preop_noALT|full','temporal|paired|endop_noALT|full']
def atomic(name,obj):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def mem():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
lock=(OUT/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(OUT/'logs').mkdir(exist_ok=True)
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','TABPFN_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_OFFLINE':'1','OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'8'})
done=[]
for route in routes:
 name=route.replace('|','_')
 if mem()<4*2**30:
  atomic('STATUS.json',{'state':'resource_hold','completed':done,'next':route,'mem_available':mem()});sys.exit(3)
 atomic('STATUS.json',{'state':'running','supervisor_pid':os.getpid(),'route':route,'completed':done,'total':len(routes)})
 with (OUT/'logs'/(name+'.log')).open('a') as log:
  child=subprocess.Popen([PY,str(OUT/'worker.py'),route],env=env,stdout=log,stderr=log,start_new_session=True)
  stopped=False
  while child.poll() is None:
   try:rss=int(next(x.split()[1] for x in Path(f'/proc/{child.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
   except (OSError,StopIteration):rss=0
   available=mem()
   atomic('RESOURCE.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'RSS_bytes':rss,'MemAvailable_bytes':available,'route':route})
   if available<2.5*2**30 or rss>4.5*2**30:
    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);stopped=True;break
   time.sleep(3)
 if child.returncode or stopped:
  atomic('STATUS.json',{'state':'resource_hold' if stopped else 'failed','completed':done,'route':route,'returncode':child.returncode});sys.exit(3 if stopped else 1)
 done.append(route);print('DONE',len(done),len(routes),name,flush=True)
atomic('STATUS.json',{'state':'predictions_complete_pending_statistics','completed':done,'total':len(routes)})
