"""Memory-bounded finite refit queue. Active main workflows are never modified."""
from fit import *
import subprocess,fcntl,time,traceback
OWNED=[]
def rss(pid):
 try:
  v=int(next(s.split()[1] for s in Path('/proc',str(pid),'status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
  for child in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():v+=rss(int(child))
  return v
 except (OSError,StopIteration):return 0
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_fixed_flow_sensitivity_preparation'
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 routes=base.read(OUT/'ROUTES.json')
 jobs=[(r,s) for s in SCENARIOS for r in routes];done=set();live={};failed=[];peaks={};reserve=3*1024**3
 def completed(j):
  d=OUT/'routes'/j[0]/j[1]/'COMPLETE.json'
  if not d.exists():return False
  rec=base.read(d);assert base.sha(Path(rec['fit_cache'])/'COMPLETE.json')==rec['fit_receipt_sha256']
  return True
 done={j for j in jobs if completed(j)}
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 pilot=('geography_paired_endop_full','matched_control')
 (OUT/'logs').mkdir(exist_ok=True)
 while len(done)<len(jobs):
  for j,(p,h) in list(live.items()):
   peaks[j]=max(peaks.get(j,0),rss(p.pid))
   if p.poll() is None:continue
   h.close();del live[j]
   if p.returncode or not completed(j):failed.append({'route':j[0],'scenario':j[1],'returncode':p.returncode})
   else:done.add(j);reserve=max(reserve,2*peaks[j])
  if failed and not live:raise RuntimeError('Failed jobs retained; diagnose '+str(failed))
  check();avail=base.mem();head=max(4*1024**3,int(avail*.25));growth=sum(max(0,reserve-rss(p.pid)) for p,h in live.values())
  for j in jobs:
   if failed or len(live)>=len(os.sched_getaffinity(0)):break
   if j in done or j in live or (pilot not in done and j!=pilot):continue
   if avail-head-growth<reserve:continue
   h=(OUT/'logs'/('_'.join(j)+'.log')).open('ab')
   p=subprocess.Popen([str(base.PYTHON),'-B',str(CODE/'fit.py'),*j],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
   live[j]=(p,h);OWNED.append((p,h));growth+=reserve
  base.atomic(OUT/'status.json',{'state':'running_fixed_flow_sensitivity_fits','completed':len(done),'total':len(jobs),
   'running':{'/'.join(j):p.pid for j,(p,h) in live.items()},'failures':failed,'time_UTC':base.stamp(),'pid':os.getpid(),'reservation_bytes_per_worker':reserve,'available_bytes':avail})
  if len(done)<len(jobs):time.sleep(15)
 receipts={str(OUT/'routes'/r/s/'COMPLETE.json'):base.sha(OUT/'routes'/r/s/'COMPLETE.json') for r,s in jobs}
 base.atomic(OUT/'STAGE_REFIT_ACCEPTANCE.json',{'status':'PASS_thirty_fixed_flow_scenario_routes_fitted','receipts':receipts,
  'unique_fit_caches':len({base.read(p)['fit_cache'] for p in receipts}),'evaluation_complete':False,'whole_project_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_fixed_flow_fits_evaluation_pending','completed':30,'total':30,'owned_workers':0,'time_UTC':base.stamp()})
 guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc()})
  for p,h in OWNED:
   p.wait()
   if not h.closed:h.close()
  raise
