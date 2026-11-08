from curves import *
import subprocess,fcntl,traceback
OWNED=[]
def rss(pid):
 try:
  v=int(next(s.split()[1] for s in Path('/proc',str(pid),'status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
  for child in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():v+=rss(int(child))
  return v
 except (OSError,StopIteration):return 0
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_preparation_and_synthetic_curve_math'
 base.gate()
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],
  'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 jobs=list(base.read(OUT/'ROUTES.json'));pilot='geography_preop_preop_full'
 done=set();live={};failed=[];reserve=2*1024**3;peaks={}
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 (OUT/'logs').mkdir(exist_ok=True)
 def completed(j):
  p=OUT/'routes'/j/'COMPLETE.json'
  if not p.exists():return False
  for f,h in base.read(p)['files'].items():assert base.sha(p.parent/f)==h
  return True
 done={j for j in jobs if completed(j)}
 while len(done)<len(jobs):
  for j,(p,h) in list(live.items()):
   peaks[j]=max(peaks.get(j,0),rss(p.pid))
   if p.poll() is None:continue
   h.close();del live[j]
   if p.returncode or not completed(j):failed.append({'route':j,'returncode':p.returncode})
   else:done.add(j);reserve=max(reserve,2*peaks[j])
  if failed and not live:raise RuntimeError(str(failed))
  check();avail=base.mem();head=max(4*1024**3,int(.25*avail))
  growth=sum(max(0,reserve-rss(p.pid)) for p,h in live.values())
  other_growth=0
  for stage in ['common6_strict_continuation_20261024_v1','common6_internal_assembly_20261024_v1']:
   status=ROOT/'outputs'/stage/'status.json';launch=ROOT/'outputs'/stage/'LAUNCH.json'
   if not status.exists() or not launch.exists():continue
   d=base.read(status);owner=base.read(launch);proc=Path('/proc')/str(owner['pid'])/'stat'
   if proc.exists() and proc.read_text().split()[21]==str(owner['start_ticks']):
    other_growth+=int(d.get('growth_reserved_bytes',0))
  growth+=other_growth
  for j in jobs:
   if failed or len(live)>=len(os.sched_getaffinity(0)):break
   if j in done or j in live or (pilot not in done and j!=pilot):continue
   if avail-head-growth<reserve:continue
   h=(OUT/'logs'/(j+'.log')).open('ab')
   p=subprocess.Popen([str(base.PYTHON),'-B',str(CODE/'curves.py'),j],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
   live[j]=(p,h);OWNED.append((p,h));growth+=reserve
  base.atomic(OUT/'status.json',{'state':'running_locked_risk_curves','completed':len(done),'total':len(jobs),
   'running':{j:p.pid for j,(p,h) in live.items()},'failures':failed,'pid':os.getpid(),'time_UTC':base.stamp(),
   'available_bytes':avail,'reservation_bytes_per_worker':reserve,'other_growth_reserved_bytes':other_growth})
  if len(done)<len(jobs):time.sleep(15)
 base.atomic(OUT/'STAGE_CURVE_ACCEPTANCE.json',{'status':'PASS_ten_locked_model_development_risk_curve_routes',
  'receipts':{str(OUT/'routes'/j/'COMPLETE.json'):base.sha(OUT/'routes'/j/'COMPLETE.json') for j in jobs},
  'SHAP_completed':False,'figures_rendered':False,'whole_project_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_locked_risk_curve_tables','completed':len(done),'total':len(jobs),'owned_workers':0,'time_UTC':base.stamp()})
 guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc()})
  for p,h in OWNED:
   p.wait()
   if not h.closed:h.close()
  raise
