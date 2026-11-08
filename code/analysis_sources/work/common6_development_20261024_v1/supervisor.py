from pathlib import Path
import os,sys,json,hashlib,datetime,fcntl,time,subprocess,traceback,shutil
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/common6_development_20261024_v1'
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
SHARED=Path('/home/lei/codex-workspaces/shared-python-envs/envs/clinical-ml-py312-cpu-xgb214-20261016-v1/bin/python')
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
OWNED=[]
def envfor(line):
 env=os.environ.copy();env.update(AKI_SCOPE_INPUT=str(OUT/'inputs'/line),AKI_SCOPE_OUT=str(OUT/'runs'/line),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 return env
def rss(pid):
 total=0
 try:
  total=int(next(s.split()[1] for s in Path('/proc',str(pid),'status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
  for c in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():total+=rss(int(c))
 except (OSError,StopIteration):pass
 return total
def ready(j):
 p=Path(j['dest'])/'COMPLETE.json'
 if not p.exists():return False
 rec=base.read(p)
 for f,h in rec.get('files',{}).items():assert base.sha(p.parent/f)==h
 return True
def check_code():
 for n,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/n)==h
 for p,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(p)==h
def phase(jobs,label):
 done={j['id'] for j in jobs if ready(j)};live={};failed=[];peak={};budget={};started=time.time()
 # First original clinical task chain / first full 48+48 search, retained as valid work.
 if label=='A':
  pilot=[j for j in jobs if j['line']=='geography' and j['task']['context']=='final' and j['task']['method']=='bayes' and (j['kind']=='impute' or j['task'].get('family')=='LR')]
 else:pilot=[j for j in jobs if j['line']=='geography' and j['task']['context']=='final' and j['task']['family']=='LR']
 pilot_ids={j['id'] for j in pilot};pilot_ok=pilot_ids<=done
 req=OUT/'requests';req.mkdir(exist_ok=True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
 for j in jobs:
  path=req/(j['id']+'.json')
  if path.exists():assert base.read(path)==j['task']
  else:base.atomic(path,j['task'])
 while len(done)<len(jobs):
  for k,(p,log,j) in list(live.items()):
   peak[k]=max(peak.get(k,0),rss(p.pid));rc=p.poll()
   if rc is not None:
    log.close();del live[k]
    if rc or not ready(j):failed.append({'id':k,'returncode':rc})
    else:
     done.add(k);r=base.read(Path(j['dest'])/'COMPLETE.json');kind=(j['kind'],j['task'].get('family',j['task'].get('method')))
     measured=max(peak[k],r.get('peak_rss_bytes',0),r.get('peak_model_rss_bytes',0))
     budget[kind]=max(budget.get(kind,0),768*1024**2,2*measured)
  if not pilot_ok and pilot_ids<=done:
   pilot_ok=True;base.atomic(OUT/(label+'_PILOT_ACCEPTANCE.json'),{'status':'PASS_original_clinical_pilot_reusable','phase':label,'receipts':{j['id']:base.sha(Path(j['dest'])/'COMPLETE.json') for j in pilot},'time_UTC':base.stamp()})
  if failed and not live:raise RuntimeError('Common6 task failure, no automatic retry '+str(failed))
  check_code();available=base.mem();head=max(4*1024**3,int(available*.25))
  def reserve(j):return budget.get((j['kind'],j['task'].get('family',j['task'].get('method'))),2*1024**3)
  growth=sum(max(0,reserve(j)-rss(p.pid)) for p,l,j in live.values())
  candidates=[j for j in jobs if j['id'] not in done and j['id'] not in live and set(j['depends'])<=done and (pilot_ok or j['id'] in pilot_ids)]
  candidates.sort(key=lambda j:(j['kind']=='impute',j['task']['context']!='final',j['id']))
  for j in candidates:
   if failed or len(live)>=base.effective_cpu():break
   if available-head-growth<reserve(j):continue
   if shutil.disk_usage(OUT).free<100*1024**3:break
   log=(logs/(j['id']+'.log')).open('ab');python=SHARED if label=='B' else PYTHON
   p=subprocess.Popen([str(python),'-B',str(CODE/'engine'/j['worker']),str(req/(j['id']+'.json')),'--threads','1'],env=envfor(j['line']),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   live[j['id']]=(p,log,j);OWNED.append((p,log));peak[j['id']]=0;growth+=reserve(j)
  base.atomic(OUT/'status.json',{'state':'running_common6_'+label if live else 'waiting_resources','phase':label,'pid':os.getpid(),'time_UTC':base.stamp(),'completed':len(done),'total':len(jobs),'running':{k:v[0].pid for k,v in live.items()},'failures':failed,'pilot_accepted':pilot_ok,'available_bytes':available,'growth_reserved_bytes':growth,'model_development_complete':False,'public_performance_accessed':False})
  if len(done)<len(jobs):time.sleep(15)
 base.atomic(OUT/(label+'_ACCEPTANCE.json'),{'status':'PASS_common6_'+label+'_only','task_count':len(jobs),'receipts':{j['id']:base.sha(Path(j['dest'])/'COMPLETE.json') for j in jobs},'seconds':time.time()-started,'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'not_unbiased_internal_validation':True,'public_performance_accessed':False,'entire_research_complete':False})
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check_code()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_isolated_six_input_engine_and_boundaries'
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 S=ROOT/'outputs/conditional_stability_20261024_v1'
 while not (S/'STAGE_STABILITY_ACCEPTANCE.json').exists():
  if (S/'FAILURE.json').exists():raise RuntimeError('Stability failed; diagnose ownership/resource status before starting common6')
  base.atomic(OUT/'status.json',{'state':'waiting_stability_completion_resource_release','pid':os.getpid(),'time_UTC':base.stamp(),'owned_workers':0,'public_performance_accessed':False})
  time.sleep(30)
 assert base.read(S/'STAGE_STABILITY_ACCEPTANCE.json')['status']=='PASS_200_planned_conditional_bootstraps_per_10_routes'
 prior=(S/'SUPERVISOR.lock').open('a')
 while True:
  try:fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 jobs=sum([base.read(OUT/'runs'/line/'A_TASKS.json') for line in ['geography','temporal']],[]);assert len(jobs)==864;phase(jobs,'A')
 for line in ['geography','temporal']:
  with (OUT/(line+'_selection.log')).open('ab') as log:
   subprocess.run([str(PYTHON),'-B',str(CODE/'select_and_build_search.py'),line],env=envfor(line),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 jobs=sum([base.read(OUT/'runs'/line/'B_TASKS.json') for line in ['geography','temporal']],[]);assert len(jobs)==108;phase(jobs,'B')
 base.atomic(OUT/'status.json',{'state':'complete_common6_coarse_and_search_only','time_UTC':base.stamp(),'owned_workers':0,'next_required':'Honest selected candidate/ensemble calibration, locked model packages, then qualified public scoring','model_locked':False,'public_performance_accessed':False,'entire_research_complete':False})
 prior.close();guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'owned_running_pids':[p.pid for p,l in OWNED if p.poll() is None]})
  for p,l in OWNED:
   p.wait()
   if not l.closed:l.close()
  raise
