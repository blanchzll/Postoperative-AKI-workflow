"""Finite expanded base-family searches after cohort-correct coarse selection."""
from core import *
import subprocess,traceback
def jobs():
 alljobs=[]
 for line in ["geography","temporal"]:
  for cohort in ["preop","paired"]:
   branches=["full","compact","selected"] if cohort=="preop" else ["full"]
   stages=["preop"] if cohort=="preop" else ["preop","endop"]
   for stage in stages:
    for branch in branches:
     for context in ["final"]+["outer_"+str(i) for i in range(5)]:
      for family in FAMILIES:
       alljobs.append({"line":line,"cohort":cohort,"stage":stage,"branch":branch,"context":context,"family":family})
 return alljobs
def directory(t):return OUT/"search_v2"/t["line"]/t["cohort"]/t["stage"]/t["branch"]/t["context"]/t["family"]
def estimate(t,peaks):
 base={"RF":2,"XGB":1.5,"SVM":1.5}.get(t["family"],1)*1024**3
 if t["branch"]=="selected":base+=.5*1024**3
 return max(base,peaks.get(t["family"],0))
def main():
 lock=(OUT/"COORDINATOR_B.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/"MODEL_ENGINE_FUNCTIONAL_QUALIFICATION.json").read_text())["status"]=="PASS"
 assert json.loads((OUT/"BRIDGE_V2_RESUME_ACCEPTANCE.json").read_text())["status"]=="PASS"
 while not (OUT/"STAGE_A_ACCEPTANCE.json").exists():
  if (OUT/"COORDINATOR_A_FAILURE.json").exists():
   atomic(OUT/"status_b.json",{"state":"blocked_stage_A_failure","not_model_completion":True});return
  pid=json.loads((OUT/"LAUNCH_A.json").read_text())["pid"]
  try:os.kill(pid,0)
  except ProcessLookupError:
   atomic(OUT/"status_b.json",{"state":"blocked_stage_A_not_alive_no_acceptance","not_model_completion":True});return
  atomic(OUT/"status_b.json",{"state":"waiting_for_cohort_correct_imputation_selection","updated":time.time(),"base_searches_planned":540,"final_model_complete":False})
  time.sleep(30)
 assert json.loads((OUT/"STAGE_A_ACCEPTANCE.json").read_text())["status"]=="complete_outer_training_coarse_selection_only"
 subprocess.run([str(PYTHON),"-B",str(CODE/"audit_imports.py")],cwd=ROOT,check=True)
 assert json.loads((OUT/"CACHE_IMPORT_OBJECT_AUDIT.json").read_text())["status"]=="PASS"
 tasks=jobs();assert len(tasks)==540
 for t in tasks:
  f=OUT/"selection"/t["line"]/t["cohort"]/t["context"]/"METHOD_SELECTION.json"
  t["method"]=json.loads(f.read_text())["selected_method"]
 atomic(OUT/"STAGE_B_MANIFEST.json",{"tasks":tasks,"count":540,"full_or_selected_targets":96,"small_finite_spaces_exhausted":True})
 atomic(OUT/"STAGE_B_CODE_LOCK.json",{n:sha(CODE/n) for n in ["model_engine.py","search_worker.py","coordinator_b.py"]})
 done=set();running={};failed=[];peaks={};start=time.time()
 order={v:i for i,v in enumerate(["LR","NB","DT","RF","KNN","SVM","SGBT","XGB","NNET"])}
 tasks.sort(key=lambda t:(order[t["family"]],t["context"]!="final",t["line"],t["cohort"],t["stage"],t["branch"],t["context"]))
 for t in tasks:
  if ready(directory(t)):done.add(digest(t))
 try:
  while len(done)<len(tasks):
   for key,v in list(running.items()):
    pp=v["process"]
    if pp.poll() is None:continue
    v["log"].close();t=v["task"]
    if pp.returncode==0 and ready(directory(t)):
     done.add(key);rec=json.loads((directory(t)/"COMPLETE.json").read_text())
     peaks[t["family"]]=max(peaks.get(t["family"],0),rec["peak_model_rss_bytes"]*1.5+256*1024**2)
    else:failed.append({"task":t,"exit":pp.returncode,"log":str(v["logpath"])})
    del running[key]
   snap=resource_snapshot();reserve=max(4*1024**3,int(snap["available_bytes"]*.25))
   growth=sum(max(0,v["reservation"]-process_rss(v["process"].pid)) for v in running.values())
   budget=snap["available_bytes"]-reserve-growth
   cpu=snap["effective_cpu"]-sum(v["threads"] for v in running.values())
   if not failed:
    for t in tasks:
     key=digest(t)
     if key in done or key in running:continue
     need=estimate(t,peaks)
     if need>budget or cpu<1:continue
     # Allocate all currently useful cores to thread-capable fits, preserving CPU slots for pending independent families.
     threadcap=t["family"] in ["RF","XGB","KNN"]
     possible=max(1,int(budget/max(need,1)))
     threads=max(1,cpu//possible) if threadcap else 1
     threads=min(threads,cpu)
     taskpath=OUT/"requests"/("search_"+key[:20]+".json");atomic(taskpath,t)
     lp=OUT/"logs"/("search_"+key[:20]+".log");handle=lp.open("a")
     env=os.environ.copy();env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]});env["PYTHONDONTWRITEBYTECODE"]="1"
     pp=subprocess.Popen([str(SHARED),"-B",str(CODE/"search_worker.py"),str(taskpath),"--threads",str(threads)],stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=ROOT)
     running[key]={"process":pp,"task":t,"log":handle,"logpath":lp,"threads":threads,"reservation":need};budget-=need;cpu-=threads
   atomic(OUT/"status_b.json",{"state":"failed_draining_owned_workers" if failed else ("running_expanded_base_family_search" if running else "waiting_memory"),"complete_studies":len(done),"total_studies":len(tasks),"running":[{"task":v["task"],"pid":v["process"].pid,"threads":v["threads"]} for v in running.values()],"failed":failed,"resources":snap,"updated":time.time(),"pid":os.getpid(),"clinical_model_lock_complete":False,"eleven_family_comparison_complete":False})
   if failed and not running:raise RuntimeError("Search failure; preserved tasks and checkpoints")
   time.sleep(5)
  rows=[]
  for t in tasks:
   rec=json.loads((directory(t)/"COMPLETE.json").read_text());rows.append({**t,"selected":rec["selected"],"fits":rec["fits"]})
  atomic(OUT/"STAGE_B_BASE_SEARCH_RESULTS.json",{"rows":rows,"status":"base_family_hyperparameter_selection_only","not_outer_validation_or_clinical_calibration":True})
  atomic(OUT/"STAGE_B_ACCEPTANCE.json",{"status":"complete540_base_family_searches_only","count":540,"seconds":time.time()-start,"voting_stacking_outer_fits_calibration_not_implied":True})
  atomic(OUT/"status_b.json",{"state":"complete_base_search_only","complete_studies":540,"total_studies":540,"updated":time.time(),"clinical_model_lock_complete":False,"eleven_family_comparison_complete":False,"task_owned_workers_remaining":0})
 finally:
  for v in running.values():v["process"].wait();v["log"].close()
  lock.close()
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"COORDINATOR_B_FAILURE.json",{"time":time.time(),"traceback":traceback.format_exc()});raise
