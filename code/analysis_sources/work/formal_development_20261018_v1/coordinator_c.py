"""Finite raw outer-base evaluation; explicitly stops before full11/calibration."""
from core import *
from coordinator_b import jobs,estimate
from raw_outer import outer_result
import subprocess,traceback,csv
def main():
 lock=(OUT/"COORDINATOR_C.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/"STAGE_B_ACCEPTANCE.json").read_text())["status"]=="complete540_base_family_searches_only"
 assert json.loads((OUT/"RAW_BUNDLE_FUNCTIONAL_QUALIFICATION.json").read_text())["status"]=="PASS"
 tasks=[t for t in jobs() if t["context"].startswith("outer_")];assert len(tasks)==450
 for t in tasks:t["method"]=json.loads((OUT/"selection"/t["line"]/t["cohort"]/t["context"]/"METHOD_SELECTION.json").read_text())["selected_method"]
 atomic(OUT/"STAGE_C_MANIFEST.json",{"tasks":tasks,"count":450,"scope":"raw base9family outer evaluation; not calibrated full11 comparison"})
 atomic(OUT/"STAGE_C_CODE_LOCK.json",{n:sha(CODE/n) for n in ["model_engine.py","raw_outer.py","coordinator_c.py"]})
 done={digest(t) for t in tasks if ready(outer_result(t))};running={};failed=[];peaks={}
 try:
  while len(done)<450:
   for key,v in list(running.items()):
    p=v["process"]
    if p.poll() is None:continue
    v["log"].close();t=v["task"]
    if p.returncode==0 and ready(outer_result(t)):
     done.add(key);r=json.loads((outer_result(t)/"COMPLETE.json").read_text())
     peaks[t["family"]]=max(peaks.get(t["family"],0),r["peak_rss_bytes"]*1.4)
    else:failed.append({"task":t,"exit":p.returncode,"log":str(v["logpath"])})
    del running[key]
   snap=resource_snapshot();reserve=max(4*1024**3,int(snap["available_bytes"]*.25))
   growth=sum(max(0,v["reservation"]-process_rss(v["process"].pid)) for v in running.values())
   budget=snap["available_bytes"]-reserve-growth;cpu=snap["effective_cpu"]-sum(v["threads"] for v in running.values())
   if not failed:
    for t in tasks:
     key=digest(t)
     if key in done or key in running:continue
     need=estimate(t,peaks)
     if cpu<1 or budget<need:continue
     threads=max(1,cpu//max(1,int(budget/need))) if t["family"] in ["RF","XGB","KNN"] else 1
     threads=min(threads,cpu)
     rp=OUT/"requests"/("raw_"+key[:20]+".json");atomic(rp,t)
     lp=OUT/"logs"/("raw_"+key[:20]+".log");handle=lp.open("a")
     env=os.environ.copy();env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]});env["PYTHONDONTWRITEBYTECODE"]="1"
     p=subprocess.Popen([str(PYTHON),"-B",str(CODE/"raw_outer.py"),str(rp),"--threads",str(threads)],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
     running[key]={"process":p,"task":t,"log":handle,"logpath":lp,"threads":threads,"reservation":need};budget-=need;cpu-=threads
   atomic(OUT/"status_c.json",{"state":"failed_draining_owned_workers" if failed else ("running_raw_outer_base_models" if running else "waiting_memory"),"complete":len(done),"total":450,"running":[{"pid":v["process"].pid,"task":v["task"]} for v in running.values()],"failed":failed,"resources":snap,"updated":time.time(),"full11family_or_calibrated_models_complete":False})
   if failed and not running:raise RuntimeError("Raw outer base task failed")
   time.sleep(5)
  import pandas as pd
  results=[]
  for t in tasks:
   d=outer_result(t);r=json.loads((d/"COMPLETE.json").read_text())
   results.append({**t,**{k:v for k,v in r["metrics"].items() if k in ["AP","Brier","AUROC","assessment_n","assessment_AKI"]}})
  df=pd.DataFrame(results);df.to_csv(OUT/"RAW_BASE_OUTER_FOLD_RESULTS.csv",index=False)
  summary=df.groupby(["line","cohort","stage","branch","family"],as_index=False)[["AP","Brier","AUROC"]].mean()
  summary.to_csv(OUT/"RAW_BASE_INTERNAL_COMPARISON.csv",index=False)
  # Do not choose a final clinical winner before Voting/Stacking and honest calibration.
  atomic(OUT/"STAGE_C_ACCEPTANCE.json",{"status":"complete450_raw_base_outer_fits_only","fits":450,"family_count":9,"metrics_uncalibrated":True,"final_clinical_model_locked":False,"next_required":"Voting/Stacking full-procedure development, calibration, model lock, then hospital/time evaluation","files":{n:sha(OUT/n) for n in ["RAW_BASE_OUTER_FOLD_RESULTS.csv","RAW_BASE_INTERNAL_COMPARISON.csv"]}})
  atomic(OUT/"status_c.json",{"state":"complete_raw_base_internal_evaluation_only","complete":450,"total":450,"updated":time.time(),"full11family_or_calibrated_models_complete":False,"task_owned_workers_remaining":0})
 finally:
  for v in running.values():v["process"].wait();v["log"].close()
  lock.close()
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"COORDINATOR_C_FAILURE.json",{"time":time.time(),"traceback":traceback.format_exc()});raise
