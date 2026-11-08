"""Finite memory-aware scheduler: outer-training original imputation selection."""
from core import *
import subprocess,traceback
def build():
 tasks=[]
 for line in ["geography","temporal"]:
  for cohort in ["preop","paired"]:
   for context in ["final"]+["outer_"+str(i) for i in range(5)]:
    for method in METHODS:
     deps=[]
     for p in parts(line,cohort,context):
      t=pair_task(line,cohort,"preop",context,p,method)
      task_id="impute_"+digest(t)[:20];deps.append(task_id)
      tasks.append({"id":task_id,"worker":"preprocess.py","task":t,"dest":str(cache_dir(t)),"depends":[]})
     for family in FAMILIES:
      t={"kind":"coarse","line":line,"cohort":cohort,"context":context,"method":method,"family":family}
      tasks.append({"id":"coarse_"+digest(t)[:20],"worker":"coarse.py","task":t,"dest":str(OUT/"coarse"/line/cohort/context/method/family),"depends":deps})
 return tasks
def select(line,cohort,context):
 dest=OUT/"selection"/line/cohort/context/"METHOD_SELECTION.json"
 rows=[]
 for family in FAMILIES:
  fam=[]
  for method in METHODS:
   d=OUT/"coarse"/line/cohort/context/method/family
   if not ready(d):return False
   r=json.loads((d/"COMPLETE.json").read_text());fam.append({"method":method,"AP":r["mean_AP"],"Brier":r["mean_Brier"]})
  # Match original average ranks; equal scientific scores share average rank.
  for metric,reverse in [("AP",True),("Brier",False)]:
   ordered=sorted(fam,key=lambda x:x[metric],reverse=reverse)
   for x in ordered:
    positions=[i+1 for i,z in enumerate(ordered) if z[metric]==x[metric]]
    x[metric+"_rank"]=sum(positions)/len(positions)
  rows.extend({"family":family,**x} for x in fam)
 ranking=[]
 for method in METHODS:
  r=[x for x in rows if x["method"]==method]
  ranking.append({"method":method,"mean_AP_rank":sum(x["AP_rank"] for x in r)/9,"mean_Brier_rank":sum(x["Brier_rank"] for x in r)/9})
 ranking.sort(key=lambda x:(x["mean_AP_rank"],x["mean_Brier_rank"],x["method"]))
 atomic(dest,{"selected_method":ranking[0]["method"],"ranking":ranking,"family_rows":rows,"scope":"full preop input original coarse rule on this training context only","line":line,"cohort":cohort,"context":context,"no_final_evaluation_used":True})
 return True
def main():
 OUT.mkdir(exist_ok=True)
 lock=(OUT/"COORDINATOR_A.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (OUT/"FUNCTIONAL_QUALIFICATION.json").exists()
 assert json.loads((OUT/"FUNCTIONAL_QUALIFICATION.json").read_text())["status"]=="PASS"
 codehash={n:sha(CODE/n) for n in ["core.py","preprocess.py","coarse.py","coordinator_a.py"]}
 atomic(OUT/"STAGE_A_CODE_LOCK.json",codehash)
 jobs=build();atomic(OUT/"STAGE_A_MANIFEST.json",{"jobs":jobs,"total":len(jobs),"selection_contexts":24})
 requests=OUT/"requests";requests.mkdir(exist_ok=True)
 logs=OUT/"logs";logs.mkdir(exist_ok=True)
 done=set();running={};failed=[];start=time.time();peak=2*1024**3;last_status=0
 for j in jobs:
  if ready(j["dest"]):done.add(j["id"])
  else:atomic(requests/(j["id"]+".json"),j["task"])
 try:
  while len(done)<len(jobs):
   for jid,v in list(running.items()):
    p=v["process"]
    if p.poll() is None:continue
    v["log"].close();j=v["job"]
    if p.returncode==0 and ready(j["dest"]):
     done.add(jid);r=json.loads((Path(j["dest"])/"COMPLETE.json").read_text())
     peak=max(512*1024**2,min(4*1024**3,max(peak*.95,r.get("peak_rss_bytes",0)*1.35)))
    else:
     failed.append({"id":jid,"exit":p.returncode,"log":str(v["logpath"])})
    del running[jid]
   snap=resource_snapshot();reserve=max(4*1024**3,int(snap["available_bytes"]*.25))
   owned_rss=0
   for v in running.values():
    owned_rss+=process_rss(v["process"].pid)
   budget=max(0,snap["available_bytes"]-reserve)
   target=min(snap["effective_cpu"],max(1,int((budget+owned_rss)/peak)))
   threads=max(1,snap["effective_cpu"]//target)
   ready_jobs=[j for j in jobs if j["id"] not in done and j["id"] not in running and set(j["depends"])<=done]
   ready_jobs.sort(key=lambda j:(0 if j["worker"]=="coarse.py" else 1,j["task"]["context"]!="final",j["id"]))
   if not failed:
    for j in ready_jobs:
     if len(running)>=target or budget<peak:break
     lp=logs/(j["id"]+".log");handle=lp.open("a")
     env=os.environ.copy()
     env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]})
     env["PYTHONDONTWRITEBYTECODE"]="1"
     pp=subprocess.Popen([str(PYTHON),"-B",str(CODE/j["worker"]),str(requests/(j["id"]+".json")),"--threads",str(threads)],stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
     running[j["id"]]={"process":pp,"job":j,"log":handle,"logpath":lp};budget-=peak
   now=time.time()
   if now-last_status>15 or failed:
    atomic(OUT/"status_a.json",{"stage":"original_coarse_imputation_selection_inside_outer_training","state":"failed_draining_owned_workers" if failed else ("running" if running else "waiting_memory"),"pid":os.getpid(),"started":start,"updated":now,"complete":len(done),"total":len(jobs),"running":[{"id":k,"pid":v["process"].pid} for k,v in running.items()],"failed":failed,"resources":snap,"estimated_peak_per_worker":peak,"target_workers":target,"threads_next_worker":threads,"final_model_development_complete":False})
    for line in ["geography","temporal"]:
     for cohort in ["preop","paired"]:
      for context in ["final"]+["outer_"+str(i) for i in range(5)]:
       target_file=OUT/"selection"/line/cohort/context/"METHOD_SELECTION.json"
       if not target_file.exists():select(line,cohort,context)
    last_status=now
   if failed and not running:raise RuntimeError("StageA task failed; preserved logs")
   time.sleep(2)
  for line in ["geography","temporal"]:
   for cohort in ["preop","paired"]:
    for context in ["final"]+["outer_"+str(i) for i in range(5)]:assert select(line,cohort,context)
  assert codehash=={n:sha(CODE/n) for n in codehash},"Active source changed"
  atomic(OUT/"STAGE_A_ACCEPTANCE.json",{"status":"complete_outer_training_coarse_selection_only","tasks":len(jobs),"selection_contexts":24,"formal_11_model_development_complete":False,"code_sha256":codehash,"seconds":time.time()-start})
  atomic(OUT/"status_a.json",{"state":"complete","complete":len(done),"total":len(jobs),"updated":time.time(),"task_owned_workers_remaining":0,"final_model_development_complete":False})
 finally:
  # Only own children; normal failures drain rather than kill completed-independent work.
  for v in running.values():
   v["process"].wait();v["log"].close()
  lock.close()
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"COORDINATOR_A_FAILURE.json",{"traceback":traceback.format_exc(),"time":time.time()})
  raise
