"""Finite, resumable resource-aware nested ensemble reconstruction DAG."""
from core import *
import subprocess,traceback,gc
G=1024**3
def routes(cohort):
 return [("preop",b) for b in ["full","compact","selected"]] if cohort=="preop" else [("preop","full"),("endop","full")]
def build(man):
 jobs=[];bank={};scopes={s["id"]:s for s in man["scopes"]}
 rootorder={r["id"]:i for i,r in enumerate(sorted(man["roots"],key=lambda r:(r["context"]=="final",r["context"],r["line"],r["cohort"])))}
 for s in man["scopes"]:
  rid=s["parent"] if s["level"]==1 else scopes[s["parent"]]["parent"]
  order=rootorder[rid];sid=s["id"];scopepath=OUT/"scopes"/sid/"SCOPE.json"
  # Coarse method choice consumes only this scope's training rows.
  cid="coarse_"+sid
  jobs.append({"id":cid,"kind":"coarse","scope":s,"dest":str(Path(s["output"])/"COARSE_ACCEPTANCE.json"),"depends":[],"priority":[order,2,s["level"],sid]})
  for stage,branch in routes(s["cohort"]):
   for family in FAMILIES:
    t={"line":s["line"],"cohort":s["cohort"],"stage":stage,"branch":branch,"context":"outer_0","family":family,"scope_id":sid,"scope_sha256":sha(scopepath)}
    key=digest(t)[:24];search="search_"+key;fit="fit_"+key
    sd=Path(s["output"])/"search_v2"/s["line"]/s["cohort"]/stage/branch/"outer_0"/family
    fd=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/family
    rank=FAMILIES.index(family)
    jobs.append({"id":search,"kind":"search","scope":s,"task":t,"dest":str(sd/"COMPLETE.json"),"depends":[cid],"priority":[order,1,s["level"],stage,branch,rank]})
    jobs.append({"id":fit,"kind":"fit","scope":s,"task":t,"dest":str(fd/"COMPLETE.json"),"depends":[search],"priority":[order,0,s["level"],stage,branch,rank]})
    bank[(sid,stage,branch,family)]=fit
 for r in man["roots"]:
  ss=[s for s in man["scopes"] if s["parent"]==r["id"] or (s["level"]==2 and scopes[s["parent"]]["parent"]==r["id"])]
  for stage,branch in routes(r["cohort"]):
   deps=[bank[(s["id"],stage,branch,f)] for s in ss for f in FAMILIES]
   key="ensemble_"+digest({"r":r["id"],"stage":stage,"branch":branch})[:24]
   d=OUT/"ensemble_results"/r["line"]/r["cohort"]/stage/branch/r["context"]
   jobs.append({"id":key,"kind":"ensemble","task":{"root":r,"stage":stage,"branch":branch},"dest":str(d/"COMPLETE.json"),"depends":deps,"priority":[rootorder[r["id"]],-1,0,stage,branch]})
 return jobs
def good(j):
 p=Path(j["dest"])
 if not p.exists():return False
 r=json.loads(p.read_text())
 if p.name=="COARSE_ACCEPTANCE.json":return r["status"]=="PASS" and r["source_scope_sha256"]==sha(OUT/"scopes"/j["scope"]["id"]/"SCOPE.json")
 return ready(p.parent)
def estimate(j,peaks):
 k=j["kind"]
 family=j.get("task",{}).get("family","")
 if k=="coarse":base=.9*G
 elif k=="ensemble":base=.7*G
 elif k=="search":base=({"RF":1.7,"XGB":1.4,"SVM":1.2}.get(family,1.0)+(.5 if j["task"]["branch"]=="selected" else 0))*G
 else:base=.9*G
 return max(base,peaks.get((k,family),0))
def launch(j,threads):
 rp=OUT/"requests"/(j["id"]+".json");env=os.environ.copy()
 env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]});env["PYTHONDONTWRITEBYTECODE"]="1"
 if j["kind"]=="ensemble":
  atomic(rp,j["task"]);cmd=[str(PYTHON),"-B",str(CODE/"ensemble_worker.py"),str(rp)]
 else:
  s=j["scope"];env.update(AKI_SCOPE_OUT=s["output"],AKI_SCOPE_INPUT=s["input"])
  sp=OUT/"scopes"/s["id"]/"SCOPE.json"
  if j["kind"]=="coarse":cmd=[str(PYTHON),"-B",str(CODE/"scope_worker.py"),"coarse",str(sp),"--threads",str(threads)]
  else:
   mp=Path(s["output"])/"selection"/s["line"]/s["cohort"]/"outer_0/METHOD_SELECTION.json"
   t={**j["task"],"method":json.loads(mp.read_text())["selected_method"]}
   atomic(rp,t)
   if j["kind"]=="search":cmd=[str(SHARED),"-B",str(CODE/"search_worker.py"),str(rp),"--threads",str(threads)]
   else:cmd=[str(PYTHON),"-B",str(CODE/"scope_worker.py"),"fit",str(sp),"--task",str(rp),"--threads",str(threads)]
 lp=OUT/"logs"/(j["id"]+".log");log=lp.open("a")
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 return {"process":p,"log":log,"logpath":lp,"job":j,"threads":threads}
def summarize(man):
 import pandas as pd,numpy as np
 from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
 base=ROOT/"outputs/formal_development_20261018_v1/RAW_BASE_OUTER_FOLD_RESULTS.csv"
 df=pd.read_csv(base)
 rows=[];calrows=[]
 for r in man["roots"]:
  if r["context"]=="final":continue
  for stage,branch in routes(r["cohort"]):
   d=OUT/"ensemble_results"/r["line"]/r["cohort"]/stage/branch/r["context"]
   sel=json.loads((d/"SELECTION.json").read_text())
   for kind in ["Voting","Stacking"]:
    rows.append({"line":r["line"],"cohort":r["cohort"],"stage":stage,"branch":branch,"family":kind,"outer_fold":int(r["context"].split("_")[1]),**sel["winners"][kind]["outer_metrics"]})
   for family in FAMILIES:
    x=pd.read_csv(d/(family+"_calibrated_outer.csv"))
    calrows.append({"line":r["line"],"cohort":r["cohort"],"stage":stage,"branch":branch,"family":family,"outer_fold":int(r["context"].split("_")[1]),"AP":float(average_precision_score(x.y,x.prob_calibrated)),"Brier":float(brier_score_loss(x.y,x.prob_calibrated)),"AUROC":float(roc_auc_score(x.y,x.prob_calibrated))})
 new=pd.DataFrame(rows);new.to_csv(OUT/"RAW_ENSEMBLE_OUTER_FOLD_RESULTS.csv",index=False)
 # Only raw probabilities compared across all eleven families.
 group=["line","cohort","stage","branch","family"]
 prior=df[group+["AP","Brier","AUROC"]]
 combined=pd.concat([prior,new[group+["AP","Brier","AUROC"]]],ignore_index=True)
 combined.groupby(group)[["AP","Brier","AUROC"]].mean().reset_index().to_csv(OUT/"RAW_ELEVEN_FAMILY_INTERNAL_COMPARISON.csv",index=False)
 pd.DataFrame(calrows).to_csv(OUT/"BASE_CALIBRATED_OUTER_FOLD_RESULTS.csv",index=False)
 atomic(OUT/"STAGE_D_ACCEPTANCE.json",{"status":"complete_nested_raw_ensembles_and_base_calibration_only","scopes":len(man["scopes"]),"raw_ensemble_outer_fits":len(rows),"calibrated_base_outer_fits":len(calrows),"strict_complete_ensemble_calibration_complete":False,"full_inner_selected_branch_family_procedure_complete":False,"final_model_locked":False,"final_hospital_time_evaluation_complete":False,"next_required":"strict ensemble calibration and complete inner selection evaluation, then model lock and final evaluation"})
def main():
 OUT.mkdir(exist_ok=True)
 lock=(OUT/"COORDINATOR_D.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/"FUNCTIONAL_QUALIFICATION.json").read_text())["status"]=="PASS"
 old=ROOT/"outputs/formal_development_20261018_v1"
 for stage in ["A","B","C"]:assert (old/("STAGE_"+stage+"_ACCEPTANCE.json")).exists()
 man=json.loads((OUT/"SCOPE_MANIFEST.json").read_text());jobs=build(man)
 for p in ["requests","logs"]: (OUT/p).mkdir(exist_ok=True)
 codehash={p.name:sha(p) for p in CODE.glob("*.py")}
 atomic(OUT/"CODE_LOCK.json",codehash)
 atomic(OUT/"JOB_MANIFEST.json",{"jobs":[{k:v for k,v in j.items() if k!="scope"}|({"scope_id":j["scope"]["id"]} if "scope" in j else {}) for j in jobs],"total":len(jobs),"counts":{k:sum(j["kind"]==k for j in jobs) for k in ["coarse","search","fit","ensemble"]}})
 done={j["id"] for j in jobs if good(j)};running={};failed=[];peaks={};start=time.time();last=0
 jobs.sort(key=lambda j:j["priority"])
 try:
  while len(done)<len(jobs):
   for key,v in list(running.items()):
    pp=v["process"]
    if pp.poll() is None:continue
    v["log"].close();j=v["job"]
    if pp.returncode==0 and good(j):
     done.add(key);r=json.loads(Path(j["dest"]).read_text())
     peak=r.get("peak_model_rss_bytes",r.get("peak_rss_bytes",0))
     peak=max(peak,v.get("observed_rss",0))
     if j["kind"]=="search":peak+=256*1024**2
     pk=(j["kind"],j.get("task",{}).get("family",""));peaks[pk]=max(peaks.get(pk,0),peak*1.4)
    else:failed.append({"id":key,"exit":pp.returncode,"log":str(v["logpath"])})
    del running[key]
   for v in running.values():
    v["observed_rss"]=max(v.get("observed_rss",0),process_rss(v["process"].pid))
   snap=resource_snapshot();headroom=max(4*G,snap["available_bytes"]*.25)
   growth=sum(max(0,v["reservation"]-process_rss(v["process"].pid)) for v in running.values())
   budget=snap["available_bytes"]-headroom-growth
   cpu=snap["effective_cpu"]-sum(v["threads"] for v in running.values())
   free=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize
   if not failed and free>60*G:
    for j in jobs:
     key=j["id"]
     if key in done or key in running or not set(j["depends"])<=done:continue
     need=estimate(j,peaks)
     if budget<need or cpu<1:continue
     threads=max(1,cpu//max(1,int(budget/need))) if j["kind"]=="coarse" or j.get("task",{}).get("family") in ["RF","XGB","KNN"] else 1
     threads=min(threads,cpu)
     v=launch(j,threads);v["reservation"]=need;running[key]=v;budget-=need;cpu-=threads
   now=time.time()
   if now-last>=15 or failed:
    counts={k:{"complete":sum(j["id"] in done and j["kind"]==k for j in jobs),"total":sum(j["kind"]==k for j in jobs)} for k in ["coarse","search","fit","ensemble"]}
    state="failed_draining_owned_workers" if failed else ("running" if running else ("waiting_disk" if free<=60*G else "waiting_memory"))
    status={"state":state,"pid":os.getpid(),"updated":now,"started":start,"completed_tasks":len(done),"total_tasks":len(jobs),"counts":counts,"running":[{"id":k,"pid":v["process"].pid,"threads":v["threads"]} for k,v in running.items()],"failed":failed,"resources":snap,"data_free_bytes":free,"final_model_locked":False,"entire_research_complete":False}
    atomic(OUT/"status_d.json",status)
    current=ROOT/"outputs/CURRENT_ANALYSIS.json";cur=json.loads(current.read_text()) if current.exists() else {}
    cur["ensemble_continuation"]={"status_file":str(OUT/"status_d.json"),"state":state,"counts":counts,"updated":now,"pid":os.getpid(),"model_lock_complete":False};atomic(current,cur)
    last=now
   if failed and not running:raise RuntimeError("Task failed; logs/checkpoints preserved, no automatic unchanged retry")
   time.sleep(5)
  assert codehash=={p.name:sha(p) for p in CODE.glob("*.py")},"Active source changed"
  summarize(man)
  atomic(OUT/"status_d.json",{"state":"complete_raw11_and_base_calibration_only","complete":len(jobs),"total":len(jobs),"updated":time.time(),"task_owned_workers_remaining":0,"strict_ensemble_calibration_complete":False,"final_model_locked":False,"entire_research_complete":False})
 finally:
  for v in running.values():v["process"].wait();v["log"].close()
  lock.close()
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"COORDINATOR_D_FAILURE.json",{"time":time.time(),"traceback":traceback.format_exc()});raise
