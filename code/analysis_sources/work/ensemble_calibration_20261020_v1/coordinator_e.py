"""Finite E DAG; never dispatch or mutate a completed D task."""
from core import *
from prepare_e import routes
import subprocess,traceback
G=1024**3
def build(man):
 jobs=[];bank={}
 roots=sorted(man["roots"],key=lambda r:(r["original_context"]=="final",r["original_context"],r["line"],r["cohort"],r["parent_fold"]))
 order={r["id"]:i for i,r in enumerate(roots)}
 by={s["id"]:s for s in man["scopes"]}
 for s in man["scopes"]:
  if s["level"]!=3:continue
  rid=by[s["parent"]]["parent"];idx=order[rid];sid=s["id"]
  scopepath=OUT/"scopes"/sid/"SCOPE.json";cid="coarse_"+sid
  jobs.append({"id":cid,"kind":"coarse","scope":s,"dest":str(Path(s["output"])/"COARSE_ACCEPTANCE.json"),"depends":[],"priority":[idx,2,sid]})
  for stage,branch in routes(s["cohort"]):
   for family in FAMILIES:
    t={"line":s["line"],"cohort":s["cohort"],"stage":stage,"branch":branch,"context":"outer_0","family":family,"scope_id":sid,"scope_sha256":sha(scopepath)}
    key=digest(t)[:24];sd=Path(s["output"])/"search_v2"/s["line"]/s["cohort"]/stage/branch/"outer_0"/family
    fd=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/family
    rank=FAMILIES.index(family)
    jobs.append({"id":"search_"+key,"kind":"search","scope":s,"task":t,"dest":str(sd/"COMPLETE.json"),"depends":[cid],"priority":[idx,1,stage,branch,rank,sid]})
    jobs.append({"id":"fit_"+key,"kind":"fit","scope":s,"task":t,"dest":str(fd/"COMPLETE.json"),"depends":["search_"+key],"priority":[idx,0,stage,branch,rank,sid]})
    bank[(sid,stage,branch,family)]="fit_"+key
 for r in roots:
  ss=[s for s in man["scopes"] if s["level"]==3 and by[s["parent"]]["parent"]==r["id"]]
  assert len(ss)==9
  for stage,branch in routes(r["cohort"]):
   deps=[bank[(s["id"],stage,branch,f)] for s in ss for f in FAMILIES]
   key="ensemble_"+digest({"root":r["id"],"stage":stage,"branch":branch})[:24]
   d=OUT/"ensemble_results"/r["line"]/r["cohort"]/stage/branch/r["id"]
   jobs.append({"id":key,"kind":"ensemble","task":{"root":r,"stage":stage,"branch":branch},"dest":str(d/"COMPLETE.json"),"depends":deps,"priority":[order[r["id"]],-1,stage,branch]})
 assert len(jobs)==29988
 return jobs
def good(j):
 p=Path(j["dest"])
 if not p.exists():return False
 if j["kind"]=="coarse":
  r=json.loads(p.read_text())
  return r["status"]=="PASS" and r["source_scope_sha256"]==sha(OUT/"scopes"/j["scope"]["id"]/"SCOPE.json")
 return ready(p.parent)
def estimate(j,peaks):
 f=j.get("task",{}).get("family","");k=j["kind"]
 if k=="coarse":b=.9*G
 elif k=="ensemble":b=.7*G
 elif k=="search":b=({"RF":1.7,"XGB":1.4,"SVM":1.2}.get(f,1.0)+(.5 if j["task"]["branch"]=="selected" else 0))*G
 else:b=.9*G
 return max(b,peaks.get((k,f),0))
def launch(j,threads):
 rp=OUT/"requests"/(j["id"]+".json");env=os.environ.copy()
 env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]})
 env["PYTHONDONTWRITEBYTECODE"]="1"
 if j["kind"]=="ensemble":
  atomic(rp,j["task"]);cmd=[str(PYTHON),"-B",str(CODE/"ensemble_e_worker.py"),str(rp)]
 else:
  s=j["scope"];env.update(AKI_SCOPE_OUT=s["output"],AKI_SCOPE_INPUT=s["input"])
  sp=OUT/"scopes"/s["id"]/"SCOPE.json"
  if j["kind"]=="coarse":cmd=[str(PYTHON),"-B",str(CODE/"scope_worker.py"),"coarse",str(sp),"--threads",str(threads)]
  else:
   mp=Path(s["output"])/"selection"/s["line"]/s["cohort"]/"outer_0/METHOD_SELECTION.json"
   t={**j["task"],"method":json.loads(mp.read_text())["selected_method"]};atomic(rp,t)
   if j["kind"]=="search":cmd=[str(SHARED),"-B",str(CODE/"search_worker.py"),str(rp),"--threads",str(threads)]
   else:cmd=[str(PYTHON),"-B",str(CODE/"scope_worker.py"),"fit",str(sp),"--task",str(rp),"--threads",str(threads)]
 lp=OUT/"logs"/(j["id"]+".log");log=lp.open("a")
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 return {"process":p,"log":log,"logpath":lp,"job":j,"threads":threads}
def main():
 lock=(OUT/"COORDINATOR_E.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/"FUNCTIONAL_QUALIFICATION.json").read_text())["status"]=="PASS"
 man=json.loads((OUT/"SCOPE_MANIFEST.json").read_text())
 assert sha(INPUT/"SOURCE_LOCK.json")==man["source_lock_sha256"]
 assert sha(OUT/"REUSED_D_RECEIPTS.json")==man["reused_receipts_sha256"]
 # Method-reference hashes verified before this corrected-data E launch; clinical source/endpoint are bound separately by SOURCE_LOCK.
 assert sha(ROOT/"outputs/research_plan_revision_20261018_094007/PROTOCOL.json")=="15156b8b29d1a92e6b41bd1b546809f01c693ebf577e9548d9578b8fdc537b9b"
 assert sha(ROOT/"outputs/research_plan_revision_20261018_094007/PARAMETER_SEARCH_SPACE.json")=="3afd015f8fa1976e3cada901fa04c15ddf1610f82120815bb0523dd575253f9e"
 assert sha(ROOT/"outputs/research_plan_revision_20261018_094007/CURRENT_RESEARCH_PLAN_ZH.md")=="c107c3002f69a09d3724f72c97f24ff335a797866e7be077026f00d3041fc66f"
 assert sha(OUT/"SCOPE_MANIFEST.json")==json.loads((OUT/"FUNCTIONAL_QUALIFICATION.json").read_text())["source_manifest_sha256"]
 jobs=build(man)
 for n in ["requests","logs"]:(OUT/n).mkdir(exist_ok=True)
 frozen={p.name:sha(p) for p in CODE.glob("*.py")}
 prior=OUT/"CODE_LOCK.json"
 if prior.exists():assert json.loads(prior.read_text())==frozen
 else:atomic(prior,frozen)
 atomic(OUT/"JOB_MANIFEST.json",{"jobs":[{k:v for k,v in j.items() if k!="scope"}|({"scope_id":j["scope"]["id"]} if "scope" in j else {}) for j in jobs],"total":len(jobs),"counts":{k:sum(j["kind"]==k for j in jobs) for k in ["coarse","search","fit","ensemble"]}})
 done={j["id"] for j in jobs if good(j)};running={};failed=[];peaks={};start=time.time();last=0
 jobs.sort(key=lambda j:j["priority"])
 try:
  while len(done)<len(jobs):
   for key,v in list(running.items()):
    p=v["process"]
    if p.poll() is None:continue
    v["log"].close();j=v["job"]
    if p.returncode==0 and good(j):
     done.add(key);rec=json.loads(Path(j["dest"]).read_text())
     peak=max(rec.get("peak_model_rss_bytes",rec.get("peak_rss_bytes",0)),v.get("observed_rss",0))
     if j["kind"]=="search":peak+=256*1024**2
     pk=(j["kind"],j.get("task",{}).get("family",""));peaks[pk]=max(peaks.get(pk,0),peak*1.4)
    else:failed.append({"id":key,"exit":p.returncode,"log":str(v["logpath"])})
    del running[key]
   for v in running.values():v["observed_rss"]=max(v.get("observed_rss",0),process_rss(v["process"].pid))
   snap=resource_snapshot();headroom=max(4*G,snap["available_bytes"]*.25)
   growth=sum(max(0,v["reservation"]-process_rss(v["process"].pid)) for v in running.values())
   budget=snap["available_bytes"]-headroom-growth
   cpu=snap["effective_cpu"]-sum(v["threads"] for v in running.values())
   stat=os.statvfs(OUT);free=stat.f_bavail*stat.f_frsize
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
    assert frozen=={p.name:sha(p) for p in CODE.glob("*.py")},"Active E source changed"
    counts={k:{"complete":sum(j["id"] in done and j["kind"]==k for j in jobs),"total":sum(j["kind"]==k for j in jobs)} for k in ["coarse","search","fit","ensemble"]}
    state="failed_draining_owned_workers" if failed else ("running" if running else ("waiting_disk" if free<=60*G else "waiting_memory"))
    atomic(OUT/"status_e.json",{"state":state,"pid":os.getpid(),"updated":now,"started":start,"completed_tasks":len(done),"total_tasks":len(jobs),"counts":counts,
     "running":[{"id":k,"pid":v["process"].pid,"threads":v["threads"]} for k,v in running.items()],"failed":failed,"resources":snap,"data_free_bytes":free,"final_model_locked":False,"entire_research_complete":False})
    current=ROOT/"outputs/CURRENT_ANALYSIS.json";cur=json.loads(current.read_text()) if current.exists() else {}
    cur["strict_ensemble_calibration"]={"state":state,"status_file":str(OUT/"status_e.json"),"counts":counts,"pid":os.getpid(),"updated":now,"model_lock_complete":False};atomic(current,cur)
    last=now
   if failed and not running:raise RuntimeError("Task failed; retained logs/checkpoints, no unchanged retry")
   time.sleep(5)
  assert frozen=={p.name:sha(p) for p in CODE.glob("*.py")}
  atomic(OUT/"COMPUTE_BATCH_ACCEPTANCE.json",{"status":"PASS","tasks":len(jobs),"new_scopes":648,"first_layer_route_tasks":180,"source_hashes_unchanged":True,"task_owned_workers_remaining":0})
  env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
  subprocess.run([str(PYTHON),"-B",str(CODE/"complete_e.py")],cwd=ROOT,env=env,check=True)
  atomic(OUT/"status_e.json",{"state":"complete_strict_calibration_internal_evaluation","updated":time.time(),"complete":len(jobs),"total":len(jobs),"task_owned_workers_remaining":0,"final_model_locked":False,"entire_research_complete":False})
 finally:
  for v in running.values():v["process"].wait();v["log"].close()
  lock.close()
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"COORDINATOR_E_FAILURE.json",{"time":time.time(),"traceback":traceback.format_exc()});raise
