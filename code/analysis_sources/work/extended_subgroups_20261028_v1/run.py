"""Observed ALT missingness and stage-specific capture with frozen risks; no clinical model refitting."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,time,fcntl,traceback,gc
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/extended_subgroups_20261028_v1"
NATIVE=ROOT/"outputs/stratified_evaluation_20261023_v1"
NATIVE_CODE=ROOT/"work/stratified_evaluation_20261023_v1"
PY=ROOT/"envs/prediction_gpu/bin/python"
KINDS=("stage1","stage2","stage3","stage2_3","ALT_observed_missing")
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+".tmp."+str(os.getpid()))
 t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def exact(p,d):
 p=Path(p)
 if p.exists():assert read(p)==d,("Changed prepared contract",str(p))
 else:save(p,d)
def prepare():
 assert sha(ROOT/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
 formal=ROOT/"outputs/formal_evaluation_20261023_v1"
 assert read(formal/"STAGE_FINAL_EVALUATION_ACCEPTANCE.json")["status"]=="PASS_four_heldout_partitions_fixed_models_and_locked_policies"
 NATIVE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 exact(NATIVE/"CODE_LOCK.json",{p.name:sha(p) for p in NATIVE_CODE.glob("*.py")})
 methods={"status":"prepared_current_source_baseline_subgroups_not_yet_evaluated",
 "tasks":24,"cohorts":4,"strata":["hospital","age_lt65_ge65","sex","diabetes","tumor_site_source_codes","preop_egfr_lt60_ge60"],
 "subgroup_assignment":"Observed baseline source values; missing separately unknown; no imputation-defined subgroup membership",
 "models":"Same locked primary input routes and policies; no subgroup refitting or selection",
 "replicates":2000,"seed":20261017,"sampling":"Patient groups within hospitals of entire evaluation population then subset",
 "allocation":"Whole-population upgrade and review recomputed in each replicate then inherited by subgroups",
 "severe_endpoint":"Current corrected clinical stages2/3 capture under original anyAKI risks",
 "dependencies":{"formal_engine_code_lock_sha256":sha(formal/"CODE_LOCK.json")},
 "source_lock_sha256":sha(ROOT/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),
 "statistics_not_heterogeneity_test":True,"entire_research_complete":False}
 exact(NATIVE/"EXECUTION_CONTRACT.json",methods)
 c={"status":"source_locked_additional_exploratory_analysis_not_completed",
 "authority":"Direct human request2026-10-28 for ALT/AKI-stage analyses",
 "timing_disclosure":"Added exploratory analyses; evaluation outcomes previously exposed; not retrospective preregistration",
 "kinds":list(KINDS),"tasks":20,"replicates":2000,"seed":20261017,
 "stage_role":"Outcome-severity capture on entire evaluation cohort; no cases-only AUROC or new severity model",
 "ALT_role":"Observed versus missing raw PreopALT; no invented normal limits or outcome-selected cutpoint",
 "allocation":"Retain full-population risk ranking and reroute policy on every hospital/patient bootstrap",
 "source_sha256":methods["source_lock_sha256"],
 "formal_acceptance_sha256":sha(formal/"STAGE_FINAL_EVALUATION_ACCEPTANCE.json"),
 "formal_code_lock_sha256":sha(formal/"CODE_LOCK.json"),
 "native_code_lock_sha256":sha(NATIVE/"CODE_LOCK.json"),
 "native_method_contract_sha256":sha(NATIVE/"EXECUTION_CONTRACT.json"),
 "runner_code_sha256":sha(__file__)}
 exact(OUT/"EXECUTION_CONTRACT.json",c)
 return c
def native():
 sys.path.insert(0,str(NATIVE_CODE));import stratified
 return stratified
def worker(line,cohort,kind):
 c=prepare();assert kind in KINDS
 s=native();dest=OUT/(line+"_"+cohort)/kind;dest.mkdir(parents=True,exist_ok=True)
 sig={"contract_sha256":sha(OUT/"EXECUTION_CONTRACT.json"),"line":line,"cohort":cohort,"kind":kind}
 if (dest/"COMPLETE.json").exists():
  old=read(dest/"COMPLETE.json");assert old["signature"]==sig
  for f,h in old["files"].items():assert sha(dest/f)==h
  return
 import numpy as np,pandas as pd
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=1):
  m,x,p,thresholds,policies,comparisons=s.load(line,cohort)
  stage=pd.to_numeric(m.stage,errors="raise").to_numpy(dtype=int)
  assert np.isin(stage,[0,1,2,3]).all()
  assert np.array_equal(stage>0,m.y.to_numpy(dtype=int)==1)
  if kind=="ALT_observed_missing":
   raw=pd.to_numeric(x.PreopALT,errors="coerce").to_numpy(dtype=float)
   level=np.where(np.isnan(raw),"ALT_missing","ALT_observed");target=stage>=2
  else:
   level=np.repeat("whole_evaluation_population",len(m))
   target=stage>=2 if kind=="stage2_3" else stage==int(kind[-1])
  result=s.evaluate(m.y.to_numpy(),p,m.group_hash.tolist(),m.excel_row.astype(str).tolist(),
    m.center.astype(str).tolist(),policies,thresholds,level,target,comparisons,replicates=2000)
  original=read(s.base.OUT/(line+"_"+cohort)/"EVALUATION.json")["receipt"]
  assert result["receipt"]["replicates"]==2000 and result["receipt"]["seed"]==20261017
  if "resample_weight_sha256" in original:
   assert result["receipt"]["resample_weight_sha256"]==original["resample_weight_sha256"]
  save(dest/"FULL_STATISTICAL_RESULT.json",result)
  point=result["point"]
  if kind.startswith("stage"):
   point=[{k:v for k,v in row.items() if k in ["model","level","n","events","severe_events"] or "severe" in k} for row in point]
  def target_labels(rows):
   if not kind.startswith("stage"):return rows
   transformed=[]
   for row in rows:
    if "metric" in row and "severe" not in str(row["metric"]):continue
    newrow={k.replace("severe","target_stage"):(v.replace("severe","target_stage") if k in ["metric","field"] and isinstance(v,str) else v) for k,v in row.items()}
    newrow["capture_target"]=kind;newrow["prediction_target"]="any_AKI_frozen_risk"
    transformed.append(newrow)
   return transformed
  pd.DataFrame(target_labels(point)).to_csv(dest/"POINT.csv",index=False)
  pd.DataFrame(target_labels(result["intervals"])).to_csv(dest/"INTERVALS.csv",index=False)
  pd.DataFrame(target_labels(result["paired_differences"])).to_csv(dest/"PAIRED_DIFFERENCES.csv",index=False)
  save(dest/"RESAMPLING_RECEIPT.json",result["receipt"])
  files=["FULL_STATISTICAL_RESULT.json","POINT.csv","INTERVALS.csv","PAIRED_DIFFERENCES.csv","RESAMPLING_RECEIPT.json"]
  save(dest/"COMPLETE.json",{"status":"PASS_current_source_exploratory_ALT_or_stage_capture","signature":sig,
    "n":len(m),"events":int(m.y.sum()),"target_stage_events":int(target.sum()),"kind":kind,
    "files":{f:sha(dest/f) for f in files},"original_any_AKI_target_and_models_unchanged":True,
    "no_subgroup_or_severity_refitting":True,"whole_population_allocation_retained":True,
    "additional_exploratory_analysis":True,"stage_only_AUROC_not_claimed":True})
 gc.collect()
def rss(pid):
 try:return int(next(x.split()[1] for x in Path("/proc",str(pid),"status").read_text().splitlines() if x.startswith("VmRSS:")))*1024
 except (OSError,StopIteration):return 0
def run_all():
 state=read(ROOT/"outputs/CURRENT_ANALYSIS.json")
 assert state["status"]=="MAIN_COMPLETE_SUPPLEMENTS_AND_DELIVERABLES_REQUIRED","Original quantitative master must finish first; main priority preserved"
 prepare()
 global_guard=(ROOT.parents[1]/"outputs/ACTIVE_REANALYSIS.lock").open("a");fcntl.flock(global_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 owner=(OUT/"SUPERVISOR.lock").open("a");fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="")
 subprocess.run([str(PY),"-B",str(NATIVE_CODE/"qualify.py")],cwd=ROOT,env=env,check=True)
 s=native()
 base_log=(OUT/"BASE24_SUPERVISOR.log").open("ab")
 base_proc=None
 if not (NATIVE/"STAGE_STRATIFIED_ACCEPTANCE.json").exists():
  base_proc=subprocess.Popen([str(PY),"-B",str(NATIVE_CODE/"supervisor.py")],cwd=ROOT,env=env,stdout=base_log,stderr=subprocess.STDOUT)
 sys.path.insert(0,str(ROOT/"work/ensemble_speedup_20261021_v2"));from resource_policy import CPUShare
 sys.path.insert(0,str(ROOT/"work/ensemble_calibration_20261020_v1"))
 from core import resource_snapshot
 cpu=CPUShare();time.sleep(2)
 jobs=[(line,cohort,kind) for line,cohort in s.base.EXPECTED for kind in KINDS]
 live={};done=[];failures=[];pilot=False;per_task=1024**3
 try:
  while jobs or live:
   if base_proc is not None and base_proc.poll() not in [None,0]:failures.append({"base24_exit":base_proc.returncode})
   for job,v in list(live.items()):
    proc,log=v
    if proc.poll() is None:continue
    log.close();del live[job]
    if proc.returncode:failures.append({"job":list(job),"exit":proc.returncode})
    else:
     d=OUT/(job[0]+"_"+job[1])/job[2];rec=read(d/"COMPLETE.json")
     for f,h in rec["files"].items():assert sha(d/f)==h
     done.append(job);pilot=True
   snap=resource_snapshot()
   budget_cpu=cpu.sample(snap.get("affinity_and_quota_cpu",snap["effective_cpu"]),[v[0].pid for v in live.values()])
   available=snap["available_bytes"];growth=sum(max(0,per_task-rss(v[0].pid)) for v in live.values())
   spare=available-max(4*1024**3,int(.25*available))-growth
   while jobs and not failures and len(live)<(budget_cpu if pilot else 1) and spare>=per_task:
    job=jobs.pop(0);log=(OUT/("_".join(job)+".log")).open("ab")
    proc=subprocess.Popen([str(PY),"-B",str(Path(__file__)),"--worker",*job],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    live[job]=(proc,log);spare-=per_task
   save(OUT/"status.json",{"state":"running_extended_subgroups" if not failures else "failed_draining",
    "pid":os.getpid(),"completed":[list(j) for j in done],"pending":[list(j) for j in jobs],
    "running":{"_".join(j):v[0].pid for j,v in live.items()},"failures":failures,"available_bytes":available,"clinical_pilot_accepted":pilot})
   if failures and not live:raise RuntimeError(str(failures))
   if jobs or live:time.sleep(5)
  if base_proc is not None:assert base_proc.wait()==0
  base_log.close()
  base_acc=read(NATIVE/"STAGE_STRATIFIED_ACCEPTANCE.json");assert base_acc["status"]=="PASS_24_prespecified_stratified_evaluations"
  assert len(done)==20
  receipts={str(NATIVE/"STAGE_STRATIFIED_ACCEPTANCE.json"):sha(NATIVE/"STAGE_STRATIFIED_ACCEPTANCE.json")}
  for job in done:
   f=OUT/(job[0]+"_"+job[1])/job[2]/"COMPLETE.json";receipts[str(f)]=sha(f)
  save(OUT/"STAGE_EXTENDED_SUBGROUP_ACCEPTANCE.json",{"status":"PASS_current_source_baseline_subgroups_ALT_and_AKI_severity",
   "source_sha256":sha(ROOT/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),"baseline_prespecified_tasks":24,
   "additional_exploratory_tasks":20,"receipts":receipts,"contract_sha256":sha(OUT/"EXECUTION_CONTRACT.json"),
   "original_models_thresholds_methods_seeds_retained":True,"scientific_review_performed":False})
  save(OUT/"status.json",{"state":"complete_baseline_and_exploratory_subgroups","owned_workers_remaining":0,"entire_research_complete":False})
 finally:
  for proc,log in live.values():proc.wait();log.close()
  if base_proc is not None and base_proc.poll() is None:base_proc.wait()
  if not base_log.closed:base_log.close()
if __name__=="__main__":
 try:
  if len(sys.argv)>1 and sys.argv[1]=="--worker":worker(*sys.argv[2:5])
  elif len(sys.argv)>1 and sys.argv[1]=="--run":run_all()
  else:prepare();print("PREPARED_CURRENT_SOURCE_PENDING_MAIN_COMPLETION",flush=True)
 except Exception:
  OUT.mkdir(parents=True,exist_ok=True);save(OUT/"FAILURE.json",{"traceback":traceback.format_exc()});raise
