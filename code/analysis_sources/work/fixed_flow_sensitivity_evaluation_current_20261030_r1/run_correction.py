"""Guarded finite correction of two stale-mask temporal sensitivity evaluations."""
from pathlib import Path
import argparse,csv,datetime,fcntl,gc,hashlib,importlib.util,json,os,resource,subprocess,sys,time,traceback
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/fixed_flow_sensitivity_evaluation_current_20261030_r1"
O=R/"outputs/fixed_flow_sensitivity_evaluation_current_20261030_r1"
C=R/"outputs/reanalysis_control/fixed_flow_mask_correction_20261030_r1"
LOCK=W/"EXECUTION_CODE_LOCK.json"
UNIT="aki-postslot-fixed-flow-evaluation-correction-20261030-r2.service"
GIB=1024**3
BUDGET=2*GIB
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 with Path(p).open("x") as f:
  json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def code_guard():
 for rel,h in read(LOCK)["files"].items():
  if sha(R/rel)!=h:raise RuntimeError("Correction dependency changed: "+rel)
def preflight():
 code_guard()
 g=module("_fixed_mask_admission",R/"work/llm_statistics_launch_20261030_r3/launch.py")
 g.code_guard();state=g.unit_state(g.FORMAL_UNIT);ap=g.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"
 reason=g.completion_reason(state,g.read(ap) if ap.exists() else None,g.read(g.F/"STATUS.json"))
 result=dict(time=now(),unit=UNIT,formal_unit=state,actual_numerical_correction_started=False)
 if reason:return dict(result,status=reason,eligible=False)
 plan=g.resource_plan(False)
 remaining=min(plan["available_for_new_growth"],plan["AKI_family_remaining_cap"])-2*GIB
 ok=plan["accepted"] and remaining>=BUDGET
 return dict(result,status="READY_FOR_TWO_CONTEXT_CORRECTION" if ok else "WAIT_JOINT_RESOURCES",
             eligible=ok,resource_plan=plan,memory_budget_bytes=BUDGET,
             remaining_after_separate_monitoring=remaining,formal_acceptance_sha256=sha(ap))
def verify_context(path,context):
 d=read(path)
 for name,h in d["files"].items():
  if sha(path.parent/name)!=h:raise RuntimeError("Context output hash mismatch")
 sig=d["signature"]
 if (sig["line"],sig["cohort"],sig["scenario"])!=(context["line"],context["cohort"],context["scenario"]):
  raise RuntimeError("Context identity changed")
 if (d["n"],d["events"])!=(context["corrected_n"],context["corrected_events"]):
  raise RuntimeError("Corrected validation denominator mismatch")
 return d
def execute():
 admission=preflight()
 if not admission["eligible"]:
  print(json.dumps(admission,ensure_ascii=False));return
 g=module("_fixed_mask_owner",R/"work/llm_statistics_launch_20261030_r3/launch.py")
 st=g.unit_state(UNIT)
 if st["ActiveState"]!="active" or int(st["MainPID"])!=os.getpid():
  raise RuntimeError("Exact owned finite unit required")
 if int(st["MemoryMax"])!=BUDGET or st["MemorySwapMax"]!="0":
  raise RuntimeError("Exact2GiB/swap0 cgroup required")
 cg=Path("/sys/fs/cgroup")/st["ControlGroup"].lstrip("/")
 wall=subprocess.check_output(["systemctl","--user","show",UNIT,"--property=RuntimeMaxUSec","--value"],text=True).strip()
 if wall!="2h":raise RuntimeError("Exact two-hour finite ceiling required")
 if (O/"RUN_STARTED.json").exists():raise RuntimeError("Existing outcome requires inspection, no blind restart")
 os.sched_setaffinity(0,{admission["resource_plan"]["CPU_affinity"][-1]})
 for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[key]="1"
 os.environ["CUDA_VISIBLE_DEVICES"]=""
 resource.setrlimit(resource.RLIMIT_CPU,(7100,7100))
 guard=(O/"OWNER.lock").open("x");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(O/"RUN_STARTED.json",dict(time=now(),pid=os.getpid(),start_ticks=Path("/proc/self/stat").read_text().split()[21],
      admission=admission,execution_lock_sha256=sha(LOCK)))
 begin=time.monotonic()
 try:
  protection=module("_mask_protection",R/"work/study_package_scope_20261030_v1/audit_fixed_policy_records.py")
  protected=protection.protection()
  ev=module("_fixed_flow_current_mask_evaluation",W/"evaluate.py")
  ev.check()
  if ev.HIGH:raise RuntimeError("Source exclusion set differs from documented current empty set")
  contexts=read(C/"CONTEXT_IMPACT_AND_RECOMPUTE_PLAN.json")
  if len(contexts)!=8 or sum(x["requires_numerical_recomputation"] for x in contexts)!=2:
   raise RuntimeError("Exact scoped correction changed")
  from threadpoolctl import threadpool_limits
  recomputed=[];retained=[];receipts={}
  with threadpool_limits(limits=1):
   for context in contexts:
    if context["requires_numerical_recomputation"]:
     ev.execute(context["line"],context["cohort"],context["scenario"])
     p=O/(context["line"]+"_"+context["cohort"])/context["scenario"]/"COMPLETE.json"
     verify_context(p,context);recomputed.append(str(p));receipts[str(p)]=sha(p)
    else:
     p=R/context["old_receipt"]
     if sha(p)!=context["old_receipt_sha256"]:raise RuntimeError("Unchanged original context changed")
     verify_context(p,context);retained.append(str(p));receipts[str(p)]=sha(p)
    gc.collect()
  if len(recomputed)!=2 or len(retained)!=6 or len(receipts)!=8:raise RuntimeError("Incomplete correction")
  if protection.protection()!=protected:raise RuntimeError("Protected manuscript changed")
  code_guard()
  # All native evidence remains unchanged; current mutable handoff/requirements
  # were backed up for traceability and are deliberately not native result files.
  snapshot=read(C/"ORIGINAL_BACKUP_MANIFEST_INTERNAL.json")
  for rel,h in snapshot.items():
   if rel in ["outputs/reanalysis_control/CURRENT_HANDOFF_ZH.md",
              "outputs/post_reanalysis_process_review_v1/REQUIRED_COMPONENTS.json"]:continue
   if sha(R/rel)!=h:raise RuntimeError("Original evidence changed: "+rel)
  save(O/"NUMERICAL_CORRECTION_COMPLETION.json",dict(time=now(),
       status="COMPUTED_TWO_CORRECTED_TEMPORAL_CONTEXTS_INDEPENDENT_VERIFICATION_REQUIRED",
       recomputed_receipts=recomputed,retained_receipts=retained,receipts=receipts,
       bootstrap_replicates=2000,seed=20261017,new_predictive_fits=0,
       original_predictions_and_92_manuscripts_unchanged=True,
       seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       cgroup_events=(cg/"memory.events").read_text(),execution_lock_sha256=sha(LOCK)))
  print("CORRECTION_COMPUTED_INDEPENDENT_VERIFICATION_REQUIRED",flush=True)
 except BaseException as error:
  if not (O/"FAILURE.json").exists():
   save(O/"FAILURE.json",dict(time=now(),error=repr(error),traceback=traceback.format_exc(),
        original_evidence_preserved=True,new_predictive_fits=0))
  raise
 finally:guard.close()
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--run",action="store_true");args=a.parse_args()
 if args.run:execute()
 else:print(json.dumps(preflight(),ensure_ascii=False))
