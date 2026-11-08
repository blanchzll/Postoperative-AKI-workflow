"""Read-only reconstruction of 24 frozen policies; run only in an admitted finite unit."""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, json, math, os, resource, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/study_package_scope_20261030_v1"
B=R/"outputs/study_package_scope_20261030_v1"
O=B/"private_policy_full_replay_v1"
SCALAR=B/"private_policy_scalar_review_v1/SCALAR_REVIEW_RECEIPT.json"
INPUT=B/"private_evaluation_dependencies_v1"
LOCK=W/"POLICY_REPLAY_CODE_LOCK.json"
UNIT="aki-postslot-private-policy-replay-20261030-v1.service"
BUDGET=768*1024**2
TOL=1e-12
def now(): return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def load_file(name,p):
 spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module);return module
def code_guard():
 locked=read(LOCK)
 for name,h in locked["files"].items(): assert sha(R/name)==h,name
 assert sha(SCALAR)==locked["scalar_receipt_sha256"]
 return locked
def preflight():
 locked=code_guard()
 g=load_file("aki_statistics_admission",R/"work/postformal_priority_admission_20261030_v1/admission.py")
 g.code_guard()
 state=g.unit_state(g.FORMAL_UNIT)
 ap=g.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"
 acceptance=g.read(ap) if ap.exists() else None
 reason=g.completion_reason(state,acceptance,g.read(g.F/"STATUS.json"))
 result={"time":now(),"code_lock_sha256":sha(LOCK),"unit":UNIT,"formal_unit_state":state,
         "actual_numeric_replay_started":False,"own_memory_budget":BUDGET}
 if reason:return dict(result,status=reason,eligible=False)
 # This conservative recheck also counts the finite replay unit if already launched.
 # It never omits another AKI or LIRI owner or future declared LIRI peak.
 plan=g.resource_plan(False)
 remaining=min(plan["available_for_new_growth"],plan["AKI_family_remaining_cap"])-2*1024**3
 ok=bool(plan["accepted"] and remaining>=BUDGET)
 result.update(formal_acceptance_sha256=sha(ap),resource_plan=plan,
   monitoring_margin_separately_retained=2*1024**3,remaining_after_monitoring=remaining)
 return dict(result,status="READY_FOR_FINITE_PRIVATE_POLICY_REPLAY" if ok else "WAIT_JOINT_RESOURCE_HEADROOM",eligible=ok)

def execute():
 admission=preflight()
 if not admission["eligible"]:
  print(json.dumps(admission,ensure_ascii=False));return
 # Caller must create the one named, bounded user unit; no unbounded interactive execution.
 g=load_file("aki_statistics_admission_execute",R/"work/postformal_priority_admission_20261030_v1/admission.py")
 state=g.unit_state(UNIT)
 assert state["ActiveState"]=="active" and int(state["MainPID"])==os.getpid(),"Exact unit owner required"
 assert 512*1024**2<=int(state["MemoryMax"])<=BUDGET,"Finite 512-768MiB cgroup required"
 cgroup=Path("/sys/fs/cgroup")/state["ControlGroup"].lstrip("/")
 assert (cgroup/"memory.swap.max").read_text().strip()=="0","Swap-free owned unit required"
 os.sched_setaffinity(0,{admission["resource_plan"]["CPU_affinity"][-1]})
 for name in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
  os.environ[name]="1"
 os.environ["CUDA_VISIBLE_DEVICES"]=""
 resource.setrlimit(resource.RLIMIT_CPU,(180,180))
 if O.exists():raise RuntimeError("Existing replay outcome must be inspected; no automatic overwrite or retry")
 O.mkdir(mode=0o700,parents=True)
 owner_file=(O/"OWNER.lock").open("w")
 fcntl.flock(owner_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.monotonic()
 save(O/"ADMISSION.json",admission)
 try:
  scalar=load_file("aki_independent_scalar_replay",W/"audit_fixed_policy_records.py")
  for p,h in scalar.PINS.items():assert sha(p)==h,str(p)
  protection=scalar.protection()
  snapshot={v["path"]:v["sha256"] for v in read(INPUT/"PRIVATE_MAIN_EVALUATION_DEPENDENCIES.json")}
  for row in read(INPUT/"UNBOUND_POLICY_RISK_FILES_REQUIRING_REPLAY.json"):snapshot[row["path"]]=row["sha256"]
  for name,h in snapshot.items():assert sha(scalar.path(name))==h,name
  # These are our own previously produced, original-SHA-bound small sklearn calibrators.
  # No unknown pickle, trained clinical predictor, torch backend or GPU is loaded.
  import numpy as np
  import joblib
  from sklearn.linear_model import LogisticRegression
  contracts=read(INPUT/"PRIVATE_PARTITION_REPLAY_CONTRACTS.json")
  rows_out=[];maximum=0.;cells=0;random_policies=0;loaded=0
  for contract in contracts:
   part=contract["partition"];dest=R/"outputs/formal_evaluation_20261023_v1"/part
   banks={}
   canonical=None;metadata=None
   for item in contract["banks"]:
    header,records=scalar.read_csv(scalar.path(item["file"]))
    keys=[x for x in header if x not in ("prob_raw","prob_calibrated")]
    values=[tuple(r[k] for k in keys) for r in records]
    if canonical is None:canonical=values;metadata=keys
    else:assert keys==metadata and canonical==values
    banks[item["model_lock"]]={"ids":scalar.identities(records),"risks":scalar.probabilities(records,"prob_calibrated")}
   for item in contract["policies"]:
    lock=read(scalar.path(item["lock"]));ident=lock["identity"];name=item["name"]
    base=banks[scalar.rel(scalar.path(lock["base_model"]["path"]))]
    upgrade=banks[scalar.rel(scalar.path(lock["upgrade_model"]["path"]))]
    ids=base["ids"];bp=base["risks"];up=upgrade["risks"];n=len(bp)
    fraction=ident["upgrade_fraction"]
    if ident["kind"]=="random":
     assert ident["mode"]=="capacity"
     k=math.ceil(n*fraction)
     canonical_indices=sorted(range(n),key=lambda i:ids[i])
     # Locked allocation seed, distinct from patient bootstrap seed 20261017.
     draws=np.random.Generator(np.random.PCG64(20261021)).choice(n,k,replace=False)
     chosen={canonical_indices[int(j)] for j in draws}
     mask=[i in chosen for i in range(n)];random_policies+=1
    else:
     assert ident["kind"]=="selective"
     mask=scalar.selective_allocation(bp,ids,lock["gate"],fraction,ident["mode"])
    _,target=scalar.read_csv(dest/(name+"_PRIVATE.csv"))
    assert [tuple(r[k] for k in metadata) for r in target]==canonical
    assert all(r["upgraded"] in ("True","False") for r in target)
    assert mask==[r["upgraded"]=="True" for r in target],(part,name,"allocation")
    mixed=[u if select else b for b,u,select in zip(bp,up,mask)]
    original_mixed=scalar.probabilities(target,"prob_raw_mixture")
    assert mixed==original_mixed,(part,name,"constituent mixture")
    calpath=scalar.path(item["calibrator"])
    assert sha(calpath)==lock["calibrator"]["sha256"]
    cal=joblib.load(calpath);loaded+=1
    assert type(cal) is LogisticRegression
    assert cal.classes_.tolist()==[0,1] and cal.coef_.shape==(1,1) and cal.intercept_.shape==(1,)
    assert cal.n_features_in_==1
    intercept=float(cal.intercept_[0]);slope=float(cal.coef_[0,0])
    assert math.isfinite(intercept) and math.isfinite(slope)
    reconstructed=[]
    for p in mixed:
     v=min(1-1e-6,max(1e-6,p))
     eta=intercept+slope*math.log(v/(1-v))
     probability=1/(1+math.exp(-eta)) if eta>=0 else math.exp(eta)/(1+math.exp(eta))
     reconstructed.append(probability)
    old=scalar.probabilities(target,"prob_calibrated")
    error=max(abs(a-b) for a,b in zip(reconstructed,old))
    assert error<=TOL,(part,name,"calibrator reconstruction",error)
    maximum=max(maximum,error);cells+=n
    rows_out.append({"partition":part,"policy":name,"n":n,"allocation_verified_exactly":True,
      "constituent_mixture_verified_exactly":True,"calibrator_sha256":sha(calpath),
      "calibrator_classes":[0,1],"calibrator_negative_slope":slope<0,
      "calibrated_risk_maximum_absolute_difference":error,"target_sha256":sha(dest/(name+"_PRIVATE.csv"))})
    del cal,target
  assert len(rows_out)==24 and random_policies==8 and loaded==24
  for name,h in snapshot.items():assert sha(scalar.path(name))==h,name
  assert scalar.protection()==protection
  save(O/"POLICY_RECONSTRUCTION_INTERNAL.json",rows_out)
  receipt={"status":"PASS_FIXED_POLICY_RECONSTRUCTION_NOT_BOOTSTRAP_OR_PACKAGE_ACCEPTANCE",
   "time":now(),"code_lock_sha256":sha(LOCK),"scalar_review_receipt_sha256":sha(SCALAR),
   "policy_variants":24,"selective_masks":16,"random_masks":8,"trusted_calibrators_reloaded":24,
   "calibrated_risk_cells_compared":cells,"maximum_absolute_calibrated_risk_difference":maximum,
   "absolute_tolerance":TOL,"relative_tolerance":0,"calibrators_or_models_refitted":0,
   "original_files_verified_unchanged":len(snapshot),"manuscript_files_verified_unchanged":92,
   "patient_bootstrap_intervals_recomputed":False,"scientific_review_or_whole_package_acceptance":False,
   "scope":"New supplementary verification of formerly unbound policy CSVs; not a historical COMPLETE hash and not a rerouted bootstrap replay.",
   "seconds":time.monotonic()-start,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   "CPU_affinity":sorted(os.sched_getaffinity(0)),"output_hashes":{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!="OWNER.lock"}}
  save(O/"FIXED_POLICY_REPLAY_ACCEPTANCE.json",receipt)
  print(json.dumps({k:receipt[k] for k in ["status","policy_variants","calibrated_risk_cells_compared","maximum_absolute_calibrated_risk_difference","seconds","peak_RSS_bytes"]}))
 except Exception as error:
  save(O/"FAILURE.json",{"time":now(),"error":type(error).__name__+": "+str(error),"automatic_retry_allowed":False,"preserve_original_inputs":True})
  raise
 finally:
  fcntl.flock(owner_file,fcntl.LOCK_UN);owner_file.close()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--run",action="store_true");args=parser.parse_args()
 if args.run:execute()
 else:print(json.dumps(preflight(),ensure_ascii=False))
if __name__=="__main__":main()
