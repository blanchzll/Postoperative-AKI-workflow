"""Finite independent scalar verification and exact same-source bootstrap reuse check."""
from pathlib import Path
import csv,datetime,hashlib,importlib.util,json,math,os,resource,signal,time
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/fixed_flow_sensitivity_evaluation_current_20261030_r1"
O=R/"outputs/fixed_flow_sensitivity_evaluation_current_20261030_r1"
C=R/"outputs/reanalysis_control/fixed_flow_mask_correction_20261030_r1"
FIT=R/"outputs/fixed_flow_sensitivity_20261024_v1"
OLD=R/"outputs/fixed_flow_sensitivity_evaluation_20261024_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 with Path(p).open("x") as f:
  json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run():
 os.umask(0o077);resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,)*2)
 resource.setrlimit(resource.RLIMIT_CPU,(90,)*2);signal.alarm(180)
 os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
 completed=O/"NUMERICAL_CORRECTION_COMPLETION.json"
 if not completed.exists():
  print("WAIT_ACTUAL_TWO_CONTEXT_NUMERICAL_CORRECTION");return
 if (O/"STAGE_EVALUATION_ACCEPTANCE.json").exists():raise RuntimeError("Existing acceptance requires inspection")
 start=time.monotonic();driver=module("_corrected_driver",W/"run_correction.py");driver.code_guard()
 for rel,h in read(W/"VERIFIER_CODE_LOCK.json")["files"].items():assert sha(R/rel)==h
 receipt=read(completed);assert len(receipt["receipts"])==8 and len(receipt["recomputed_receipts"])==2
 scalar=module("_corrected_scalar",R/"work/study_package_scope_20261030_v1/audit_fixed_policy_records.py")
 protection=scalar.protection();checks=0;maximum=0.0;scope=[]
 def same(a,b):
  nonlocal checks,maximum
  checks+=1
  if a is None or b is None:assert a is None and b is None;return
  if isinstance(a,bool) or isinstance(b,bool):assert a==b;return
  if isinstance(a,(float,int)) and isinstance(b,(float,int)):
   assert math.isfinite(a) and math.isfinite(b)
   error=abs(a-b);maximum=max(maximum,error);assert error<=1e-12,(a,b,error)
  else:assert a==b
 def same_payload(a,b):
  assert set(a)==set(b)
  for k in a:same(a[k],b[k])
 snapshot=read(C/"ORIGINAL_BACKUP_MANIFEST_INTERNAL.json")
 for rel,h in snapshot.items():
  if rel in ["outputs/reanalysis_control/CURRENT_HANDOFF_ZH.md","outputs/post_reanalysis_process_review_v1/REQUIRED_COMPONENTS.json"]:continue
  assert sha(R/rel)==h
 contexts=read(C/"CONTEXT_IMPACT_AND_RECOMPUTE_PLAN.json")
 for ctx in contexts:
  p=O/(ctx["line"]+"_"+ctx["cohort"])/ctx["scenario"]/"COMPLETE.json" if ctx["requires_numerical_recomputation"] else R/ctx["old_receipt"]
  assert sha(p)==receipt["receipts"][str(p)]
  d=driver.verify_context(p,ctx)
  if not ctx["requires_numerical_recomputation"]:
   scope.append(dict(line=ctx["line"],cohort=ctx["cohort"],scenario=ctx["scenario"],disposition="retained_original_numerically_unchanged"))
   continue
  current=read(p.parent/"EVALUATION.json")
  reference_path=OLD/(ctx["line"]+"_"+ctx["cohort"])/"omit_ALT"/"COMPLETE.json"
  reference_receipt=read(reference_path)
  assert sha(reference_path)==snapshot[str(reference_path.relative_to(R))]
  for name,h in reference_receipt["files"].items():assert sha(reference_path.parent/name)==h
  reference=read(reference_path.parent/"EVALUATION.json")
  a,b=current["receipt"],reference["receipt"]
  for key in ["replicates","seed","resample_weight_sha256","zero_event_replicates","all_event_replicates","group_unit_within_hospital"]:
   same(a[key],b[key])
  assert a["replicates"]==2000 and a["seed"]==20261017 and a["group_unit_within_hospital"]
  interval_ref={(x["model"],x["metric"]):x for x in reference["intervals"]}
  def control(name):
   route,scenario=name.rsplit("__",1)
   assert scenario in ["matched_control","exclude_high_baseline_Cr"]
   return route+"__matched_control"
  for name,point in current["point"].items():
   route,scenario=name.rsplit("__",1)
   rec=read(FIT/"routes"/route/scenario/"COMPLETE.json")
   ctrl=read(FIT/"routes"/route/"matched_control"/"COMPLETE.json")
   assert rec["fit_cache"]==ctrl["fit_cache"] and rec["exclude_evaluation_rows"]==[]
   cache=Path(rec["fit_cache"]);accepted=read(cache/"COMPLETE.json")
   assert sha(cache/"COMPLETE.json")==rec["fit_receipt_sha256"]
   predpath=cache/"PRIVATE_HELDOUT_PREDICTIONS.csv"
   assert sha(predpath)==accepted["files"]["PRIVATE_HELDOUT_PREDICTIONS.csv"]
   data=rows(predpath);ids=scalar.identities(data);y=[int(x["y"]) for x in data]
   pred=scalar.probabilities(data,"prob_calibrated")
   assert len(data)==ctx["corrected_n"] and sum(y)==ctx["corrected_events"]
   independently=scalar.scalar_metrics(y,pred,ids)
   for metric,value in independently.items():same(value,point[metric])
   same_payload(point,reference["point"][control(name)])
  for item in current["intervals"]:
   original=interval_ref[(control(item["model"]),item["metric"])]
   same_payload({k:v for k,v in item.items() if k!="model"},{k:v for k,v in original.items() if k!="model"})
  pairs={(x["model"],x["metric"]):x for x in reference["paired_differences"]}
  for item in current["paired_differences"]:
   left,right=item["model"].split(" MINUS ");left,right=control(left),control(right);metric=item["metric"]
   if left!=right:
    original=pairs[(left+" MINUS "+right,metric)]
    same_payload({k:v for k,v in item.items() if k!="model"},{k:v for k,v in original.items() if k!="model"})
   else:
    valid=interval_ref[(left,metric)]["valid_replicates"]
    expected=dict(estimate=0.0 if reference["point"][left][metric] is not None else None,
                  lower95=0.0 if valid else None,upper95=0.0 if valid else None,
                  valid_replicates=valid,requested_replicates=2000,metric=metric)
    same_payload({k:v for k,v in item.items() if k!="model"},expected)
  scope.append(dict(line=ctx["line"],cohort=ctx["cohort"],scenario=ctx["scenario"],
       disposition="recomputed_and_verified",
       reference_same_source_matched_control=str(reference_path.relative_to(R)),
       reference_receipt_sha256=sha(reference_path),
       bootstrap_weight_sha256=a["resample_weight_sha256"],
       not_independent_additional_patient_cohort=True))
 assert scalar.protection()==protection
 proof=dict(status="PASS_SCALAR_RECOMPUTATION_AND_SAME_INPUT_BOOTSTRAP_CONCORDANCE",
     time=datetime.datetime.now(datetime.timezone.utc).isoformat(),numeric_checks=checks,max_abs=maximum,
     contexts=scope,original92_unchanged=True,new_fits=0,
     independent_scalar_metrics=True,
     interval_verification="Original current-source matched-control2000 results with exact input/prediction and bootstrap-weight identity, not a new independent cohort or new third bootstrap.",
     seconds=time.monotonic()-start,max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 write(O/"INDEPENDENT_CORRECTION_VERIFICATION.json",proof)
 write(O/"STAGE_EVALUATION_ACCEPTANCE.json",dict(
     status="PASS_eight_current_source_matched_refit_sensitivity_evaluations",
     receipts=receipt["receipts"],recomputed_contexts=2,retained_unchanged_contexts=6,
     current_source_unverified_high_baseline_count=0,
     empty_exclusion_scenario_adds_no_independent_robustness_evidence=True,
     correction_diagnosis_sha256=sha(C/"DIAGNOSIS.json"),
     correction_completion_sha256=sha(completed),
     independent_verification_sha256=sha(O/"INDEPENDENT_CORRECTION_VERIFICATION.json"),
     source_sha256="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3",
     whole_project_complete=False,final_scientific_review=False,
     downstream_figures_report_package_must_use_corrected_context_paths=True))
 print(json.dumps({"status":proof["status"],"checks":checks,"max_abs":maximum}))
if __name__=="__main__":run()
