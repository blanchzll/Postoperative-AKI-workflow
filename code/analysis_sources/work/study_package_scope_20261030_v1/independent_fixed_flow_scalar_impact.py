"""Independent scalar impact of the confirmed stale source-exclusion bug."""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,resource,signal,time
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
C=R/"outputs/reanalysis_control/fixed_flow_mask_correction_20261030_r1"
F=R/"outputs/fixed_flow_sensitivity_20261024_v1"
OLD=R/"outputs/fixed_flow_sensitivity_evaluation_20261024_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def run():
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CPU,(60,)*2)
 signal.alarm(120);os.sched_setaffinity(0,{max(os.sched_getaffinity(0))});start=time.monotonic()
 output=C/"INDEPENDENT_SCALAR_IMPACT.json"
 if output.exists():raise RuntimeError("Existing impact calculation must be inspected")
 path=R/"work/study_package_scope_20261030_v1/audit_fixed_policy_records.py"
 assert sha(path)=="3df4e9868d848a6cecddbe2b5bd43262a9b5ce69766c356a9925d2c15172e39b"
 spec=importlib.util.spec_from_file_location("_scalar_impact",path);scalar=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)
 protected=scalar.protection();snapshot=read(C/"ORIGINAL_BACKUP_MANIFEST_INTERNAL.json")
 routes=read(F/"ROUTES.json");result=[];sources={};checks=0;err=0.
 for cohort in ["preop","paired"]:
  parent=OLD/("temporal_"+cohort)
  for scenario in ["omit_ALT","exclude_high_baseline_Cr"]:
   complete=parent/scenario/"COMPLETE.json"
   assert sha(complete)==snapshot[str(complete.relative_to(R))]
   d=read(complete);ep=complete.parent/"EVALUATION.json"
   assert sha(ep)==d["files"]["EVALUATION.json"];sources[str(ep.relative_to(R))]=sha(ep)
  correct=read(parent/"omit_ALT/EVALUATION.json")["point"]
  historical=read(parent/"exclude_high_baseline_Cr/EVALUATION.json")["point"]
  for route,ref in routes.items():
   if ref["identity"]["line"]!="temporal" or ref["identity"]["cohort"]!=cohort:continue
   rec=read(F/"routes"/route/"exclude_high_baseline_Cr/COMPLETE.json")
   control=read(F/"routes"/route/"matched_control/COMPLETE.json")
   assert rec["fit_cache"]==control["fit_cache"] and not rec["exclude_evaluation_rows"]
   cache=Path(rec["fit_cache"]);accepted=read(cache/"COMPLETE.json")
   assert sha(cache/"COMPLETE.json")==rec["fit_receipt_sha256"]
   pred=cache/"PRIVATE_HELDOUT_PREDICTIONS.csv"
   assert sha(pred)==accepted["files"]["PRIVATE_HELDOUT_PREDICTIONS.csv"]
   sources[str(pred.relative_to(R))]=sha(pred)
   _,rows=scalar.read_csv(pred);y=[int(x["y"]) for x in rows]
   current=scalar.scalar_metrics(y,scalar.probabilities(rows,"prob_calibrated"),scalar.identities(rows))
   reference=correct[route+"__matched_control"];old=historical[route+"__exclude_high_baseline_Cr"]
   for metric,value in current.items():
    if value is None:assert reference[metric] is None
    else:
     difference=abs(value-reference[metric]);err=max(err,difference);assert difference<=1e-12
    checks+=1
    result.append({"route":route,"cohort":cohort,"metric":metric,"old_wrong_subset":old[metric],
                   "current_full_validation_scalar":value,
                   "delta_current_minus_old":value-old[metric] if value is not None and old[metric] is not None else None})
 assert len({x["route"] for x in result})==5 and scalar.protection()==protected
 proof={"status":"PASS_INDEPENDENT_SCALAR_IMPACT_ONLY_BOOTSTRAP_CORRECTION_PENDING",
 "scalar_metrics_checked":checks,"unique_routes":5,"max_abs_difference_from_existing_valid_current_control":err,
 "source_hashes":sources,"all_metrics_no_outcome_selection":result,"new_fits":0,"new_bootstraps":0,
 "formal_37_component_correction_accepted":False,"92_protected_unchanged":True,
 "seconds":time.monotonic()-start,"max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 with output.open("x") as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 print(json.dumps({k:v for k,v in proof.items() if k not in ["all_metrics_no_outcome_selection","source_hashes"]}))
if __name__=="__main__":run()
