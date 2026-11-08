"""Read-only completed-current-POD1 identity reuse. Never regenerate inputs or pilots."""
from pathlib import Path
import json,hashlib,time,os
import pandas as pd
R=Path(__file__).resolve().parents[2]
F=R/"outputs/pod1_corrected_fixed_update_20261024_v1"
A=R/"outputs/resource_acceleration_20261026_v1/pod1_fixed_advance"
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024**2),b""):h.update(b)
 return h.hexdigest()
def verify_files(parent,files):
 for f,h in files.items():assert sha(Path(parent)/f)==h,str(Path(parent)/f)
def assert_prediction_alignment(data,valid_rows,predictions):
 ids=predictions.excel_row.tolist()
 assert len(ids)==len(valid_rows) and len(set(ids))==len(ids)
 assert set(ids)==set(valid_rows),"Predictions must contain exactly the requested records"
 assert predictions.y.tolist()==data.loc[ids,"y"].tolist(),"Prediction/source label mismatch by record"
 assert predictions.group_hash.tolist()==data.loc[ids,"group_hash"].tolist(),"Prediction/source group mismatch by record"
def verify():
 advanced=read(A/"FIXED_ADVANCE_ACCEPTANCE.json")
 assert advanced["status"]=="PASS_original_CUDA_POD1_fixed_fits_completed_evaluation_pending"
 acceptance=F/"STAGE_POD1_FIT_ACCEPTANCE.json"
 assert sha(acceptance)==advanced["fit_receipt_sha256"]
 assert sha(R/"outputs/pod1_update_20261017/DEVELOPMENT_REUSE_SEAL.json")==advanced["development_seal_sha256"]
 code_sha=sha(F/"CODE_LOCK.json");input_sha=sha(F/"INPUT_ACCEPTANCE.json")
 verify_files(R/"work/pod1_corrected_fixed_update_20261024_v1",read(F/"CODE_LOCK.json"))
 verify_files(R,read(F/"DEPENDENCIES.json"))
 inp=read(F/"INPUT_ACCEPTANCE.json")
 assert inp["status"]=="PASS_POD1_2947_39_corrected_source_and_rebuilt_group_partitions"
 assert (inp["eligible_n"],inp["events"])==(2947,39)
 verify_files(F,inp["files"]);verify_files(R,read(F/"SOURCE_FILES.json"))
 q=read(F/"QUALIFICATION.json")
 assert q["status"]=="PASS_POD1_current_inputs_fixed_choices_nested_group_boundaries_and_CUDA"
 assert q["code_lock_sha256"]==code_sha and q["actual_XGB_device"].startswith("cuda")
 launch=read(A/"ACCELERATION_LAUNCH_ACCEPTANCE.json")
 assert sha(F/"QUALIFICATION.json")==launch["qualification_sha256"]
 pilot=read(F/"PILOT_ACCEPTANCE.json")
 assert pilot["status"]=="PASS_real_development_LR_and_nested_stack_pilots"
 assert pilot["actual_nested_raw_requests"]>0 and pilot["all_parent_and_child_target_groups_excluded"]
 verify_files(F/"tasks",{str(i).zfill(3)+"/COMPLETE.json":h for i,h in pilot["reusable_complete_receipts"].items()})
 data=pd.read_csv(F/"PRIVATE_CURRENT_POD1_COHORT.csv",dtype={"group_hash":str},float_precision="round_trip").set_index("excel_row")
 assert data.index.is_unique and len(data)==2947 and int(data.y.sum())==39
 cohort_sha=sha(F/"PRIVATE_CURRENT_POD1_COHORT.csv")
 tasks=read(F/"TASKS.json");fit=read(acceptance)
 assert fit["status"]=="PASS_132_fixed_choice_POD1_fitting_contexts"
 assert len(tasks)==len(fit["receipts"])==q["contexts"]
 verify_files(R,fit["receipts"]);raw_refs={};finals=[];max_roundtrip=0.
 for i,t in enumerate(tasks):
  tr=t["train_rows"];va=t["valid_rows"]
  assert len(tr)==len(set(tr)) and len(va)==len(set(va))
  assert set(data.loc[tr,"group_hash"]).isdisjoint(data.loc[va,"group_hash"])
  d=F/"tasks"/str(i).zfill(3);rec=read(d/"COMPLETE.json")
  assert rec["signature"]=={"task":t,"code":code_sha,"inputs":input_sha}
  verify_files(d,rec["files"]);verify_files(R,rec["raw_fit_receipts"]);raw_refs.update(rec["raw_fit_receipts"])
  z=pd.read_csv(d/"PRIVATE_PREDICTIONS.csv",dtype={"group_hash":str},float_precision="round_trip")
  assert_prediction_alignment(data,va,z)
  assert rec["heldout_scores_not_evaluated"]
  if t["role"]=="final":
   finals.append(i);max_roundtrip=max(max_roundtrip,float(rec.get("diagnostic",{}).get("max_single_batch_abs",0)))
 counts={};upstream={}
 for area in ["raw_cache","imputation_cache"]:
  records=list((F/area).glob("*/COMPLETE.json"));counts[area]=len(records)
  for p in records:
   r=read(p);sig=r["signature"]
   assert sig["cohort"]==cohort_sha and sig["code"]==code_sha
   assert set(data.loc[sig["train"],"group_hash"]).isdisjoint(data.loc[sig["valid"],"group_hash"])
   verify_files(p.parent,r["files"])
   refs=r.get("upstream_receipts",{});verify_files(R,refs);upstream.update(refs)
 assert len(finals)==fit["final_models"]
 locks=read(F/"MODEL_LOCKS.json");assert len(locks["models"])==len(finals)
 for i,z in locks["models"].items():
  assert z["task"]==tasks[int(i)] and sha(z["receipt"])==z["sha256"]
 return {"status":"PASS_completed_current_source_POD1_fixed_fits_reverified_without_refitting",
  "contexts":len(tasks),"final_models":len(finals),"unique_direct_raw_receipts":len(raw_refs),
  "cache_receipts_checked":counts,"upstream_receipts_checked":len(upstream),
  "fit_acceptance_sha256":sha(acceptance),"code_lock_sha256":code_sha,"input_acceptance_sha256":input_sha,
  "POD1_development_seal_sha256":advanced["development_seal_sha256"],
  "single_batch_diagnostic_max_absolute_difference":max_roundtrip,
  "single_batch_diagnostic_is_not_calculator_qualification":True,
  "all_task_labels_groups_and_patient_boundaries_checked":True,
  "qualified_pilots_reused_without_regenerating_trace":True,"no_model_or_input_file_rewritten":True,
  "heldout_metrics_not_evaluated":True,"whole_POD1_evaluation_complete":False,"scientific_review_performed":False,
  "at":time.time()}
if __name__=="__main__":
 result=verify();p=A/"FIXED_CHECKPOINT_REUSE_VERIFICATION.json";q=p.with_suffix(".tmp")
 q.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(q,p)
 print("POD1_FIXED_CHECKPOINTS_VERIFIED",result["contexts"],result["final_models"],result["cache_receipts_checked"])
