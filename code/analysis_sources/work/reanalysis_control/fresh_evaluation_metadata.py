"""Current-source fresh evaluation metadata; historical prediction/evaluation recovery is disabled."""
from pathlib import Path
import json,hashlib,csv,math,os
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def ensure_fresh_metadata(root):
 root=Path(root);out=root/"outputs/formal_evaluation_20261023_v1";source=root/"outputs/v4_development_inputs_20261018"
 assert sha(source/"SOURCE_LOCK.json")=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
 models=root/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json"
 policy=root/"outputs/selective_policy_lock_20261023_v1/STAGE_POLICY_LOCK_ACCEPTANCE.json"
 a=json.loads(models.read_text());b=json.loads(policy.read_text())
 assert a["status"]=="PASS" and a["final_model_locked"] and b["final_hospital_time_scoring_permitted"]
 assert b["policy_variant_count"]==24
 for ref in a["models"]+b["policy_manifests"]:assert sha(ref["path"])==ref["sha256"]
 codes=json.loads((out/"CODE_LOCK.json").read_text())
 assert codes=={p.name:sha(p) for p in (root/"work/formal_evaluation_20261023_v1").glob("*.py")}
 metadata={}
 lock=json.loads((source/"SOURCE_LOCK.json").read_text())
 for cohort in ["preop","paired"]:
  path=source/(cohort+"_metadata.csv");assert sha(path)==lock["input_sha256"][path.name]
  with path.open(newline="") as f:metadata[cohort]=list(csv.DictReader(f))
 valid_models={ref["sha256"] for ref in a["models"]}
 for recpath in out.glob("*/private_predictions/*/COMPLETE.json"):
  rec=json.loads(recpath.read_text());sig=rec["signature"];key=sig["signature"]
  assert key=={"code_lock":sha(out/"CODE_LOCK.json"),"model_acceptance":sha(models),"policy_acceptance":sha(policy),"source":sha(source/"SOURCE_LOCK.json"),"line":key["line"],"cohort":key["cohort"]}
  assert sig["model_lock"] in valid_models
  pred=recpath.parent/"PREDICTIONS.csv";assert sha(pred)==rec["sha256"]
  with pred.open(newline="") as f:rows=list(csv.DictReader(f))
  expected=[z for z in metadata[key["cohort"]] if z[key["line"]+"_partition_v2"]=="validation"]
  fields=["excel_row","group_hash","y"]
  assert [[z[k] for k in fields] for z in rows]==[[z[k] for k in fields] for z in expected]
  assert len(rows)==rec["n"] and sum(int(z["y"]) for z in rows)==rec["events"]
  for z in rows:
   for name in ["prob_raw","prob_calibrated"]:
    value=float(z[name]);assert math.isfinite(value) and 0<=value<=1
 config={"status":"fresh_current_source_evaluation_no_historical_recovery",
 "source_lock_sha256":sha(source/"SOURCE_LOCK.json"),
 "current_code_lock_sha256":sha(out/"CODE_LOCK.json"),
 "model_acceptance_sha256":sha(models),"policy_acceptance_sha256":sha(policy),
 "completed_evaluations":{},"prediction_receipts":{},"previous_code_lock_sha256":None,
 "unchanged_statistics_sha256":sha(root/"work/formal_evaluation_20261023_v1/evaluation_stats.py"),
 "legacy_prediction_or_completed_evaluation_reuse_allowed":False,
 "current_signature_predictions_reused_only_after_hash_and_source_identity_verification":True,
 "all_bootstrap_evaluations_must_be_computed_on_current_data":True}
 target=out/"DATE_RECOVERY_COMPATIBILITY.json"
 if target.exists():assert json.loads(target.read_text())==config,"Changed fresh-evaluation metadata"
 else:
  tmp=target.with_name(target.name+".tmp");tmp.write_text(json.dumps(config,ensure_ascii=False,indent=2)+chr(10));os.replace(tmp,target)
 return config
