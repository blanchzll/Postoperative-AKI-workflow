"""Seal this analysis component only after actual result and pixel reading; not full-study scientific acceptance."""
import argparse
from pathlib import Path
import common as c
def run():
 c.check_code();c.numeric_guard()
 with c.owned_lock("finalize"):
  names=("INDEPENDENT_RESULT_VERIFICATION.json","RESOURCE_REPORT_ACCEPTANCE.json","RENDER_AND_ANALYSIS_REPORT.json")
  receipts={}
  for name in names:
   p=c.O/name;a=c.read(p);assert a["code_lock_sha256"]==c.check_code()
   c.verify_files(c.O,a["inputs_sha256"]);c.verify_files(c.O,a["files"]);receipts[str(p)]=c.sha(p)
  rendered=c.read(c.O/"RENDER_AND_ANALYSIS_REPORT.json")
  p=c.O/"ANALYST_INTERPRETATION_AND_VISUAL_REVIEW.json"
  assert p.exists(),"Actual result reading and six PNG inspections must be recorded; this program cannot perform scientific judgment"
  review=c.read(p)
  assert review["status"]=="COMPLETE_ACTUAL_LLM_RESULT_AND_PIXEL_READING"
  assert review["all_24_calibrated_routes_read"] is True
  assert review["all_54_predeclared_comparisons_read"] is True
  assert review["raw_and_calibrated_diagnostics_read"] is True
  assert review["negative_null_and_heterogeneous_results_retained"] is True
  assert review["primary_winner_unchanged"] is True
  expected={k:v for k,v in rendered["files"].items() if k.endswith(".png")}
  assert len(expected)==6 and review["viewed_png_sha256"]==expected
  c.verify_files(c.O,expected);c.verify_files(c.O,review["read_artifacts_sha256"])
  required={str(c.O/"LLM_ANALYSIS_REPORT_ZH.md"),str(c.O/"SOURCE_CALIBRATION_INDEPENDENT.csv"),
            str(c.O/"MODEL_RESOURCE_SUMMARY.csv")}
  required|={str(c.S/l/n) for l in c.LINES for n in ("POINT_ESTIMATES.csv","INTERVALS.csv","PAIRED_DIFFERENCES.csv")}
  assert required<=set(review["read_artifacts_sha256"])
  assert set(review["context_interpretation"])==set(c.LINES)
  for line,observations in review["context_interpretation"].items():
   assert isinstance(observations,list) and observations
   for item in observations:
    assert all(isinstance(item.get(k),str) and item[k].strip() for k in ("observation","evidence","scope","remaining_limit"))
  assert isinstance(review["pixel_observations"],list) and len(review["pixel_observations"])==6
  assert review["full_study_scientific_review_complete"] is False
  receipts[str(p)]=c.sha(p)
  destination=c.L/"COMPARISON_ACCEPTANCE.json"
  if destination.exists():
   old=c.read(destination);assert old["receipts"]==receipts
   print("UNCHANGED_ALREADY_SEALED");return
  c.save(destination,{"status":"PASS_three_locked_LLMs_pretrained_finetuned_paired_equal_capacity_supplement",
    "time":c.now(),"source_lock_sha256":c.SOURCE_SHA,"formal_training_contract_sha256":c.FORMAL_SHA,
    "formal_training_acceptance":str(c.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"),
    "formal_training_acceptance_sha256":c.sha(c.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"),
    "numerical_acceptance":str(c.S/"NUMERICAL_COMPARISON_ACCEPTANCE.json"),
    "numerical_acceptance_sha256":c.sha(c.S/"NUMERICAL_COMPARISON_ACCEPTANCE.json"),
    "code_lock":str(c.W/"DELIVERY_CODE_LOCK.json"),"code_lock_sha256":c.check_code(),
    "receipts":receipts,"models":list(c.MODELS),"result_routes":24,"formal_fits":48,"before_banks":6,
    "prior_evaluation_exposure_and_exploratory_multiplicity_disclosed":True,
    "source_specific_calibration_not_label_free_zero_shot":True,"main_winner_unchanged":True,
    "actual_analyst_record_does_not_replace_final_three_stage_scientific_review":True,
    "final_scientific_review_complete":False})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");a=p.parse_args()
 if not a.run:raise SystemExit("Use --run only after actual result and visual reading")
 run()
