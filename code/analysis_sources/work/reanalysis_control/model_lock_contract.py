"""Build current-source model-lock metadata without changing numerical methods or importing historical results."""
from pathlib import Path
import json,hashlib,csv,os
SOURCE_SHA="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
METHOD_REFS={
"outputs/research_plan_revision_20261018_094007/PROTOCOL.json":"15156b8b29d1a92e6b41bd1b546809f01c693ebf577e9548d9578b8fdc537b9b",
"outputs/research_plan_revision_20261018_094007/PARAMETER_SEARCH_SPACE.json":"3afd015f8fa1976e3cada901fa04c15ddf1610f82120815bb0523dd575253f9e",
"outputs/research_plan_revision_20261018_094007/CURRENT_RESEARCH_PLAN_ZH.md":"c107c3002f69a09d3724f72c97f24ff335a797866e7be077026f00d3041fc66f"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ensure_contract(root):
 root=Path(root)
 source=root/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"
 assert sha(source)==SOURCE_SHA,"Changed corrected source"
 for rel,h in METHOD_REFS.items():assert sha(root/rel)==h,("Changed method reference",rel)
 e=root/"outputs/ensemble_calibration_20261020_v1"
 assert json.loads((e/"STAGE_E_ACCEPTANCE.json").read_text())["status"]=="PASS"
 st=json.loads((e/"status_e.json").read_text())
 assert st["state"]=="complete_strict_calibration_internal_evaluation"
 assert st["complete"]==st["total"]==29988 and st["task_owned_workers_remaining"]==0
 current=json.loads((e/"CODE_LOCK.json").read_text())
 assert current=={p.name:sha(p) for p in (root/"work/ensemble_calibration_20261020_v1").glob("*.py")}
 counts={}
 for cohort,expected in [("preop",(3710,205)),("paired",(2428,138))]:
  with (source.parent/(cohort+"_metadata.csv")).open(newline="") as f:rows=list(csv.DictReader(f))
  actual=(len(rows),sum(int(row["y"]) for row in rows))
  assert actual==expected,(cohort,actual)
  counts[cohort]={"n":actual[0],"events":actual[1]}
 spec=Path(__file__).with_name("model_lock_method_contract.json")
 method=json.loads(spec.read_text())
 contract={**method["method_fields"],
 "status":"current_corrected_source_execution_contract_before_final_model_fits",
 "run_root":str(root),"source_lock_sha256":SOURCE_SHA,"cohort_counts":counts,
 "current_method_reference_sha256":METHOD_REFS,
 "reference_text_contains_historical_counts_only":True,
 "E_source_code_lock_sha256":sha(e/"CODE_LOCK.json"),
 "E_acceptance_sha256":sha(e/"STAGE_E_ACCEPTANCE.json"),
 "method_contract_spec_sha256":sha(spec),
 "historical_method_template_sha256":method["historical_method_template_sha256"],
 "metadata_preparation_code_sha256":sha(__file__),
 "new_cases_experiments_or_scientific_scope_changes":False,
 "historical_model_prediction_or_outcome_receipts_imported":False,
 "evaluation_exposure":"Previously exposed; not new blinded validation",
 "pending_analyses":"All authorized current 36 components; POD1 and TabPFN saved fits reused subject to their current-source guards",
 "manuscript_prose_work_authorized_in_this_chat":False}
 out=root/"outputs/model_lock_20261022_v1/EXECUTION_CONTRACT.json"
 if out.exists():assert json.loads(out.read_text())==contract,"Changed existing execution contract"
 else:
  out.parent.mkdir(parents=True,exist_ok=True)
  temp=out.with_name(out.name+".tmp")
  temp.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n")
  os.replace(temp,out)
 return contract
