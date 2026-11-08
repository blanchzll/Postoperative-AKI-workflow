from pathlib import Path
import csv,gzip,hashlib,json,os,datetime,fcntl,resource
from zoneinfo import ZoneInfo
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot");A=R/"outputs/novelty_strengthening_20261101_v1"
def now():return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
def save(p,j):
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+"\n");tmp.chmod(0o600);os.replace(tmp,p)
with (A/"RUNNER.lock").open("a+") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pid=os.getpid();s=Path("/proc/self/stat").read_text();start=s[s.rfind(")")+2:].split()[19]
 save(A/"RUN_OWNER.json",{"pid":pid,"start_ticks":start,"unit":"aki-novelty-source-schema-20261101-v1.service","time":now(),"scope":"Read-only source presence, schema and frozen manuscript-source identity audit; no outcomes or prediction scoring."})
 N=R.parents[1]/"writing_runs/novelty_revision_20261101_v1"
 baseline=json.loads((N/"BASELINE_INPUTS.json").read_text())
 matches=[]
 for ps,info in baseline["source_files"].items():
  p=Path(ps);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==info["sha256"],ps;matches.append({"path":ps,"sha256":h,"bytes":p.stat().st_size})
 z=Path(baseline["final_delivery_receipt"]["archive"]);h=hashlib.sha256(z.read_bytes()).hexdigest();assert h==baseline["final_delivery_receipt"]["archive_sha256"]
 candidates={
 "MIMIC_local_existing":{"root":Path("/home/lei/physionet.org/files/mimic"),"files":["hosp/patients.csv.gz","hosp/admissions.csv.gz","hosp/diagnoses_icd.csv.gz","hosp/d_icd_diagnoses.csv.gz","hosp/procedures_icd.csv.gz","hosp/d_icd_procedures.csv.gz","hosp/labevents.csv.gz","hosp/d_labitems.csv.gz","hosp/transfers.csv.gz","hosp/hcpcsevents.csv.gz","icu/icustays.csv.gz","icu/procedureevents.csv.gz","icu/d_items.csv.gz"]},
 "eICU_local_existing":{"root":Path("/home/lei/physionet.org/files/eicu/eicu-collaborative-research-database-2.0"),"files":["patient.csv.gz","admissionDx.csv.gz","diagnosis.csv.gz","treatment.csv.gz","pastHistory.csv.gz","lab.csv.gz","apachePatientResult.csv.gz","apachePredVar.csv.gz","apacheApsVar.csv.gz"]}
 }
 output={}
 for name,cfg in candidates.items():
  rec=[]
  for fn in cfg["files"]:
   p=cfg["root"]/fn
   if not p.exists():rec.append({"relative_path":fn,"exists":False});continue
   with gzip.open(p,"rt",newline="") as f:header=next(csv.reader(f))
   rec.append({"relative_path":fn,"exists":True,"resolved_path":str(p.resolve()),"compressed_bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns,"columns":header,"header_sha256":hashlib.sha256(json.dumps(header).encode()).hexdigest(),"patient_rows_read_or_saved":0})
  output[name]={"source_root":str(cfg["root"]),"resolved_root":str(cfg["root"].resolve()),"files":rec,"qualified_for_independent_external_evaluation":False,"pending":["source exact version/provenance and current access terms","whether previously evaluated in this research or overlapping derivative","GI cancer resection phenotype in available diagnosis/procedure information","valid preoperative or operative end timestamp rather than assumed midnight","baseline and postoperative SCr availability and comparable endpoint","mapping of frozen6/9 inputs and case/patient unit; source-only preprocessing","cohort protocol freeze before new performance evaluation"]}
 save(A/"SOURCE_SCHEMA_AUDIT.json",{"status":"PASS_EXISTING_SOURCE_PRESENCE_AND_HEADERS_ONLY","time":now(),"candidates":output,"new_independent_cohort_qualified":False,"new_clinical_outcomes_or_predictions_computed":False,"INSPIRE_previously_evaluated_not_new_independent_validation":True,"new_datasets_downloaded":0,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
 save(A/"BASELINE_PROTECTION_CHECK.json",{"status":"PASS_EXACT_SIX_BASELINE_FILE_AND_SUBMISSION_ARCHIVE_HASHES","time":now(),"files":matches,"archive":str(z),"archive_sha256":h,"manuscript_prose_read_or_reviewed":False,"manuscript_files_changed":False})
 status={"status":"SOURCE_SCHEMA_PREFLIGHT_COMPLETE_FULL_ENHANCEMENT_NOT_COMPLETE","time":now(),"analysis_owner_thread_id":"01a170c0-3584-74e2-b44c-4d0711ee1860","source_writing_thread_id":"01a192de-cc29-7d52-9dfa-42603428fc4e","latest_user_task_source":"writing_runs/novelty_revision_20261101_v1/ANALYSIS_REQUEST_ZH.md","three_aspects":{"single_resource_decision_protocol":"PENDING_FREEZE","independent_external_evaluation":"EXISTING_MIMIC_EICU_HEADERS_AUDITED_COHORT_AND_TIME_QUALIFICATION_PENDING","decision_boundaries_and_robustness":"PENDING_NEW_AUTHORIZED_CALCULATIONS"},"old37_and_four_supplement_acceptances_are_not_this_task_acceptance":True,"new_task_acceptance_exists":False,"new_analyses_completed":False,"new_model_fits":0,"no_claim_that_preflight_is_analysis_completion":True}
 save(A/"STATUS.json",status)
 print(json.dumps({"status":status["status"],"candidate_header_files":sum(len(x["files"]) for x in output.values()),"baseline_hash_files":len(matches),"new_independent_evaluation_complete":False,"new_decision_analysis_complete":False}),flush=True)
