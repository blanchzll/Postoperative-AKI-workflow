"""Finite calculator reproducibility snapshot. This is not final whole-study package acceptance."""
from pathlib import Path
import json,hashlib,zipfile,shutil,os,subprocess,sys,datetime,time,re,traceback
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
C=R/"outputs/calculator_release_20261025_v1";D=R/"outputs/final_analysis_delivery_20261025_v1"
P=D/"calculator_repro_snapshot_20261029_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def save(p,d):p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
def cp(src,root,rel=None):
 src=Path(src);dst=root/(rel if rel is not None else src.relative_to(R));dst.parent.mkdir(mode=0o700,parents=True,exist_ok=True);shutil.copyfile(src,dst);os.chmod(dst,0o600)
def extract_verified(zpath,dst):
 dst.mkdir(mode=0o700)
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  manifest=json.loads(z.read("MANIFEST_SHA256.json"))
  for entry in z.infolist():
   p=Path(entry.filename);assert not p.is_absolute() and ".." not in p.parts and not entry.is_dir()
   out=dst/p;out.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
   h=hashlib.sha256()
   with z.open(entry) as f,out.open("wb") as g:
    for b in iter(lambda:f.read(1024*1024),b""):g.write(b);h.update(b)
   os.chmod(out,0o600)
   if entry.filename!="MANIFEST_SHA256.json":assert h.hexdigest()==manifest[entry.filename]
 return len(manifest)
def seal(root,target):
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
 save(root/"MANIFEST_SHA256.json",files)
 with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for name in sorted(files):z.write(root/name,name)
  z.write(root/"MANIFEST_SHA256.json","MANIFEST_SHA256.json")
 os.chmod(target,0o600)
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  for n,h in files.items():
   hh=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(1024*1024),b""):hh.update(b)
   assert hh.hexdigest()==h,n
 return {"path":str(target),"bytes":target.stat().st_size,"sha256":sha(target),"members":len(files),"CRC_pass":True,"each_member_SHA256_pass":True}
def cli(root,fixture,validate=False):
 a=[sys.executable,"-B",str(root/"outputs/calculator_release_20261025_v1/app/cli.py"),"--project-root",str(root),"--input",str(fixture)]
 if validate:a.append("--validate-only")
 p=subprocess.run(a,capture_output=True,text=True,timeout=180,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
 if p.returncode:raise RuntimeError("Relocation CLI failed:"+p.stderr[-2500:])
 return json.loads(p.stdout)
def main():
 start=time.monotonic()
 assert not P.exists(),"Never overwrite an existing snapshot"
 mount=subprocess.run(["findmnt","-n","-o","SOURCE,FSTYPE,TARGET","--target",str(D.resolve())],capture_output=True,text=True,check=True).stdout.strip()
 assert mount.endswith("/data") and "ext4" in mount
 free=shutil.disk_usage(D.resolve()).free
 closure=json.loads((C/"CURRENT_PRIVATE_CALCULATOR_STATE_CLOSURE.json").read_text())
 for p,ref in closure.items():assert sha(p)==ref["sha256"]
 assert json.loads((C/"CALCULATOR_ACCEPTANCE.json").read_text())["status"]=="PASS_complete_research_calculator_qualification"
 assert free>2*1024**3
 P.mkdir(mode=0o700);shared=P/"shared_review";private=P/"private_DO_NOT_SHARE";shared.mkdir(mode=0o700);private.mkdir(mode=0o700)
 # Explicit small code/aggregate allowlist. No raw rows, hashed patient ledger, prompts or fitted states in shared.
 methods=["v2_stable_imputation_20261017","python_six_outcomes_20261017","ensemble_calibration_20261020_v1","calculator_current_20261029_v1","process_resolution_current_20261029_v1","analysis_figures_current_20261029_v1"]
 codefiles=[p for name in methods for p in (R/"work"/name).glob("*") if p.suffix in [".py",".js"] and p.is_file()]
 docs=["CALCULATOR_ACCEPTANCE.json","ROUTE_ACCEPTANCE.json","IO_ACCEPTANCE.json","BROWSER_ACCEPTANCE.json","HTTP_SERVER_EXIT.json","PERFORMANCE_ACCEPTANCE.json","PERFORMANCE_COLD_WARM.csv","README_ZH.md","VARIABLE_DICTIONARY.csv","requirements-core.txt","ENVIRONMENT_SNAPSHOT.json","CALCULATOR_SCREENSHOT_VISUAL_REVIEW.json"]
 for root in [shared,private]:
  for p in codefiles:cp(p,root)
  for p in (C/"app").iterdir():
   if p.is_file() and p.suffix in [".py",".js",".html",".json"]:cp(p,root)
  for p in (C/"synthetic_examples").glob("*.json"):cp(p,root)
  for n in docs:cp(C/n,root)
  for n in ["UI_DESKTOP.png","UI_MOBILE.png"]:cp(C/n,root)
  for p in (D/"figures").rglob("*"):
   if p.is_file() and p.suffix in [".png",".pdf",".svg"]:cp(p,root)
  for n in ["FIGURE_NOTES_ZH.md","TABLE_1_ALL_PRIMARY_MODELS.csv","PRIMARY_FIGURE_RENDER_ACCEPTANCE.json"]:
   cp(D/n,root)
  for n in ["ALT_SUPPORT.csv","CALIBRATION_UPDATE_SUPPORT.csv","CALIBRATION_POINT_DIAGNOSTICS.csv","CALIBRATION_FOLD_PARAMETERS.csv","NONOVERLAP_SUPPORT.csv","PARTITION_CROSS_TAB.csv","SELECTIVE_VS_RANDOM_MC.csv","RESOLUTION_REPORT_ZH.md","TARGETED_ISSUES_FOR_FINAL_REVIEW.json","POLICY_CALIBRATION_DIRECTION_DIAGNOSTIC.json"]:
   cp(R/"outputs/process_resolution_20261025_v1"/n,root)
  for ctx in ["geography_preop","temporal_preop","geography_paired","temporal_paired"]:
   p=R/"outputs/formal_evaluation_20261023_v1"/ctx/"EVALUATION.json";d=json.loads(p.read_text())
   save(root/"aggregate_results"/(ctx+".json"),{k:d[k] for k in ["point","intervals","paired_differences"]})
  cp(R/"outputs/v4_development_inputs_20261018/FEATURE_SPECS.json",root)
  save(root/"SOURCE_INPUT_MANIFEST.json",{"SOURCE_LOCK_sha256":sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),"clinical_main":[3710,205],"paired":[2428,138],"record_level_data_included":False,"patient_identity_ledger_included":False})
  save(root/"SCOPE.json",{"scope":"Corrected-source10-route research calculator plus current aggregate support figures; finite review snapshot","full_study_or_LLM_reproduction_complete":False,"LLM_formal_training_ongoing":True,"final_full_study_packages_pending":True,"original_or_gated_pretrained_LLM_weights_included":False,"public_release_authorized":False})
  (root/"README.md").write_text("""# Corrected-source calculator reproducibility review snapshot
This finite local review snapshot verifies the ten corrected-source calculator routes and current aggregate support figures. It is not a public release, final whole-study archive or completed LLM comparison.
Use the existing Python3.12 environment recorded in outputs/calculator_release_20261025_v1/ENVIRONMENT_SNAPSHOT.json; no installation or package upgrade was performed.
From the extracted root, always pass the extracted root explicitly:
  python -B outputs/calculator_release_20261025_v1/app/cli.py --project-root . --input outputs/calculator_release_20261025_v1/synthetic_examples/geography_preop_preop_compact.json --validate-only
The shared archive includes no fitted state, original clinical rows or identity ledger. Prediction requires the PRIVATE archive. In PRIVATE, remove --validate-only. All ten synthetic fixtures were compared with the original workspace in fresh CLI processes; missingness uses the exact frozen keyed imputer and canonical100 inference.
The private models/imputers contain learned training information and must not be shared publicly. The shared archive also remains a review artifact; license and external release are separate decisions.
Source-state bytes and all archived members are SHA256 checked; ZIP CRC and a clean extraction are verified. This does not prove full retraining from the archive, independent external validation or scientific acceptance. Original prediction models, calibration, source thresholds, methods and seeds were not changed. The target-label updating supports and conditional random-allocation diagnostic remain separate from deployed calculator routes. Explicit adverse results and the negative comparator-calibration slope are retained.
The LLM supplement, final integrated report and complete study-level packages remain pending. Refer to the current project handoff for subsequent versions. Never execute historical entrypoints solely because their code is included.
""")
  (root/"LICENSE_STATUS.md").write_text("Local review artifact; no new public license or distribution authorization. Original rights remain. Private trained information is not public data.\n")
 for p,ref in closure.items():cp(p,private)
 forbidden=[];credentials=[]
 pattern=re.compile(r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*=\s*['\"][A-Za-z0-9_./+-]{12,}['\"]")
 for p in shared.rglob("*"):
  if not p.is_file():continue
  if p.suffix.lower() in [".joblib",".pkl",".npz",".xlsx",".sqlite",".db"] or any(x in p.name for x in ["PRIVATE","PREDICTIONS","metadata.csv","features.csv","OVERLAP_LEDGER"]):forbidden.append(str(p.relative_to(shared)))
  if p.suffix in [".py",".js",".json",".md",".txt"] and pattern.search(p.read_text(errors="replace")):credentials.append(str(p.relative_to(shared)))
 assert not forbidden and not credentials,(forbidden,credentials)
 save(P/"PRIVACY_REVIEW.json",{"status":"PASS_explicit_allowlist_and_actual_content_scan","forbidden":[],"literal_credential_flags":[],"shared_fitted_information":False,"shared_individual_rows":False,"private_imputer_training_information":True,"public_release_authorized":False})
 records=[];checks=[]
 sys.path.insert(0,str(C/"app"));from engine import Calculator
 original=Calculator(R)
 for root,name,isprivate in [(shared,"AKI_CALCULATOR_SHARED_REVIEW_20261029_v1.zip",False),(private,"AKI_CALCULATOR_PRIVATE_DO_NOT_SHARE_20261029_v1.zip",True)]:
  save(P/"STATUS.json",{"status":"SEALING","archive":name,"time":datetime.datetime.now().astimezone().isoformat()})
  rec=seal(root,P/name);records.append(rec)
  target=P/("verified_extracted_"+root.name);count=extract_verified(P/name,target)
  mismatches=[];routes=0;worst=0.
  for f in sorted((target/"outputs/calculator_release_20261025_v1/synthetic_examples").glob("*.json")):
   actual=cli(target,f,not isprivate)
   if isprivate:
    expected=original.predict(json.loads(f.read_text()))
    assert actual["route"]==expected["route"] and actual["thresholds"]==expected["thresholds"]
    for a,b in zip(actual["predictions"],expected["predictions"]):
     diff=max(abs(a["risk"]-b["risk"]),abs(a["raw_risk"]-b["raw_risk"]));worst=max(worst,diff)
     assert diff<=1e-12 and a["case_key"]==b["case_key"] and a["alerts"]==b["alerts"] and a["missing_fields"]==b["missing_fields"],f.name
   else:assert actual["status"]=="VALID_INPUT" and actual["model_loaded"] is False
   routes+=1
  assert routes==10
  checks.append({"archive":name,"extracted_members_SHA256_pass":count,"synthetic_routes_checked":routes,"private_prediction_parity":isprivate,"max_private_numeric_difference":worst if isprivate else None,"shared_no_models_loaded":not isprivate,"extracted_root":str(target)})
  # Extracted copies retained as review evidence. Owned finite processes have exited; original source unchanged.
 assert all(sha(p)==ref["sha256"] for p,ref in closure.items())
 save(P/"REPRO_SNAPSHOT_ACCEPTANCE.json",{"status":"PASS_current_calculator_two_archives_SHA_CRC_clean_extraction_and10_route_relocation","archives":records,"clean_extraction_checks":checks,"original_private_state_closure_unchanged":True,"state_files":len(closure),"state_bytes":sum(x["bytes"] for x in closure.values()),"data_mount":mount,"free_bytes_before":free,"elapsed_seconds":time.monotonic()-start,"code_sha256":sha(__file__),"final_full_study_packages_complete":False,"final_scientific_review_complete":False,"PACKAGE_ACCEPTANCE_not_claimed":True})
 save(C/"REPRO_SNAPSHOT_CURRENT.json",{"receipt":str(P/"REPRO_SNAPSHOT_ACCEPTANCE.json"),"sha256":sha(P/"REPRO_SNAPSHOT_ACCEPTANCE.json"),"status":"CALCULATOR_REPRODUCIBILITY_SNAPSHOT_READY_FINAL_FULL_STUDY_PACKAGE_PENDING"})
 save(P/"STATUS.json",{"status":"COMPLETE_CALCULATOR_REPRO_SNAPSHOT_ONLY","elapsed_seconds":time.monotonic()-start,"time":datetime.datetime.now().astimezone().isoformat()})
 print("PASS_CALCULATOR_SNAPSHOT_ONLY",time.monotonic()-start,flush=True)
if __name__=="__main__":
 try:main()
 except Exception:
  if P.exists():save(P/"FAILURE.json",{"error":traceback.format_exc(),"time":datetime.datetime.now().astimezone().isoformat()})
  raise
