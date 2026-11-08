"""Finite appendix delivery: safe aggregate bundle, private audit bundle, clean extraction/replay."""
from pathlib import Path
import os,json,hashlib,zipfile,shutil,subprocess,time,resource,fcntl
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");D=A/"deliveries";S=A/"replay_scratch"
PY="/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/envs/prediction_gpu/bin/python"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 start=time.time();D.mkdir(mode=0o700,exist_ok=True)
 with (D/"PACKAGE_RUNNER.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (A/"DELIVERY_PACKAGE_ACCEPTANCE.json").exists()
  avail=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))*1024
  assert avail>2*2**30+max(4*2**30,.25*avail)
  assert shutil.disk_usage(A.resolve()).free>2*2**30
  assert read(A/"PRESERVATION_ACCEPTANCE.json")["status"].startswith("PASS")
  assert read(A/"figures/VISUAL_REVIEW_ACCEPTANCE.json")["status"].startswith("PASS")
  assert read(A/"SCIENTIFIC_REVIEW.json")["unresolved_confirmed_computational_errors_current_chain"]==0
  rootfiles=["ANALYSIS_PROTOCOL_LOCK.json","ANALYSIS_PLAN_ZH.md","COMPARISON_MATRIX.csv","ANALYSIS_RESULTS_AND_REVIEW_ZH.md","SCIENTIFIC_REVIEW.json","ISSUE_AND_REMEDIATION.csv","CLAIM_TABLE_VERSION_MATRIX.csv","EVIDENCE_SOURCE_TABLE_LEDGER.csv","EVIDENCE_ASSEMBLY_ACCEPTANCE.json","PUBLIC_METHODS_AND_PRIOR_ART_REVIEW.json","MIMIC_VIEW_AND_MODEL_IDENTITY.json","FIGURE_CONTRACT.json","ENVIRONMENT_VERSIONS.json","FIGURE_NOTES_ZH.md","MANUSCRIPT_HANDOFF_ZH.md","DELIVERY_README_ZH.md"]
  public=[A/n for n in rootfiles]+list((A/"tables").glob("*.csv"))
  public+=list((A/"figures").glob("N*.*"))+[A/"figures/gen_fig_novelty.py",A/"figures/VISUAL_REVIEW_ACCEPTANCE.json",A/"figures/RENDER_RECEIPT.json",A/"figures/SOURCE_PREFLIGHT.json"]
  public += [A/"resource_comparison"/n for n in ["RESOURCE_BUDGET_GRID.csv","BREAK_EVEN_BOUNDARIES.csv","INDEPENDENT_NUMERICAL_VERIFICATION.json","NUMERICAL_COMPUTATION_ACCEPTANCE.json","MATH_QUALIFICATION.json"]]
  public += [A/"mimic_calendar_v2"/n for n in ["POINT_ESTIMATES.csv","INTERVALS.csv","SOURCE_THRESHOLD_BURDEN.csv","INVERSE_COVERAGE.csv","INPUT_QUALIFICATION.json","INDEPENDENT_NUMERICAL_VERIFICATION.json","INDEPENDENT_INPUT_ENDPOINT_REVIEW.json","EXACT_REBUILD_REPLAY.json"]]
  public += [A/"qualification"/n for n in ["COHORT_QUALIFICATION_AUDIT.json","EICU_AUDIT_CLARIFICATION.json","MIMIC_RESECTION_CODELIST.csv","MIMIC_PHENOTYPE_PRELOCK.json","MIMIC_LAB_AND_ENDPOINT_PRELOCK.json"]]
  public+=list(A.glob("*.py"))
  public=sorted(set(public))
  for p in public:
   assert p.is_file(),p
   assert "PRIVATE" not in p.name and p.suffix not in [".npz",".pkl",".joblib",".bin"],p
   if p.suffix==".csv":
    import csv
    with p.open() as f:fields=next(csv.reader(f))
    assert not ({"record_key","group_hash","subject_id","hadm_id","charttime","storetime"}&set(fields)),p
  excludes={"deliveries","replay_scratch","outputs"}
  private=sorted(p for p in A.rglob("*") if p.is_file() and p.relative_to(A).parts[0] not in excludes and p.name not in ["STATUS.json","MIMIC_CURRENT.json","CURRENT_HANDOFF_ZH.md"] and not p.name.endswith((".lock",".tmp")))
  packages=[]
  for name,files in [("summary_replay",public),("private_audit",private)]:
   out=D/(name+".zip");assert not out.exists()
   manifest={str(p.relative_to(A)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in files}
   with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,str(p.relative_to(A)))
    z.writestr("PACKAGE_MEMBER_MANIFEST.json",json.dumps(manifest,indent=2,ensure_ascii=False))
   out.chmod(0o600)
   with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    target=S/name;target.mkdir(mode=0o700,parents=True);z.extractall(target)
   for n,j in manifest.items():assert sha(target/n)==j["sha256"]
   packages.append({"name":name,"path":str(out),"sha256":sha(out),"bytes":out.stat().st_size,"members":len(manifest)+1,"verified_payload_members":len(manifest),"source_manifest":manifest,"CRC_and_clean_extraction_hashes_pass":True})
  # Rebase only the plotting root, no statistical changes, and compare every quantitative source CSV and rendered PNG pixel.
  root=S/"summary_replay";p=root/"figures/gen_fig_novelty.py";before=sha(p);src=p.read_text();old='A=Path("'+str(A)+'")';assert src.count(old)==1
  p.write_text(src.replace(old,'A=Path(os.environ["AKI_EXTENSION_ROOT"])'))
  env=dict(os.environ,AKI_EXTENSION_ROOT=str(root),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
  r=subprocess.run([PY,"-B",str(p)],env=env,capture_output=True,text=True,timeout=180);(D/"CLEAN_PLOT_REPLAY.log").write_text(r.stdout+r.stderr);assert r.returncode==0,(r.returncode,r.stderr)
  from PIL import Image,ImageChops
  assert ImageChops.difference(Image.new("RGB",(1,1),(0,0,0)),Image.new("RGB",(1,1),(1,0,0))).getbbox() is not None
  plots=[]
  for name in ["N1_equal_budget_capture","N2_signed_equal_capture_boundary","N3_MIMIC_calendar_transport"]:
   assert sha(root/"figures"/(name+"_SOURCE_DATA.csv"))==sha(A/"figures"/(name+"_SOURCE_DATA.csv"))
   left=Image.open(root/"figures"/(name+".png")).convert("RGB");right=Image.open(A/"figures"/(name+".png")).convert("RGB")
   assert left.size==right.size and ImageChops.difference(left,right).getbbox() is None,name
   plots.append({"name":name,"source_CSV_SHA_equal":True,"all_PNG_pixels_equal":True,"size":left.size});left.close();right.close()
  # Private input/endpoint re-execution uses only clean extracted raw subset and candidate/codebook files.
  root2=S/"private_audit";p2=root2/"verify_mimic_calendar_inputs_v2.py";b2=sha(p2);src=p2.read_text();assert src.count(old)==1
  p2.write_text(src.replace("from pathlib import Path","from pathlib import Path\nimport os").replace(old,'A=Path(os.environ["AKI_EXTENSION_ROOT"])'))
  r=subprocess.run([PY,"-B",str(p2)],env=dict(env,AKI_EXTENSION_ROOT=str(root2)),capture_output=True,text=True,timeout=60);(D/"CLEAN_PRIVATE_INPUT_REPLAY.log").write_text(r.stdout+r.stderr);assert r.returncode==0
  result=read(root2/"mimic_calendar_v2/INDEPENDENT_INPUT_ENDPOINT_REVIEW.json");assert result["disagreement_records"]==0 and result["exact_rational_events"]==59 and result["cases"]==759
  # Package records are compact at top level; full membership hash catalogs are separate.
  for pk in packages:
   (D/(pk["name"]+"_MEMBERS.json")).write_text(json.dumps(pk.pop("source_manifest"),indent=2,ensure_ascii=False)+"\n")
  rec={"status":"PASS_NEW_EXTENSION_APPENDIX_PACKAGE_CRC_HASH_CLEAN_REPLAY","packages":packages,"shared_no_patient_rows_or_raw_lab_or_model_states_or_credentials":True,"shared_scope_verified_by_explicit_file_allowlist_and_CSV_headers":True,"private_must_not_be_external_shared":True,"clean_plot_replay":plots,"root_only_code_adaptations":{"plot_before_sha256":before,"plot_after_sha256":sha(p),"private_input_before_sha256":b2,"private_input_after_sha256":sha(p2)},"private_clean_exact_input_replay":{k:v for k,v in result.items() if k not in ["source_sha256"]},"not_a_from_zero_full_study_training_or_restricted_source_download_claim":True,"full_new_statistics_already_independently_recomputed_in_native_frozen_version":True,"source_models_environment_raw_data_remain_original_licensed_locations":True,"elapsed_seconds":time.time()-start,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"code_sha256":sha(__file__)}
  (A/"DELIVERY_PACKAGE_ACCEPTANCE.json").write_text(json.dumps(rec,indent=2,ensure_ascii=False)+"\n")
  # Only owned replay scratch, recoverable exactly from verified ZIPs, no live subprocess remains.
  assert S.resolve().parent==A.resolve();shutil.rmtree(S)
  print(json.dumps({"status":"APPENDIX_PACKAGES_PASS","packages":[{"name":x["name"],"bytes":x["bytes"],"members":x["members"]} for x in packages],"seconds":time.time()-start}),flush=True)
if __name__=="__main__":main()
