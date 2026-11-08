"""Post-extension read-only preservation check, hashes only for manuscript files."""
from pathlib import Path
import json,hashlib,time,resource
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");O=A.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
expected={};types={}
def add(p,h,kind):
 p=str(p)
 if p in expected:assert expected[p]==h
 expected[p]=h;types.setdefault(kind,set()).add(p)
source=read(O/"v4_development_inputs_20261018/SOURCE_LOCK.json")
add(O/"v4_development_inputs_20261018/SOURCE_LOCK.json","afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3","original_source_lock")
for p,h in source["input_sha256"].items():add(O/"v4_development_inputs_20261018"/p,h,"source_inputs")
prot=read(O/"post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json")
for p,v in prot["files"].items():add(p,v["sha256"],"92_historical_manuscript_files")
base=read(A/"BASELINE_PROTECTION_CHECK.json")
for v in base["files"]:add(v["path"],v["sha256"],"six_latest_baseline_files")
add(base["archive"],base["archive_sha256"],"latest_submission_archive")
for stage in ["model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json","common6_model_lock_20261024_v1/STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json"]:
 j=read(O/stage)
 for it in j["models"]:
  add(it["path"],it["sha256"],"26_core_locks" if "common6" not in stage else "two_common6_locks")
  def traverse(x):
   if isinstance(x,dict):
    if "path" in x and "sha256" in x:add(x["path"],x["sha256"],"locked_model_components")
    for v in x.values():traverse(v)
   elif isinstance(x,list):
    for v in x:traverse(v)
  traverse(read(it["path"]))
t=time.time();result=[]
for p,h in expected.items():
 actual=sha(p);assert actual==h,(p,h,actual)
 result.append({"path":p,"sha256":actual,"bytes":Path(p).stat().st_size})
(A/"PRESERVATION_ACCEPTANCE.json").write_text(json.dumps({"status":"PASS_ORIGINAL_SOURCE_MODELS_THRESHOLDS_AND_MANUSCRIPT_HASHES_UNCHANGED","counts":{k:len(v) for k,v in types.items()},"verified_unique_files":len(result),"files":result,"manuscript_content_not_read_or_reviewed":True,"manuscript_files_not_written":True,"old_acceptance_not_reused_as_new_task_acceptance":True,"code_sha256":sha(__file__),"elapsed_seconds":time.time()-t,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},ensure_ascii=False,indent=2)+"\n")
print("Preservation verified",len(result),{k:len(v) for k,v in types.items()})
