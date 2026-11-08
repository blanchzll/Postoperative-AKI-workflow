"""Inspect current control JSON structures and derive a minimal component evidence index.
This does not export raw JSON, patient values, fitted states, or declare scientific acceptance.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,resource,time,re,csv,io
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
B=R/"outputs/study_package_scope_20261030_v1"
O=B/"control_json_content_review_v1"
INVENTORY=B/"result_role_review_v2/RESULT_FILE_CANDIDATES_INTERNAL.json"
REQUIRED=R/"outputs/post_reanalysis_process_review_v1/REQUIRED_COMPONENTS.json"
GATE=R/"outputs/post_reanalysis_process_review_v1/REVIEW_GATE_STATUS.json"
EXISTING_INDEX=R/"outputs/research_repro_preparation_20261029_v1/bundle/ANALYSIS_REPORT_EVIDENCE_INDEX.json"
EXISTING_PACKAGE_ACCEPTANCE=R/"outputs/research_repro_preparation_20261029_v1/PREPARATION_ACCEPTANCE.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def shape(v):return type(v).__name__
def segment(k):
 if re.fullmatch(r"[0-9a-fA-F]{24,}",str(k)):return "{opaque_key}"
 if str(k).isdigit():return "{number_key}"
 return str(k)
record_keys={"record_key","record_keys","train_record_keys","prediction_record_keys","assessment_record_keys","group_hash","group_hashes","excel_row","excel_rows","train_rows","assessment_rows","test_rows","train_idx","test_idx","patient_id","patient_ids","subject_id","subject_ids","clinical_text"}
credential_keys={"hf_token","api_key","access_token","cookie","password","secret_key","client_secret"}
def inspect(d):
 stack=[("$",d)];types=Counter();records={};credentials={};path_types=set();absolute_paths=0;nodes=0
 while stack:
  path,node=stack.pop();nodes+=1;types[shape(node)]+=1
  if isinstance(node,dict):
   for key,value in node.items():
    where=path+"."+segment(key);low=str(key).lower()
    if low in record_keys:
     v=records.setdefault(where,{"field":low,"types":set(),"occurrences":0,"max_container_length":0})
     v["types"].add(shape(value));v["occurrences"]+=1
     if isinstance(value,(list,dict)):v["max_container_length"]=max(v["max_container_length"],len(value))
    if low in credential_keys:credentials[where]={"type":shape(value),"nonempty":bool(value)}
    stack.append((where,value))
  elif isinstance(node,list):
   for value in node:stack.append((path+"[*]",value))
  elif isinstance(node,str):
   if node.startswith(("/home/","/data/","/Users/")):absolute_paths+=1
  if len(path_types)<1000:path_types.add((path,shape(node)))
 for v in records.values():v["types"]=sorted(v["types"])
 return {"node_count":nodes,"types":dict(types),"record_identifier_or_clinical_text_field_structures":records,
 "credential_named_field_structures_require_manual_semantics":credentials,"absolute_filesystem_path_values_count":absolute_paths,
 "schema_path_sample_count":len(path_types),"schema_path_sample_capped":len(path_types)==1000}
def main():
 started=time.monotonic()
 # A bounded standard-library inspection inside the existing monitoring allowance.
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
 assert not O.exists(),"Inspect existing output instead of overwriting"
 O.mkdir(mode=0o700,parents=True)
 refs={str(p.relative_to(R)):sha(p) for p in [INVENTORY,REQUIRED,GATE,EXISTING_INDEX,EXISTING_PACKAGE_ACCEPTANCE]}
 items=[x for x in read(INVENTORY) if x["candidate_role"]=="CONTROL_OR_NUMERICAL_JSON_REQUIRES_CONTENT_REVIEW"]
 assert len(items)==392,len(items)
 rows=[];status_counts=Counter()
 for item in items:
  p=R/item["path"];before=sha(p)
  assert before==item["actual_sha256"],item["path"]
  assert p.stat().st_size<=10*1024**2
  data=read(p);obs=inspect(data)
  native=data.get("status") if isinstance(data,dict) else None
  if native is not None:assert isinstance(native,str) and len(native)<240,item["path"]
  status_counts[native or "(no scalar top-level status)"]+=1
  rows.append({"path":item["path"],"sha256":before,"bytes":p.stat().st_size,"top_level_type":shape(data),
   "top_level_keys":sorted(data.keys()) if isinstance(data,dict) else None,"native_status":native,
   **obs,"raw_file_shared_disposition":"PRIVATE_PROVENANCE_BY_DEFAULT_NOT_SHARED_RAW",
   "no_identifier_values_extracted":True,"automated_structure_screen_is_not_scientific_or_privacy_release":True})
  assert sha(p)==before,item["path"]
  del data,obs
 required=read(REQUIRED)["components"];gate=read(GATE)
 assert gate["required_components"]==37 and len(required)==37
 gate_rows=gate["components"]
 if isinstance(gate_rows,list):gate_by={x["component"]:x for x in gate_rows}
 else:gate_by=gate_rows
 index=[]
 for x in required:
  rel="outputs/"+x["receipt"];p=R/rel
  native=None;digest=None
  if p.exists():
   digest=sha(p);payload=read(p)
   if isinstance(payload,dict):native=payload.get("status")
   assert native is None or isinstance(native,str)
   del payload
  g=gate_by[x["component"]]
  index.append({"component":x["component"],"required":bool(x.get("required",True)),"receipt_relative_path":rel,
   "receipt_present":p.exists(),"receipt_sha256":digest,"native_status":native,
   "current_gate_status":g.get("status"),"gate_checked_at":gate["checked_Asia_Shanghai"],
   "scientific_review_completed":False,"scope":"Calculation/component evidence only; raw receipt omitted"})
 assert all(sha(R/k)==v for k,v in refs.items()),"Upstream snapshot changed during scan"
 save(O/"CONTROL_FILE_STRUCTURAL_OBSERVATIONS_INTERNAL.json",rows)
 save(O/"MINIMAL_COMPONENT_EVIDENCE_INDEX_CANDIDATE.json",index)
 summary={"status":"CONTROL_METADATA_SCOPE_PREPARATION_REQUIRES_FINAL_PACKAGE_AND_SCIENCE",
 "time":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"inputs_sha256":refs,"code_sha256":sha(__file__),
 "files_inspected":len(rows),"total_bytes":sum(x["bytes"] for x in rows),"nodes_inspected":sum(x["node_count"] for x in rows),
 "files_with_record_or_clinical_text_named_fields":sum(bool(x["record_identifier_or_clinical_text_field_structures"]) for x in rows),
 "files_with_credential_named_fields_needing_manual_semantics":sum(bool(x["credential_named_field_structures_require_manual_semantics"]) for x in rows),
 "raw_control_json_files_selected_for_shared_export":0,"minimal_component_index_rows":len(index),
 "component_receipts_present":sum(x["receipt_present"] for x in index),"reported_current_gate_count":gate["complete_components"],
 "native_status_counts":dict(status_counts),"whole_research_package_created":False,"scientific_review_performed":False,
 "no_new_fits_or_bootstrap":True,"original_files_unchanged":True,"original_62_file_partial_package_reused_as_reference_not_rebuilt":True,
 "outputs_sha256":{p.name:sha(p) for p in O.iterdir() if p.is_file()},
 "wall_seconds":time.monotonic()-started,"peak_host_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 save(O/"CONTENT_REVIEW_PREPARATION.json",summary)
 print(json.dumps({k:summary[k] for k in ["status","files_inspected","nodes_inspected","files_with_record_or_clinical_text_named_fields","files_with_credential_named_fields_needing_manual_semantics","minimal_component_index_rows","wall_seconds","peak_host_RSS_bytes"]}))
if __name__=="__main__":main()
