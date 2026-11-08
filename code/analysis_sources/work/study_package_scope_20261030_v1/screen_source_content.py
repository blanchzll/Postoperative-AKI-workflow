#!/usr/bin/env python3
"""Bounded source-content screening and specific reviewed private-source decisions.
Does not execute candidate sources, approve a shared release, or validate science.
"""
from pathlib import Path
import ast, datetime, hashlib, json, os, re, resource, signal, time
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
B=R/"outputs/study_package_scope_20261030_v1"
OUT=B/"source_content_disposition_v1"
INDEX=B/"source_literal_role_review_v1/SOURCE_SCREENED_HASH_INDEX_INTERNAL.json"
INDEX_SHA="10153225edcb5c1a0ca75a22d0aa44ca8b731062fd53c977dc5f4a4214fafa64"
ROLE=B/"source_literal_role_review_v1/SOURCE_ROLE_REVIEW_RECEIPT.json"
ROLE_SHA="8228e94e5964d85ae8bb0e529cc310b32fc44aa47771e66ee74bb9bc5bd2836d"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def save(p,d):
 with Path(p).open("x",encoding="utf-8") as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write("\n")
 os.chmod(p,0o600)
def main():
 started=time.monotonic()
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
 resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
 if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 assert sha(INDEX)==INDEX_SHA and sha(ROLE)==ROLE_SHA
 assert sha(R/"AGENTS.md")=="b550409c427153e92b13dfd92870ae4838e76de12808f84c4f90630e16c9bbf1"
 rows=json.loads(INDEX.read_text());assert len(rows)==307
 protections=R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json"
 protected=json.loads(protections.read_text())["files"];assert len(protected)==92
 protected_sha=sha(protections)
 for p,h in protected.items():assert sha(p)==(h["sha256"] if isinstance(h,dict) else h),p
 fields={"age","sex","gender","bmi","alb","albumin","creatinine","scr","albu","alt","bun","egfr","preopalb","preopcr","preopalt","preopbun","preophb","height","weight"}
 secrets=[r"hf_[A-Za-z0-9]{20,}",r"sk-[A-Za-z0-9_-]{20,}",r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",r"(?:AKIA|ASIA)[A-Z0-9]{16}"]
 report=[];parsed=0;bytes_total=0;nonpython=[];credential_count=0
 originals={r["path"]:r["sha256"] for r in rows}
 for row in rows:
  p=R/row["path"];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==row["sha256"],row["path"]
  text=raw.decode("utf-8");bytes_total+=len(raw);flags=[]
  for pattern in secrets:
   for m in re.finditer(pattern,text):
    flags.append({"kind":"possible_credential_literal_REDACTED","line":text.count("\n",0,m.start())+1,"length":len(m.group())});credential_count+=1
  if p.suffix!=".py":
   nonpython.append({"path":row["path"],"sha256":row["sha256"],"review":"Full source text read: calculator HTML/CSS/JS with constructed DEMO defaults or browser QA using the constructed example. No candidate source executed."})
  else:
   tree=ast.parse(text);parsed+=1
   for n in ast.walk(tree):
    if isinstance(n,ast.Dict):
     ks={str(k.value).lower() for k in n.keys if isinstance(k,ast.Constant)}
     nums=[v for v in n.values if isinstance(v,ast.Constant) and isinstance(v.value,(float,int))]
     vals=[v for v in n.values if isinstance(v,ast.Constant) and isinstance(v.value,(float,int,str))]
     if len(ks&fields)>=3 and (len(nums)>=2 or len(vals)>=3):
      flags.append({"kind":"clinical_key_literal_dictionary","line":n.lineno,"matched_key_names":sorted(ks&fields),"number_literals":len(nums),"scalar_literals":len(vals)})
    if isinstance(n,(ast.List,ast.Tuple,ast.Set)):
     ints=[x.value for x in n.elts if isinstance(x,ast.Constant) and isinstance(x.value,int) and not isinstance(x.value,bool)]
     if len(ints)>=3 and len(ints)==len(n.elts) and any(1000<=x<=10000 for x in ints):
      flags.append({"kind":"numeric_vector_possible_record_keys","line":n.lineno,"count":len(ints)})
     count=sum(isinstance(x,ast.Constant) and isinstance(x.value,(float,int,str)) for x in n.elts)
     if count>40:flags.append({"kind":"large_flat_literal_collection","line":n.lineno,"count":count})
    if isinstance(n,ast.Assign) and isinstance(n.value,(ast.List,ast.Tuple,ast.Set)):
     names=[x.id for x in n.targets if isinstance(x,ast.Name)]
     if any(re.search(r"(?:row|patient|subject|record|case|HIGH)",name,re.I) for name in names) and any(isinstance(x,ast.Constant) and isinstance(x.value,(str,int)) for x in n.value.elts):
      flags.append({"kind":"named_identifier_vector","line":n.lineno,"names":names,"count":len(n.value.elts)})
    if isinstance(n,ast.Constant):
     if isinstance(n.value,bytes) and len(n.value)>128:flags.append({"kind":"bytes_blob","line":n.lineno,"length":len(n.value)})
     if isinstance(n.value,str) and len(n.value)>2000:flags.append({"kind":"long_text_literal","line":n.lineno,"length":len(n.value)})
  report.append({"path":row["path"],"sha256":row["sha256"],"bytes":len(raw),"flags":flags,"shared_approval":"NOT_INFERRED_FROM_SCREEN"})
 assert parsed==305 and len(nonpython)==2 and credential_count==0
 legacy="work/fixed_flow_sensitivity_evaluation_20261024_v1/evaluate.py"
 original_tree=ast.parse((R/legacy).read_text())
 high_assign=[n for n in ast.walk(original_tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="HIGH" for t in n.targets)]
 assert len(high_assign)==1
 original_identifiers=ast.literal_eval(high_assign[0].value);assert len(original_identifiers)==3
 identifier_occurrences=[]
 for row in rows:
  text=(R/row["path"]).read_text()
  line_numbers=[]
  for i,line in enumerate(text.splitlines(),1):
   if any(re.search(r"(?<![0-9A-Za-z])"+re.escape(str(v))+r"(?![0-9A-Za-z])",line) for v in original_identifiers):line_numbers.append(i)
  if line_numbers:identifier_occurrences.append({"path":row["path"],"sha256":row["sha256"],"lines":line_numbers,"literal_identifiers_not_copied_into_receipt":True})
 private={
  legacy:{"classification":"historical_private","reason":"Contains three real historical source-row identifiers in the obsolete HIGH exclusion. Preserve original privately; current corrected evaluation is a separate version."},
  "work/fixed_flow_sensitivity_evaluation_20261024_v1/qualify.py":{"classification":"historical_private","reason":"The constructed mask test reused those three real historical source-row identifiers. Synthetic test labels do not anonymize the embedded identifiers; do not copy this source into a shared review archive."},
  "work/study_package_scope_20261030_v1/diagnose_fixed_flow_exclusion.py":{"classification":"private_only","reason":"Diagnostic source embeds the same historical source-row identifier set. Preserve diagnostic provenance privately; share only a summary without individual identifiers."},
  "work/fixed_flow_sensitivity_review_20261024_v1/review.py":{"classification":"historical_private","reason":"The obsolete report-generation string embeds the three actual historical source-row identifiers. It also belongs to the previously prohibited old-count/old-interpretation entrypoint; preserve privately and do not execute as current review."}
 }
 assert {x["path"] for x in identifier_occurrences}==set(private),identifier_occurrences
 for occurrence in identifier_occurrences:
  row=next(x for x in report if x["path"]==occurrence["path"])
  row["flags"].append({"kind":"confirmed_historical_source_identifiers","lines":occurrence["lines"],"identifier_values_omitted":True})
 decisions={
  "work/calculator_current_20261029_v1/benchmark_process.py":"The integer vector is constructed throughput batch sizes. Synthetic cases are generated from synthetic_examples; old entrypoint role restriction still applies.",
  "work/calculator_current_20261029_v1/native_route_qualification.py":"The vector is timing batch sizes. Actual replay records are loaded at runtime and written to private outputs; no record data are embedded in the inspected vector.",
  "work/llm_delivery_current_20261029_v1/qualify.py":"The vector declares selected bootstrap replicate indices, not patient identifiers.",
  "work/llm_delivery_current_20261029_v1/verify_results.py":"The vector is the same independent bootstrap replicate schedule, not patient identifiers.",
  "work/llm_supplement_20261028_v1/export_ngc_bf16_tensors_v1.py":"The vectors are documented model tensor/chunk dimensions, not clinical records or patient identifiers.",
  "work/post_master_delivery_20261028_v1/benchmark_calculator_current_explicit_root.py":"The vector is constructed throughput batch sizes; generated DEMO cases have deterministic synthetic values.",
  "work/post_master_delivery_20261028_v1/review_published_current_v1.py":"The vector selects bootstrap replicate indices for independent scalar verification.",
  "work/public_patient_qualification_20261024_v1/audit.py":"The dictionary maps lab field names to model feature names. It contains no measured lab values or patient rows.",
  "work/published_benchmark_comparison_20261024_v1/review.py":"The vector selects bootstrap replicate indices; existing historical-entrypoint restrictions remain.",
  "work/published_benchmark_comparison_20261024_v1/review_locked_route_support.py":"The vector selects bootstrap replicate indices. Prior unqualified-entrypoint disposition remains unchanged.",
  "work/study_package_scope_20261030_v1/review_control_json_content.py":"The set consists of field names used to detect private records, not actual patient/group identifiers.",
  "work/post_master_delivery_20261028_v1/build_calculator_repro_snapshot_current.py":"The long string is the calculator-only archive README. Full string read; it contains no clinical case data. This old snapshot builder is not the full-study package entrypoint.",
  "work/analysis_figures_current_20261029_v1/render_curves.py":"The dictionary contains axis display labels and measurement units, not measured clinical values.",
  "work/analysis_figures_current_20261029_v1/render_shap.py":"The dictionary maps feature identifiers to display labels, not clinical observations.",
  "work/llm_supplement_20261028_v1/prepare_clinical_text_and_folds.py":"The dictionary defines measurement units for permitted fields. It does not contain patient clinical text or observed values.",
  "work/next_stage_preparation_20261022_v1/prepare.py":"The dictionary specifies units in a historical source-contract constructor, not individual values. Prior prohibition on running this old population-preparation entrypoint remains."
 }
 assert {x["path"] for x in report if x["flags"]}==set(decisions)|set(private)
 OUT.mkdir(mode=0o700)
 save(OUT/"SOURCE_CONTENT_SCREEN_INTERNAL.json",{"source_index_sha256":INDEX_SHA,"files":report,"python_files":parsed,"non_python_manual_review":nonpython,"matched_credential_patterns":credential_count,"credential_scope":"Only listed explicit patterns; zero matches is not a complete credential or privacy acceptance."})
 save(OUT/"PRIVATE_SOURCE_CONTENT_DISPOSITIONS.json",{
  "status":"FOUR_CONFIRMED_IDENTIFIER_BEARING_SOURCES_EXCLUDED_FROM_SHARED_REVIEW",
  "source_index_sha256":INDEX_SHA,"source_role_receipt_sha256":ROLE_SHA,
  "private_sources":[dict(path=p,sha256=originals[p],shared_review_allowed=False,**v) for p,v in private.items()],
  "identifier_occurrences_without_values":identifier_occurrences,
  "original_files_modified":False,"shared_sanitized_code_copies_created":False,
  "remaining_sources_automatically_shared_approved":False,"public_release_authorized":False,
  "current_numeric_replacement":"work/fixed_flow_sensitivity_evaluation_current_20261030_r1/evaluate.py",
  "replacement_evaluation_completion_not_claimed":True})
 save(OUT/"MANUAL_FLAG_INTERPRETATIONS.json",{
  "review_type":"Actual bounded source-context reading, with full two non-Python sources and the complete long literal.",
  "flags":[{"path":p,"sha256":originals[p],"interpretation":reason,"whole_file_shared_approval":False} for p,reason in decisions.items()],
  "private_source_dispositions":"PRIVATE_SOURCE_CONTENT_DISPOSITIONS.json",
  "all_remaining_files_deep_manually_reviewed":False})
 for p,h in originals.items():assert sha(R/p)==h,p
 assert sha(INDEX)==INDEX_SHA and sha(ROLE)==ROLE_SHA and sha(protections)==protected_sha
 for p,h in protected.items():assert sha(p)==(h["sha256"] if isinstance(h,dict) else h),p
 output_names=["SOURCE_CONTENT_SCREEN_INTERNAL.json","PRIVATE_SOURCE_CONTENT_DISPOSITIONS.json","MANUAL_FLAG_INTERPRETATIONS.json"]
 receipt={"status":"PASS_BOUNDED_SOURCE_CONTENT_SCREEN_AND_FOUR_SPECIFIC_PRIVATE_DISPOSITIONS",
  "time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  "code_sha256":sha(__file__),"source_index_sha256":INDEX_SHA,"source_role_receipt_sha256":ROLE_SHA,
  "source_files":len(rows),"source_bytes":bytes_total,"python_files_parsed":parsed,"other_source_files_manually_read":len(nonpython),
  "flagged_files":sum(bool(x["flags"]) for x in report),"private_source_files":len(private),"sources_unchanged":True,
  "protected_manuscript_files":len(protected),"protected_hashes_unchanged":True,
  "candidate_sources_imported_or_executed":0,"new_fits":0,"new_clinical_inputs":0,
  "whole_shared_source_content_approval":False,"whole_study_package_acceptance":False,"scientific_review_acceptance":False,
  "limits":["Focused literal/identifier screening does not prove every remaining source is shareable.",
  "Current final selection must apply these exact-hash exclusions and merge existing entrypoint roles and subsequent source addenda.",
  "All record-level runtime outputs, model/imputer state and clinical prompts remain private independently of source-file classification."],
  "outputs":{n:sha(OUT/n) for n in output_names},
  "elapsed_seconds":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 save(OUT/"SOURCE_CONTENT_DISPOSITION_RECEIPT.json",receipt)
 print(json.dumps({"receipt":str(OUT/"SOURCE_CONTENT_DISPOSITION_RECEIPT.json"),"sha256":sha(OUT/"SOURCE_CONTENT_DISPOSITION_RECEIPT.json"),"private_sources":len(private),"elapsed_seconds":receipt["elapsed_seconds"],"peak_rss_bytes":receipt["peak_rss_bytes"]}))
if __name__=="__main__":main()
