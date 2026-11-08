"""Read-only code/scope audit for a future complete study review archive.
No analysis entrypoint is imported or executed; no patient-level data is read.
"""
from pathlib import Path
import ast, collections, fcntl, hashlib, json, os, re, resource, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=Path(__file__).parent
O=R/"outputs/study_package_scope_20261030_v1"
SOURCE_SHA="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
EXCLUDED={
 "work/calculator_release_20261025_v1/build_review_packages.py":"Historical171/122 constants, old R1-state reference, shared OVERLAP_LEDGER and destructive replace; never execute for current release",
 "work/benchmark_and_leadtime_20261024_v1/prepare_manuscript_insertion.py":"Manuscript insertion is forbidden in this chat",
 "work/published_benchmark_comparison_20261024_v1/report.py":"Historical hardcoded interpretation; replaced by current evidence and reports",
 "work/llm_supplement_20261028_v1/setup_environment_and_sources.py":"Completed environment/source acquisition; not an allowed reproduction entrypoint",
}
def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 p=Path(path);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def protection():
 files=read(R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json")["files"]
 assert len(files)==92
 result={}
 for name,ref in files.items():
  result[name]=sha(name)
  assert result[name]==ref["sha256"],name
 return result
def run():
 started=time.monotonic()
 # This is a bounded metadata/AST scan, not a resource-heavy analysis.
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
 resource.setrlimit(resource.RLIMIT_CPU,(120,120))
 O.mkdir(mode=0o700,parents=True,exist_ok=True)
 with (O/"OWNER.lock").open("a+") as handle:
  fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (O/"SCOPE_AUDIT.json").exists(),"Do not overwrite an accepted audit; inspect changes first"
  before=protection()
  source=R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"
  assert sha(source)==SOURCE_SHA
  required_path=R/"outputs/post_reanalysis_process_review_v1/REQUIRED_COMPONENTS.json"
  required=read(required_path)["components"];mapping=read(W/"COMPONENT_CODE_SCOPES.json")
  assert len(required)==37 and set(mapping)=={x["component"] for x in required}
  gate_path=R/"outputs/post_reanalysis_process_review_v1/REVIEW_GATE_STATUS.json"
  gate=read(gate_path);by_component={x["component"]:x for x in gate["components"]}
  seeds=set();coverage=[];missing_dirs=[]
  for row in required:
   component=row["component"];files=[]
   for directory in mapping[component]:
    root=R/"work"/directory
    if not root.is_dir():missing_dirs.append(str(root));continue
    files.extend(p for p in root.iterdir() if p.is_file() and p.suffix in [".py",".js",".sh"])
   if component=="secondary_tabpfn":
    files.extend((R/"outputs/tabpfn_benchmark_20261025_v1").glob("*.py"))
   if component in ["calculator","calculator_browser"]:
    files.extend(p for p in (R/"outputs/calculator_release_20261025_v1/app").iterdir()
                 if p.is_file() and p.suffix in [".py",".js",".html"])
   selected=[p for p in files if str(p.relative_to(R)) not in EXCLUDED and "manuscript" not in p.name.lower()]
   seeds.update(selected)
   receipt=R/"outputs"/row["receipt"];current=by_component[component]
   rh=sha(receipt) if receipt.exists() else None
   if current["status"]=="ACCEPTED":assert rh==current["sha256"],component
   coverage.append({"component":component,"receipt":str(receipt),"receipt_sha256":rh,
                    "gate_status_at_audit":current["status"],"code_candidates":[str(p.relative_to(R)) for p in sorted(set(selected))],
                    "coverage_is_candidate_source_inventory_not_execution_proof":True})
  assert not missing_dirs,missing_dirs
  catalogue=list((R/"work").glob("*/*.py"))+list((R/"outputs/tabpfn_benchmark_20261025_v1").glob("*.py"))+list((R/"outputs/calculator_release_20261025_v1/app").glob("*.py"))
  names=collections.defaultdict(list)
  for p in catalogue:names[p.stem].append(p)
  info=[];missing_local=[];ambiguous=[];literal_code_paths=[];historical_literals=[]
  for p in sorted(seeds):
   rel=str(p.relative_to(R));text=p.read_text();row={"path":rel,"sha256":sha(p),"bytes":p.stat().st_size,
          "scope":"INTERNAL_SOURCE_CANDIDATE_ONLY_NOT_SHARED_CONTENT_APPROVAL","code_executed":False}
   if p.suffix==".py":
    tree=ast.parse(text,filename=str(p));row["python_AST_parsed"]=True
    constants={x.value for x in ast.walk(tree) if isinstance(x,ast.Constant) and isinstance(x.value,str)}
    relevant={d for ds in mapping.values() for d in ds if d in constants}
    for node in ast.walk(tree):
     modules=[]
     if isinstance(node,ast.Import):modules=[n.name for n in node.names]
     elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]
     for module in modules:
      base=module.split(".")[0]
      if base not in names:continue
      local=names[base]
      same=[x for x in local if x.parent==p.parent]
      if same:local=same
      else:
       hinted=[x for x in local if x.parent.name in relevant]
       if hinted:local=hinted
      item={"importer":rel,"line":node.lineno,"module":module,"candidates":[str(x.relative_to(R)) for x in local]}
      if len(local)>1:ambiguous.append(item)
      elif local[0] not in seeds:missing_local.append(item)
     if isinstance(node,(ast.List,ast.Tuple)):
      vals=[x.value if isinstance(x,ast.Constant) else None for x in node.elts]
      if vals in [[3710,171],[2428,122]]:
       historical_literals.append({"path":rel,"line":node.lineno,"literal":vals,
          "meaning_requires_context_review":"Historical count literal; do not execute or treat as corrected result without checking call path"})
    for value in constants:
     if value.endswith(".py") and ("\n" not in value) and len(value)<500:
      literal_code_paths.append({"source":rel,"literal":value,"automatically_executed_or_copied":False})
   info.append(row)
  excluded=[]
  for rel,reason in EXCLUDED.items():
   p=R/rel
   if p.exists():excluded.append({"path":rel,"sha256":sha(p),"reason":reason,"source_file_unchanged":True})
  old=R/"work/calculator_release_20261025_v1/build_review_packages.py"
  old_lines=old.read_text().splitlines()
  old_findings=[]
  for i,line in enumerate(old_lines,1):
   if any(s in line for s in ['"main":[3710,171]','"paired":[2428,122]','OVERLAP_LEDGER.csv','R1_FROZEN_STATE_RECHECK','shutil.rmtree']):
    old_findings.append({"line":i,"source_excerpt":line,"scope":"old builder not executed"})
  prior=read(R/"outputs/research_repro_preparation_20261029_v1/bundle/CODE_INDEX.json")
  prior_paths={str(Path(x["source"]).relative_to(R)) for x in prior}
  for x in prior:assert sha(x["source"])==x["sha256"]
  contract={
   "status":"FULL_STUDY_PACKAGE_SCOPE_DEFINED_NOT_BUILT",
   "source_lock_sha256":SOURCE_SHA,"required_components_sha256":sha(required_path),"component_count":37,
   "source_counts_from_current_gate":gate["source"]["counts"],"known_POD1":[2947,39],
   "public_data_common6":[6043,253],"all_current_results_and_negative_results_required":True,
   "archive_prerequisite_components":[x["component"] for x in required if x["component"] not in ["review_packages","integrated_analysis_report"]],
   "dependency_order":["Real LLM24-route comparison and all other numerical components",
     "Complete current result tables, process analysis report content and all figures",
     "Actual complete visual acceptance","Two complete review archives and clean extraction/reproduction checks",
     "Integrated analysis delivery acceptance referencing those artifacts","Refresh all37 then actual statistical/scientific review",
     "Version affected artifacts and packages if real review identifies necessary fixes; final writing gate remains separate"],
   "avoid_acceptance_dependency_cycle":"Do not require PACKAGE_ACCEPTANCE or ANALYSIS_DELIVERY_ACCEPTANCE before building their own artifacts; require real current report content and its hash instead.",
   "shared_scope":["Current numerical/control code references with explicit runnable/archival roles",
     "Source counts, field dictionary, frozen methods/seeds/model-version and environment specifications",
     "All current aggregate results and actual reviewed figures including common6/POD1/albumin44subgroups/benchmarks/scenarios/windows/SHAP/curves/LLM",
     "Current calculator app and constructed examples, validation-only route",
     "Source-to-table/figure/report and package member hashes; explicit limitations and private-input requirements"],
   "private_scope":["All shared review material",
     "Exact current calculator fitted-state dependency closure",
     "Complete deidentified fixed-prediction inputs, source calibration/threshold definitions and bootstrap draws needed for authorized internal numerical replay, if admitted by explicit file/content manifest",
     "Current formal model/adapter and checkpoint provenance; pretrained model weights are source/version references, not redistributed",
     "Explicit original clinical source manifest; raw source workbooks and patient identity ledger are not automatically copied"],
   "never_shared":["Individual record/patient/group keys or clinical rows","OVERLAP_LEDGER.csv",
     "PRIVATE_FIXED_EVALUATION_INPUTS.npz","Fitted imputer/model/adapter states","Clinical JSONL prompts or labels",
     "Credentials, browser cookies, HF cache files","Manuscript prose and protected92 files","Gated pretrained model weights"],
   "reproduction_checks_required":["ZIP CRC and every member SHA256","Path-safe clean extraction",
     "All10 calculator validation-only shared examples","All10 private relocated calculator prediction comparisons",
     "Fixed-prediction metric/interval replay across completed analysis scope with explicit per-analysis coverage and residual limits",
     "Source/table/figure inventory matches actual outputs and current acceptance",
     "Source model/input bytes and protected92 hashes remain unchanged"],
   "not_claimed":["New patient cohort","All original models retrained from ZIP","Independent new validation","Public distribution authorization","Scientific review complete"],
   "final_receipt":"outputs/calculator_release_20261025_v1/PACKAGE_ACCEPTANCE.json",
   "final_receipt_expected_status":"PASS_two_review_archives_SHA_CRC_and_relocation",
   "unresolved_source_dependencies_are_blockers_to_claiming_full_code_closure":True,
  }
  save(O/"PACKAGE_SCOPE_CONTRACT.json",contract)
  save(O/"COMPONENT_SCOPE_INVENTORY.json",coverage)
  save(O/"CODE_CANDIDATES_INTERNAL.json",info)
  save(O/"LOCAL_CODE_DEPENDENCY_FINDINGS.json",{"missing_local_candidates":missing_local,"ambiguous_local_modules":ambiguous,
       "literal_code_path_candidates":literal_code_paths,"static_analysis_is_not_complete_runtime_dependency_proof":True,
       "historical_count_literals":historical_literals,"excluded_entrypoints":excluded})
  save(O/"OLD_BUILDER_DIAGNOSIS.json",{"source":str(old),"sha256":sha(old),"findings":old_findings,
       "old_builder_executed":False,"current_final_package_receipt_exists":(R/"outputs/calculator_release_20261025_v1/PACKAGE_ACCEPTANCE.json").exists(),
       "no_external_release_or_privacy_incident_claimed":True})
  after=protection();assert before==after
  result={"status":"PASS_BOUNDED_SCOPE_AND_STATIC_SOURCE_AUDIT_ONLY","time":now(),"code_sha256":sha(__file__),
    "component_entries":len(coverage),"code_candidate_files":len(info),"code_candidate_bytes":sum(x["bytes"] for x in info),
    "prior_preparation_code_files":len(prior_paths),"new_source_candidates_not_in_prior_preparation":len({x["path"] for x in info}-prior_paths),
    "missing_local_dependency_candidates":len(missing_local),"ambiguous_local_import_sites":len(ambiguous),
    "historical_count_literal_sites":len(historical_literals),"unsafe_old_builder_findings":len(old_findings),
    "protected92_hashes_unchanged":True,"all_numerical_entrypoints_imported_or_executed":False,
    "source_gate_snapshot":gate["checked_Asia_Shanghai"],"full_package_built":False,"full_code_closure_claimed":False,
    "final_PACKAGE_ACCEPTANCE_written":False,"real_LLM_metrics_computed":False,"final_scientific_review":False,
    "elapsed_seconds":time.monotonic()-started,"max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
  result["artifacts_sha256"]={p.name:sha(p) for p in O.glob("*.json")}
  save(O/"SCOPE_AUDIT.json",result)
  print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":run()
