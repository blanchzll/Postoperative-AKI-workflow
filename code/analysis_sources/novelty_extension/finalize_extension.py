"""Finalize only after genuine numeric, source, scientific, figure, protection and clean package acceptance."""
from pathlib import Path
import os,json,hashlib,csv,datetime,shutil,subprocess
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");C=A.parent/"reanalysis_control"
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(p,j):
 p=Path(p);t=p.with_name(p.name+".tmp");t.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+"\n");os.replace(t,p)
def main():
 assert not (A/"ACCEPTANCE.json").exists()
 assert sha(A/"ANALYSIS_PROTOCOL_LOCK.json")=="1afa3e481704e2ea993b2ea4bd44ba13866c13843e4218e2895a624f583edb10"
 assert sha(A/"qualification/MIMIC_LAB_AND_ENDPOINT_PRELOCK.json")=="5fda7a45acc64165f0ecb6b93b9dc2736f228ef34a4f788aef45dc10ac7d20ce"
 r=read(A/"resource_comparison/INDEPENDENT_NUMERICAL_VERIFICATION.json");assert r["status"].startswith("PASS")
 n=read(A/"mimic_calendar_v2/INDEPENDENT_NUMERICAL_VERIFICATION.json");assert n["status"].startswith("PASS") and n["replicates_each"]==2000 and n["events"]==59 and n["cases"]==759
 i=read(A/"mimic_calendar_v2/INDEPENDENT_INPUT_ENDPOINT_REVIEW.json");assert i["disagreement_records"]==0 and i["exact_rational_events"]==59
 sc=read(A/"SCIENTIFIC_REVIEW.json");assert sc["status"].startswith("SCIENTIFIC_REVIEW_COMPLETED") and sc["unresolved_confirmed_computational_errors_current_chain"]==0
 for f,h in sc["reviewed_evidence_hashes"].items():assert sha(A/f)==h,(f,"science_evidence_changed")
 fig=read(A/"figures/VISUAL_REVIEW_ACCEPTANCE.json");assert fig["status"].startswith("PASS") and fig["final_source_preflight"]["counts"]["FAIL"]==0
 assert sha(A/"figures/gen_fig_novelty.py")==fig["plot_source_sha256"]
 pk=read(A/"DELIVERY_PACKAGE_ACCEPTANCE.json");assert pk["status"].startswith("PASS")
 assert sha(A/"build_delivery_packages.py")==pk["code_sha256"] and '.convert("RGBA")' not in (A/"build_delivery_packages.py").read_text()
 for q in pk["packages"]:assert sha(q["path"])==q["sha256"] and q["CRC_and_clean_extraction_hashes_pass"]
 assert all(q["source_CSV_SHA_equal"] and q["all_PNG_pixels_equal"] for q in pk["clean_plot_replay"])
 assert pk["private_clean_exact_input_replay"]["disagreement_records"]==0
 assert not (A/"replay_scratch").exists()
 p=read(A/"PRESERVATION_ACCEPTANCE.json");assert p["status"].startswith("PASS") and p["counts"]["92_historical_manuscript_files"]==92 and p["counts"]["26_core_locks"]==26
 # Verify fixed-source/model/manuscript hashes at final release without rewriting the earlier receipt.
 for v in p["files"]:assert sha(v["path"])==v["sha256"],("preservation",v["path"])
 for row in csv.DictReader((A/"CLAIM_TABLE_VERSION_MATRIX.csv").open()):
  for s,h in json.loads(row["source_hashes"]).items():assert sha(A/s)==h
 for row in csv.DictReader((A/"EVIDENCE_SOURCE_TABLE_LEDGER.csv").open()):
  assert sha(row["source"])==row["source_sha256"] and sha(A/row["delivered_table"])==row["delivered_sha256"]
 # Actual owned process/lock closure; unloaded transient units are not represented as fresh success receipts.
 units=["aki-novelty-source-schema-20261101-v1.service","aki-novelty-resource-comparison-20261101-v1.service","aki-novelty-resource-verify-20261101-v1.service","aki-novelty-cohort-time-audit-20261101-v1.service","aki-novelty-mimic-calendar-inputs-20261101-v1.service","aki-novelty-mimic-calendar-evaluation-20261101-v1.service","aki-novelty-mimic-calendar-evaluation-20261101-v2.service","aki-novelty-mimic-independent-verify-20261101-v2.service","aki-novelty-figures-20261101-v1.service","aki-novelty-figures-20261101-v2.service","aki-novelty-figures-20261101-v3.service","aki-novelty-preservation-20261101-v1.service","aki-novelty-delivery-package-20261101-v1.service","aki-novelty-delivery-package-20261101-v2.service"]
 states={}
 for u in units:
  raw=subprocess.run(["systemctl","--user","show",u,"--property=LoadState,ActiveState,SubState,MainPID,Result,ExecMainStatus"],capture_output=True,text=True)
  d=dict(x.split("=",1) for x in raw.stdout.splitlines() if "=" in x);assert d.get("ActiveState") not in ["active","activating","reloading","deactivating"];assert int(d.get("MainPID","0"))==0
  if d.get("LoadState")=="not-found":d["scope"]="Transient unit unloaded; normal execution documented by prior run receipts/logs; no restart."
  states[u]=d
 live=[]
 for pp in Path("/proc").iterdir():
  if not pp.name.isdigit() or int(pp.name)==os.getpid():continue
  try:
   parts=(pp/"cmdline").read_bytes().split(b"\0")
   if any(s.decode(errors="ignore").startswith(str(A)+"/") and s.endswith(b".py") for s in parts[1:]):live.append(int(pp.name))
  except (OSError,PermissionError):continue
 assert not live,live
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 save(A/"OWNED_PROCESS_CLOSURE.json",{"status":"PASS_NO_LIVE_EXTENSION_BATCH_PROCESS_OR_ACTIVE_UNIT","time":now,"states":states,"live_owned_script_processes":live,"all_other_projects_left_untouched":True,"old_lease_or_automation_not_reenabled":True})
 save(A/"QUALIFICATION_CURRENT.json",{"MIMIC":{"qualified":"independent-source calendar-anchor six-input sensitivity only","n":759,"events":59,"source":"MIMIC-IV3.1","current":"mimic_calendar_v2","exact_original_surgical_clock_endpoint_qualified":False,"full27_38_LLM_external_validation_qualified":False},"eICU":{"qualified_for_requested_surgical_validation":False,"scores_generated":False,"broad_neoplasm_counts_corrected":"1412GIpath+356otherGU; not eligible cohort"},"INSPIRE":{"already_evaluated":True,"new_independent_cohort":False},"time":now})
 # Hash-backed continuation-state backup, followed by a final static artifact manifest.
 H=A/"history_control_before_completion";H.mkdir(mode=0o700)
 for root,label,names in [(A,"analysis",["STATUS.json","CURRENT_HANDOFF_ZH.md","MIMIC_CURRENT.json"]),(C,"control",["CURRENT_HANDOFF_ZH.md","NOVELTY_STRENGTHENING_CURRENT.json"])]:
  for name in names:
   pth=root/name
   if pth.exists():shutil.copy2(pth,H/(label+"_"+name))
 save(H/"HASHES.json",{p.name:sha(p) for p in H.iterdir() if p.name!="HASHES.json"})
 manifest={}
 skip={"ACCEPTANCE.json","SOURCE_AND_ARTIFACT_MANIFEST.json","STATUS.json","CURRENT_HANDOFF_ZH.md","MIMIC_CURRENT.json"}
 for f in sorted(A.rglob("*")):
  if not f.is_file() or f.name in skip or f.name.endswith((".lock",".tmp")):continue
  rel=str(f.relative_to(A))
  manifest[rel]={"sha256":sha(f),"bytes":f.stat().st_size,"history_only":("history" in rel or rel.startswith("mimic_calendar/"))}
 save(A/"SOURCE_AND_ARTIFACT_MANIFEST.json",{"status":"STATIC_NEW_EXTENSION_SNAPSHOT","time":now,"files":manifest,"source_provenance_receipts":["qualification/COHORT_QUALIFICATION_AUDIT.json","mimic_calendar_v2/INPUT_QUALIFICATION.json","ANALYSIS_PROTOCOL_LOCK.json","PRESERVATION_ACCEPTANCE.json"],"mutable_control_files_excluded":sorted(skip),"original_models_raw_data_and_shared_environment_not_duplicated":True})
 decisive=["ANALYSIS_PROTOCOL_LOCK.json","QUALIFICATION_CURRENT.json","resource_comparison/NUMERICAL_COMPUTATION_ACCEPTANCE.json","resource_comparison/INDEPENDENT_NUMERICAL_VERIFICATION.json","mimic_calendar_v2/COMPUTATION_ACCEPTANCE.json","mimic_calendar_v2/INDEPENDENT_INPUT_ENDPOINT_REVIEW.json","mimic_calendar_v2/INDEPENDENT_NUMERICAL_VERIFICATION.json","SCIENTIFIC_REVIEW.json","ISSUE_AND_REMEDIATION.csv","CLAIM_TABLE_VERSION_MATRIX.csv","ANALYSIS_RESULTS_AND_REVIEW_ZH.md","MANUSCRIPT_HANDOFF_ZH.md","figures/VISUAL_REVIEW_ACCEPTANCE.json","DELIVERY_PACKAGE_ACCEPTANCE.json","PRESERVATION_ACCEPTANCE.json","OWNED_PROCESS_CLOSURE.json","SOURCE_AND_ARTIFACT_MANIFEST.json"]
 accept={"status":"PASS_NEW_NOVELTY_STRENGTHENING_ANALYSIS_REVIEW_AND_HANDOFF_WITH_EXPLICIT_SCOPE_LIMITS","time":now,"analysis_root":str(A),"new_task_complete":True,"source_human_turn":"01a1956d-ca57-7d83-a49b-0da3de36fecb","analysis_owner_thread":"01a170c0-3584-74e2-b44c-4d0711ee1860","existing_writing_owner_thread":"01a192de-cc29-7d52-9dfa-42603428fc4e","target_journal":"JAMA Network Open","completion":{"new_joint_budget_grid_rows":5208,"new_signed_boundaries":168,"comparisons":14,"bootstrap_replicates":2000,"new_predictive_model_fits":0,"new_independent_source_sensitivity":{"source":"MIMIC-IV3.1","n":759,"events":59,"source_frozen_RF_models":2,"probability_views":4,"scope":"Calendar-day+1..+7 observed SCr ratio sensitivity; not exact original endpoint or full27/38input validation"},"new_figures":3,"scientific_issues_reviewed":13,"claim_source_version_rows":10,"aggregate_and_private_packages_clean_replayed":True},"not_estimable_or_not_claimed":["Exact original surgical-clock7dayKDIGO publicvalidation","Full27/38orLLM validation innewMIMIC","Qualified eICU surgical validation","Measured acquisition/review time/cost/fungibility","Actual avoided AKI or clinical deployment effect","Absolute first or new cost-sensitive algorithm"],"retained_boundaries":["All negative/heterogeneous results","Pointwise fixed-model CIs not simultaneous guarantees","D equal-capture review-equivalent units distinct fromoldδ7eventloss units","Historical holdout overlap273records/272groups andevaluationexposure","MIMIC Alb missing465/759,selection andrecord-window limits"],"confirmed_new_MIMIC_float_error_corrected":True,"current_MIMIC_directory":"mimic_calendar_v2","original_core_and_four_supplements_preserved_not_substituted_for_this_acceptance":True,"manuscript_files_changed":False,"manuscript_prose_reviewed_or_written":False,"writing_integration_executed":False,"writing_integration_permitted_for_existing_final_thread":True,"new_chat_created_or_message_sent":False,"original_source_model_manuscript_files_reverified_at_release":p["verified_unique_files"],"files":{n:sha(A/n) for n in decisive},"packages":pk["packages"]}
 save(A/"ACCEPTANCE.json",accept)
 ah=sha(A/"ACCEPTANCE.json")
 save(A/"STATUS.json",{"status":accept["status"],"time":now,"this_task_complete":True,"new_task_acceptance_exists":True,"acceptance_sha256":ah,"new_budget_grid_rows":5208,"new_exact_boundaries":168,"MIMIC":{"directory":"mimic_calendar_v2","n":759,"events":59,"scope":"calendar six-input sensitivity only"},"active_units":[],"manuscript_prose_untouched":True,"writing_integration_done":False,"existing_final_writing_thread":accept["existing_writing_owner_thread"]})
 save(A/"MIMIC_CURRENT.json",{"status":"ACCEPTED_LIMITED_CALENDAR_SENSITIVITY_WITH_INDEPENDENT_AND_SCIENTIFIC_REVIEW","directory":str(A/"mimic_calendar_v2"),"n":759,"events":59,"superseded_float_candidate":str(A/"mimic_calendar"),"new_task_acceptance_sha256":ah})
 short="# 当前新增分析已真实验收，交给既有final写作对话\n唯一分析根目录："+str(A)+"。本次三方面增强完成，ACCEPTANCE SHA256="+ah+"。\n必须先读（仅3项）：ACCEPTANCE.json、ANALYSIS_RESULTS_AND_REVIEW_ZH.md、CLAIM_TABLE_VERSION_MATRIX.csv。之后按需读MANUSCRIPT_HANDOFF_ZH.md点名文件。\n新增5208资源情景/168边界及3图、MIMIC759/59六输入日历敏感性、真实独立复算/科学审读和追加复核包已完成。精确术时原终点/全输入外验未取得；eICU不合格；实际工时/干预效应不可估。只用MIMIC_v2，不沿用56事件候选或宽eICU初筛名。原模型/源/稿件基线保持；无新拟合/有利赢家替换。\n写作整合由既有预测模型manuscript final(01a192de-cc29-7d52-9dfa-42603428fc4e)负责；本对话不写改审正文、不另建聊天/发跨对话消息。无活动批次，不恢复旧心跳/独占租约，不干扰LIRI。旧范围完成证据和本轮修前状态均保留。\n"
 (A/"CURRENT_HANDOFF_ZH.md").write_text(short)
 (C/"CURRENT_HANDOFF_ZH.md").write_text(short)
 save(C/"NOVELTY_STRENGTHENING_CURRENT.json",{"status":accept["status"],"time":now,"directory":str(A),"handoff":str(A/"MANUSCRIPT_HANDOFF_ZH.md"),"acceptance":str(A/"ACCEPTANCE.json"),"acceptance_sha256":ah,"status_file":str(A/"STATUS.json"),"source_user_turn_id":accept["source_human_turn"],"old_scope_acceptances_not_new_task_acceptance":True,"new_task_complete":True,"current_new_MIMIC":"759patients/59events; calendar6inputonly","manuscript_prose_untouched":True,"writing_integration_done":False,"existing_writing_owner_thread":accept["existing_writing_owner_thread"],"active_units":[],"no_cross_thread_message_or_new_automation":True})
 # Final readback: static evidence, entrypoints and exact acceptance agree.
 assert read(A/"STATUS.json")["acceptance_sha256"]==sha(A/"ACCEPTANCE.json")==read(C/"NOVELTY_STRENGTHENING_CURRENT.json")["acceptance_sha256"]
 for n,h in accept["files"].items():assert sha(A/n)==h
 print(json.dumps({"status":accept["status"],"acceptance_sha256":ah,"manifest_files":len(manifest),"source_model_manuscript_files_verified":p["verified_unique_files"],"active_owned_batches":0,"new_task_complete":True},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
