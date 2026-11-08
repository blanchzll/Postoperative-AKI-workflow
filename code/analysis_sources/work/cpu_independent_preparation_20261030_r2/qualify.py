"""Bounded qualification of resource/owner admission; no numerical audit or training."""
from pathlib import Path
import importlib.util,json,os,subprocess,resource,time,sys
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/cpu_independent_preparation_20261030_r2"
spec=importlib.util.spec_from_file_location("aki_audit_r2",W/"dispatch.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
started=time.monotonic();checks=[]
def check(name,condition):
    if not condition: raise AssertionError(name)
    checks.append(name)
g=m.GIB
d={"AKI_existing_LLM_unit_cap":16,"LIRI_scrna_max":32,"LIRI_virtual_CPU_total":4,"LIRI_drug_preparation":2,"OS_other_headroom_min":4}
def row(unit,cap,anon,family=None): return {"unit":unit,"cap":cap*g,"anon":anon*g,"family":family}
m.code_guard();check("actual_qualified_helper_and_all_original_source_pins_match",True)
p=m.calculate(80*g,80*g,d,[],list(range(10)))
check("closed_main_sufficient_joint_headroom_permits_finite_audit",p["accepted"] and p["remaining_bytes"]==12*g)
check("inactive_LIRI_future32_4_2_still_reserved",p["other_growth"]==38*g)
p=m.calculate(64*g,64*g,d,[row(m.FORMAL,16,1)],list(range(10)))
check("active_formal_consumes_joint_AKI_commitment",not p["accepted"] and p["AKI_family_remaining_cap"]==0)
check("anonymous_growth_not_reclaimable_file_cache",p["other_growth"]==53*g)
check("legacy_AKI_only_formula_would_incorrectly_admit_counterexample",64*g-15*g-16*g-8*g-2*g>0 and p["remaining_bytes"]<0)
p=m.calculate(80*g,12*g,d,[],list(range(10)))
check("shared_ancestor_limit_blocks_despite_host_RAM",not p["accepted"])
p=m.calculate(80*g,80*g,d,[row("liri-virtual-x",4,1,"LIRI_virtual_CPU_total")],list(range(10)))
check("resident_LIRI_anonymous_pages_subtracted_once",p["other_growth"]==37*g)
p=m.calculate(100*g,100*g,d,[row("liri-star-a",32,4,"LIRI_scrna_max"),row("liri-nac-b",32,5,"LIRI_scrna_max")],list(range(10)))
check("concurrent_family_caps_cannot_hide_above_phase_maximum",p["LIRI_commitments"]["LIRI_scrna_max"]["declared_or_active_peak"]==64*g and p["other_growth"]==61*g)
p=m.calculate(80*g,80*g,d,[row("liri-unknown-x",3,1)],list(range(10)))
check("unclassified_live_research_growth_is_additional",p["other_growth"]==40*g)
p=m.calculate(80*g,80*g,d,[row("aki-some-other-unit",12,2)],list(range(10)))
check("other_AKI_cap_preserved",not p["accepted"] and p["AKI_family_remaining_cap"]==4*g)
p=m.calculate(80*g,80*g,d,[],[])
check("empty_CPU_allocation_blocks",not p["accepted"])
p=m.calculate(2*g,2*g,d,[],[0])
check("OS_minimum_and_monitor_reserve_preserved",p["OS_margin"]==4*g and p["monitoring_reserve"]==2*g)
p=m.calculate(80*g,80*g,d,[row("aki-own",8,1)],list(range(10)),own_unit="aki-own")
check("child_own_anonymous_bytes_not_double_reserved",p["accepted"] and p["additional_audit_growth"]==7*g and p["other_AKI_caps"]==0)
try:m.calculate(80*g,80*g,d,[{"unit":"liri-unbounded","cap":None,"anon":0}],list(range(10)))
except ValueError:check("unbounded_live_unit_requires_review",True)
else:raise AssertionError("unbounded unit admitted")
for name,status in [("unknown_launch","LAUNCH_OUTCOME_UNKNOWN"),("incomplete_launch","LAUNCHING_AUDIT_R2"),("unfinished_child","EXECUTING_AUDIT_R2"),("failed_helper","AUDIT_FAILED_NEEDS_DIAGNOSIS")]:
 check(name+"_not_automatically_retried",m.owner_reason({"status":status},{"ActiveState":"inactive","MainPID":"0","Result":"success","ExecMainStatus":"0"}) is not None)
check("live_PID_blocks_even_with_inactive_state",m.owner_reason({},{"ActiveState":"inactive","MainPID":"123"})=="WAIT_EXISTING_AUDIT_OWNER")
check("terminal_failure_blocks",m.owner_reason({},{"ActiveState":"inactive","MainPID":"0","Result":"exit-code","ExecMainStatus":"1"})=="HOLD_PRIOR_AUDIT_UNIT_FAILED")
check("collected_successful_historical_unit_can_be_reconciled",m.owner_reason({"status":"DISPATCHED_INCREMENTAL_CPU_STRUCTURAL_AUDIT_ONLY"},{"LoadState":"not-found","ActiveState":"inactive","MainPID":"0","Result":"success","ExecMainStatus":"0"}) is None)
oa={"status":"DISPATCHED_INCREMENTAL_CPU_STRUCTURAL_AUDIT_ONLY","time":"2026-10-29T20:22:33+08:00"}
ac={"status":"PASS_INCREMENTAL_SEALED_STRUCTURAL_AUDIT_ONLY","code_sha256":m.HELPER_SHA,"time":"2026-10-29T20:22:53+08:00"}
check("collected_unit_requires_post_dispatch_receipt",m.completion_receipt_reason(oa,ac) is None)
check("collected_timeout_with_old_receipt_requires_diagnosis",m.completion_receipt_reason(oa,{**ac,"time":"2026-10-29T18:00:00+08:00"})=="HOLD_PRIOR_OWNER_RECEIPT_PREDATES_DISPATCH")
check("collected_unit_without_receipt_requires_diagnosis",m.completion_receipt_reason(oa,None)=="HOLD_PRIOR_OWNER_NO_QUALIFIED_COMPLETION_RECEIPT")
check("old_helper_receipt_wrong_code_rejected",m.completion_receipt_reason(oa,{**ac,"code_sha256":"wrong"})=="HOLD_PRIOR_OWNER_NO_QUALIFIED_COMPLETION_RECEIPT")
check("child_recheck_with_no_helper_start_can_be_rechecked",m.completion_receipt_reason({"status":"AUDIT_NOT_STARTED_AFTER_CHILD_RECHECK"},None) is None)
oldread=m.read
m.read=lambda p: {"state":"CHANGED"} if p==m.PHASE else oldread(p)
try:m.resource_plan()
except RuntimeError as e:check("changed_joint_authority_blocks_before_launch","Joint authority" in str(e))
else:raise AssertionError("phase ignored")
finally:m.read=oldread
s=m.stats_module()
expected=[{"model":model,"stage":stage,"line":line,"fit":str(fit),"kind":"fit"} for model in s.MODELS for stage in ["preop","endop"] for line in ["geography","temporal"] for fit in [0,1,2,"final"]]
expected += [{"model":model,"stage":stage,"line":"all","fit":"before","kind":"before"} for model in s.MODELS for stage in ["preop","endop"]]
a={"formal_fits":48,"original_checkpoint_banks":6,"contract_sha256":s.PINS[m.F/"FORMAL_TRAINING_CONTRACT.json"],"tasks":expected}
st={"LoadState":"loaded","ActiveState":"inactive","MainPID":"0","Result":"success","ExecMainStatus":"0"}
fs={"status":"COMPLETE_TRAINING_AND_PREDICTION_BANKS_STATISTICAL_COMPARISON_REQUIRED"}
check("only_closed_exact54_native_tasks_eligible",s.completion_reason(st,a,fs) is None)
check("counts_alone_without_exact_task_set_rejected",s.completion_reason(st,{**a,"tasks":expected[:-1]},fs)=="FORMAL_BANK_SET_MISMATCH")
check("missing_controller_handle_not_assumed_success",s.completion_reason({**st,"LoadState":"not-found"},a,fs)=="FORMAL_UNIT_MISSING_VERIFY_OWNERSHIP")
before=set(os.sched_getaffinity(0));selected=sorted(before)[-min(2,len(before)):]
child=subprocess.run([sys.executable,"-B","-c","import os,json;cp="+repr(selected)+";os.sched_setaffinity(0,cp);print(json.dumps(sorted(os.sched_getaffinity(0))))"],capture_output=True,text=True,check=True)
check("actual_child_task_local_affinity_applied",json.loads(child.stdout)==selected)
check("parent_affinity_unchanged",set(os.sched_getaffinity(0))==before)
owner_before=m.sha(m.OWNER);index_before=m.sha(m.A/"AUDIT_INDEX.json")
main=m.state(m.FORMAL);start=m.ticks(main["MainPID"])
actual=m.preflight()
check("actual_live_boundary_waits_with_pending_sealed_tasks",actual["status"]=="WAIT_SHARED_RESOURCE_ADMISSION" and len(actual["pending_tasks"])>0)
check("actual_current_joint_AKI_cap_not_exceeded","AKI_JOINT_COMMITMENT_ALREADY_ALLOCATED" in actual["resource_plan"]["reasons"])
check("actual_owner_and_audit_index_unchanged",m.sha(m.OWNER)==owner_before and m.sha(m.A/"AUDIT_INDEX.json")==index_before)
check("main_controller_same_PID_and_start_ticks",m.state(m.FORMAL)["MainPID"]==main["MainPID"] and m.ticks(main["MainPID"])==start)
m.code_guard()
out={"status":"PASS_RESOURCE_OWNER_GUARD_ONLY_NOT_AUDIT_OR_SCIENCE","time":m.now(),"checks":checks,"count":len(checks),"wrapper_sha256":m.sha(W/"dispatch.py"),
"qualification_code_sha256":m.sha(__file__),"lock_sha256":m.sha(W/"LAUNCHER_LOCK.json"),"actual_preflight":actual,"original_audit_helper_unchanged":True,
"no_formal_or_audit_worker_launched":True,"no_statistical_result_or_model_changed":True,"wall_seconds":time.monotonic()-started,"peak_host_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
m.save(m.O/"GUARD_QUALIFICATION.json",out)
print(json.dumps({k:out[k] for k in ["status","count","wall_seconds","peak_host_RSS_bytes","wrapper_sha256"]}))
