"""Rebuild current-source decision inputs, qualify, and run four frozen workload contexts.
Only the source/control adapter is renewed; native decision algebra and bootstrap are unchanged.
"""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,traceback
R=Path(__file__).resolve().parents[2];P=R.parents[1]
O=R/"outputs/decision_workload_20261024_v1";W=R/"work/decision_workload_20261024_v1"
C=R/"outputs/decision_continuation_20261024_v1";H=R/"outputs/hypothesis_aligned_extensions_20261028_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700);t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def child(script,args,log):
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
 with log.open("ab") as stream:
  p=subprocess.Popen([sys.executable,"-B",str(script),*args],env=env,stdout=stream,stderr=stream)
  try:rc=p.wait(timeout=2700)
  except subprocess.TimeoutExpired:p.kill();p.wait();raise TimeoutError("Owned finite decision worker exceeded allotted wallclock")
 assert rc==0,"Inspect bounded worker log before retry"
def main():
 O.mkdir(parents=True,exist_ok=True,mode=0o700);C.mkdir(parents=True,exist_ok=True,mode=0o700)
 with (O/"SUPERVISOR.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  original=P/"outputs/research_plan_revision_20261018_094007/DECISION_PROTOCOL_AMENDMENT_V3.json"
  assert sha(original)=="4450ec6f40431a38af99d5738a47868601fa362789d51877394c7051480d1d0c"
  source=R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json";assert sha(source)=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
  protocol=read(original)
  assert protocol["workload"]["fraction_of_planned_tests_genuinely_additional"]==[0,.5,1]
  assert protocol["workload"]["review_minutes_per_alert"]==[5,10,15]
  assert protocol["clinical_effect_stress_test"]["assumed_relative_risk_reduction"]==[-.1,0,.1,.2,.3]
  assert protocol["clinical_effect_stress_test"]["effective_uptake"]==[.5,.75,1]
  current={"status":"CURRENT_SOURCE_LINKAGE_AMENDMENT_NUMERIC_DECISION_GRIDS_UNCHANGED","original_method_protocol":str(original),"original_method_protocol_sha256":sha(original),"current_input_lock":str(source),"current_input_lock_sha256":sha(source),"clinical_main":[3710,205],"paired":[2428,138],"POD1_secondary":[2947,39],"true_added_RRT":14,"new_relative_creatinine_events_separate":34,"population_text_and_endpoint_counts_in_original_protocol_are_historical":True,"primary_models_calibration_thresholds_and_heldout_evaluations":"Current completed corrected-source stage receipts only","nonlinear_LR_status":"Current A/B/C/D/E locked model-selection includes its original LR candidates; no assertion of an additional old60-study nonlinear-LR report. This workload does not fit a predictive LR or identify a dose treatment effect.","individual_inference_boundary":"Accepted frozen-prediction replay; current SHAP adapters independently match all ten locked route predictions. Calculator/browser qualification remains separate.","original_preceding_phase_gates":[*protocol["phase_gates"]["prior_to_new_decision_results"]],"no_dependency_on_unrelated_old_optional_extension_results":True,"hypothetical_effect_for_preop_only":True,"not_virtual_trial_or_observed_prevention":True,"prior_evaluation_exposure_disclosed":True,"time":now()}
  current_path=O/"CURRENT_SOURCE_SCOPE_AMENDMENT.json"
  if current_path.exists():assert {k:v for k,v in read(current_path).items() if k!="time"}=={k:v for k,v in current.items() if k!="time"}
  else:save(current_path,current)
  code={n:sha(W/n) for n in ["clinical.py","clinical_adapter.py","decision_math.py","qualify.py"]}
  deps={str(p):sha(p) for p in [original,current_path,source,R/"work/formal_evaluation_20261023_v1/common.py",R/"work/formal_evaluation_20261023_v1/evaluation_stats.py",R/"work/final_evaluation_engine_20261022_v1/fixed_prediction_metrics.py",R/"work/selective_information_20261021_v1/policy_core.py",R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json",R/"outputs/selective_policy_lock_20261023_v1/STAGE_POLICY_LOCK_ACCEPTANCE.json",R/"outputs/formal_evaluation_20261023_v1/STAGE_FINAL_EVALUATION_ACCEPTANCE.json"]}
  for name,value in [("CODE_LOCK.json",code),("DEPENDENCIES.json",deps)]:
   path=O/name
   if path.exists():assert read(path)==value
   else:save(path,value)
  save(C/"status.json",{"state":"current_source_synthetic_and_source_replay_qualification","time":now(),"PID":os.getpid()})
  if not (O/"QUALIFICATION.json").exists():child(W/"qualify.py",[],O/"QUALIFICATION.log")
  if not (O/"SOURCE_ADAPTER_QUALIFICATION.json").exists():child(W/"clinical_adapter.py",[],O/"SOURCE_ADAPTER_QUALIFICATION.log")
  q=read(O/"SOURCE_ADAPTER_QUALIFICATION.json")
  assert q["status"]=="PASS_26_model_and_24_policy_frozen_predictions_replayed"
  expected={"geography_preop":(1159,72),"temporal_preop":(1683,66),"geography_paired":(1158,72),"temporal_paired":(1011,36)}
  assert len(q["proof"])==4
  for v in q["proof"]:assert (v["n"],v["events"])==expected[v["partition"]] and v["capacity_and_threshold_counts_match_accepted_results"]
  preceding={}
  for name,path in {"final_evaluation":R/"outputs/formal_evaluation_20261023_v1/STAGE_FINAL_EVALUATION_ACCEPTANCE.json","model_lock":R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json","policy_lock":R/"outputs/selective_policy_lock_20261023_v1/STAGE_POLICY_LOCK_ACCEPTANCE.json","common6":R/"outputs/common6_public_evaluation_20261024_v1/REVIEW_ACCEPTANCE.json","current_SHAP_source_replay_and_arithmetic":R/"outputs/locked_pipeline_shap_20261024_v1/REVIEW_ACCEPTANCE.json"}.items():
   rec=read(path);assert rec["status"].startswith("PASS")
   preceding[name]={"receipt":str(path),"sha256":sha(path),"status":rec["status"]}
  release={"status":"PASS_required_preceding_stages_accounted_for","code_lock_sha256":sha(O/"CODE_LOCK.json"),"nonlinear_LR_status_explicit":True,"public_and_necessary_supporting_stage_statuses":preceding,"current_source_scope_amendment":str(current_path),"current_source_scope_amendment_sha256":sha(current_path),"no_old_optional_gate_or_pre_correction_receipt_used":True,"technical_stage_acceptance_is_not_full_scientific_report_delivery":True}
  rp=O/"RELEASE_FOR_CLINICAL_SCENARIOS.json"
  if rp.exists():assert read(rp)==release
  else:save(rp,release)
  save(O/"PREPARATION_ACCEPTANCE.json",{"status":"PASS_math_and_source_adapter_preparation_only","current_input_sha256":sha(source),"source_adapter_qualification_sha256":sha(O/"SOURCE_ADAPTER_QUALIFICATION.json"),"mathematical_qualification_sha256":sha(O/"QUALIFICATION.json"),"source_amendment_sha256":sha(current_path),"code_lock_sha256":sha(O/"CODE_LOCK.json")})
  active=[];contexts=list(expected)
  try:
   for ctx in contexts:
    if (O/ctx/"COMPLETE.json").exists():continue
    line,cohort=ctx.split("_");stream=(C/(ctx+".log")).open("ab")
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    proc=subprocess.Popen([sys.executable,"-B",str(W/"clinical.py"),line,cohort],env=env,stdout=stream,stderr=stream);active.append((ctx,proc,stream))
   save(C/"status.json",{"state":"running_four_current_source_frozen_clinical_workloads","time":now(),"PID":os.getpid(),"owned_workers":{ctx:p.pid for ctx,p,s in active},"all_available_CPU_affinity":sorted(os.sched_getaffinity(0)),"replicates":2000,"seed":20261017,"scientific_review_completed":False})
   for ctx,proc,stream in active:
    try:rc=proc.wait(timeout=2700)
    except subprocess.TimeoutExpired:proc.kill();proc.wait();raise TimeoutError("Owned decision worker exceeded finite budget")
    finally:stream.close()
    assert rc==0,"Decision worker failed: "+ctx
  finally:
   for ctx,proc,stream in active:
    if proc.poll() is None:proc.wait()
    stream.close()
  receipts={}
  import pandas as pd,numpy as np
  allburden=[];alleffect=[];allwork=[]
  for ctx in contexts:
   folder=O/ctx;receipt=read(folder/"COMPLETE.json")
   assert receipt["status"]=="PASS_frozen_workload_scenarios_not_clinical_efficacy"
   assert (receipt["n"],receipt["events"])==expected[ctx]
   assert receipt["signature"]["release_sha256"]==sha(rp)
   for f,h in receipt["files"].items():assert sha(folder/f)==h
   boot=read(folder/"NEW_THRESHOLD_INTERVALS.json")["result"]["receipt"]
   final=read(R/"outputs/formal_evaluation_20261023_v1"/ctx/"EVALUATION.json")["receipt"]
   assert boot["replicates"]==2000 and boot["seed"]==20261017 and boot["resample_weight_sha256"]==final["resample_weight_sha256"]
   burden=pd.read_csv(folder/"OPERATING_BURDEN.csv")
   work=pd.read_csv(folder/"WORKLOAD_SCENARIOS.csv")
   effect=pd.read_csv(folder/"HYPOTHETICAL_PREOPERATIVE_EFFECT_ONLY.csv")
   assert sorted(work.additional_fraction.unique())==[0,.5,1] and sorted(work.minutes_per_review.unique())==[5,10,15]
   assert sorted(effect.assumed_RRR.unique())==[-.1,0,.1,.2,.3] and sorted(effect.assumed_uptake.unique())==[.5,.75,1]
   assert np.allclose(work.additional_tests_per1000,work.reviewed_per1000*work.additional_fraction)
   assert np.allclose(work.review_hours_per1000,work.reviewed_per1000*work.minutes_per_review/60)
   for table in [burden,work,effect]:table.insert(0,"context",ctx)
   allburden.append(burden);allwork.append(work);alleffect.append(effect)
   receipts[str(folder/"COMPLETE.json")]=sha(folder/"COMPLETE.json")
  for tables,name in [(allburden,"ALL_OPERATING_BURDEN.csv"),(allwork,"ALL_WORKLOAD_SCENARIOS.csv"),(alleffect,"ALL_HYPOTHETICAL_PREOP_EFFECT_SCENARIOS.csv")]:pd.concat(tables,ignore_index=True).to_csv(O/name,index=False)
  save(C/"STAGE_DECISION_ACCEPTANCE.json",{"status":"PASS_four_frozen_clinical_decision_workload_contexts","receipts":receipts,"source_release_sha256":sha(rp),"no_threshold_optimisation_or_effect_assumption_selection":True,"no_claim_of_observed_AKI_prevention":True,"interpretation_pending":True,"whole_project_complete":False})
  save(C/"status.json",{"state":"complete_four_current_source_workload_scenarios_analysis_review_pending","time":now(),"completed":4,"owned_workers":0})
  save(H/"CLINICAL_SCENARIO_ACCEPTANCE.json",{"status":"PASS_hypothesis_aligned_equal_capacity_clinical_scenarios_and_workload","time":now(),"current_input_lock_sha256":sha(source),"native_stage_receipt":str(C/"STAGE_DECISION_ACCEPTANCE.json"),"native_stage_sha256":sha(C/"STAGE_DECISION_ACCEPTANCE.json"),"source_amendment":str(current_path),"source_amendment_sha256":sha(current_path),"qualifications":{str(O/n):sha(O/n) for n in ["QUALIFICATION.json","SOURCE_ADAPTER_QUALIFICATION.json"]},"receipts":receipts,"generated_summary_files":{str(O/n):sha(O/n) for n in ["ALL_OPERATING_BURDEN.csv","ALL_WORKLOAD_SCENARIOS.csv","ALL_HYPOTHETICAL_PREOP_EFFECT_SCENARIOS.csv"]},"negative_and_null_hypothetical_effects_preserved":True,"same_patient_same_capacity_and_original_group_bootstrap":True,"real_intervention_effect_estimated":False,"final_scientific_review_completed":False,"controller_code_sha256":sha(__file__)})
if __name__=="__main__":
 try:main()
 except BaseException as e:
  save(C/"CURRENT_SOURCE_FAILURE.json",{"time":now(),"error_class":type(e).__name__,"error":str(e)[:500],"stack":[{"file":str(Path(v.filename)),"line":v.lineno,"function":v.name} for v in traceback.extract_tb(e.__traceback__)[-8:]]})
  raise
