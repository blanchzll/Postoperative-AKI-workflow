"""Advance scoped predictor repair, then original CUDA POD1 fixed fits. Main evaluation stays ordered."""
from secondary_runtime import *
import pod1_development_reuse as reuse
import fcntl,traceback
A=R/"outputs/resource_acceleration_20261026_v1/pod1_fixed_advance"
B=A/"predictor_bridge_repair"
D=R/"outputs/pod1_update_20261017"
def check_retained():
 prep=read(B/"REPAIR_PREPARED.json")
 for rel,h in prep["preserved_model_files_sha256"].items():assert sha(D/rel)==h,rel
 for e in prep["retained_imputation_caches"]:
  cache=D/e["cache_relative"]
  assert sha(cache/"data.joblib")==e["data_joblib_sha256_before"]
  assert sha(cache/"state.joblib")==e["state_joblib_sha256_before"]
 return prep
if __name__=="__main__":
 try:
  with (R/"work/.pod1_fixed_advance.lock").open("a+") as guard:
   fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if not (B/"REPAIR_ACCEPTANCE.json").exists():
    check_retained()
    run("pod1_update_20261017/pipeline.py",gpu=True)
    prep=check_retained()
    old=read(B/"before_development/protocol.json");current=read(D/"protocol.json");assert old==current
    for line in ["geography","temporal"]:
     before=read(B/"before_development"/line/"MODEL_LOCK.json");after=read(D/line/"MODEL_LOCK.json")
     for branch in ["all_preop","all_pod1","matched_preop"]:assert before["final_candidates"][branch]==after["final_candidates"][branch]
    run("reanalysis_control/pod1_development_reuse.py",args=("seal",))
    seal=reuse.verify()
    save(B/"REPAIR_ACCEPTANCE.json",{"status":"PASS_source_documented_predictor_repair_and_unaffected_model_reuse_verified",
     "development_seal_sha256":sha(reuse.SEAL),"retained_development_contexts":prep["retained_development_contexts"],
     "recomputed_development_contexts":prep["invalidated_development_contexts"],
     "retained_imputation_caches":len(prep["retained_imputation_caches"]),
     "recomputed_imputation_caches":len(prep["invalidated_imputation_caches"]),
     "bridge_sha256":sha(D/"PREDICTOR_SOURCE_BRIDGE.json"),"all_current_development_contexts":seal["development_contexts"],
     "full_POD1_evaluation_complete":False,"scientific_review_performed":False,"main_E_not_interrupted":True,"at":time.time()})
   seal=reuse.verify()
   actual=reuse.sha;tests=[]
   for suffix in ["SOURCE_LOCK.json","worker.py","MODEL_LOCK.json","model.joblib","data.joblib","PREDICTOR_SOURCE_BRIDGE.json"]:
    target=next(k for k in seal["files"] if k.endswith(suffix))
    reuse.sha=lambda p,target=target: "0"*64 if Path(p)==R/target else actual(p)
    try:reuse.verify_files({target:seal["files"][target]});raise RuntimeError("Changed identity accepted")
    except AssertionError:tests.append({"target_type":suffix,"changed_hash_rejected":True})
    finally:reuse.sha=actual
   save(A/"CHECKPOINT_REUSE_REVALIDATION.json",{"status":"PASS_current_bridged_source_seal_and_tamper_guards",
    "verified_files":len(seal["files"]),"tests":tests,"seal_sha256":sha(reuse.SEAL),"scientific_review_performed":False,"at":time.time()})
   stage="pod1_corrected_fixed_update_20261024_v1";lock(stage);deps(stage)
   run(stage+"/prepare.py")
   assert read(R/"outputs"/stage/"INPUT_ACCEPTANCE.json")["events"]==39
   run(stage+"/qualify.py",gpu=True)
   run(stage+"/supervisor.py",gpu=True)
   receipt=R/"outputs"/stage/"STAGE_POD1_FIT_ACCEPTANCE.json"
   assert read(receipt)["status"]=="PASS_132_fixed_choice_POD1_fitting_contexts"
   save(A/"FIXED_ADVANCE_ACCEPTANCE.json",{"status":"PASS_original_CUDA_POD1_fixed_fits_completed_evaluation_pending",
    "fit_receipt":str(receipt),"fit_receipt_sha256":sha(receipt),"development_seal_sha256":sha(reuse.SEAL),
    "heldout_metrics_evaluated":False,"whole_POD1_complete":False,"main_not_interrupted":True,"at":time.time()})
 except Exception:
  save(A/"FAILURE.json",{"traceback":traceback.format_exc(),"at":time.time(),"no_unchanged_automatic_retry":True});raise
