from secondary_runtime import *
import pandas as pd
from pod1_fixed_reuse import assert_prediction_alignment
# An advanced finite fixed-fit owner must finish before the main step reuses it.
owner=R/"outputs/reanalysis_control/POD1_FIXED_PARALLEL_OWNER.json"
if owner.exists():
 unit=read(owner)["unit"]
 save(R/"outputs/resource_acceleration_20261026_v1/pod1_fixed_advance/MAIN_JOIN_REQUEST.json",{"pid":os.getpid(),"start_ticks":Path("/proc/self/stat").read_text().split()[21],"at":time.time()})
 while True:
  state=subprocess.check_output(["systemctl","--user","show",unit,"-p","ActiveState","--value"],text=True).strip()
  if state not in {"active","activating","deactivating"}:break
  time.sleep(15)
 assert read(R/"outputs/pod1_corrected_fixed_update_20261024_v1/STAGE_POD1_FIT_ACCEPTANCE.json")["status"]=="PASS_132_fixed_choice_POD1_fitting_contexts","Diagnose advanced POD1 fixed-fit failure; do not repeat unchanged"
seal=R/"outputs/pod1_update_20261017/DEVELOPMENT_REUSE_SEAL.json"
if seal.exists():
 run("reanalysis_control/pod1_development_reuse.py")
else:
 run("reanalysis_control/prepare_pod1_source.py")
 run("pod1_update_20261017/pipeline.py",gpu=True)
 run("reanalysis_control/pod1_development_reuse.py",args=("seal",))
stage="pod1_corrected_fixed_update_20261024_v1"
if owner.exists():
 # Existing pilots have cached task receipts; rerunning their trace would be empty.
 # Verify the original qualification and entire saved bank, preserving all fits.
 run("reanalysis_control/pod1_fixed_reuse.py")
else:
 lock(stage);deps(stage)
 run(stage+"/prepare.py")
 assert read(R/"outputs"/stage/"INPUT_ACCEPTANCE.json")["events"]==39
 run(stage+"/qualify.py",gpu=True);run(stage+"/supervisor.py",gpu=True)
fit=R/"outputs"/stage;tasks=read(fit/"TASKS.json")
data=pd.read_csv(fit/"PRIVATE_CURRENT_POD1_COHORT.csv",dtype={"group_hash":str},float_precision="round_trip").set_index("excel_row")
assert len(data)==2947 and int(data.y.sum())==39
for i,t in enumerate(tasks):
 assert set(data.loc[t["train_rows"],"group_hash"]).isdisjoint(data.loc[t["valid_rows"],"group_hash"])
 dest=fit/"tasks"/str(i).zfill(3);rec=read(dest/"COMPLETE.json")
 assert rec["signature"]=={"task":t,"code":sha(fit/"CODE_LOCK.json"),"inputs":sha(fit/"INPUT_ACCEPTANCE.json")}
 for f,h in rec["files"].items():assert sha(dest/f)==h
 for f,h in rec["raw_fit_receipts"].items():assert sha(f)==h
 z=pd.read_csv(dest/"PRIVATE_PREDICTIONS.csv",dtype={"group_hash":str},float_precision="round_trip")
 assert_prediction_alignment(data,t["valid_rows"],z)
stage="pod1_corrected_evaluation_20261024_v1";lock(stage);deps(stage)
assert read(R/"outputs/formal_evaluation_20261023_v1/FUNCTIONAL_QUALIFICATION.json")["status"]=="PASS_synthetic_metrics_calibration_policy_bootstrap"
save(R/"outputs"/stage/"QUALIFICATION.json",{"status":"PASS_POD1_evaluation_source_alignment_and_original_statistics_reuse","contexts_checked":len(tasks),"new_secondary_n":2947,"new_secondary_events":39,"all_current_prediction_receipts_verified":True,"no_heldout_winner_selection":True,"no_performance_score_computed_in_this_check":True})
run(stage+"/evaluate.py")
