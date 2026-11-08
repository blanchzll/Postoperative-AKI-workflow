
from pathlib import Path
import sys,json,subprocess,os
R=Path(__file__).resolve().parents[2];C=R/"work/formal_development_20261018_v1"
sys.path.insert(0,str(C));from core import *
method=json.loads((OUT/"selection/geography/preop/final/METHOD_SELECTION.json").read_text())["selected_method"]
task={"line":"geography","cohort":"preop","stage":"preop","branch":"compact","context":"final","family":"LR","method":method}
req=OUT/"requests/bridge_current_release.json";atomic(req,task)
dest=OUT/"search_v2/geography/preop/preop/compact/final/LR"
cmd=[str(SHARED),"-B",str(C/"search_worker.py"),str(req),"--threads","1"]
subprocess.run(cmd+["--target","4"],check=True,cwd=R)
first=json.loads((dest/"TRIALS.json").read_text())["trials"]
assert len(first)==4 and not (dest/"COMPLETE.json").exists()
subprocess.run(cmd,check=True,cwd=R)
full=json.loads((dest/"TRIALS.json").read_text())["trials"]
assert full[:4]==first and len(full)==96
assert len({r["candidate_id"] for r in full})==96
assert [r["phase"] for r in full]==["random"]*48+["tpe"]*48
assert ready(dest)
atomic(OUT/"BRIDGE_V2_RESUME_ACCEPTANCE.json",{"status":"PASS","valid_unique_trials":96,"fits":288,"first_four_trials_identical_after_resume":True,"exact48random48TPE_check":"PASS","source_lock_sha256":sha(INPUT/"SOURCE_LOCK.json"),"code_sha256":{n:sha(C/n) for n in ["model_engine.py","search_worker.py"]},"no_final_evaluation_used":True})
