"""Finite original-runtime finalizer; preserves running jobs and diagnostics."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,fcntl,traceback
ROOT=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
OUT=ROOT/"outputs/ensemble_continuation_20261018_v1"
CODE=ROOT/"work/ensemble_continuation_20261018_v1"
def atomic(p,value):
 q=Path(str(p)+".tmp")
 with q.open("w") as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def live(pid):
 try:
  state=(Path("/proc")/str(pid)/"stat").read_text().split()[2]
  return state!="Z"
 except FileNotFoundError:return False
def main():
 lock=(OUT/"FINALIZER.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=json.loads((OUT/"LAUNCH_D.json").read_text());pid=launch["pid"]
 while live(pid):
  atomic(OUT/"finalizer_status.json",{"state":"waiting_for_finite_D_batch","controller_pid":pid,"updated":time.time(),"no_compute_restarted":True})
  time.sleep(60)
 if (OUT/"STAGE_D_ACCEPTANCE.json").exists():
  atomic(OUT/"finalizer_status.json",{"state":"D_acceptance_already_present","updated":time.time(),"no_refits":True});return
 failure=OUT/"COORDINATOR_D_FAILURE.json"
 if not failure.exists():raise RuntimeError("Controller exited without scientific acceptance or diagnostics")
 f=json.loads(failure.read_text())
 if "ModuleNotFoundError: No module named 'pandas'" not in f["traceback"] or "summarize(man)" not in f["traceback"]:
  raise RuntimeError("Unqualified failure; preserved for diagnosis, not automatically retried")
 # Only an orchestration-only end-of-batch summary failure can be repaired here.
 # All model/search processes used their unchanged original/shared explicit paths.
 program="""
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from core import *
from coordinator_d import build,good,summarize
man=json.loads((OUT/'SCOPE_MANIFEST.json').read_text());jobs=build(man)
assert all(good(j) for j in jobs),'Not all scientific artifacts accepted'
lock=(OUT/'COORDINATOR_D.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
frozen=json.loads((OUT/'CODE_LOCK.json').read_text())
assert frozen=={p.name:sha(p) for p in CODE.glob('*.py')},'Active code changed'
summarize(man)
atomic(OUT/'FINALIZATION_RECOVERY_RECEIPT.json',{'status':'PASS','cause':'launcher resolved venv symlink for lightweight coordinator; model/search interpreters were correct','repair':'reconstructed summaries only using original frozen Python; no fitting repeated','prior_diagnostic_sha256':sha(OUT/'COORDINATOR_D_FAILURE.json'),'all_task_receipts_verified':len(jobs),'original_model_runtime':sys.executable,'scientific_results_unchanged':True})
atomic(OUT/'status_d.json',{'state':'complete_raw11_and_base_calibration_only','complete':len(jobs),'total':len(jobs),'updated':time.time(),'task_owned_workers_remaining':0,'strict_ensemble_calibration_complete':False,'final_model_locked':False,'entire_research_complete':False,'finalization_recovery_verified':True})
"""
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
 subprocess.run([str(ROOT/"envs/prediction_gpu/bin/python"),"-B","-c",program,str(CODE)],cwd=ROOT,env=env,check=True)
 atomic(OUT/"finalizer_status.json",{"state":"verified_summary_recovered_without_refits","updated":time.time(),"no_compute_restarted":True})
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"FINALIZER_FAILURE.json",{"time":time.time(),"traceback":traceback.format_exc()});raise
