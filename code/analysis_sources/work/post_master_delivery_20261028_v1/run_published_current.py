"""Qualify unchanged published equations, then calculate current-source locked comparisons."""
from pathlib import Path
import json,subprocess,sys,datetime,hashlib,fcntl,os
R=Path(__file__).resolve().parents[2];O=R/"outputs/published_benchmark_comparison_20261024_v1"
def main():
 with (O/"CURRENT_CONTROLLER.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
  for script,log in [(R/"work/published_benchmark_comparison_20261024_v1/qualify.py","CURRENT_QUALIFICATION.log"),(Path(__file__).parent/"published_benchmark_current_v1.py","CURRENT_EXECUTION.log")]:
   with (O/log).open("ab") as stream:
    child=subprocess.Popen([sys.executable,"-B",str(script)],env=env,stdout=stream,stderr=stream)
    (O/"CURRENT_CONTROLLER_STATUS.json").write_text(json.dumps({"status":"RUNNING_EQUATION_QUALIFICATION" if "qualify" in script.name else "RUNNING_CURRENT_FIXED_PUBLISHED_COMPARISONS","time":datetime.datetime.now().astimezone().isoformat(),"worker_PID":child.pid,"script":str(script),"independent_review_still_required":True},indent=2)+"\n")
    try:rc=child.wait(timeout=6500)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned finite published comparator budget exceeded")
   assert rc==0,"Inspect current qualification/execution log before a new retry"
  (O/"CURRENT_CONTROLLER_STATUS.json").write_text(json.dumps({"status":"COMPUTED_INDEPENDENT_REVIEW_REQUIRED","time":datetime.datetime.now().astimezone().isoformat(),"not_a_final_scientific_acceptance":True},indent=2)+"\n")
if __name__=="__main__":main()
