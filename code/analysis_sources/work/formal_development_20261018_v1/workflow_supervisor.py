"""Finite A->B->C continuation and truthful current-state bookkeeping."""
from core import *
import subprocess,traceback
def alive(pid):
 try:
  os.kill(int(pid),0)
  stat=Path(f"/proc/{pid}/stat").read_text().split()
  return stat[2]!="Z"
 except (OSError,ValueError):return False
def load(p):return json.loads(p.read_text()) if p.exists() else None
def main():
 lock=(OUT/"WORKFLOW_SUPERVISOR.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic(OUT/"WORKFLOW_CAPABILITY_BOUNDARY.json",{"automated_sequence":["A:24cohort-specific six-imputer/nine-base coarse selections","B:540 expanded base-family hyperparameter studies","C:450 raw base-family outer fits"],"not_yet_executed_or_claimed":["Voting/Stacking full-procedure qualification/development","honest full-procedure calibration","final model lock","current hospital/time evaluation","SHAP/POD1/sensitivity updates","decision scenarios","albumin association","web calculator/public release"],"no_full_research_completion_implied_by_batch_exit":True})
 while True:
  a=load(OUT/"status_a.json");b=load(OUT/"status_b.json");c=load(OUT/"status_c.json")
  failures=[str(p.relative_to(OUT)) for p in OUT.glob("COORDINATOR_*_FAILURE.json")]
  if (OUT/"STAGE_B_ACCEPTANCE.json").exists() and not (OUT/"LAUNCH_C.json").exists() and not failures:
   log=(OUT/"COORDINATOR_C.log").open("a");env=os.environ.copy()
   env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]});env["PYTHONDONTWRITEBYTECODE"]="1"
   pp=subprocess.Popen([str(PYTHON),"-B",str(CODE/"coordinator_c.py")],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True,close_fds=True);log.close()
   atomic(OUT/"LAUNCH_C.json",{"pid":pp.pid,"launched":time.time(),"scope":"raw base outer validation only"})
  running=[]
  for n in ["A","B","C"]:
   l=load(OUT/("LAUNCH_"+n+".json"))
   if l and alive(l["pid"]):running.append({"stage":n,"pid":l["pid"]})
  compute_running=any((x and x.get("state","").startswith("running")) for x in [a,b,c])
  state=load(ROOT/"outputs/CURRENT_ANALYSIS.json")
  state["status"]="formal_development_batches_running" if running and not failures else ("formal_development_failure_preserved" if failures else "formal_development_raw_base_batch_complete_next_methods_pending")
  state["active_compute"]={"task_owned_compute_running":compute_running,"task_owned_controllers_alive":running,"supervisor_pid":os.getpid(),"status_files":[str((OUT/n).relative_to(ROOT)) for n in ["status_a.json","status_b.json","status_c.json"]],"updated_UTC_epoch":time.time()}
  state.setdefault("formal_development",{}).update({"stage_A":a,"stage_B":b,"stage_C":c,"capability_boundary":str((OUT/"WORKFLOW_CAPABILITY_BOUNDARY.json").relative_to(ROOT)),"failures":failures,"final_model_lock":"not_complete","hospital_time_evaluation":"not_started"})
  atomic(ROOT/"outputs/CURRENT_ANALYSIS.json",state)
  atomic(OUT/"workflow_status.json",{"state":state["status"],"controllers":running,"stage_A_complete":bool((OUT/"STAGE_A_ACCEPTANCE.json").exists()),"stage_B_complete":bool((OUT/"STAGE_B_ACCEPTANCE.json").exists()),"stage_C_complete":bool((OUT/"STAGE_C_ACCEPTANCE.json").exists()),"failure_files":failures,"updated":time.time(),"entire_research_complete":False})
  if failures or (OUT/"STAGE_C_ACCEPTANCE.json").exists():break
  if not running:
   atomic(OUT/"WORKFLOW_STOPPED_WITHOUT_ACCEPTANCE.json",{"time":time.time(),"reason":"no task controllers alive; no automatic identical retry"})
   break
  time.sleep(30)
 lock.close()
if __name__=="__main__":
 try:main()
 except Exception:
  atomic(OUT/"WORKFLOW_SUPERVISOR_FAILURE.json",{"time":time.time(),"traceback":traceback.format_exc()});raise
