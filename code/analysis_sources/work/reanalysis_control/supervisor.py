
from pathlib import Path
import os,sys,json,subprocess,time,fcntl,traceback,datetime,hashlib
R=Path(__file__).resolve().parents[2];P=R.parents[1];O=R/"outputs/reanalysis_control";C=R/"work/formal_development_20261018_v1";B=R/"outputs/formal_development_20261018_v1";PY=R/"envs/prediction_gpu/bin/python"
def save(p,x):
 q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(q,p)
def update(stage,status,**kw):
 s=json.loads((R/"outputs/CURRENT_ANALYSIS.json").read_text());s.update(status=status,current_stage=stage,controller_PID=os.getpid(),updated_UTC=time.time(),**kw)
 for p in [R/"outputs/CURRENT_ANALYSIS.json",P/"outputs/CURRENT_ANALYSIS.json",P/"outputs/ALT_CORRECTION_IN_PROGRESS.json"]:save(p,s)
def run(stage,script,receipt,expected=None):
 gate=json.loads(Path("/home/lei/codex-workspaces/MAINTENANCE_HOLD_20261025.json").read_text())
 assert gate.get("new_research_launches_allowed") and not gate.get("no_automatic_resume"),"User hardware hold active"
 assert hashlib.sha256((R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json").read_bytes()).hexdigest()=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3","Input source changed"
 if receipt.exists():
  rec=json.loads(receipt.read_text())
  if expected:assert rec["status"]==expected
  return
 update(stage,"RUNNING_CORRECTED_DATA_REANALYSIS",model_training_started=True)
 log=O/(stage+".log")
 with log.open("a") as f:
  p=subprocess.Popen([str(PY),"-B",str(script)],cwd=R,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  save(O/(stage+"_LAUNCH.json"),{"PID":p.pid,"stage":stage,"script":str(script),"time":time.time()})
  if stage=="A":save(B/"LAUNCH_A.json",{"pid":p.pid,"time":time.time()})
  code=p.wait()
 assert code==0,stage+" exit="+str(code)+" log="+str(log)
 assert receipt.exists(),stage+" missing receipt"
 if expected:assert json.loads(receipt.read_text())["status"]==expected
 save(O/(stage+"_ACCEPTANCE.json"),{"status":"PASS","receipt":str(receipt),"time":time.time()})
def main():
 owner=(P/"outputs/ACTIVE_REANALYSIS.lock").open("a");fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.environ["CUDA_VISIBLE_DEVICES"]="" # Main pipeline uses the validated CPU implementations; TabPFN is a separate GPU batch.
 lock=(O/"RUN.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="1"
 os.environ["PYTHONDONTWRITEBYTECODE"]="1"
 run("QUALIFY_A",C/"qualify_a.py",B/"FUNCTIONAL_QUALIFICATION.json","PASS")
 run("A",C/"coordinator_a.py",B/"STAGE_A_ACCEPTANCE.json","complete_outer_training_coarse_selection_only")
 run("QUALIFY_MODELS",C/"qualify_models.py",B/"MODEL_ENGINE_FUNCTIONAL_QUALIFICATION.json","PASS")
 run("QUALIFY_BRIDGE",R/"work/reanalysis_control/qualify_bridge.py",B/"BRIDGE_V2_RESUME_ACCEPTANCE.json","PASS")
 run("QUALIFY_BUNDLE",C/"qualify_bundle.py",B/"RAW_BUNDLE_FUNCTIONAL_QUALIFICATION.json","PASS")
 run("B",C/"coordinator_b.py",B/"STAGE_B_ACCEPTANCE.json","complete540_base_family_searches_only")
 run("C",C/"coordinator_c.py",B/"STAGE_C_ACCEPTANCE.json","complete450_raw_base_outer_fits_only")
 # This stage hands control to a separately qualified downstream recipe. Never infer whole-project completion.
 plan=O/"DOWNSTREAM_RECIPE.json"
 if plan.exists():
  for row in json.loads(plan.read_text())["steps"]:run(row["stage"],Path(row["script"]),Path(row["receipt"]),row.get("expected"))
 update("MAIN_MODEL_EVALUATION_COMPLETE","MAIN_COMPLETE_SUPPLEMENTS_AND_DELIVERABLES_REQUIRED",whole_reanalysis_complete=False)
if __name__=="__main__":
 try:main()
 except BaseException:
  save(O/"FAILURE.json",{"traceback":traceback.format_exc(),"time":time.time()})
  update("FAILED_REQUIRES_DIAGNOSIS","REANALYSIS_FAILURE_CHECKPOINTS_PRESERVED",whole_reanalysis_complete=False)
  raise
