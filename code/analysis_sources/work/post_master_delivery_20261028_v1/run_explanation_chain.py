"""Finite current-source SHAP and risk-curve continuation; no model refitting."""
from pathlib import Path
import os,sys,json,hashlib,datetime,fcntl,subprocess
R=Path(__file__).resolve().parents[2]
C=R/"outputs/reanalysis_control"
D=R/"outputs/post_master_delivery_20261028_v1"
PY=R/"envs/prediction_gpu/bin/python"
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 t=Path(p).with_name(Path(p).name+".tmp");t.write_text(json.dumps(d,indent=2)+"\n");os.replace(t,p)
def gate():
 h=read("/home/lei/codex-workspaces/MAINTENANCE_HOLD_20261025.json")
 assert h["new_research_launches_allowed"] and not h["no_automatic_resume"]
 assert sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
 assert read(R/"outputs/CURRENT_ANALYSIS.json")["status"]=="MAIN_COMPLETE_SUPPLEMENTS_AND_DELIVERABLES_REQUIRED"
def main():
 guard=(R.parents[1]/"outputs/ACTIVE_REANALYSIS.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 own=(C/"EXPLANATION_CHAIN.lock").open("a");fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gate();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
 prepare=R/"work/post_master_delivery_20261028_v1/prepare_locked_explanations_current.py"
 assert sha(prepare)=="a8acf1299a814afb6771907de8667835609a796d8156b2c22cc2e49bae43a631"
 if not (D/"EXPLANATION_PREPARATION.json").exists():
  save(C/"EXPLANATION_CHAIN_STATUS.json",{"status":"RUNNING_CURRENT_SOURCE_MATH_AND_BOUNDARY_PREPARATION","PID":os.getpid(),"time":now()})
  subprocess.run([str(PY),"-u","-B",str(prepare)],cwd=R,env=env,check=True)
 receipts={}
 for stage,name in [("locked_pipeline_shap_20261024_v1","STAGE_SHAP_ACCEPTANCE.json"),("locked_risk_curves_20261024_v1","STAGE_CURVE_ACCEPTANCE.json")]:
  gate();out=R/"outputs"/stage;receipt=out/name
  if not receipt.exists():
   save(C/"EXPLANATION_CHAIN_STATUS.json",{"status":"RUNNING_NATIVE_"+stage,"PID":os.getpid(),"time":now(),"previous_receipts":receipts,"models_refitted":False})
   subprocess.run([str(PY),"-u","-B",str(R/"work"/stage/"supervisor.py")],cwd=R,env=env,check=True)
  result=read(receipt);assert len(result["receipts"])==10
  for f,h in result["receipts"].items():
   assert sha(f)==h
   for name,digest in read(f)["files"].items():assert sha(Path(f).parent/name)==digest
  receipts[str(receipt)]=sha(receipt)
 save(C/"EXPLANATION_CHAIN_STATUS.json",{"status":"COMPLETE_NATIVE_SHAP_AND_RISK_CURVE_TABLES_RENDERING_AND_REVIEW_STILL_REQUIRED","PID":os.getpid(),"time":now(),"receipts":receipts,"whole_project_complete":False,"owned_workers_remaining":0})
 save(D/"EXPLANATION_NATIVE_CONTINUATION_ACCEPTANCE.json",{"status":"PASS_CURRENT_SOURCE_NATIVE_SHAP_AND_RISK_CURVE_COMPUTATION_ONLY","time":now(),"receipts":receipts,"manuscript_changed":False,"whole_project_complete":False})
if __name__=="__main__":
 try:main()
 except BaseException as e:
  save(C/"EXPLANATION_CHAIN_FAILURE.json",{"time":now(),"error_class":type(e).__name__,"diagnostic":str(e)[:800],"valid_outputs_preserved":True});raise
