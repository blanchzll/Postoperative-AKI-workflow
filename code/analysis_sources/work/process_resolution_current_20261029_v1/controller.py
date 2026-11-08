"""Current corrected-source four-issue preparation, real qualification, finite batch and independent QC."""
from pathlib import Path
import json,hashlib,datetime,subprocess,sys,os,fcntl,traceback
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;O=R/"outputs/process_resolution_20261025_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 t=Path(str(p)+".tmp");t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n");os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def check():
 for name,digest in read(O/"CURRENT_CODE_LOCK.json").items():assert sha(W/name)==digest
def run(script,limit):
 check();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
 with (O/(script+".log")).open("ab") as log:
  child=subprocess.Popen([sys.executable,"-B",str(W/script)],env=env,stdout=log,stderr=log)
  save(O/"CURRENT_CONTROLLER_STATUS.json",{"status":"RUNNING_CURRENT_SOURCE_FOUR_ISSUE_"+script,"time":now(),"worker_PID":child.pid,"main_LLM_not_interrupted":True,"final_scientific_review_performed":False})
  try:rc=child.wait(timeout=limit)
  except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned finite support-stage wallclock exceeded; valid checkpoints preserved")
 assert rc==0,"Inspect concrete child failure before retry: "+script
def main():
 with (O/"CURRENT_CONTROLLER.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
  if not (O/"INPUT_ACCEPTANCE.json").exists():run("prepare.py",600)
  else:
   rec=read(O/"INPUT_ACCEPTANCE.json");assert sha(W/"PRIVATE_INPUTS.pkl")==rec["prepared_pickle_sha256"]
   assert sha(O/"ANALYSIS_CONTRACT.json")==rec["contract_sha256"]
   for p,h in rec["sources_sha256"].items():assert sha(p)==h
  if not (O/"QUALIFICATION.json").exists():run("qualify.py",1200)
  else:
   for name,digest in read(O/"QUALIFICATION.json")["code_sha256"].items():assert sha(W/name)==digest
  if not (O/"RUN_STATUS.json").exists() or read(O/"RUN_STATUS.json")["status"]!="completed":run("runner.py",14000)
  state=read(O/"RUN_STATUS.json")
  if state["status"].startswith("WAIT_RESOURCE"):
   save(O/"CURRENT_CONTROLLER_STATUS.json",{"status":"WAIT_RESOURCE_VALID_INPUTS_QUALIFICATION_PILOTS_CHUNKS_PRESERVED","time":now(),"native_status":state,"requires_user_intervention":False});return
  assert state["status"]=="completed"
  if not (O/"RESULT_ACCEPTANCE.json").exists():run("aggregate.py",1200)
  if not (O/"INDEPENDENT_RESULT_VERIFICATION.json").exists():run("verify_results.py",1200)
  save(O/"CURRENT_CONTROLLER_STATUS.json",{"status":"COMPLETE_FOUR_CURRENT_SOURCE_ISSUE_CALCULATIONS_AND_INDEPENDENT_VERIFICATION_REPORT_AND_FINAL_SCIENCE_PENDING","time":now(),"receipts":{str(O/n):sha(O/n) for n in ["INPUT_ACCEPTANCE.json","QUALIFICATION.json","RUN_MANIFEST.json","RESULT_ACCEPTANCE.json","INDEPENDENT_RESULT_VERIFICATION.json"]},"final_scientific_review_completed":False,"whole_project_complete":False})
if __name__=="__main__":
 try:main()
 except BaseException as e:
  save(O/"CURRENT_CONTROLLER_FAILURE.json",{"time":now(),"error_class":type(e).__name__,"message":str(e)[:500],"stack":[{"file":str(Path(v.filename)),"line":v.lineno,"function":v.name} for v in traceback.extract_tb(e.__traceback__)[-10:]],"existing_current_primary_models_and_LLM_not_changed":True});raise
