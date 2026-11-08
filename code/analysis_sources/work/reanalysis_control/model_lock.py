from pathlib import Path
import os,sys,json,subprocess,hashlib,time
R=Path(__file__).resolve().parents[2]
PY=R/"envs/prediction_gpu/bin/python"
CTL=R/"outputs/reanalysis_control"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(q,p)
def run(script,scope=None):
 env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=""
 if scope:env["AKI_SCOPE_OUT"]=str(R/"outputs"/scope)
 env["AKI_SCOPE_INPUT"]=str(R/"outputs/v4_development_inputs_20261018")
 subprocess.run([str(PY),"-B",str(R/"work"/script)],cwd=R,env=env,check=True)
out=R/"outputs/model_lock_20261022_v1";out.mkdir(parents=True,exist_ok=True)
code=R/"work/model_lock_20261022_v1";lock={p.name:sha(p) for p in code.glob("*.py")}
save(out/"CODE_LOCK.json",lock)
save(R/"outputs/ensemble_speedup_20261021_v2/BASELINE_CPU_SHARE.json",{"ceil_baseline":1,"reason":"No historical reserved CPU quota; dynamically use actual available affinity and quota."})
from model_lock_contract import ensure_contract
ensure_contract(R)
run("model_lock_20261022_v1/qualify_worker.py","ensemble_calibration_20261020_v1")
q=json.loads((out/"FUNCTIONAL_QUALIFICATION.json").read_text())
assert q["status"]=="PASS_model_lock_engineering_only" and q["code_sha256"]==lock
run("model_lock_20261022_v1/lock_models.py","ensemble_calibration_20261020_v1")
