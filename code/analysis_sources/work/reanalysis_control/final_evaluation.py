from pathlib import Path
import os,sys,json,subprocess,hashlib,time
R=Path(__file__).resolve().parents[2]
PY=R/"envs/prediction_gpu/bin/python"
CTL=R/"outputs/reanalysis_control"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(q,p)
def run(script,scope=None,args=()):
 env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=""
 if scope:env["AKI_SCOPE_OUT"]=str(R/"outputs"/scope)
 env["AKI_SCOPE_INPUT"]=str(R/"outputs/v4_development_inputs_20261018")
 subprocess.run([str(PY),"-B",str(R/"work"/script),*args],cwd=R,env=env,check=True)
def lock_code(stage):
 out=R/"outputs"/stage;out.mkdir(parents=True,exist_ok=True)
 lock={p.name:sha(p) for p in (R/"work"/stage).glob("*.py")}
 save(out/"CODE_LOCK.json",lock)
lock_code("formal_evaluation_20261023_v1")
from fresh_evaluation_metadata import ensure_fresh_metadata
ensure_fresh_metadata(R)
run("formal_evaluation_20261023_v1/qualify.py")
run("formal_evaluation_20261023_v1/clinical.py",args=("--qualify",))
run("formal_evaluation_20261023_v1/supervisor.py")
