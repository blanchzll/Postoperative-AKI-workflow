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
run("ensemble_calibration_20261020_v1/prepare_e.py","ensemble_calibration_20261020_v1")
