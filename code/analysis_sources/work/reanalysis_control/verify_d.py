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
os.environ["AKI_SCOPE_OUT"]=str(R/"outputs/ensemble_continuation_20261018_v1")
sys.path.insert(0,str(R/"work/ensemble_continuation_20261018_v1"))
import coordinator_d as d
man=json.loads((d.OUT/"SCOPE_MANIFEST.json").read_text());jobs=d.build(man)
assert len(jobs)==13308
assert all(d.good(j) for j in jobs)
frozen=json.loads((d.OUT/"CODE_LOCK.json").read_text())
assert frozen=={p.name:sha(p) for p in d.CODE.glob("*.py")}
save(d.OUT/"FINALIZATION_VERIFICATION_RECEIPT.json",{"status":"PASS","all_task_receipts_verified":len(jobs),"source_lock_sha256":sha(d.INPUT/"SOURCE_LOCK.json"),"D_acceptance_sha256":sha(d.OUT/"STAGE_D_ACCEPTANCE.json"),"code_lock_sha256":sha(d.OUT/"CODE_LOCK.json"),"time":time.time()})
