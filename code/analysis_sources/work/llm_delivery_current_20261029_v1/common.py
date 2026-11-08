"""Read-only inputs and finite delivery helpers. No model training or manuscript writes."""
from pathlib import Path
import csv, datetime, fcntl, hashlib, json, os, subprocess
from contextlib import contextmanager
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
L=R/"outputs/llm_supplement_20261028_v1"
F=L/"formal_v1"; S=L/"statistics_v1"; O=L/"delivery_v1"
W=Path(__file__).resolve().parent
MODELS=("biomedlm_2_7b","medgemma_4b_it","llama3_8b_instruct_ngc_v1")
LINES=("geography","temporal"); STAGES=("preop","endop")
SOURCE_SHA="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
FORMAL_SHA="eb27122860f1500b848efeea0ce1a19846e7d94aa5d0de7adbabc2c8db2041de"
FORMAL_CODE_SHA="05040a02b79880f47ea7c480c34e4f4db62e85494cb89143ea5f3bbb79e2ddce"
STATS_LOCK_SHA="488e00e7f2c1004f2309e86a6eb1f86a82896bb355735b79bdf83383f6b7c851"
def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 temp=p.with_name(p.name+".tmp."+str(os.getpid()))
 with temp.open("w") as f:
  json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def write_csv(p,rows):
 assert rows
 p=Path(p);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def verify_files(base,files):
 for p,h in files.items():
  p=Path(p);p=p if p.is_absolute() else Path(base)/p
  assert sha(p)==h,("file_hash_mismatch",str(p))
def check_code():
 p=W/"DELIVERY_CODE_LOCK.json";lock=read(p)
 verify_files(W,lock["files"])
 return sha(p)
def formal_guard():
 props=subprocess.run(["systemctl","--user","show","aki-postslot-llm-formal-20261029-v1.service","-p","ActiveState,Result,ExecMainStatus"],capture_output=True,text=True,check=True,timeout=10).stdout
 state=dict(x.split("=",1) for x in props.splitlines() if "=" in x)
 assert state=={"ActiveState":"inactive","Result":"success","ExecMainStatus":"0"},"FORMAL_NOT_CLOSED_NO_REAL_COMPARISON_ALLOWED"
 assert sha(F/"FORMAL_TRAINING_CONTRACT.json")==FORMAL_SHA
 assert sha(R/"work/llm_supplement_20261028_v1/formal_benchmark_v1.py")==FORMAL_CODE_SHA
 assert sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")==SOURCE_SHA
 a=read(F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json")
 assert a["formal_fits"]==48 and a["original_checkpoint_banks"]==6 and a["engineering_pilots"]==3
 assert a["contract_sha256"]==FORMAL_SHA
 assert read(F/"STATUS.json")["status"]=="COMPLETE_TRAINING_AND_PREDICTION_BANKS_STATISTICAL_COMPARISON_REQUIRED"
 expected={(m,s,l,str(f)) for m in MODELS for s in STAGES for l in LINES for f in (0,1,2,"final")}
 fits=[(x["model"],x["stage"],x["line"],str(x["fit"])) for x in a["tasks"] if x["kind"]=="fit"]
 assert len(fits)==len(set(fits))==48 and set(fits)==expected
 assert {(x["model"],x["stage"]) for x in a["tasks"] if x["kind"]=="before"}=={(m,s) for m in MODELS for s in STAGES}
 for x in a["tasks"]:
  p=Path(x["receipt"]);assert sha(p)==x["sha256"]
  r=read(p);verify_files(p.parent,r["files"])
  for k in ("model","stage","line","fit","kind"):assert str(r[k])==str(x[k])
 return a
def numeric_guard():
 formal=formal_guard();p=R/"work/llm_comparison_current_20261029_v1/STATISTICS_CODE_LOCK.json"
 assert sha(p)==STATS_LOCK_SHA;verify_files(p.parent,read(p)["files"])
 a=read(S/"NUMERICAL_COMPARISON_ACCEPTANCE.json")
 assert a["LLM_result_routes"]==24 and a["bootstrap_replicates_per_context"]==2000
 assert a["formal_training_acceptance_sha256"]==sha(F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json")
 assert a["statistics_protocol_sha256"]==sha(S/"STATISTICS_PROTOCOL.json")
 assert a["code_lock_sha256"]==STATS_LOCK_SHA
 assert {r["line"] for r in a["receipts"]}==set(LINES)
 for ref in a["receipts"]:
  p=Path(ref["receipt"]);assert sha(p)==ref["sha256"];verify_files(p.parent,read(p)["files"])
  d=S/ref["line"];prep=read(d/"PREPARATION_ACCEPTANCE.json")
  verify_files(d,prep["input_files_sha256"]);verify_files(d,prep["sources"])
 return formal,a
@contextmanager
def owned_lock(name):
 O.mkdir(mode=0o700,parents=True,exist_ok=True)
 with (O/("."+name+".lock")).open("a+") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield
def existing_receipt(name,inputs):
 p=O/name
 if not p.exists():return False
 a=read(p)
 assert a["code_lock_sha256"]==check_code() and a["inputs_sha256"]==inputs,"Preserve old evidence; new version or diagnosed repair required"
 verify_files(O,a["files"])
 return True
