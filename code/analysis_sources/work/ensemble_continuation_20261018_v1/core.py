"""Versioned training-boundary utilities; original sklearn runtime only."""
from pathlib import Path
import os,sys,json,hashlib,time,fcntl
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=Path(os.environ.get("AKI_SCOPE_OUT",str(ROOT/"outputs/ensemble_continuation_20261018_v1")))
INPUT=Path(os.environ.get("AKI_SCOPE_INPUT",str(ROOT/"outputs/v4_development_inputs_20261018")))
PLAN=ROOT/"outputs/research_plan_revision_20261018_094007"
PYTHON=ROOT/"envs/prediction_gpu/bin/python"
SHARED=Path("/home/lei/codex-workspaces/shared-python-envs/envs/clinical-ml-py312-cpu-xgb214-20261016-v1/bin/python")
FAMILIES=["LR","DT","RF","KNN","SVM","NB","XGB","SGBT","NNET"]
METHODS=["bayes","pmm","cart","knn","mean_mode","random_forest"]
COMPACT=["Age","Sex","PreopCr","PreopBUN","PreopAlb","PreopHb","Diabetes","Hypertension","HeartDisease"]
@lru_cache(maxsize=4096)
def _sha(p,mtime,size):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha(p):
 p=Path(p);st=p.stat();return _sha(str(p),st.st_mtime_ns,st.st_size)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+"."+str(os.getpid())+".tmp")
 with q.open("w") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def ready(d,signature=None):
 p=Path(d)/"COMPLETE.json"
 if not p.exists():return False
 r=json.loads(p.read_text())
 if signature is not None and r["signature"]!=signature:raise ValueError("Stale signature:"+str(d))
 for f,h in r.get("files",r.get("outputs_sha256",{})).items():
  if sha(Path(d)/f)!=h:raise ValueError("Corrupt artifact:"+str(Path(d)/f))
 return True
def spec_for(stage,columns=None):
 sp=json.loads((INPUT/"FEATURE_SPECS.json").read_text())["endop" if stage=="endop" else "preop"]
 if columns is not None:
  assert set(columns)<=set(sp["num"]+sp["cat"])
  sp={"num":[c for c in sp["num"] if c in columns],"cat":[c for c in sp["cat"] if c in columns],"cat_levels":{c:sp["cat_levels"][c] for c in sp["cat"] if c in columns}}
 return sp
def parts(line,cohort,context):
 allp=json.loads((INPUT/(line+"_"+cohort+"_resampling.json")).read_text())
 if context=="final":return [p for p in allp if p["scheme"]=="final_development_inner"]
 outer=int(context.split("_")[1])
 return [p for p in allp if p["scheme"]=="inner" and p["parent_outer"]==outer]
def pair_task(line,cohort,stage,context,part,method,branch="full"):
 return {"kind":"impute","line":line,"cohort":cohort,"stage":stage,"context":context,"fold":part["fold"],"method":method,"branch":branch,"train_rows":part["train_excel_rows"],"assessment_rows":part["assessment_excel_rows"]}
def cache_signature(t):
 cols=COMPACT if t["branch"]=="compact" else t.get("columns")
 sp=spec_for(t["stage"],cols)
 features=INPUT/(t["cohort"]+"_"+t["stage"]+"_features.csv")
 return {"engine":"stable_v2_original_sklearn1.5.2","input_sha256":sha(features),"spec":sp,"train_rows":t["train_rows"],"assessment_rows":t["assessment_rows"],"method":t["method"],"seed":42,"iterations":5,"trees":200,"neighbors":5,"imputer_code":{f:sha(ROOT/"work"/f) for f in ["v2_stable_imputation_20261017/stable_imputer.py","python_six_outcomes_20261017/mixed_imputer.py","python_six_outcomes_20261017/donor_helpers.py"]}}
def cache_dir(t):return OUT/"cache"/digest(cache_signature(t))[:24]
def frame_and_meta(t,sp):
 import pandas as pd
 meta=pd.read_csv(INPUT/(t["cohort"]+"_metadata.csv"),dtype={"group_hash":str}).set_index("excel_row",drop=False)
 tr=meta.loc[t["train_rows"]];va=meta.loc[t["assessment_rows"]]
 if set(tr.group_hash)&set(va.group_hash):raise ValueError("Patient-group leakage")
 for m in [tr,va]:
  if not m[t["line"]+"_partition_v2"].eq("development").all():raise ValueError("Final evaluation used in training operation")
 x=pd.read_csv(INPUT/(t["cohort"]+"_"+t["stage"]+"_features.csv"),dtype={c:"string" for c in sp["cat"]},float_precision="round_trip").set_index("excel_row")
 cols=sp["num"]+sp["cat"]
 return x.loc[tr.index,cols],tr,x.loc[va.index,cols],va
def keys(meta):return (meta.group_hash.astype(str)+"|"+meta.excel_row.astype(str)).tolist()
def read_cache(t):
 import pandas as pd
 d=cache_dir(t);sig=cache_signature(t)
 assert ready(d,sig),str(d)
 sp=sig["spec"];frames=[]
 for f in ["train.csv","assessment.csv"]:
  frames.append(pd.read_csv(d/f,dtype={c:"string" for c in sp["cat"]},float_precision="round_trip").set_index("excel_row"))
 meta=pd.read_csv(INPUT/(t["cohort"]+"_metadata.csv"),dtype={"group_hash":str}).set_index("excel_row",drop=False)
 tm=meta.loc[t["train_rows"]];vm=meta.loc[t["assessment_rows"]]
 assert frames[0].index.tolist()==tm.index.tolist() and frames[1].index.tolist()==vm.index.tolist()
 assert set(tm.group_hash).isdisjoint(vm.group_hash)
 assert list(frames[0])==sp["num"]+sp["cat"]
 return frames[0],tm,frames[1],vm,sp
def resource_snapshot():
 mem=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
 # Resolve cgroup membership on both unified and legacy controllers.
 lines=Path("/proc/self/cgroup").read_text().splitlines()
 for line in lines:
  _,controllers,rel=line.split(":",2)
  if not controllers:
   base=Path("/sys/fs/cgroup")/rel.lstrip("/")
   for parent in [base,*base.parents]:
    if parent==Path("/sys"):break
    lim=parent/"memory.max";used=parent/"memory.current"
    if lim.exists() and used.exists() and lim.read_text().strip()!="max":
     mem=min(mem,max(0,int(lim.read_text())-int(used.read_text())))
  elif "memory" in controllers.split(","):
   base=Path("/sys/fs/cgroup/memory")/rel.lstrip("/")
   if (base/"memory.limit_in_bytes").exists():
    mem=min(mem,max(0,int((base/"memory.limit_in_bytes").read_text())-int((base/"memory.usage_in_bytes").read_text())))
 return {"available_bytes":mem,"effective_cpu":len(os.sched_getaffinity(0)),"time":time.time()}

def process_rss(pid):
 total=0
 try:
  for line in Path(f"/proc/{pid}/status").read_text().splitlines():
   if line.startswith("VmRSS:"):total+=int(line.split()[1])*1024;break
  children=Path(f"/proc/{pid}/task/{pid}/children")
  if children.exists():
   for child in children.read_text().split():total+=process_rss(int(child))
 except (OSError,ProcessLookupError):pass
 return total

# Execution-only resource admission for the post-memory-upgrade run.

_base_resource_snapshot=resource_snapshot
_cpu_sample=None
_other_cpu_ewma=None
def resource_snapshot():
 global _cpu_sample,_other_cpu_ewma
 import math
 snap=_base_resource_snapshot();effective=snap["effective_cpu"]
 rel=next(x.split(":",2)[2] for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
 cg=Path("/sys/fs/cgroup")/rel.lstrip("/")
 for parent in [cg,*cg.parents]:
  q=parent/"cpu.max"
  if q.exists():
   quota,period=q.read_text().split()
   if quota!="max":effective=min(effective,max(1,int(quota)//int(period)))
 rows={}
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:
   v=(p/"stat").read_text().split(") ",1)[1].split()
   rows[int(p.name)]=(int(v[11])+int(v[12]),int(v[1]),v[19])
  except (OSError,ValueError):pass
 own={os.getpid()}
 while True:
  n=len(own);own.update(pid for pid,v in rows.items() if v[1] in own)
  if len(own)==n:break
 now=time.monotonic()
 if _cpu_sample is not None and now-_cpu_sample[0]>=1:
  oldtime,oldrows=_cpu_sample
  ticks=sum(max(0,v[0]-oldrows[p][0]) for p,v in rows.items() if p not in own and p in oldrows and v[2]==oldrows[p][2])
  other=ticks/os.sysconf("SC_CLK_TCK")/(now-oldtime)
  _other_cpu_ewma=other if _other_cpu_ewma is None else .65*_other_cpu_ewma+.35*other
  _cpu_sample=(now,rows)
 elif _cpu_sample is None:_cpu_sample=(now,rows)
 snap["affinity_and_quota_cpu"]=effective
 snap["other_measured_cpu_cores"]=_other_cpu_ewma or 0
 snap["effective_cpu"]=max(1,effective-int(math.ceil(_other_cpu_ewma or 0)))
 return snap
