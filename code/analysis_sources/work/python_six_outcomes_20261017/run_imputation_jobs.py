"""Frozen input signatures, finite-output checks and live resource snapshots."""
from pathlib import Path
import json,hashlib,csv,os
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"outputs/python_six_outcomes_20261017"
CODE=ROOT/"work/python_six_outcomes_20261017"
METHODS=["mean_mode","random_forest","pmm","bayes","cart","knn"];MIB=1024**2
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def resource_snapshot():
    mem={k:int(v.split()[0])*1024 for k,v in (s.split(":",1) for s in Path("/proc/meminfo").read_text().splitlines())}
    cg=Path("/sys/fs/cgroup")/Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
    free=mem["MemAvailable"];threads=len(os.sched_getaffinity(0));limits=[]
    for d in [cg,*cg.parents]:
        if not str(d).startswith("/sys/fs/cgroup"):continue
        item={"path":str(d)}
        for n in ["memory.max","memory.high","memory.current","cpu.max"]:
            if (d/n).exists():item[n]=(d/n).read_text().strip()
        for n in ["memory.max","memory.high"]:
            if item.get(n,"max")!="max":free=min(free,max(0,int(item[n])-int(item.get("memory.current","0"))))
        if "cpu.max" in item:
            quota,period=item["cpu.max"].split()
            if quota!="max":threads=min(threads,max(1,int(int(quota)/int(period))))
        limits.append(item)
    return {"time":__import__("time").time(),"mem_available":mem["MemAvailable"],"effective_free":free,"threads":threads,"limits":limits}
def signature(pair,method):
    files=[ROOT/pair["files"][s]["path"] for s in ["train_features","valid_features","train_meta","valid_meta"]]
    files += [BASE/"fold_inputs/feature_spec.json",BASE/"imputation_protocol.json"]
    files += [CODE/n for n in ["mixed_imputer.py","donor_helpers.py","impute_worker.py"]]
    return {"engine":"Python","method":method,"seed":42,"files":{str(f.relative_to(ROOT)):sha(f) for f in files},"cohort_sha256":sha(BASE/"cohort.csv")}
def check_outputs(out,pair):
    s=json.loads((out/"SUCCESS.json").read_text());assert s["status"]=="SUCCESS"
    headers=[]
    for side in ["train","valid"]:
        with (out/(side+".csv")).open() as f:
            r=csv.reader(f);header=next(r);n=0
            for row in r:
                assert len(row)==len(header) and all(v.strip() and v.lower() not in ["na","nan","inf","-inf"] for v in row)
                n+=1
            assert n==pair[side+"_n"];headers.append(header)
    assert headers[0]==headers[1] and len(headers[0])==s["nfeatures"]
    assert (out/"state.joblib").stat().st_size>0 and (out/"metadata.json").exists()
