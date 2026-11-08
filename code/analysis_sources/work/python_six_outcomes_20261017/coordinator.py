"""Finite Python-only 120-imputation/270-model queue, then selection and audit."""
from run_imputation_jobs import *
from pathlib import Path
import subprocess,signal,time,sys,traceback
HERE=Path(__file__).resolve().parent;LOG=ROOT/"logs/python_six_outcomes_20261017"
PY=str(ROOT/"envs/prediction_gpu/bin/python")
FAMS=["LR","DT","RF","KNN","SVM","NB","XGB","SGBT","NNET"]
LIVE={};VERIFIED=set();SIGS={}
RESERVE=2048*MIB;MODEL_BUDGET=640*MIB
IMPUTE_BUDGET={m:(1536*MIB if m=="random_forest" else 640*MIB) for m in METHODS}
def sig(p,m):
    key=(p["name"],m)
    if key not in SIGS:SIGS[key]=signature(p,m)
    return SIGS[key]
def ready(p,m):
    d=BASE/"imputed"/m/p["name"]
    if str(d) in VERIFIED:return True
    if not (d/"provenance.json").exists() or not (d/"SUCCESS.json").exists():return False
    assert json.loads((d/"provenance.json").read_text())==sig(p,m)
    check_outputs(d,p);VERIFIED.add(str(d));return True
def done(task,pairs):
    m,i,f=task;d=BASE/"six_by_nine"/m/f"dev_fold_{i}"/f
    if not (d/"SUCCESS.json").exists():return False
    p=next(x for x in pairs if x["role"]=="development_outer" and x["outer"]==i)
    assert json.loads((d/"SUCCESS.json").read_text())=={"protocol_sha256":sha(BASE/"six_by_nine/protocol.json"),"outer_signature":sig(p,m),"family":f}
    assert (d/"predictions.csv").exists()
    return True
def rss(p):
    try:return int(next(x for x in Path(f"/proc/{p.pid}/status").read_text().splitlines() if x.startswith("VmRSS:")).split()[1])*1024
    except (FileNotFoundError,StopIteration):return 0
def status(**kw):atomic(BASE/"status.json",{"time":time.time(),"supervisor_pid":os.getpid(),**kw})
def stop():
    for c in LIVE.values():
        p=c["proc"]
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
def interrupt(s,f):stop();status(state="stopped_by_signal",signal=s);raise SystemExit(128+s)
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
def launch(key,args,budget,threads,payload):
    e=os.environ.copy();e.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS=str(threads))
    with (LOG/(key.replace("/","__").replace(":","__")+".log")).open("ab") as f:
        p=subprocess.Popen(args,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    LIVE[key]={"proc":p,"budget":budget,"threads":threads,"payload":payload,"started":time.time(),"peak":0}
    print("START",key,p.pid,"threads",threads,flush=True)
def room(budget):
    snap=resource_snapshot();unused=sum(max(0,x["budget"]-rss(x["proc"])) for x in LIVE.values())
    return snap["effective_free"]>=RESERVE+unused+budget
def post(script,name):
    while resource_snapshot()["effective_free"]<2560*MIB:
        status(state="waiting_for_memory",phase=name,resource=resource_snapshot());time.sleep(30)
    p=launch(name,[PY,"-u",str(script)],MODEL_BUDGET,1,None)
    q=LIVE[name]["proc"]
    while q.poll() is None:time.sleep(5)
    rc=q.returncode;LIVE.pop(name)
    if rc:raise RuntimeError(f"{name} failed exit{rc}")
def main():
    assert json.loads((BASE/"functional_acceptance.json").read_text())["status"]=="passed"
    frozen=json.loads((BASE/"code_freeze_manifest.json").read_text())
    for p,h in frozen["files"].items():assert sha(ROOT/p)==h,p
    conf=json.loads((BASE/"launch_config.json").read_text())
    manifest=json.loads((BASE/"fold_inputs/fold_manifest.json").read_text())
    assert manifest["source_cohort_sha256"]==sha(BASE/"cohort.csv")
    pairs=[p for p in manifest["pairs"] if p["role"].startswith("development_")]
    jobs=[(p,m) for m in METHODS for p in pairs]
    tasks=[(m,i,f) for m in METHODS for i in range(5) for f in FAMS]
    gpu_free=0;gpu="";last_gpu=0
    while True:
        snap=resource_snapshot()
        for key,c in list(LIVE.items()):
            c["peak"]=max(c["peak"],rss(c["proc"]));rc=c["proc"].poll()
            if rc is None:continue
            if rc:raise RuntimeError(f"Task {key} failed exit{rc}; inspect scoped log")
            if key.startswith("IP:"):assert ready(*c["payload"]),key
            else:assert done(c["payload"],pairs),key
            atomic(BASE/"job_resources"/(key.replace("/","__").replace(":","__")+".json"),{"job":key,"peak_main_rss_bytes":c["peak"],"elapsed_seconds":time.time()-c["started"],"threads":c["threads"],"exit_code":rc})
            LIVE.pop(key);print("DONE",key,flush=True)
        if snap["effective_free"]<1536*MIB:
            stop();status(state="paused_memory",resource=snap,unfinished=list(LIVE));raise SystemExit(75)
        pending_ip=[(p,m) for p,m in jobs if not ready(p,m)]
        pending_ml=[t for t in tasks if not done(t,pairs)]
        if time.time()-last_gpu>30:
            gpu=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.free,memory.used,utilization.gpu,temperature.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
            try:gpu_free=int(gpu.splitlines()[0].split(",")[2].strip())*MIB
            except (IndexError,ValueError):gpu_free=0
            last_gpu=time.time()
        for p,m in pending_ip:
            key="IP:"+m+"/"+p["name"]
            if key in LIVE:continue
            th=conf["imputer_RF_threads"] if m=="random_forest" else 1;budget=IMPUTE_BUDGET[m]
            if sum(c["threads"] for c in LIVE.values())+th>snap["threads"]:continue
            if not room(budget+MODEL_BUDGET):continue
            launch(key,[PY,"-u",str(HERE/"impute_worker.py"),m,p["name"],"--threads",str(th)],budget,th,(p,m))
        for t in pending_ml:
            m,i,f=t;key=f"ML:{m}/{i}/{f}"
            if key in LIVE:continue
            needed=[p for p in pairs if p["outer"]==i];assert len(needed)==4
            if not all(ready(p,m) for p in needed):continue
            th=conf["model_threads"]
            if sum(c["threads"] for c in LIVE.values())+th>snap["threads"] or not room(MODEL_BUDGET):continue
            if f=="XGB" and gpu_free<1024*MIB+768*MIB*(1+sum(k.endswith("/XGB") for k in LIVE)):continue
            launch(key,[PY,"-u",str(HERE/"run_six_by_nine.py"),"--methods",m,"--outer",str(i),"--families",f,"--threads",str(th)],MODEL_BUDGET,th,t)
        # State is written from verified artifacts, not merely process exit counts.
        status(state="running" if LIVE else "waiting_for_resources",phase="six_Python_imputations_and_nine_models",completed_imputations=120-len(pending_ip),total_imputations=120,completed_model_tasks=270-len(pending_ml),total_model_tasks=270,active=[{"job":k,"pid":c["proc"].pid,"threads":c["threads"],"rss":rss(c["proc"])} for k,c in LIVE.items()],resource=snap,gpu=gpu)
        if not pending_ip and not pending_ml and not LIVE:break
        time.sleep(5)
    post(HERE/"summarize_six_by_nine.py","selection")
    post(HERE/"audit_and_report.py","quality_audit_and_report")
    winner=json.loads((BASE/"six_by_nine/imputation_selection.json").read_text())["selected_method"]
    status(state="completed_six_method_selection",selected_method=winner,downstream_main_models_started=False)
    pointer=json.loads((ROOT/"outputs/CURRENT_ANALYSIS.json").read_text())
    pointer.update(status="Python_six_method_comparison_complete",active_compute={},supervisor_pid=None,completed_benchmark=str((BASE/"six_by_nine").relative_to(ROOT)),current_report=str((BASE/"six_by_nine/IMPUTATION_SELECTION_REPORT_ZH.md").relative_to(ROOT)),main_imputation_method_not_yet_confirmed=False,selected_imputation_method=winner)
    atomic(ROOT/"outputs/CURRENT_ANALYSIS.json",pointer)
    print("PYTHON_SIX_METHOD_SELECTION_COMPLETE",winner,flush=True)
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        stop()
        if not isinstance(exc,SystemExit):status(state="needs_attention",error=repr(exc));traceback.print_exc()
        raise

