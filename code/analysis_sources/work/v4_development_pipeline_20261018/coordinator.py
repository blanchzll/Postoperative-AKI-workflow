"""Finite v4 imputation -> nine-family development selection; owned resources only."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,fcntl,signal
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
INPUT=ROOT/"outputs/v4_development_inputs_20261018"
OUT=ROOT/"outputs/v4_development_pipeline_20261018";LOG=ROOT/"logs/v4_development_pipeline_20261018"
G=1024**3;ACTIVE=[];STOP=False
METHODS=["random_forest","mean_mode","knn","cart","bayes","pmm"]
FAMILIES=["RF","XGB","LR","DT","KNN","SVM","NB","SGBT","NNET"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
    q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n");os.replace(q,p)
def available():
    free=next(int(s.split()[1])*1024 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))
    details={}
    for base,maxname,currentname in [("/sys/fs/cgroup","memory.max","memory.current"),("/sys/fs/cgroup/memory","memory.limit_in_bytes","memory.usage_in_bytes")]:
        limit=Path(base)/maxname;usage=Path(base)/currentname
        if limit.exists():
            val=limit.read_text().strip();cur=usage.read_text().strip() if usage.exists() else None
            details[str(limit)]={"limit":val,"usage":cur}
            if val.isdigit() and cur and cur.isdigit() and int(val)<2**60:free=min(free,max(0,int(val)-int(cur)))
    return free,details
def rss(pid):
    try:return next(int(s.split()[1])*1024 for s in (Path("/proc")/str(pid)/"status").read_text().splitlines() if s.startswith("VmRSS:"))
    except (FileNotFoundError,StopIteration):return 0
def stat(name,**kw):atomic(OUT/"status.json",{"state":name,"pid":os.getpid(),"epoch":time.time(),**kw})
def donepath(j):
    if j["kind"]=="impute":return OUT/j["line"]/("inner_"+str(j["fold"]))/j["method"]/"COMPLETE.json"
    return OUT/"comparison"/j["line"]/j["method"]/j["family"]/"COMPLETE.json"
def label(j):return "_".join(str(j[k]) for k in ["kind","line","method","fold","family"] if k in j)
def reserve(j):
    if j["kind"]=="impute" and j["method"]=="random_forest":return 2*G
    if j["kind"]=="compare" and j["family"]=="RF":return 1536*1024**2
    if j["kind"]=="compare" and j["family"]=="XGB":return G
    return 768*1024**2
def threaded(j):return (j["kind"]=="impute" and j["method"]=="random_forest") or (j["kind"]=="compare" and j["family"] in ["RF","XGB","KNN"])
def check_done(j):
    d=donepath(j);r=json.loads(d.read_text())
    for f,h in r["outputs_sha256"].items():assert sha(d.parent/f)==h
    if j["kind"]=="impute":assert r["source_input_acceptance_sha256"]==sha(INPUT/"INPUT_ACCEPTANCE.json")
    else:assert r["source_lock_sha256"]==sha(INPUT/"SOURCE_LOCK.json") and r["protocol_sha256"]==sha(OUT/"COMPARISON_PROTOCOL.json")
    return r
def stop(sig,frame):
    global STOP
    STOP=True
    for r in ACTIVE:
        if r["proc"].poll() is None:r["proc"].terminate()
def phase(jobs,name,offset,total):
    completed=[];waiting=[];errors=[]
    for j in jobs:
        if donepath(j).exists():check_done(j);completed.append(j)
        else:waiting.append(j)
    affinity=len(os.sched_getaffinity(0))
    while waiting or ACTIVE:
        for r in list(ACTIVE):
            if r["proc"].poll() is not None:
                r["log"].close();ACTIVE.remove(r)
                if r["proc"].returncode==0 and donepath(r["job"]).exists():check_done(r["job"]);completed.append(r["job"])
                else:errors.append({"job":r["job"],"returncode":r["proc"].returncode});waiting.clear()
        if STOP:waiting.clear()
        while waiting and not errors and not STOP:
            freecpu=affinity-sum(r["threads"] for r in ACTIVE)
            if freecpu<1:break
            free,_=available();free-=sum(max(0,r["reserve"]-rss(r["proc"].pid)) for r in ACTIVE)
            chosen=next((j for j in waiting if free>3*G+reserve(j)),None)
            if chosen is None:break
            if threaded(chosen):
                slots=max(1,min(sum(threaded(j) for j in waiting),max(1,int((free-3*G)//reserve(chosen)))))
                threads=max(1,freecpu//slots)
            else:threads=1
            if chosen["kind"]=="impute":args=[chosen["line"],str(chosen["fold"]),chosen["method"],str(threads)];script="impute_worker.py"
            else:args=[chosen["line"],chosen["method"],chosen["family"],str(threads)];script="compare_worker.py"
            fp=(LOG/(label(chosen)+".log")).open("a")
            env=os.environ.copy()
            for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:env[key]="1"
            p=subprocess.Popen([sys.executable,str(CODE/script)]+args,cwd=ROOT,stdout=fp,stderr=subprocess.STDOUT,env=env)
            ACTIVE.append({"job":chosen,"proc":p,"log":fp,"threads":threads,"reserve":reserve(chosen)});waiting.remove(chosen)
        free,limits=available()
        if free<768*1024**2 and ACTIVE:
            r=max(ACTIVE,key=lambda v:rss(v["proc"].pid))
            if r["proc"].poll() is None:r["proc"].terminate()
            errors.append({"job":r["job"],"reason":"Critical available RAM; stopped own unfinished worker"});waiting.clear()
        stat("running" if not STOP and not errors else "draining_own_workers",phase=name,phase_completed=len(completed),phase_total=len(jobs),overall_completed=offset+len(completed),overall_total=total,waiting=len(waiting),active=[{"job":r["job"],"pid":r["proc"].pid,"threads":r["threads"],"rss_bytes":rss(r["proc"].pid)} for r in ACTIVE],effective_available_bytes=free,cgroup_memory=limits,errors=errors,best_method_selected=False,final_hospital_time_evaluation_fitted=False)
        if waiting or ACTIVE:time.sleep(5)
    if STOP or errors:
        stat("interrupted_checkpointed" if STOP else "failed_checkpointed",phase=name,phase_completed=len(completed),phase_total=len(jobs),errors=errors,owned_workers_exited=True)
        raise RuntimeError("Inspect scoped logs; no unchanged automatic retry")
    return [check_done(j) for j in jobs]
def summarize_comparison(jobs):
    import pandas as pd
    rows=[]
    for j in jobs:
        r=check_done(j)
        rows.append({k:r[k] for k in ["line","method","family","selected_candidate","mean_fold_AP","mean_fold_Brier","mean_fold_AUROC","n_candidates","fits"]}|{"warning_count":len(r["warnings"])})
    df=pd.DataFrame(rows);assert len(df)==108
    df["AP_rank"]=df.groupby(["line","family"]).mean_fold_AP.rank(ascending=False,method="average")
    df["Brier_rank"]=df.groupby(["line","family"]).mean_fold_Brier.rank(ascending=True,method="average")
    df.to_csv(OUT/"DEVELOPMENT_SELECTION_METRICS_108.csv",index=False)
    selected={}
    ranking=df.groupby(["line","method"]).agg(mean_AP_rank=("AP_rank","mean"),mean_Brier_rank=("Brier_rank","mean"),mean_AP=("mean_fold_AP","mean"),mean_Brier=("mean_fold_Brier","mean")).reset_index()
    ranking=ranking.sort_values(["line","mean_AP_rank","mean_Brier_rank","method"])
    ranking.to_csv(OUT/"IMPUTATION_RANKING.csv",index=False)
    for line,g in ranking.groupby("line"):
        assert len(g)==6
        selected[line]=g.iloc[0].method
    atomic(OUT/"METHOD_SELECTION.json",{"state":"development_selection_complete_outer_pipeline_evaluation_pending","selected_family_by_line":selected,"criterion":"Lowest meanAP rank across9families; tie lowest meanBrier rank then method name","scores_are_not_unbiased_validation":True,"final_hospital_time_labels_used":False,"source_unit_default_not_all_cells_adjudicated":True,"ALT_free_sensitivity_required":True,"clinical_model_lock_complete":False,"protocol_sha256":sha(OUT/"COMPARISON_PROTOCOL.json"),"source_lock_sha256":sha(INPUT/"SOURCE_LOCK.json")})
    return selected
def main():
    assert str(OUT.resolve()).startswith("/data/") and os.path.ismount("/data")
    LOG.mkdir(mode=0o700,exist_ok=True)
    source=json.loads((INPUT/"SOURCE_LOCK.json").read_text())
    for f,h in source["input_sha256"].items():assert sha(INPUT/f)==h
    q=json.loads((ROOT/"outputs/v2_stable_imputation_20261017/FUNCTIONAL_QUALIFICATION.json").read_text())
    assert sha(ROOT/"work/v2_stable_imputation_20261017/stable_imputer.py")==q["new_source_sha256"]
    freeze=OUT/"CODE_FREEZE.json"
    if not freeze.exists():atomic(freeze,{str(p.relative_to(ROOT)):sha(p) for p in CODE.glob("*.py")})
    for p,h in json.loads(freeze.read_text()).items():assert sha(ROOT/p)==h
    try:os.nice(5)
    except OSError:pass
    free,limits=available()
    atomic(OUT/"RESOURCE_ADMISSION.json",{"affinity":len(os.sched_getaffinity(0)),"available_at_start":free,"memory_limits":limits,"cpu_quota_files":{str(p):p.read_text().strip() for p in [Path("/sys/fs/cgroup/cpu.max"),Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")] if p.exists()},"loadavg":Path("/proc/loadavg").read_text().strip(),"headroom":3*G,"actual_filesystem":str(OUT.resolve()),"data_free_bytes":os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize,"planned_storage_upper_bytes":8*G,"GPU":"driver/library mismatch; CPU consistent across new comparison, no driver/service changes","nice":os.nice(0),"scientific_methods_seeds_preserved":True})
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (ROOT/"work/.pod1_update.lock").open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("Other active project stage owns lock; no duplicate launched")
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+"\n");lock.flush()
        try:
            ijobs=[{"kind":"impute","line":line,"fold":fold,"method":method} for method in METHODS for line in ["geography","temporal"] for fold in range(3)]
            cjobs=[{"kind":"compare","line":line,"method":method,"family":family} for family in FAMILIES for line in ["geography","temporal"] for method in METHODS]
            imputed=phase(ijobs,"six_imputation_training_only_preparation",0,144)
            atomic(OUT/"IMPUTATION_ACCEPTANCE.json",{"state":"36_input_tasks_complete_nine_model_comparison_next","completed":len(imputed),"all_observed_preserved":all(r["observed_predictors_preserved"] for r in imputed),"all_case_stability_checks":all(r["case_stable_real_data_checks_passed"] for r in imputed),"no_best_method_claim_from_preparation":True})
            compared=phase(cjobs,"six_imputations_by_nine_models_development_comparison",36,144)
            selected=summarize_comparison(cjobs)
            atomic(OUT/"STAGE_ACCEPTANCE.json",{"state":"development_imputation_selection_complete_full_model_development_pending","imputation_tasks":36,"comparison_tasks":108,"candidate_model_fits":sum(r["fits"] for r in compared),"all_development_only":all(not r["final_hospital_or_time_evaluation_used"] for r in compared),"natural_prevalence_no_SMOTE":True,"selected_methods":selected,"final_models_or_clinical_thresholds_locked":False,"owned_worker_pools_and_processes_exited":True})
            stat("complete",phase="development_selection_complete_outer_pipeline_evaluation_pending",overall_completed=144,overall_total=144,selected_methods=selected,best_method_selected_for_development=True,final_hospital_time_evaluation_fitted=False,owned_workers_exited=True)
        finally:
            for r in ACTIVE:
                if r["proc"].poll() is None:r["proc"].terminate()
            for r in ACTIVE:
                try:r["proc"].wait(timeout=30)
                except subprocess.TimeoutExpired:r["proc"].kill();r["proc"].wait()
                r["log"].close()
            ACTIVE.clear()
if __name__=="__main__":main()
