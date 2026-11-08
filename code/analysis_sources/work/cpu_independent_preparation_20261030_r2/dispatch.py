"""Resource/owner boundary for the unchanged CPU sealed-LLM integrity audit."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, fcntl, hashlib, importlib.util, json, os, resource, shutil
import subprocess, sys, time

R = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
C = R/"outputs/reanalysis_control"
F = R/"outputs/llm_supplement_20261028_v1/formal_v1"
A = R/"outputs/cpu_sealed_llm_audit_20261029_v1"
O = R/"outputs/cpu_sealed_audit_admission_20261030_r2"
W = R/"work/cpu_independent_preparation_20261030_r2"
SCRIPT = R/"work/cpu_independent_preparation_20261029_v1/run_sealed_audit.py"
ENV = "/home/lei/codex-workspaces/shared-python-envs/envs/clinical-llm-py312-torch260-cu118-peft018-20261028-v1/bin/python"
PHASE = Path("/home/lei/codex-workspaces/liri/logs/priority_after_aki_20261023/CURRENT_RESOURCE_PHASE.json")
AUTH = PHASE.with_name("PARALLEL_LIRI_AUTHORIZATION_20261029.json")
STATS = R/"work/llm_statistics_launch_20261030_v1/launch.py"
OWNER = C/"CPU_SEALED_AUDIT_OWNER.json"
FORMAL = "aki-postslot-llm-formal-20261029-v1.service"
GIB = 1024**3
BUDGET = 8*GIB
AUTH_SHA = "5ffd3e1ddb11b6ac4b21ea34803a52f39cc7a60092ebe9d147deba92d0e42bc8"
HELPER_SHA = "8897e965552e5f2ffafe337e1b22738d7faa8785d83dac16c8bd1ed4e6061104"
STATS_SHA = "6b918fa6c387a986a4f4d81e1027decaf61baa60326a7e32279a020a4a1d7e47"
LIVE = {"active", "activating", "deactivating", "reloading"}

def now(): return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    t=p.with_name(p.name+".tmp."+str(os.getpid()))
    with t.open("w") as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def backup(p):
    p=Path(p)
    if not p.exists(): return None
    digest=sha(p);d=O/"preserved"/(digest+"__"+p.name)
    if not d.exists(): d.parent.mkdir(mode=0o700,parents=True,exist_ok=True);d.write_bytes(p.read_bytes())
    if sha(d)!=digest: raise RuntimeError("Backup hash mismatch")
    return {"path":str(d),"sha256":digest}
def state(unit):
    p=subprocess.run(["systemctl","--user","show",unit,"--property=LoadState,ActiveState,SubState,MainPID,ControlGroup,MemoryMax,Result,ExecMainStatus"],text=True,capture_output=True)
    if p.returncode: raise RuntimeError("Cannot establish unit state: "+unit)
    return dict(x.split("=",1) for x in p.stdout.splitlines() if "=" in x)
def ticks(pid):
    if not pid or str(pid)=="0": return None
    s=Path("/proc")/str(pid)/"stat"
    return int(s.read_text().split(") ",1)[1].split()[19])
def code_guard():
    lock=read(W/"LAUNCHER_LOCK.json")
    for p,h in lock["files_sha256"].items():
        if sha(p)!=h: raise RuntimeError("Pinned code/source mismatch: "+p)
    if sha(SCRIPT)!=HELPER_SHA or sha(STATS)!=STATS_SHA or sha(AUTH)!=AUTH_SHA: raise RuntimeError("Authority/helper mismatch")
def stats_module():
    spec=importlib.util.spec_from_file_location("aki_qualified_statistics_launch_boundary",STATS)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def identity(x): return "/".join(str(x[k]) for k in ["model","line","stage","fit","kind"])
def pending_tasks():
    idx=read(A/"AUDIT_INDEX.json") if (A/"AUDIT_INDEX.json").exists() else {"tasks":{}}
    items=[x for x in read(F/"ACCEPTED_TASKS.json")["tasks"] if x["kind"] in ("fit","before")]
    if len({identity(x) for x in items})!=len(items): raise RuntimeError("Duplicate accepted identity")
    pending=[]
    for x in items:
        if sha(x["receipt"])!=x["sha256"]: raise RuntimeError("Accepted COMPLETE no longer matches")
        old=idx["tasks"].get(identity(x));valid=False
        if old and old["COMPLETE_sha256"]==x["sha256"] and old["input_signature"]==idx.get("input_signature"):
            if sha(old["audit_receipt"])!=old["audit_sha256"]: raise RuntimeError("Cached audit receipt changed")
            saved=read(old["audit_receipt"]).get("sealed_file_stats",{})
            actual={}
            for p in saved:
                st=Path(p).stat()
                actual[p]={"bytes":st.st_size,"mtime_ns":st.st_mtime_ns,"ctime_ns":st.st_ctime_ns,"inode":st.st_ino,"device":st.st_dev}
            valid=bool(saved) and actual==saved
        if not valid: pending.append(identity(x))
    return {"accepted_tasks":len(items),"cached_tasks":len(idx["tasks"]),"pending_tasks":pending}
def calculate(mem,headroom,declared,rows,affinity,budget=BUDGET,own_unit=None):
    lirikeys=["LIRI_scrna_max","LIRI_virtual_CPU_total","LIRI_drug_preparation"]
    groups={k:{"caps":0,"anon":0} for k in lirikeys}
    other_aki_caps=0;growth=0;own_anon=0
    for row in rows:
        cap=row["cap"];anon=row["anon"]
        if not isinstance(cap,int) or cap<=0 or anon<0: raise ValueError("Unknown memory budget")
        if row["unit"]==own_unit: own_anon=anon;continue
        key=row.get("family")
        if key in groups: groups[key]["caps"]+=cap;groups[key]["anon"]+=anon
        else:
            growth+=max(0,cap-anon)
            if row["unit"].startswith("aki-"): other_aki_caps+=cap
    commitments={}
    for key,g in groups.items():
        peak=max(int(declared[key]*GIB),g["caps"])
        commitments[key]={"declared_or_active_peak":peak,"resident_anon":g["anon"],"future_growth":max(0,peak-g["anon"])}
        growth+=commitments[key]["future_growth"]
    osmargin=max(4*GIB,int(declared["OS_other_headroom_min"]*GIB),mem//4)
    monitor=2*GIB;new_growth=max(0,budget-own_anon)
    remainder=min(mem,headroom)-growth-osmargin-monitor-new_growth
    family_remaining=max(0,int(declared["AKI_existing_LLM_unit_cap"]*GIB)-other_aki_caps)
    reasons=[]
    if budget>family_remaining: reasons.append("AKI_JOINT_COMMITMENT_ALREADY_ALLOCATED")
    if remainder<0: reasons.append("INSUFFICIENT_SHARED_GROWTH_AND_OS_HEADROOM")
    if not affinity: reasons.append("NO_CPU_ALLOCATION")
    return {"accepted":not reasons,"reasons":reasons,"MemAvailable":mem,"shared_ancestor_headroom":headroom,"other_growth":growth,
            "LIRI_commitments":commitments,"OS_margin":osmargin,"monitoring_reserve":monitor,"audit_budget":budget,
            "own_resident_anon":own_anon,"additional_audit_growth":new_growth,"remaining_bytes":remainder,
            "other_AKI_caps":other_aki_caps,"AKI_family_remaining_cap":family_remaining,"CPU_affinity":affinity}
def resource_plan(own_unit=None):
    ph=read(PHASE)
    if ph.get("state")!="AKI_AND_LIRI_PARALLEL_BY_DIRECT_USER_AUTHORIZATION" or ph.get("authorization_sha256")!=AUTH_SHA or ph.get("no_automatic_resume",False):
        raise RuntimeError("Joint authority changed; inspect before adapting")
    if read(C/"AKI_RESOURCE_LEASE.json").get("status")!="SHARED_AKI_LIRI_BY_LATER_DIRECT_USER_AUTHORIZATION": raise RuntimeError("AKI lease changed")
    m=stats_module()
    out=subprocess.check_output(["systemctl","--user","list-units","--type=service","--state=running,activating,deactivating","--plain","--no-legend","aki-*","liri-*"],text=True)
    rows=[]
    for line in out.splitlines():
        if not line.strip(): continue
        name=line.split()[0];s=state(name)
        if s["ActiveState"] not in LIVE: continue
        cap=s.get("MemoryMax")
        if not cap or not cap.isdigit(): raise RuntimeError("Unbounded live research unit: "+name)
        cg=Path("/sys/fs/cgroup")/s["ControlGroup"].lstrip("/")
        st=dict(x.split() for x in (cg/"memory.stat").read_text().splitlines())
        rows.append({"unit":name,"cap":int(cap),"anon":int(st["anon"]),"family":m.family(name),"PID":s["MainPID"],"start_ticks":ticks(s["MainPID"])})
    mem=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
    cg=Path("/sys/fs/cgroup")/state("app.slice")["ControlGroup"].lstrip("/");headroom=mem;ancestors=[]
    while cg.is_relative_to("/sys/fs/cgroup"):
        mx=(cg/"memory.max").read_text().strip() if (cg/"memory.max").exists() else "max"
        cur=int((cg/"memory.current").read_text()) if (cg/"memory.current").exists() else 0
        if mx!="max": headroom=min(headroom,max(0,int(mx)-cur))
        ancestors.append({"path":str(cg),"max":mx,"current":cur})
        if cg==Path("/sys/fs/cgroup"): break
        cg=cg.parent
    sharing=ph["CPU_initial_sharing"];allowed=sorted(os.sched_getaffinity(0))
    outside=sum(int(sharing[k]) for k in ["LIRI_scrna_max_threads","LIRI_virtual_total_threads","LIRI_drug_preparation_max_threads"])
    n=max(0,min(len(allowed),int(ph["effective_CPU_snapshot"])-outside,int(sharing["AKI_and_system_remaining"])))
    plan=calculate(mem,headroom,ph["reservations_GiB"],rows,allowed[-n:] if n else [],own_unit=own_unit)
    resolved=A.resolve();mount=json.loads(subprocess.check_output(["findmnt","-J","-T",str(resolved)],text=True))["filesystems"][0]
    if mount["target"]!="/data" or not resolved.is_relative_to("/data"): raise RuntimeError("Wrong output filesystem")
    free=shutil.disk_usage(resolved).free
    if free<256*1024**2: plan["accepted"]=False;plan["reasons"].append("OUTPUT_STORAGE_HEADROOM")
    plan.update(time=now(),phase_sha256=sha(PHASE),active_units=rows,ancestors=ancestors,data_free_bytes=free,mount=mount,
                expected_output_bound_bytes=128*1024**2)
    return plan
def owner_reason(d,s):
    if s.get("ActiveState") in LIVE or s.get("MainPID","0")!="0": return "WAIT_EXISTING_AUDIT_OWNER"
    if d.get("status") in {"LAUNCHING_AUDIT_R2","LAUNCH_OUTCOME_UNKNOWN","EXECUTING_AUDIT_R2","AUDIT_FAILED_NEEDS_DIAGNOSIS"}: return "HOLD_PREVIOUS_AUDIT_OUTCOME_NEEDS_RECONCILIATION"
    if s.get("Result") not in (None,"","success") or s.get("ExecMainStatus") not in (None,"","0"): return "HOLD_PRIOR_AUDIT_UNIT_FAILED"
    return None
def completion_receipt_reason(owner, acceptance):
    if owner.get("status")=="AUDIT_NOT_STARTED_AFTER_CHILD_RECHECK": return None
    if not acceptance or acceptance.get("status")!="PASS_INCREMENTAL_SEALED_STRUCTURAL_AUDIT_ONLY" or acceptance.get("code_sha256")!=HELPER_SHA:
        return "HOLD_PRIOR_OWNER_NO_QUALIFIED_COMPLETION_RECEIPT"
    try:
        if datetime.fromisoformat(acceptance["time"]) < datetime.fromisoformat(owner["time"]):
            return "HOLD_PRIOR_OWNER_RECEIPT_PREDATES_DISPATCH"
    except (KeyError,TypeError,ValueError):
        return "HOLD_PRIOR_OWNER_COMPLETION_TIME_UNRESOLVED"
    return None
def preflight(own_unit=None):
    code_guard()
    if (A/"FAILURE.json").exists(): return {"status":"HOLD_DIAGNOSE_PRIOR_AUDIT_FAILURE"}
    if not own_unit and OWNER.exists():
        d=read(OWNER);reason=owner_reason(d,state(d["unit"]))
        if reason: return {"status":reason,"unit":d["unit"]}
        acceptance=read(A/"INTEGRITY_PREPARATION_ACCEPTANCE.json") if (A/"INTEGRITY_PREPARATION_ACCEPTANCE.json").exists() else None
        reason=completion_receipt_reason(d,acceptance)
        if reason: return {"status":reason,"unit":d["unit"]}
        if d.get("audit_acceptance_sha256") and sha(A/"INTEGRITY_PREPARATION_ACCEPTANCE.json")!=d["audit_acceptance_sha256"]:
            return {"status":"HOLD_PRIOR_OWNER_COMPLETION_HASH_CHANGED","unit":d["unit"]}
    main=state(FORMAL)
    if main.get("ActiveState")=="active" and int(main.get("MainPID","0"))>0:
        main["start_ticks"]=ticks(main["MainPID"])
    else:
        acceptance=read(F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json") if (F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json").exists() else None
        reason=stats_module().completion_reason(main,acceptance,read(F/"STATUS.json"))
        if reason: return {"status":reason}
    live_names=subprocess.check_output(["systemctl","--user","list-units","--type=service","--state=running,activating,deactivating","--plain","--no-legend","aki-postslot-sealed-llm-cpu-audit-*"],text=True)
    other=[line.split()[0] for line in live_names.splitlines() if line.strip() and line.split()[0]!=own_unit]
    if other: return {"status":"WAIT_EXISTING_AUDIT_UNIT_WITH_OR_WITHOUT_OWNER_FILE","units":other}
    with (A/"OWNER.lock").open("a") as probe:
        try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return {"status":"WAIT_EXISTING_AUDIT_LOCK_HOLDER"}
    p=pending_tasks()
    if not p["pending_tasks"]: return {"status":"NO_NEW_SEALED_TASKS",**p,"formal_state":main}
    plan=resource_plan(own_unit)
    return {"status":"READY_FINITE_AUDIT" if plan["accepted"] else "WAIT_SHARED_RESOURCE_ADMISSION",**p,"formal_state":main,"resource_plan":plan}
def execute(receipt):
    d=read(receipt);code_guard()
    if d["launcher_sha256"]!=sha(__file__) or d["unit"] not in Path("/proc/self/cgroup").read_text(): raise RuntimeError("Execution owner mismatch")
    if sha(PHASE)!=d["preflight"]["resource_plan"]["phase_sha256"]: raise RuntimeError("Joint phase changed after dispatch")
    p=preflight(d["unit"])
    if p["status"]!="READY_FINITE_AUDIT":
        d.update(status="AUDIT_NOT_STARTED_AFTER_CHILD_RECHECK",child_preflight=p,time=now());save(OWNER,d);print(json.dumps(d));return
    cpus=p["resource_plan"]["CPU_affinity"];os.sched_setaffinity(0,cpus)
    if set(os.sched_getaffinity(0))!=set(cpus): raise RuntimeError("CPU affinity did not apply")
    d.update(status="EXECUTING_AUDIT_R2",child_preflight=p,PID=os.getpid(),start_ticks=ticks(os.getpid()));save(OWNER,d)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    proc=subprocess.run([ENV,"-B",str(SCRIPT)],env=env)
    d.update(status="AUDIT_HELPER_FINISHED_REQUIRES_UNIT_EXIT" if proc.returncode==0 else "AUDIT_FAILED_NEEDS_DIAGNOSIS",helper_exit_code=proc.returncode,finished_time=now())
    if proc.returncode==0:
        ac=read(A/"INTEGRITY_PREPARATION_ACCEPTANCE.json")
        if ac["code_sha256"]!=HELPER_SHA: raise RuntimeError("Audit helper changed")
        d.update(audit_acceptance_sha256=sha(A/"INTEGRITY_PREPARATION_ACCEPTANCE.json"),audit_counts=ac["counts"])
    save(OWNER,d)
    raise SystemExit(proc.returncode)
def dispatch(launch=False):
    O.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (A/"DISPATCH_OWNER.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=preflight();save(O/"LATEST_ADMISSION.json",p)
        if p["status"]!="READY_FINITE_AUDIT" or not launch: print(json.dumps(p,ensure_ascii=False));return
        stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
        unit="aki-postslot-sealed-llm-cpu-audit-r2-"+stamp+".service"
        log=R/"logs"/("cpu_sealed_llm_audit_r2_"+stamp+".log")
        rcpt=O/"launches"/(stamp+".json")
        d={"status":"LAUNCHING_AUDIT_R2","time":now(),"unit":unit,"preflight":p,"launcher":str(Path(__file__)),"launcher_sha256":sha(__file__),
           "script":str(SCRIPT),"script_sha256":HELPER_SHA,"log":str(log),"prior_owner_backup":backup(OWNER),
           "prior_acceptance_backup":backup(A/"INTEGRITY_PREPARATION_ACCEPTANCE.json"),"no_clinical_metrics_or_new_fit":True}
        save(rcpt,d);save(OWNER,d)
        cmd=["systemd-run","--user","--collect","--unit="+unit,"--property=MemoryMax="+str(BUDGET),"--property=MemorySwapMax=0","--property=CPUWeight=200",
             "--property=TasksMax=128","--property=RuntimeMaxSec=1800","--property=StandardOutput=append:"+str(log),"--property=StandardError=append:"+str(log),
             "/usr/bin/python3","-B",str(Path(__file__)),"--execute",str(rcpt)]
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        except subprocess.TimeoutExpired:
            d.update(status="LAUNCH_OUTCOME_UNKNOWN",actual_unit_state=state(unit));save(OWNER,d);raise
        # Do not overwrite a child-produced owner receipt if the child has already started.
        current=read(OWNER)
        if current["status"]=="LAUNCHING_AUDIT_R2":
            current.update(status="DISPATCHED_INCREMENTAL_CPU_AUDIT_R2" if result.returncode==0 else "LAUNCH_OUTCOME_UNKNOWN",launch_returncode=result.returncode,launch_output=(result.stdout+result.stderr)[:3000],actual_unit_state=state(unit))
            save(OWNER,current)
        print(json.dumps({"status":read(OWNER)["status"],"unit":unit,"launch_returncode":result.returncode}))
        if result.returncode: raise SystemExit(result.returncode)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--launch",action="store_true");ap.add_argument("--execute");args=ap.parse_args()
    if args.execute: execute(args.execute)
    else: dispatch(args.launch)
if __name__=="__main__": main()
