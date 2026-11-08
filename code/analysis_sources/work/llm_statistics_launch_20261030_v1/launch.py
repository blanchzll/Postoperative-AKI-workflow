"""Finite, CPU-only launch boundary for the unchanged locked LLM statistics."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, resource, shutil, subprocess, sys
from datetime import datetime
from zoneinfo import ZoneInfo

R = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
C = R / "outputs/reanalysis_control"
F = R / "outputs/llm_supplement_20261028_v1/formal_v1"
S = R / "outputs/llm_supplement_20261028_v1/statistics_v1"
W = R / "work/llm_comparison_current_20261029_v1"
O = R / "outputs/llm_statistics_launch_20261030_v1"
PHASE = Path("/home/lei/codex-workspaces/liri/logs/priority_after_aki_20261023/CURRENT_RESOURCE_PHASE.json")
AUTH = Path("/home/lei/codex-workspaces/liri/logs/priority_after_aki_20261023/PARALLEL_LIRI_AUTHORIZATION_20261029.json")
FORMAL_UNIT = "aki-postslot-llm-formal-20261029-v1.service"
UNIT = "aki-postslot-llm-statistics-20261030-v1.service"
OWNER = C / "LLM_STATISTICS_OWNER.json"
GIB = 1024**3
PINS = {
    W/"STATISTICS_CODE_LOCK.json": "488e00e7f2c1004f2309e86a6eb1f86a82896bb355735b79bdf83383f6b7c851",
    R/"work/llm_delivery_current_20261029_v1/DELIVERY_CODE_LOCK.json": "dc12c0ac124d2c77592a5a486f1250628092becddbeb991b06e234f1362d7039",
    F/"FORMAL_TRAINING_CONTRACT.json": "eb27122860f1500b848efeea0ce1a19846e7d94aa5d0de7adbabc2c8db2041de",
    R/"work/llm_supplement_20261028_v1/formal_benchmark_v1.py": "05040a02b79880f47ea7c480c34e4f4db62e85494cb89143ea5f3bbb79e2ddce",
    R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json": "afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3",
    AUTH: "5ffd3e1ddb11b6ac4b21ea34803a52f39cc7a60092ebe9d147deba92d0e42bc8",
}
MODELS = ["biomedlm_2_7b", "medgemma_4b_it", "llama3_8b_instruct_ngc_v1"]

def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def unit_state(unit):
    text = subprocess.check_output(
        ["systemctl", "--user", "show", unit,
         "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,MainPID,ControlGroup,MemoryMax,MemorySwapMax"],
        text=True)
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)

def code_guard():
    checked = {}
    for path, expected in PINS.items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError("Pinned input changed: " + str(path))
        checked[str(path)] = actual
    for path, expected in read(W/"STATISTICS_CODE_LOCK.json")["files"].items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError("Frozen statistics dependency changed: " + path)
        checked[path] = actual
    return checked

def completion_reason(state, acceptance, status):
    if state.get("LoadState") == "not-found":
        return "FORMAL_UNIT_MISSING_VERIFY_OWNERSHIP"
    if state.get("ActiveState") != "inactive" or state.get("MainPID", "0") != "0":
        return "WAIT_FORMAL_PROCESS_EXIT"
    if state.get("Result") != "success" or state.get("ExecMainStatus") != "0":
        return "FORMAL_TERMINAL_FAILURE_NEEDS_DIAGNOSIS"
    if not acceptance:
        return "WAIT_FORMAL_ACCEPTANCE"
    if (acceptance.get("formal_fits") != 48 or acceptance.get("original_checkpoint_banks") != 6
            or acceptance.get("contract_sha256") != PINS[F/"FORMAL_TRAINING_CONTRACT.json"]):
        return "FORMAL_ACCEPTANCE_COUNTS_OR_CONTRACT_MISMATCH"
    if status.get("status") != "COMPLETE_TRAINING_AND_PREDICTION_BANKS_STATISTICAL_COMPARISON_REQUIRED":
        return "FORMAL_STATUS_NOT_COMPLETE"
    expected = {(m, s, line, str(k)) for m in MODELS for s in ["preop", "endop"]
                for line in ["geography", "temporal"] for k in [0, 1, 2, "final"]}
    fits = [(t["model"], t["stage"], t["line"], str(t["fit"]))
            for t in acceptance.get("tasks", []) if t["kind"] == "fit"]
    banks = [(t["model"], t["stage"]) for t in acceptance.get("tasks", []) if t["kind"] == "before"]
    if len(fits) != 48 or set(fits) != expected:
        return "FORMAL_TASK_SET_MISMATCH"
    if len(banks) != 6 or set(banks) != {(m, s) for m in MODELS for s in ["preop", "endop"]}:
        return "FORMAL_BANK_SET_MISMATCH"
    return None

def family(unit):
    if unit.startswith(("liri-star-", "liri-nac-", "liri-full-input-")):
        return "LIRI_scrna_max"
    if unit.startswith("liri-virtual-"):
        return "LIRI_virtual_CPU_total"
    if unit.startswith("liri-drug-") or (unit.startswith("liri-") and "docking" in unit):
        return "LIRI_drug_preparation"
    return None

def calculate_plan(mem_available, ancestor_headroom, declared, anonymous, extra_growth, affinity, own_anon=0, other_aki_caps=0):
    os_margin = max(4*GIB, int(declared["OS_other_headroom_min"]*GIB), mem_available//4)
    growth = extra_growth
    for key in ["LIRI_scrna_max", "LIRI_virtual_CPU_total", "LIRI_drug_preparation"]:
        growth += max(0, int(declared[key]*GIB) - anonymous.get(key, 0))
    effective = min(mem_available, ancestor_headroom)
    remainder = effective - os_margin - growth
    # The new statistics unit replaces the closed formal unit's AKI commitment.
    total_capacity = max(0, remainder + own_anon)
    quantum = 128*1024**2
    family_remaining = max(0, int(declared["AKI_existing_LLM_unit_cap"]*GIB)-other_aki_caps)
    cap = min(family_remaining, total_capacity)//quantum*quantum
    return {
        "accepted": cap >= 2*GIB and len(affinity) > 0,
        "MemAvailable": mem_available, "shared_ancestor_headroom": ancestor_headroom,
        "OS_margin": os_margin, "other_committed_growth": growth,
        "available_for_new_growth": remainder, "own_already_resident_anon": own_anon,
        "memory_max_bytes": cap, "minimum_preparation_and_pilot_budget": 2*GIB,
        "formal_commitment_replaced_not_added": True, "CPU_affinity": affinity,
        "other_AKI_task_caps": other_aki_caps, "AKI_family_remaining_cap": family_remaining,
    }

def resource_plan(include_self=False):
    ph = read(PHASE)
    if ph.get("state") != "AKI_AND_LIRI_PARALLEL_BY_DIRECT_USER_AUTHORIZATION":
        raise RuntimeError("Joint resource authority changed; inspect before adapting")
    if ph.get("authorization_sha256") != PINS[AUTH] or ph.get("no_automatic_resume", False):
        raise RuntimeError("Joint authority gate not open")
    if read(C/"AKI_RESOURCE_LEASE.json")["status"] != "SHARED_AKI_LIRI_BY_LATER_DIRECT_USER_AUTHORIZATION":
        raise RuntimeError("AKI lease changed; inspect current direct instructions")
    declared = ph["reservations_GiB"]
    output = subprocess.check_output(
        ["systemctl", "--user", "list-units", "--type=service", "--state=running,activating",
         "--no-legend", "--plain", "aki-*", "liri-*"], text=True)
    names = [line.split()[0] for line in output.splitlines() if line.strip()]
    anonymous = {}; rows = []; extra_growth = 0; own_anon = 0; other_aki_caps = 0
    for name in names:
        d = unit_state(name)
        if name == FORMAL_UNIT:
            raise RuntimeError("Formal unit became active during resource admission")
        cg = Path("/sys/fs/cgroup")/d["ControlGroup"].lstrip("/")
        stat = dict(line.split() for line in (cg/"memory.stat").read_text().splitlines())
        anon = int(stat.get("anon", 0))
        cap = d.get("MemoryMax", "infinity")
        if cap in ["infinity", "[not set]"]:
            raise RuntimeError("Unbounded active research unit needs explicit measured budget: " + name)
        cap = int(cap); key = family(name)
        if name == UNIT:
            if not include_self:
                raise RuntimeError("Statistics writer already running")
            own_anon = anon
        elif key:
            anonymous[key] = anonymous.get(key, 0) + anon
        else:
            extra_growth += max(0, cap-anon)
            if name.startswith("aki-"):
                other_aki_caps += cap
        rows.append({"unit": name, "family": key, "anonymous_bytes": anon, "memory_max_bytes": cap,
                     "PID": d["MainPID"], "control_group": d["ControlGroup"]})
    mem = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemAvailable:")))*1024
    app = unit_state("app.slice")["ControlGroup"]
    cg = Path("/sys/fs/cgroup")/app.lstrip("/")
    ancestors = []; headroom = mem
    while cg.is_relative_to("/sys/fs/cgroup"):
        maximum = (cg/"memory.max").read_text().strip() if (cg/"memory.max").exists() else "max"
        current = int((cg/"memory.current").read_text()) if (cg/"memory.current").exists() else 0
        ancestors.append({"path": str(cg), "max": maximum, "current": current})
        if maximum != "max":
            headroom = min(headroom, max(0, int(maximum)-current))
        if cg == Path("/sys/fs/cgroup"):
            break
        cg = cg.parent
    allowed = sorted(os.sched_getaffinity(0))
    sharing = ph["CPU_initial_sharing"]
    outside_threads = sum(int(sharing[k]) for k in [
        "LIRI_scrna_max_threads", "LIRI_virtual_total_threads", "LIRI_drug_preparation_max_threads"])
    host_effective = int(ph["effective_CPU_snapshot"])
    cpu_count = max(0, min(len(allowed), host_effective-outside_threads,
                          int(sharing["AKI_and_system_remaining"])))
    selected = allowed[-cpu_count:] if cpu_count else []
    plan = calculate_plan(mem, headroom, declared, anonymous, extra_growth, selected, own_anon, other_aki_caps)
    resolved = (R/"outputs").resolve()
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(resolved)], text=True))["filesystems"][0]
    free = shutil.disk_usage(resolved).free
    if mount["target"] != "/data" or not resolved.is_relative_to("/data"):
        raise RuntimeError("Output path is not on the verified mounted data disk")
    if free < 4*GIB:
        plan.update(accepted=False, storage_wait="Need 4GiB free for bounded output and scratch estimate")
    plan.update(time=now(), phase_sha256=sha(PHASE), active_units=rows, ancestors=ancestors,
                phase_declared_reservations=declared, mount=mount, data_free_bytes=free,
                expected_new_outputs_and_scratch_bytes=2*GIB, CPU_policy="joint declared future threads; numerical workers remain single-threaded")
    return plan

def preflight(include_self=False):
    pins = code_guard()
    state = unit_state(FORMAL_UNIT)
    acceptance = read(F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json") if (F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json").exists() else None
    status = read(F/"STATUS.json")
    reason = completion_reason(state, acceptance, status)
    result = {"time": now(), "code_sha256": sha(__file__), "frozen_inputs": pins,
              "formal_unit": state, "formal_state": status, "real_statistics_started": False}
    if reason:
        return dict(result, status=reason, eligible=False)
    # Small receipt integrity first; native unchanged guard verifies all large payloads
    # inside the admitted finite unit before calibration or clinical metrics.
    for item in acceptance["tasks"]:
        if sha(item["receipt"]) != item["sha256"]:
            raise RuntimeError("Formal receipt hash mismatch: " + item["receipt"])
    result["formal_acceptance_sha256"] = sha(F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json")
    plan = resource_plan(include_self)
    return dict(result, status="READY_FOR_FINITE_STATISTICS_UNIT" if plan["accepted"] else "WAIT_JOINT_RESOURCES",
                eligible=plan["accepted"], resource_plan=plan)

def apply_affinity(selected):
    requested = set(selected)
    if not requested or not requested.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Planned task-local CPUs are no longer permitted")
    os.sched_setaffinity(0, requested)
    if set(os.sched_getaffinity(0)) != requested:
        raise RuntimeError("Task-local CPU affinity was not applied")

def execute(receipt_path):
    receipt = read(receipt_path)
    if receipt["code_sha256"] != sha(__file__):
        raise RuntimeError("Launcher changed after admission")
    apply_affinity(receipt["resource_plan"]["CPU_affinity"])
    with (C/"LLM_STATISTICS_ACTIVE.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = preflight(include_self=True)
        if not result["eligible"]:
            raise RuntimeError("Worker preflight no longer eligible: " + result["status"])
        planned = receipt["resource_plan"]
        if result["resource_plan"]["phase_sha256"] != planned["phase_sha256"]:
            raise RuntimeError("Joint phase changed after launch; inspect before continuation")
        if result["resource_plan"]["memory_max_bytes"] < planned["memory_max_bytes"]:
            raise RuntimeError("Memory headroom decreased before preparation")
        own = unit_state(UNIT)
        if int(own["MemoryMax"]) != planned["memory_max_bytes"] or own["MemorySwapMax"] != "0":
            raise RuntimeError("Actual unit cap or no-swap property differs")
        if sorted(os.sched_getaffinity(0)) != planned["CPU_affinity"]:
            raise RuntimeError("Actual CPU affinity differs")
        if (S/"NUMERICAL_COMPARISON_ACCEPTANCE.json").exists():
            raise RuntimeError("Existing numerical acceptance must be reviewed, not rerun")
        owner = dict(receipt, status="RUNNING_FIXED_STATISTICS", PID=os.getpid(), execution_started=now())
        owner["start_ticks"] = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        save(OWNER, owner)
        cmd = [str(R/"envs/prediction_gpu/bin/python"), "-B", str(W/"bootstrap.py"), "--run"]
        child = subprocess.Popen(cmd, cwd=R)
        owner.update(child_PID=child.pid, command=cmd)
        save(OWNER, owner)
        rc = child.wait()
        numerical = S/"NUMERICAL_COMPARISON_ACCEPTANCE.json"
        if rc != 0 or not numerical.exists():
            raise RuntimeError("Native finite statistics failed; preserve partial outputs and diagnose, exit=" + str(rc))
        a = read(numerical)
        if a["LLM_result_routes"] != 24 or a["bootstrap_replicates_per_context"] != 2000:
            raise RuntimeError("Numerical receipt scope mismatch")
        owner.update(status="NUMERICAL_COMPLETE_REVIEW_REQUIRED", completed=now(),
                     child_returncode=rc, numerical_receipt_sha256=sha(numerical),
                     final_scientific_review=False, full37_complete=False)
        save(OWNER, owner)

def launch():
    with (C/"LLM_STATISTICS_LAUNCH.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        existing = unit_state(UNIT)
        if existing.get("LoadState") != "not-found":
            raise RuntimeError("Statistics unit already exists; inspect its real terminal/live state before any recovery")
        if OWNER.exists() or (S/"NUMERICAL_COMPARISON_ACCEPTANCE.json").exists():
            raise RuntimeError("Existing owner or result requires scoped inspection; no duplicate launch")
        qualification = read(O/"LAUNCH_GUARD_QUALIFICATION.json")
        if qualification["code_sha256"] != sha(__file__) or qualification["status"] != "PASS_OPERATIONAL_GUARDS_ONLY":
            raise RuntimeError("Current launcher qualification missing")
        check = preflight()
        save(O/"LATEST_PREFLIGHT.json", check)
        if not check["eligible"]:
            print(check["status"], flush=True)
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
        receipt_path = O/("LAUNCH_"+stamp+".json")
        log = R/"logs"/("llm_statistics_"+stamp+".log")
        receipt = dict(check, unit=UNIT, status="LAUNCHING", log=str(log), launch_receipt=str(receipt_path))
        save(receipt_path, receipt); save(OWNER, receipt)
        plan = receipt["resource_plan"]
        args = ["systemd-run", "--user", "--unit="+UNIT, "--property=Type=exec",
                "--property=MemoryMax="+str(plan["memory_max_bytes"]), "--property=MemorySwapMax=0",
                "--property=CPUWeight=1000", "--property=RuntimeMaxSec=43200",
                "--property=WorkingDirectory="+str(R),
                "--property=StandardOutput=append:"+str(log), "--property=StandardError=append:"+str(log)]
        for name, value in {"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
                            "NUMEXPR_NUM_THREADS":"1", "PYTHONDONTWRITEBYTECODE":"1",
                            "CUDA_VISIBLE_DEVICES":""}.items():
            args.append("--setenv="+name+"="+value)
        args += ["/usr/bin/python3", "-B", str(Path(__file__)), "--execute", str(receipt_path)]
        done = subprocess.run(args, capture_output=True, text=True)
        save(O/("DISPATCH_"+stamp+".json"), {"time":now(), "returncode":done.returncode,
             "stdout":done.stdout, "stderr":done.stderr, "unit":UNIT,
             "unknown_outcome_requires_same_unit_inspection":True})
        print("DISPATCH_RECORDED", UNIT, done.returncode, flush=True)

def qualify():
    checks = 0
    state = {"LoadState":"loaded", "ActiveState":"inactive", "MainPID":"0", "Result":"success", "ExecMainStatus":"0"}
    status = {"status":"COMPLETE_TRAINING_AND_PREDICTION_BANKS_STATISTICAL_COMPARISON_REQUIRED"}
    tasks = [{"kind":"fit","model":m,"stage":s,"line":line,"fit":str(k)}
             for m in MODELS for s in ["preop","endop"] for line in ["geography","temporal"] for k in [0,1,2,"final"]]
    tasks += [{"kind":"before","model":m,"stage":s} for m in MODELS for s in ["preop","endop"]]
    a = {"formal_fits":48,"original_checkpoint_banks":6,"contract_sha256":PINS[F/"FORMAL_TRAINING_CONTRACT.json"],"tasks":tasks}
    assert completion_reason(state,a,status) is None; checks+=1
    for change in [{"ActiveState":"active"},{"MainPID":"123"},{"Result":"exit-code"},{"ExecMainStatus":"1"},{"LoadState":"not-found"}]:
        assert completion_reason(dict(state,**change),a,status) is not None; checks+=1
    for change in [{"formal_fits":47},{"original_checkpoint_banks":5},{"contract_sha256":"bad"},{"tasks":tasks[:-1]},
                   {"tasks":tasks[:-2]+[tasks[-1],tasks[-1]]},{"tasks":tasks[1:]+[tasks[1]]}]:
        assert completion_reason(state,dict(a,**change),status) is not None; checks+=1
    assert completion_reason(state,None,status) is not None; checks+=1
    assert completion_reason(state,a,{"status":"RUNNING"}) is not None; checks+=1
    d={"AKI_existing_LLM_unit_cap":16,"LIRI_scrna_max":32,"LIRI_virtual_CPU_total":4,"LIRI_drug_preparation":2,"OS_other_headroom_min":4}
    p=calculate_plan(60*GIB,60*GIB,d,{},0,list(range(10)))
    assert p["memory_max_bytes"]==7*GIB and p["other_committed_growth"]==38*GIB and p["accepted"];checks+=1
    p=calculate_plan(40*GIB,40*GIB,d,{},0,list(range(10)))
    assert not p["accepted"] and p["memory_max_bytes"]==0;checks+=1
    p=calculate_plan(40*GIB,40*GIB,d,{"LIRI_scrna_max":28*GIB,"LIRI_virtual_CPU_total":GIB},0,list(range(10)))
    assert p["memory_max_bytes"]==16*GIB and p["other_committed_growth"]==9*GIB;checks+=1
    p=calculate_plan(40*GIB,10*GIB,d,{"LIRI_scrna_max":28*GIB,"LIRI_virtual_CPU_total":GIB},0,list(range(10)))
    assert not p["accepted"];checks+=1
    p=calculate_plan(40*GIB,40*GIB,d,{"LIRI_scrna_max":28*GIB,"LIRI_virtual_CPU_total":GIB},20*GIB,list(range(10)))
    assert not p["accepted"];checks+=1
    p=calculate_plan(60*GIB,60*GIB,d,{},0,[])
    assert not p["accepted"];checks+=1
    p=calculate_plan(60*GIB-GIB//16,60*GIB-GIB//16,d,{},0,list(range(10)),GIB//16)
    assert p["memory_max_bytes"]==7*GIB;checks+=1
    p=calculate_plan(40*GIB,40*GIB,d,{"LIRI_scrna_max":28*GIB,"LIRI_virtual_CPU_total":GIB},0,list(range(10)),0,12*GIB)
    assert p["memory_max_bytes"]==4*GIB and p["other_AKI_task_caps"]==12*GIB;checks+=1
    p=calculate_plan(40*GIB,40*GIB,d,{"LIRI_scrna_max":28*GIB,"LIRI_virtual_CPU_total":GIB},0,list(range(10)),0,16*GIB)
    assert not p["accepted"] and p["memory_max_bytes"]==0;checks+=1
    # This host does not delegate cpuset to the user manager. Qualify supported
    # per-process affinity in a new, finite, stdlib-only child instead.
    parent_affinity = sorted(os.sched_getaffinity(0))
    subset = parent_affinity[-min(10, len(parent_affinity)):]
    child_code = ("import importlib.util,json,os,sys;"
                  "s=importlib.util.spec_from_file_location('guard',sys.argv[1]);"
                  "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                  "m.apply_affinity(json.loads(sys.argv[2]));"
                  "print(json.dumps(sorted(os.sched_getaffinity(0))))")
    actual = json.loads(subprocess.check_output(
        ["/usr/bin/python3","-B","-c",child_code,str(Path(__file__)),json.dumps(subset)],
        text=True,timeout=15))
    assert actual==subset and sorted(os.sched_getaffinity(0))==parent_affinity;checks+=1
    # Real active-formal check must refuse before touching clinical metrics or NumPy.
    native_before={str(p):sha(p) for p in S.rglob("*") if p.is_file()}
    live=preflight()
    if live["formal_unit"].get("ActiveState")=="active":
        assert not live["eligible"] and live["status"]=="WAIT_FORMAL_PROCESS_EXIT";checks+=1
    assert "numpy" not in sys.modules and "torch" not in sys.modules;checks+=1
    native_after={str(p):sha(p) for p in S.rglob("*") if p.is_file()}
    assert native_before==native_after;checks+=1
    save(O/"LAUNCH_GUARD_QUALIFICATION.json",{"status":"PASS_OPERATIONAL_GUARDS_ONLY","time":now(),
         "code_sha256":sha(__file__),"checks":checks,"live_preflight":live,
         "native_statistics_files_unchanged":native_after,
         "max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         "no_new_training_or_clinical_metrics":True,"no_new_scheduler_or_other_unit_changes":True})
    print("QUALIFIED_OPERATIONAL_ONLY",checks,live["status"],flush=True)

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--qualify",action="store_true")
    mode.add_argument("--launch",action="store_true")
    mode.add_argument("--execute")
    args=parser.parse_args()
    O.mkdir(parents=True,exist_ok=True,mode=0o700)
    if args.qualify:qualify()
    elif args.launch:launch()
    elif args.execute:execute(args.execute)
    else:
        result=preflight();save(O/"LATEST_PREFLIGHT.json",result)
        print(result["status"],flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as error:
        save(O/("FAILURE_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")+".json"),
             {"time":now(),"error":type(error).__name__+": "+str(error),"code_sha256":sha(__file__),
              "automatic_retry_allowed":False,"preserve_existing_results":True})
        raise
