"""Qualify new finite entry guard without loading any scientific model."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import traceback

R = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W = R/"work/study_package_scope_20261030_v1"
B = R/"outputs/study_package_scope_20261030_v1/portable_locked_route_replay_entry_preparation_v1"
OLD = R/"outputs/study_package_scope_20261030_v1/portable_pod1_replay_entry_preparation_v1"
ENTRY_SHA = "5d16e31024541e734f9a8e1954f91adb8acd82e6500a41317fdc08385af0a583"
OLD_ENTRY_SHA = "5b1b41c8fdf09ec9908d607e4b5e86af6c462f518e272a5febb79f6d34c6f99c"
LOCK_SHA = "b9f1f9895900eb6891278329a9f0faa17f6338148282a30c99c584c1b56b871d"
OLD_GUARD_SHA = "cb15fd00e8009abb7aaa7018b31bce2679b4d3cc739137cab83ff2114f58c725"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def save(path,value):
    with path.open("x") as h:
        json.dump(value,h,ensure_ascii=False,indent=2);h.write("\n")
        h.flush();os.fsync(h.fileno())

def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    start=time.monotonic()
    if (B/"ENTRY_PREPARATION_QUALIFICATION.json").exists() or (B/"FAILURE.json").exists():
        raise RuntimeError("Inspect existing result before rerun")
    try:
        before_modules=set(sys.modules)
        entry_path=W/"replay_portable_locked_routes.py"
        assert sha(entry_path)==ENTRY_SHA
        assert sha(W/"LOCKED_ROUTE_PORTABLE_REPLAY_CODE_LOCK.json")==LOCK_SHA
        assert sha(W/"replay_portable_pod1.py")==OLD_ENTRY_SHA
        assert sha(OLD/"ENTRY_PREPARATION_QUALIFICATION.json")==OLD_GUARD_SHA
        old_receipt=json.loads((OLD/"ENTRY_PREPARATION_QUALIFICATION.json").read_text())
        assert old_receipt["entry_sha256"]==OLD_ENTRY_SHA
        original_checks=old_receipt["fresh_child_audit_hook_checks"]
        assert len(original_checks["checks"])==7 and original_checks["forbidden_attempts_rejected"]==3
        old_tree=ast.parse((W/"replay_portable_pod1.py").read_text())
        new_tree=ast.parse(entry_path.read_text())
        same={}
        for name in ["audit_open","sample_owned_gpu","gpu_snapshot"]:
            a=[n for n in ast.walk(old_tree) if isinstance(n,ast.FunctionDef) and n.name==name]
            b=[n for n in ast.walk(new_tree) if isinstance(n,ast.FunctionDef) and n.name==name]
            assert len(a)==len(b)==1
            aa=ast.dump(a[0],include_attributes=False);bb=ast.dump(b[0],include_attributes=False)
            assert aa==bb
            same[name]=hashlib.sha256(aa.encode()).hexdigest()
        protection=load("_new_entry_protection",W/"audit_fixed_policy_records.py")
        original_protection=protection.protection()
        entry=load("_new_locked_replay_entry",entry_path)
        assert not entry.O.exists()
        entry.code_guard()
        script = (
            "import sys,runpy,json\n"
            "before=set(sys.modules)\n"
            "sys.argv=["+repr(str(entry_path))+",'--run']\n"
            "runpy.run_path("+repr(str(entry_path))+",run_name='__main__')\n"
            "added=set(sys.modules)-before\n"
            "numeric=sorted(m for m in added if m.split('.')[0] in {'numpy','pandas','scipy','sklearn','xgboost','torch','joblib'})\n"
            "print(json.dumps({'numerical_imports_added':numeric}))\n"
        )
        child=subprocess.run([sys.executable,"-B","-c",script],capture_output=True,text=True,timeout=30)
        for name,text in [("FINITE_RUN_GUARD_STDOUT.txt",child.stdout),("FINITE_RUN_GUARD_STDERR.txt",child.stderr)]:
            with (B/name).open("x") as h:h.write(text)
        assert child.returncode==0,"Inspect saved child stderr"
        emitted=[json.loads(line) for line in child.stdout.splitlines() if line.strip()]
        assert len(emitted)==2
        admission,modules=emitted
        assert admission["status"]=="WAIT_FORMAL_PROCESS_EXIT" and not admission["eligible"]
        assert admission["formal_unit_state"]["ActiveState"]=="active"
        assert int(admission["formal_unit_state"]["MainPID"])>0
        assert not admission["actual_numeric_replay_started"]
        assert modules["numerical_imports_added"]==[]
        assert not entry.O.exists()
        numeric=sorted(m for m in set(sys.modules)-before_modules
            if m.split(".")[0] in {"numpy","pandas","scipy","sklearn","xgboost","torch","joblib"})
        assert not numeric
        assert protection.protection()==original_protection
        result={
            "status":"PASS_FINITE_LOCKED_ROUTE_ENTRY_RUN_GUARD_ONLY_NUMERICAL_REPLAY_PENDING",
            "entry_sha256":ENTRY_SHA,"code_lock_sha256":LOCK_SHA,
            "qualification_code_sha256":sha(__file__),"actual_run_switch_guard":admission,
            "child_exit_code":child.returncode,"actual_model_loads":0,
            "actual_prediction_replays":0,"new_fits":0,"new_bootstraps":0,
            "numerical_imports_added_parent":numeric,"numerical_imports_added_child":[],
            "numeric_output_directory_created":False,"manuscript92_unchanged":True,
            "unchanged_guard_ASTs":same,
            "prior_actual_path_read_guard_receipt_sha256":OLD_GUARD_SHA,
            "prior_path_guard_cases_reused_by_exact_AST_identity":7,
            "new_path_guard_cases_claimed":0,
            "GPU_sampler_actual_runtime_qualification_in_this_preparation":False,
            "owned_child_exited":True,
            "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "seconds":time.monotonic()-start,"not_package_or_scientific_acceptance":True}
        save(B/"ENTRY_PREPARATION_QUALIFICATION.json",result)
        print(json.dumps({k:result[k] for k in ["status","actual_model_loads","actual_prediction_replays","prior_path_guard_cases_reused_by_exact_AST_identity","peak_RSS_bytes","seconds"]}))
    except BaseException:
        if not (B/"FAILURE.json").exists():
            save(B/"FAILURE.json",{"traceback":traceback.format_exc(),"numeric_entry_not_authorized_by_this_failure":True})
        raise

if __name__=="__main__":
    main()
