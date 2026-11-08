
from pathlib import Path
import ast,hashlib,importlib.util,json,os,resource,subprocess,sys,tempfile,time
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/study_package_scope_20261030_v1"
B=R/"outputs/study_package_scope_20261030_v1/portable_pod1_replay_entry_preparation_v1"
resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(60,60))
os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
started=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
scalar=load("_aki_protection_for_entry_preparation",W/"audit_fixed_policy_records.py")
protected=scalar.protection()
modules_before=set(sys.modules)
entry=load("_aki_pod1_replay_entry_prepare",W/"replay_portable_pod1.py")
admission=entry.preflight()
assert admission["status"]=="WAIT_FORMAL_PROCESS_EXIT" and not admission["eligible"]
assert not entry.O.exists()
added=set(sys.modules)-modules_before
numeric=[m for m in added if m.split(".")[0] in {"numpy","pandas","scipy","sklearn","xgboost","torch","joblib"}]
assert not numeric,numeric
tree=ast.parse((W/"replay_portable_pod1.py").read_text())
nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="audit_open"]
assert len(nodes)==1
audit_source=ast.unparse(nodes[0])
with tempfile.TemporaryDirectory(prefix="owned-read-guard-",dir=B) as temporary:
 t=Path(temporary).resolve();orig=t/"original";out=t/"out";pkg=out/"package";outside=t/"outside"
 for p in [orig,pkg,outside]:p.mkdir(parents=True,exist_ok=True)
 for p in [orig/"old.txt",pkg/"copy.txt",out/"log.txt",outside/"library.txt"]:p.write_text("test-only sentinel")
 (pkg/"redirect.txt").symlink_to(orig/"old.txt")
 script="from pathlib import Path\nimport os,sys,json\n"+audit_source+"\n"
 script+="scratch=Path("+repr(str(pkg))+")\nown_output=Path("+repr(str(out))+")\noriginal_roots=[Path("+repr(str(orig))+")]\n"
 script+='audit={"enforce":True,"forbidden_original_reads":[],"allowed_package_reads":0}\nsys.addaudithook(audit_open)\nchecks=[]\n'
 script+="allowed="+repr([str(pkg/"copy.txt"),str(out/"log.txt"),str(outside/"library.txt")])+"\n"
 script+='for p in allowed:\n assert Path(p).read_text()=="test-only sentinel"\n checks.append("allowed")\n'
 script+="denied="+repr([(str(orig/"old.txt"),"r"),(str(orig/"new.txt"),"w"),(str(pkg/"redirect.txt"),"r")])+"\n"
 script+='for p,mode in denied:\n try:\n  open(p,mode).close()\n except RuntimeError as e:\n  assert "fallback" in str(e)\n  checks.append("rejected")\n else: raise AssertionError("forbidden open was not rejected")\n'
 script+='audit["enforce"]=False\n'
 script+="assert Path("+repr(str(orig/"old.txt"))+").read_text()=='test-only sentinel'\nchecks.append('explicit_post_replay_hash_window')\n"
 script+='assert len(checks)==7 and len(audit["forbidden_original_reads"])==3\nprint(json.dumps({"checks":checks,"forbidden_attempts_rejected":3,"clinical_files_read":0}))\n'
 cp=subprocess.run([sys.executable,"-B","-c",script],capture_output=True,text=True,timeout=30,check=False)
 (B/"READ_GUARD_CHILD_STDOUT.txt").write_text(cp.stdout)
 (B/"READ_GUARD_CHILD_STDERR.txt").write_text(cp.stderr)
 assert cp.returncode==0,"Audit child failed; inspect persisted stderr"
 guard=json.loads(cp.stdout)
assert scalar.protection()==protected
result={"status":"PASS_FINITE_ENTRY_PREFLIGHT_AND_ORIGINAL_PATH_READ_GUARD_ONLY",
 "entry_sha256":sha(W/"replay_portable_pod1.py"),
 "code_lock_sha256":sha(W/"POD1_PORTABLE_REPLAY_CODE_LOCK.json"),
 "preflight":admission,"fresh_child_audit_hook_checks":guard,
 "numerical_library_imports_added":numeric,"actual_model_loads":0,
 "actual_prediction_replays":0,"new_fits":0,"new_bootstraps":0,
 "numeric_output_directory_created":entry.O.exists(),
 "manuscript_92_hashes_unchanged":True,"owned_guard_fixture_removed":not t.exists(),
 "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 "seconds":time.monotonic()-started,
 "not_full_package_or_scientific_acceptance":True}
with (B/"ENTRY_PREPARATION_QUALIFICATION.json").open("x") as h:json.dump(result,h,ensure_ascii=False,indent=2);h.write("\n")
print(json.dumps({k:result[k] for k in ["status","actual_model_loads","actual_prediction_replays","peak_RSS_bytes","seconds","owned_guard_fixture_removed"]},ensure_ascii=False))
