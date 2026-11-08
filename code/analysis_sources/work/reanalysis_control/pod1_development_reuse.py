"""Strong identity-only reuse guard for completed current-source POD1 development."""
from pathlib import Path
import json,hashlib,os,sys,fcntl,time,collections
R=Path(__file__).resolve().parents[2]
D=R/"outputs/pod1_update_20261017"
SEAL=D/"DEVELOPMENT_REUSE_SEAL.json"
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024**2),b""):h.update(b)
 return h.hexdigest()
def current_identity():
 proto=read(D/"protocol.json");ph=sha(D/"protocol.json")
 source=R/proto["source"];assert sha(source)==proto["source_sha256"]
 source_lock=R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"
 assert sha(source_lock)==proto["source_lock_sha256"]
 extra={source,source_lock}
 for rel,h in read(source_lock)["input_sha256"].items():
  p=source_lock.parent/rel;assert sha(p)==h;extra.add(p)
 for rel,h in read(D/"code_freeze_manifest.json")["files"].items():
  p=R/rel;assert sha(p)==h,rel;extra.add(p)
 a=read(D/"DEVELOPMENT_SELECTION_ACCEPTANCE.json")
 assert a["status"]=="PASS_fresh_corrected_POD1_development_selection_only"
 assert a["models_locked_before_fixed_refit"] and a["heldout_predictions_not_computed_here"]
 z=read(D/"input_acceptance.json")
 assert z["status"]=="passed" and (z["eligible_n"],z["eligible_events"])==(2947,39)
 assert z["source_sha256"]==proto["source_sha256"]
 bridge=read(D/"PREDICTOR_SOURCE_BRIDGE.json")
 assert bridge["status"]=="PASS_only_source_documented_surgical_text_recoveries"
 assert z["predictor_bridge_receipt_sha256"]==sha(D/"PREDICTOR_SOURCE_BRIDGE.json")
 assert bridge["source_lock_sha256"]==sha(source_lock)
 assert bridge["latest_features_sha256"]==sha(source_lock.parent/"full_source_endop_features.csv")
 assert bridge["recovery_ledger_sha256"]==sha(source_lock.parent/"SURGICAL_TEXT_RECOVERY_LEDGER.csv")
 assert bridge["changed_n"]==len(set(bridge["eligible_recovery_rows"]))==11
 worker=sha(R/"work/pod1_update_20261017/worker.py")
 jobs=0;imputations=0
 for line,up in [("geography","preop_outcomes_python_20261017"),("temporal","preop_temporal_python_20261017")]:
  choice=R/"outputs/formal_development_20261018_v1/selection"/line/"preop/final/METHOD_SELECTION.json"
  assert read(choice)["selected_method"]==proto["imputation"][line];extra.add(choice)
  original=R/"outputs"/up/"fold_inputs/fold_manifest.json";extra.add(original)
  manifest=read(D/line/"fold_manifest.json");assert sha(original)==manifest["original_manifest_sha256"]
  for pair in read(original)["pairs"]:
   for spec in pair["files"].values():
    if isinstance(spec,dict) and "path" in spec:extra.add(R/spec["path"])
  model_lock=read(D/line/"MODEL_LOCK.json")
  assert model_lock["protocol_sha256"]==ph and model_lock["no_heldout_winner_selection"]
  for p in manifest["pairs"]:
   if not p["role"].startswith("development"):continue
   branch=p["branch"];name=p["name"];cache=D/line/"imputed"/branch/name
   expected={"source_sha256":sha(D/line/"pairs"/branch/(name+".joblib")),"protocol_sha256":ph,
    "method":proto["imputation"][line],"branch":branch,"line":line,"pair":name}
   assert read(cache/"SUCCESS.json")==expected
   assert (cache/"data.joblib").is_file() and (cache/"state.joblib").is_file();imputations+=1
   if p["role"]!="development_outer":continue
   for family in proto["families"][branch]:
    dest=D/line/"development"/branch/name/family
    expected={"protocol_sha256":ph,"worker_sha256":worker,"cache_sha256":sha(cache/"data.joblib"),
     "mode":"development","branch":branch,"pair":name,"family":family}
    assert read(dest/"SUCCESS.json")==expected
    for n in ["predictions.csv","model.joblib","selected_inner_oof.csv","tuning.json"]:assert (dest/n).is_file()
    jobs+=1
 assert jobs==110 and imputations==200
 return extra,{"development_contexts":jobs,"imputation_pairs":imputations,"eligible_n":2947,"events":39,
  "verification_scope":"Computational source/code/checkpoint identity only; not scientific-review acceptance"}
def verify_files(files):
 for rel,h in files.items():
  assert not Path(rel).is_absolute() and ".." not in Path(rel).parts
  assert sha(R/rel)==h,"Completed POD1 checkpoint/dependency changed: "+rel
def verify():
 s=read(SEAL);assert s["status"]=="PASS_current_source_POD1_development_identity_sealed"
 verify_files(s["files"]);current_identity();return s
def seal():
 assert not SEAL.exists(),"Existing seal must be verified, never silently replaced"
 extra,summary=current_identity()
 ignored={"status.json","resource_admission.json","DEVELOPMENT_REUSE_SEAL.json"}
 files={p for p in D.rglob("*") if p.is_file() and p.name not in ignored and not p.name.endswith(".lock")}
 files|=extra
 value={"status":"PASS_current_source_POD1_development_identity_sealed","created_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
  **summary,"files":{str(p.relative_to(R)):sha(p) for p in sorted(files)},
  "MODEL_LOCK_timestamps_preserved":True,"no_model_refit":True}
 q=SEAL.with_suffix(".tmp");q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n");os.replace(q,SEAL)
 return verify()
if __name__=="__main__":
 with (R/"work/.pod1_update.lock").open("a+") as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  v=seal() if len(sys.argv)>1 and sys.argv[1]=="seal" else verify()
 print("POD1_DEVELOPMENT_REUSE_VERIFIED",len(v["files"]),v["development_contexts"],v["imputation_pairs"])
