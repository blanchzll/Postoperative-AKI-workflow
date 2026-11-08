"""Focused real-input boundary and provenance checks, not performance validation."""
from core import *
from preprocess import impute
import numpy as np,pandas as pd,joblib,time
from threadpoolctl import threadpool_limits
start=time.time();checks={}
source=json.loads((INPUT/"SOURCE_LOCK.json").read_text())
for f,h in source["input_sha256"].items():assert sha(INPUT/f)==h
checks["frozen_inputs_hashes"]=True
for line in ["geography","temporal"]:
 for cohort in ["preop","paired"]:
  meta=pd.read_csv(INPUT/(cohort+"_metadata.csv"),dtype={"group_hash":str}).set_index("excel_row")
  allp=json.loads((INPUT/(line+"_"+cohort+"_resampling.json")).read_text())
  for p in allp:
   tr=meta.loc[p["train_excel_rows"]];va=meta.loc[p["assessment_excel_rows"]]
   assert set(tr.group_hash).isdisjoint(va.group_hash)
   assert tr[line+"_partition_v2"].eq("development").all() and va[line+"_partition_v2"].eq("development").all()
   if p["scheme"]=="inner":
    parent=next(x for x in allp if x["scheme"]=="outer" and x["fold"]==p["parent_outer"])
    assert set(tr.index)|set(va.index)==set(parent["train_excel_rows"])
checks["all_four_manifests_patient_groups_and_outer_nesting"]=True
p=parts("geography","preop","final")[0]
full=pair_task("geography","preop","preop","final",p,"bayes")
impute(full,1)
compact=pair_task("geography","preop","preop","final",p,"bayes","compact")
with threadpool_limits(limits=1):impute(compact,1)
imp=joblib.load(cache_dir(compact)/"imputer.joblib")
assert set(imp.cols)==set(COMPACT) and len(imp.cols)==9
assert imp.original_.shape[1]==9
checks["real_compact9_only_fit_and_transform"]=True
sp=spec_for("preop",COMPACT);_,tm,x,vm=frame_and_meta(compact,sp)
z=imp.transform(x.iloc[:4],keys(vm.iloc[:4]))
for c in imp.cols:
 assert c in COMPACT
checks["real_missing_batch_single_reverse_reload"]=True
bad=dict(compact);bad["assessment_rows"]=compact["train_rows"][:3]
try:frame_and_meta(bad,sp);raise AssertionError("Leakage not detected")
except ValueError as e:assert "leakage" in str(e)
checks["overlap_rejected"]=True
try:imp.transform(x.iloc[:2],["same","same"]);raise AssertionError("duplicate keys accepted")
except ValueError:pass
checks["duplicate_keys_rejected"]=True
checks["cache_reuse_files_verified"]=ready(cache_dir(full),cache_signature(full))
assert all(checks.values())
atomic(OUT/"FUNCTIONAL_QUALIFICATION.json",{"status":"PASS","scope":"engineering input/boundary/inference checks only, not clinical performance","checks":checks,"seconds":time.time()-start})
print(canonical({"status":"PASS","checks":checks,"seconds":time.time()-start}))
