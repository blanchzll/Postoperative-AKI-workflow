"""Strict calibrated candidate and inner-selected procedure internal evaluation."""
from core import *
from ensemble_math import *
from prepare_e import routes
from functools import lru_cache
DOUT=ROOT/"outputs/ensemble_continuation_20261018_v1"
BASE=ROOT/"outputs/formal_development_20261018_v1"
SOURCE=ROOT/"outputs/v4_development_inputs_20261018"
DM=json.loads((DOUT/"SCOPE_MANIFEST.json").read_text())
SM={s["id"]:s for s in DM["scopes"]}
EM=json.loads((OUT/"SCOPE_MANIFEST.json").read_text())
ER={r["id"]:r for r in EM["roots"]}
ALL=FAMILIES+["Voting","Stacking"]
@lru_cache(maxsize=2)
def meta(cohort):return pd.read_csv(SOURCE/(cohort+"_metadata.csv"),dtype={"group_hash":str}).set_index("excel_row",drop=False)
def read_prediction(p,receipt,rows,line,cohort):
 assert sha(p)==receipt["files"][p.name]
 x=pd.read_csv(p,dtype={"group_hash":str},float_precision="round_trip")
 assert x.excel_row.is_unique and set(x.excel_row)==set(rows)
 m=meta(cohort).loc[x.excel_row]
 assert m.y.tolist()==x.y.tolist() and m.group_hash.tolist()==x.group_hash.tolist()
 assert m[line+"_partition_v2"].eq("development").all()
 assert np.isfinite(x.prob_raw).all() and x.prob_raw.between(0,1).all()
 return x
def esource(s,stage,branch,family):
 if family in FAMILIES:
  d=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/family
  name="predictions.csv"
 else:
  d=OUT/"ensemble_results"/s["line"]/s["cohort"]/stage/branch/s["id"];name=family+"_outer.csv"
 assert ready(d)
 rec=json.loads((d/"COMPLETE.json").read_text())
 return read_prediction(d/name,rec,s["assessment_rows"],s["line"],s["cohort"]),d/"COMPLETE.json"
def choose(pool):
 ap=max(x["inner_AP"] for x in pool);pool=[x for x in pool if x["inner_AP"]>=ap-1e-12]
 br=min(x["inner_Brier"] for x in pool)
 return min([x for x in pool if x["inner_Brier"]<=br+1e-12],key=lambda x:x["candidate_id"])
def training_pool(s,stage,branches,lineage):
 pool=[]
 for branch in branches:
  for family in ALL:
   if family in FAMILIES:
    d=Path(s["output"])/"search_v2"/s["line"]/s["cohort"]/stage/branch/"outer_0"/family
    assert ready(d)
    rpath=d/"COMPLETE.json";rec=json.loads(rpath.read_text());best=rec["selected"]
    assert rec["signature"]["task"]["scope_id"]==s["id"]
    assert rec["signature"]["task"]["scope_sha256"]==sha(DOUT/"scopes"/s["id"]/"SCOPE.json")
    assert sha(best["prediction_file"])==best["prediction_sha256"]
    with np.load(best["prediction_file"],allow_pickle=False) as z:
     assert len(z["excel_row"])==len(s["train_rows"]) and set(z["excel_row"])==set(s["train_rows"])
     m=meta(s["cohort"]).loc[z["excel_row"]]
     assert m.y.tolist()==z["y"].tolist() and m.group_hash.tolist()==z["group_hash"].tolist()
   else:
    d=OUT/"ensemble_results"/s["line"]/s["cohort"]/stage/branch/s["id"];assert ready(d)
    rpath=d/"COMPLETE.json";rec=json.loads(rpath.read_text());best=json.loads((d/"SELECTION.json").read_text())["winners"][family]
    read_prediction(d/("inner_"+best["candidate_id"][:16]+".csv"),rec,s["train_rows"],s["line"],s["cohort"])
   lineage[str(rpath)]=sha(rpath)
   cid=canonical({"branch":branch,"family":family,"parameter_candidate":best["candidate_id"]})
   pool.append({"branch":branch,"family":family,"inner_AP":float(best["mean_AP"]),"inner_Brier":float(best["mean_Brier"]),"candidate_id":cid,"source":str(rpath)})
 return pool
def save_calibration(d,train,held):
 assert train.excel_row.is_unique and train.y.nunique()==2
 model=calibrator(train)
 d.mkdir(parents=True,exist_ok=True)
 train.to_csv(d/"PRIVATE_HONEST_CALIBRATION_OOF.csv",index=False)
 joblib.dump(model,d/"calibrator.joblib",compress=3)
 loaded=joblib.load(d/"calibrator.joblib")
 if held is not None:
  assert set(train.group_hash).isdisjoint(held.group_hash)
  q=held.copy();q["prob_calibrated"]=apply_calibration(model,q.prob_raw)
  np.testing.assert_allclose(q.prob_calibrated,apply_calibration(loaded,q.prob_raw),atol=1e-12,rtol=1e-12)
  np.testing.assert_allclose(q.prob_calibrated,apply_calibration(loaded,q.prob_raw.iloc[::-1])[::-1],atol=1e-12,rtol=1e-12)
  q.to_csv(d/"PRIVATE_CALIBRATED_OUTER.csv",index=False)
  return q
 return None
def main():
 assert json.loads((OUT/"COMPUTE_BATCH_ACCEPTANCE.json").read_text())["status"]=="PASS"
 dest=OUT/"calibrated_internal";dest.mkdir(mode=0o700,exist_ok=True)
 if (dest/"ACCEPTANCE.json").exists():
  rec=json.loads((dest/"ACCEPTANCE.json").read_text())
  assert all(sha(dest/f)==h for f,h in rec["files"].items());return
 decisions_source=DOUT/"interim_all_development_20261019/INNER_SELECTION_DECISIONS.json"
 frozen=json.loads(decisions_source.read_text())["decisions"]
 assert len(frozen)==30
 lineage={str(decisions_source):sha(decisions_source)}
 per_child=[]
 # Freeze training-only child branch/family decisions before reading outer held scores.
 for r in EM["roots"]:
  s=SM[r["id"]]
  procedures=[("preop",["full","compact","selected"])] if s["cohort"]=="preop" else [("preop",["full"]),("endop",["full"])]
  for stage,branches in procedures:
   pool=training_pool(s,stage,branches,lineage);w=choose(pool)
   assert len(pool)==(33 if s["cohort"]=="preop" else 11)
   per_child.append({"scope":s["id"],"line":s["line"],"cohort":s["cohort"],"stage":stage,"selected":w,"candidates":pool,"training_only":True})
 atomic(dest/"TRAINING_ONLY_CHILD_SELECTIONS.json",{"decisions":per_child,"final_evaluation_used":False})
 child_winner={(x["scope"],x["stage"]):x["selected"] for x in per_child}
 ensemble_metrics=[];base_metrics=[];proc_metrics=[];proc_all=[];cal_count=0;rootcount=0
 for root in DM["roots"]:
  line,cohort,context=root["line"],root["cohort"],root["context"]
  top=sorted([s for s in DM["scopes"] if s["parent"]==root["id"]],key=lambda s:s["parent_fold"])
  assert len(top)==3
  for stage,branch in routes(cohort):
   original=DOUT/"ensemble_results"/line/cohort/stage/branch/context
   assert ready(original);orec=json.loads((original/"COMPLETE.json").read_text())
   if context!="final":
    for fam in FAMILIES:
     fn=original/(fam+"_calibrated_outer.csv")
     bx=read_prediction(fn,orec,root["assessment_rows"],line,cohort)
     assert np.isfinite(bx.prob_calibrated).all() and bx.prob_calibrated.between(0,1).all()
     base_metrics.append({"line":line,"cohort":cohort,"stage":stage,"branch":branch,"family":fam,"outer_fold":int(context.split("_")[1]),"n":len(bx),"events":int(bx.y.sum()),**score(bx.y,bx.prob_calibrated)})
     lineage[str(fn)]=sha(fn)
   for family in ["Voting","Stacking"]:
    tr=[];sources={}
    for s in top:
     x,rec=esource(s,stage,branch,family);tr.append(x.assign(fold=s["parent_fold"]));sources[str(rec)]=sha(rec)
    train=pd.concat(tr,ignore_index=True)
    assert set(train.excel_row)==set(root["train_rows"]) and train.excel_row.is_unique
    held=None
    if context!="final":
     held=read_prediction(original/(family+"_outer.csv"),orec,root["assessment_rows"],line,cohort)
    d=dest/"candidate_calibration"/line/cohort/stage/branch/context/family
    q=save_calibration(d,train,held);cal_count+=1
    sources[str(original/"COMPLETE.json")]=sha(original/"COMPLETE.json")
    atomic(d/"LINEAGE.json",sources)
    if q is not None:
     ensemble_metrics.append({"line":line,"cohort":cohort,"stage":stage,"branch":branch,"family":family,"outer_fold":int(context.split("_")[1]),"n":len(q),"events":int(q.y.sum()),**score(q.y,q.prob_calibrated)})
  procedures=[("preop",["full","compact","selected"])] if cohort=="preop" else [("preop",["full"]),("endop",["full"])]
  for stage,branches in procedures:
   tr=[];sources={}
   for s in top:
    w=child_winner[(s["id"],stage)]
    x,rec=esource(s,stage,w["branch"],w["family"])
    tr.append(x.assign(fold=s["parent_fold"],selected_family=w["family"],selected_branch=w["branch"]))
    sources[str(rec)]=sha(rec)
   train=pd.concat(tr,ignore_index=True)
   assert train.excel_row.is_unique and set(train.excel_row)==set(root["train_rows"])
   held=None;w=None
   if context!="final":
    fold=int(context.split("_")[1])
    dec=next(v for v in frozen if v["line"]==line and v["cohort"]==cohort and v["stage"]==stage and v["fold"]==fold)
    w=dec["selected"]
    if w["family"] in FAMILIES:
     d=BASE/"outer_raw"/line/cohort/stage/w["branch"]/context/w["family"];name="predictions.csv"
    else:
     d=DOUT/"ensemble_results"/line/cohort/stage/w["branch"]/context;name=w["family"]+"_outer.csv"
    assert ready(d);rec=json.loads((d/"COMPLETE.json").read_text())
    held=read_prediction(d/name,rec,root["assessment_rows"],line,cohort)
    sources[str(d/"COMPLETE.json")]=sha(d/"COMPLETE.json")
   d=dest/"procedure_calibration"/line/cohort/stage/context
   q=save_calibration(d,train,held);rootcount+=1
   atomic(d/"LINEAGE.json",sources)
   if q is not None:
    proc_metrics.append({"line":line,"cohort":cohort,"stage":stage,"outer_fold":fold,"selected_family":w["family"],"selected_branch":w["branch"],"n":len(q),"events":int(q.y.sum()),**score(q.y,q.prob_calibrated)})
    proc_all.append(q.assign(line=line,cohort=cohort,stage=stage,outer_fold=fold,selected_family=w["family"],selected_branch=w["branch"]))
 assert len(ensemble_metrics)==100 and len(proc_metrics)==30 and cal_count==120 and rootcount==36
 base=pd.DataFrame(base_metrics)
 assert len(base)==450
 combined=pd.concat([base,pd.DataFrame(ensemble_metrics)],ignore_index=True)
 group=["line","cohort","stage","branch","family"]
 assert combined.groupby(group).size().eq(5).all() and len(combined)==550
 combined.to_csv(dest/"CALIBRATED_ELEVEN_FAMILY_FOLD_METRICS.csv",index=False)
 summary=combined.groupby(group)[["AP","AUROC","Brier"]].agg(["mean","std"]);summary.columns=["_".join(x) for x in summary.columns]
 summary.reset_index().to_csv(dest/"CALIBRATED_ELEVEN_FAMILY_SUMMARY.csv",index=False)
 procs=pd.DataFrame(proc_metrics);procs.to_csv(dest/"INNER_SELECTED_CALIBRATED_PROCEDURE_FOLD_METRICS.csv",index=False)
 summary=procs.groupby(["line","cohort","stage"])[["AP","AUROC","Brier"]].agg(["mean","std"]);summary.columns=["_".join(x) for x in summary.columns]
 summary.reset_index().to_csv(dest/"INNER_SELECTED_CALIBRATED_PROCEDURE_SUMMARY.csv",index=False)
 pd.concat(proc_all,ignore_index=True).to_csv(dest/"PRIVATE_INNER_SELECTED_CALIBRATED_PREDICTIONS.csv",index=False)
 atomic(dest/"SOURCE_LINEAGE.json",lineage)
 files={str(p.relative_to(dest)):sha(p) for p in dest.rglob("*") if p.is_file() and p.name!="ACCEPTANCE.json"}
 acceptance={"status":"PASS","calibrated_candidate_fold_metrics":550,"calibrated_inner_selected_procedure_fold_metrics":30,"ensemble_calibration_contexts":120,"procedure_calibration_contexts":36,
 "all_calibration_rows_excluded_from_all_upstream_selection":True,"final_model_locked":False,"final_hospital_time_evaluation_accessed":False,"files":files,"postprocessor_sha256":sha(__file__)}
 atomic(dest/"ACCEPTANCE.json",acceptance)
 atomic(OUT/"STAGE_E_ACCEPTANCE.json",{k:v for k,v in acceptance.items() if k!="files"}|{"detail_acceptance_sha256":sha(dest/"ACCEPTANCE.json"),"next_required":"lock complete pipelines and training thresholds before final hospital/time assessment"})
 print(canonical({k:v for k,v in acceptance.items() if k!="files"}),flush=True)
if __name__=="__main__":main()
