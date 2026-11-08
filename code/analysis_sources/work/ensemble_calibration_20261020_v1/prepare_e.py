"""Finite third-level scope manifest; frozen D results are read-only dependencies."""
from core import *
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
DOUT=ROOT/"outputs/ensemble_continuation_20261018_v1"
SOURCE=ROOT/"outputs/v4_development_inputs_20261018"
def routes(cohort):
 return [("preop",b) for b in ["full","compact","selected"]] if cohort=="preop" else [("preop","full"),("endop","full")]
def nested(meta,train,sid):
 m=meta.loc[train];seed0=int(digest({"seed":20261017,"scope":sid})[:8],16)
 # Only new, uncomputed deepest CV folds: deterministic training-label feasibility.
 for attempt in range(128):
  seed=(seed0+attempt)%(2**32);cv=[];feasible=True
  for f,(i,j) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(m,m.y,m.group_hash)):
   a=m.iloc[i];b=m.iloc[j]
   assert set(a.group_hash).isdisjoint(b.group_hash)
   if a.y.nunique()!=2 or b.y.nunique()!=2:feasible=False
   cv.append({"scheme":"inner","parent_outer":0,"fold":f,"train_excel_rows":a.excel_row.tolist(),"assessment_excel_rows":b.excel_row.tolist()})
  if feasible:return cv,seed
 raise ValueError("No feasible three-fold assignment: "+sid)
def make(s,p,meta):
 line,cohort=s["line"],s["cohort"];train=p["train_excel_rows"];test=p["assessment_excel_rows"]
 sid="s_"+digest({"line":line,"cohort":cohort,"train":train,"test":test})[:24]
 d=OUT/"scopes"/sid;inp=d/"inputs";inp.mkdir(parents=True,exist_ok=True)
 for name in ["FEATURE_SPECS.json","SOURCE_LOCK.json","INPUT_ACCEPTANCE.json",cohort+"_metadata.csv",cohort+"_preop_features.csv"]+([cohort+"_endop_features.csv"] if cohort=="paired" else []):
  if not (inp/name).exists():(inp/name).symlink_to(SOURCE/name)
 cv,seed=nested(meta,train,sid)
 part={"scheme":"outer","parent_outer":None,"fold":0,"train_excel_rows":train,"assessment_excel_rows":test}
 atomic(inp/(line+"_"+cohort+"_resampling.json"),cv+[part])
 a=meta.loc[train];b=meta.loc[test]
 assert set(a.group_hash).isdisjoint(b.group_hash)
 assert a[line+"_partition_v2"].eq("development").all() and b[line+"_partition_v2"].eq("development").all()
 result={"id":sid,"line":line,"cohort":cohort,"level":3,"parent":s["id"],"parent_fold":p["fold"],"train_rows":train,"assessment_rows":test,
 "train_groups_sha256":digest(a.group_hash.tolist()),"assessment_groups_sha256":digest(b.group_hash.tolist()),"input":str(inp),"output":str(d/"results"),"seed_deeper_folds":seed,"inner":cv,
 "train_n":len(a),"train_AKI":int(a.y.sum()),"assessment_n":len(b),"assessment_AKI":int(b.y.sum()),"new_E_scope":True}
 atomic(d/"SCOPE.json",result);return result
def main():
 man=json.loads((DOUT/"SCOPE_MANIFEST.json").read_text())
 assert json.loads((DOUT/"STAGE_D_ACCEPTANCE.json").read_text())["raw_ensemble_outer_fits"]==100
 assert json.loads((DOUT/"FINALIZATION_VERIFICATION_RECEIPT.json").read_text())["all_task_receipts_verified"]==13308
 oldby={s["id"]:s for s in man["scopes"]};oldroots={r["id"]:r for r in man["roots"]}
 roots=[];scopes=[];receipts={}
 metas={c:pd.read_csv(SOURCE/(c+"_metadata.csv"),dtype={"group_hash":str}).set_index("excel_row",drop=False) for c in ["preop","paired"]}
 for s in man["scopes"]:
  sp=DOUT/"scopes"/s["id"]/"SCOPE.json"
  assert json.loads(sp.read_text())==s
  receipts[str(sp)]=sha(sp)
  # Completed D is immutable; artifacts are hash-checked again at each read in E.
  rec=Path(s["output"])/"COARSE_ACCEPTANCE.json"
  assert json.loads(rec.read_text())["source_scope_sha256"]==sha(sp)
  receipts[str(rec)]=sha(rec)
  for stage,branch in routes(s["cohort"]):
   for fam in FAMILIES:
    rec=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/fam/"COMPLETE.json"
    rr=json.loads(rec.read_text());assert rr["signature"]["task"]["scope_sha256"]==sha(sp)
    receipts[str(rec)]=sha(rec)
  if s["level"]==1:
   parent=oldroots[s["parent"]]
   roots.append({"id":s["id"],"line":s["line"],"cohort":s["cohort"],"context":s["id"],
      "original_root":s["parent"],"original_context":parent["context"],"parent_fold":s["parent_fold"],
      "train_rows":s["train_rows"],"assessment_rows":s["assessment_rows"],"base_output":s["output"],
      "children":[t["id"] for t in man["scopes"] if t["parent"]==s["id"]]})
  elif s["level"]==2:
   scopes.append(s)
   for p in s["inner"]:scopes.append(make(s,p,metas[s["cohort"]]))
 assert len(roots)==72 and len(scopes)==864 and sum(s["level"]==3 for s in scopes)==648
 assert len({s["id"] for s in scopes})==864
 by={s["id"]:s for s in scopes}
 for r in roots:
  children=[by[k] for k in r["children"]]
  assert sorted(v for c in children for v in c["assessment_rows"])==sorted(r["train_rows"])
  for c in children:
   new=[s for s in scopes if s["parent"]==c["id"]]
   assert len(new)==3
   assert sorted(v for t in new for v in t["assessment_rows"])==sorted(c["train_rows"])
   for t in new:assert set(t["train_rows"])|set(t["assessment_rows"])==set(c["train_rows"])
 atomic(OUT/"REUSED_D_RECEIPTS.json",receipts)
 atomic(OUT/"SCOPE_MANIFEST.json",{"roots":roots,"scopes":scopes,"new_scopes":648,"reused_scopes":216,"source_D_manifest_sha256":sha(DOUT/"SCOPE_MANIFEST.json"),
 "D_acceptance_sha256":sha(DOUT/"STAGE_D_ACCEPTANCE.json"),"source_lock_sha256":sha(SOURCE/"SOURCE_LOCK.json"),"reused_receipts_sha256":sha(OUT/"REUSED_D_RECEIPTS.json"),
 "no_final_assessment_in_training":True})
 new=[s for s in scopes if s["level"]==3]
 atomic(OUT/"BOUNDARY_QUALIFICATION.json",{"status":"PASS","new_scopes":len(new),"first_layer_ensemble_roots":len(roots),
 "new_searches":sum(len(routes(s["cohort"]))*9 for s in new),"first_layer_route_tasks":sum(len(routes(r["cohort"])) for r in roots),
 "min_train_AKI":min(s["train_AKI"] for s in new),"min_assessment_AKI":min(s["assessment_AKI"] for s in new),"patient_group_disjoint":True,"development_only":True})
 print(canonical(json.loads((OUT/"BOUNDARY_QUALIFICATION.json").read_text())),flush=True)
if __name__=="__main__":main()
