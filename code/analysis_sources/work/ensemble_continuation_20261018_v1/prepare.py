"""Predeclared nested scope DAG; preserve outer/inner memberships."""
from core import *
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
ORIGINAL=ROOT/"outputs/formal_development_20261018_v1"
SOURCE=ROOT/"outputs/v4_development_inputs_20261018"
def nested(meta,train,scope_id):
 m=meta.loc[train]
 # New deeper folds only; existing five-by-three manifests are unchanged.
 seed=int(digest({"seed":20261017,"scope":scope_id})[:8],16)
 splitter=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
 rows=[]
 for f,(i,j) in enumerate(splitter.split(m,m.y,m.group_hash)):
  a=m.iloc[i];b=m.iloc[j]
  assert set(a.group_hash).isdisjoint(b.group_hash)
  assert a.y.nunique()==2 and b.y.nunique()==2
  rows.append({"scheme":"inner","parent_outer":0,"fold":f,"train_excel_rows":a.excel_row.tolist(),"assessment_excel_rows":b.excel_row.tolist()})
 return rows,seed
def create(line,cohort,train,test,level,parent,fold,meta):
 sid="s_"+digest({"line":line,"cohort":cohort,"train":train,"test":test})[:24]
 d=OUT/"scopes"/sid;inp=d/"inputs";inp.mkdir(parents=True,exist_ok=True)
 for name in ["FEATURE_SPECS.json","SOURCE_LOCK.json","INPUT_ACCEPTANCE.json",cohort+"_metadata.csv",cohort+"_preop_features.csv"]+([cohort+"_endop_features.csv"] if cohort=="paired" else []):
  f=inp/name
  if not f.exists():f.symlink_to(SOURCE/name)
 cv,seed=nested(meta,train,sid)
 outer={"scheme":"outer","parent_outer":None,"fold":0,"train_excel_rows":train,"assessment_excel_rows":test}
 atomic(inp/(line+"_"+cohort+"_resampling.json"),cv+[outer])
 a=meta.loc[train];b=meta.loc[test]
 assert set(a.group_hash).isdisjoint(b.group_hash)
 assert a[line+"_partition_v2"].eq("development").all() and b[line+"_partition_v2"].eq("development").all()
 r={"id":sid,"line":line,"cohort":cohort,"level":level,"parent":parent,"parent_fold":fold,"train_rows":train,"assessment_rows":test,"train_groups_sha256":digest(a.group_hash.tolist()),"assessment_groups_sha256":digest(b.group_hash.tolist()),"input":str(inp),"output":str(d/"results"),"seed_deeper_folds":seed,"inner":cv,"train_n":len(a),"train_AKI":int(a.y.sum()),"assessment_n":len(b),"assessment_AKI":int(b.y.sum())}
 atomic(d/"SCOPE.json",r);return r
def main():
 if (OUT/"SCOPE_MANIFEST.json").exists():
  print("MANIFEST_EXISTS");return
 scopes=[];roots=[]
 for line in ["geography","temporal"]:
  for cohort in ["preop","paired"]:
   meta=pd.read_csv(SOURCE/(cohort+"_metadata.csv"),dtype={"group_hash":str}).set_index("excel_row",drop=False)
   allp=json.loads((SOURCE/(line+"_"+cohort+"_resampling.json")).read_text())
   for context in ["final"]+["outer_"+str(i) for i in range(5)]:
    pp=parts(line,cohort,context);members=[]
    root={"id":line+"_"+cohort+"_"+context,"line":line,"cohort":cohort,"context":context,"children":[]}
    if context=="final":
     root["train_rows"]=sorted(set(v for p in pp for v in p["assessment_excel_rows"]));root["assessment_rows"]=[]
    else:
     op=next(p for p in allp if p["scheme"]=="outer" and p["fold"]==int(context.split("_")[1]))
     root["train_rows"]=op["train_excel_rows"];root["assessment_rows"]=op["assessment_excel_rows"]
    for p in pp:
     s=create(line,cohort,p["train_excel_rows"],p["assessment_excel_rows"],1,root["id"],p["fold"],meta)
     root["children"].append(s["id"]);members+=s["assessment_rows"];scopes.append(s)
     for p2 in s["inner"]:
      sub=create(line,cohort,p2["train_excel_rows"],p2["assessment_excel_rows"],2,s["id"],p2["fold"],meta)
      scopes.append(sub)
    assert sorted(members)==sorted(root["train_rows"]) and len(set(members))==len(members)
    roots.append(root)
 assert len(scopes)==288 and len({s["id"] for s in scopes})==288
 # Every deeper selection/refit row remains in its parent's training set.
 byid={s["id"]:s for s in scopes}
 for s in scopes:
  if s["level"]==2:
   parent=byid[s["parent"]]
   assert set(s["train_rows"])|set(s["assessment_rows"])==set(parent["train_rows"])
 atomic(OUT/"SCOPE_MANIFEST.json",{"roots":roots,"scopes":scopes,"count":288,"first_level":72,"second_level":216,"source_lock_sha256":sha(SOURCE/"SOURCE_LOCK.json"),"old_A_B_C_unchanged":True,"no_final_assessment_rows_in_any_scope":True})
 atomic(OUT/"METHOD_BOUNDARY_ADDENDUM.json",{"status":"execution_detail_not_new_clinical_design","reason":"refit upstream imputer selection, feature selection and full parameter search inside ensemble validation; never resplit a globally constructed base OOF matrix","original_outer_inner_memberships_unchanged":True,"deeper_folds":"three stratified patient-group folds, seed deterministically derived from 20261017 and scope id","parameter_search_budget":"same frozen 48 unique random plus 48 unique TPE; finite NB/KNN exhausted","ensemble_grid":{"members":[3,5,9],"Voting":"equal weights","Stacking_C":[0.01,0.1,1,10]},"automated_scope":"honest nested base bank, raw Voting/Stacking internal comparison, strict base-family calibration","not_included":"strict complete-ensemble calibration and full inner branch/family-winner crossfit, final locking/evaluation, sensitivities and downstream deliverables remain separate stages","planned_initial_storage_allowance_bytes":200*1024**3,"storage_guard":"do not admit new jobs below 60GiB free on verified /data; drain owned work"})
 print(canonical({"scopes":len(scopes),"roots":len(roots),"status":"PASS"}))
if __name__=="__main__":main()
