"""First-layer ensemble predictions with all selection inside the training boundary."""
from core import *
from ensemble_math import *
import argparse
MAN=json.loads((OUT/"SCOPE_MANIFEST.json").read_text())
SCOPES={s["id"]:s for s in MAN["scopes"]}
def children(sid):return sorted([s for s in MAN["scopes"] if s["parent"]==sid],key=lambda s:s["parent_fold"])
def load_scope(s,stage,branch):
 frames={}
 for f in FAMILIES:
  d=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/f
  assert ready(d),str(d)
  frames[f]=pd.read_csv(d/"predictions.csv",dtype={"group_hash":str})
 return matrix(frames)
def oof(ss,stage,branch):
 result=pd.concat([load_scope(s,stage,branch).assign(fold=s["parent_fold"]) for s in ss],ignore_index=True)
 assert result.excel_row.is_unique
 return result
def destination(r,stage,branch):return OUT/"ensemble_results"/r["line"]/r["cohort"]/stage/branch/r["id"]
def choose(rows):
 ap=max(x["mean_AP"] for x in rows);rows=[x for x in rows if x["mean_AP"]>=ap-1e-12]
 br=min(x["mean_Brier"] for x in rows)
 return min([x for x in rows if x["mean_Brier"]<=br+1e-12],key=lambda x:x["candidate_id"])
def run(r,stage,branch):
 d=destination(r,stage,branch);d.mkdir(parents=True,exist_ok=True)
 sig={"root":r,"stage":stage,"branch":branch,"worker":sha(__file__),"math":sha(CODE/"ensemble_math.py"),"manifest":sha(OUT/"SCOPE_MANIFEST.json")}
 if ready(d,sig):return
 top=children(r["id"]);root_oof=oof(top,stage,branch)
 assert set(root_oof.excel_row)==set(r["train_rows"])
 folds=[];proof=[]
 for s in top:
  train=oof(children(s["id"]),stage,branch);valid=load_scope(s,stage,branch)
  assert set(train.excel_row)==set(s["train_rows"]) and set(valid.excel_row)==set(s["assessment_rows"])
  assert set(train.group_hash).isdisjoint(valid.group_hash)
  folds.append((train,valid))
  proof.append({"scope":s["id"],"train_groups":digest(sorted(train.group_hash)),"held_groups":digest(sorted(valid.group_hash)),"all_upstream_selection_excludes_held_rows":True})
 candidates=[];files=[]
 for kind in ["Voting","Stacking"]:
  for k in [3,5,9]:
   for C in ([None] if kind=="Voting" else [.01,.1,1.,10.]):
    scores=[];pred=[]
    for f,(tr,va) in enumerate(folds):
     p,_,members=infer(tr,va,kind,k,C)
     scores.append(score(va.y,p));q=va[["excel_row","group_hash","center","y"]].copy();q["fold"]=f;q["prob_raw"]=p;pred.append(q)
    conf={"kind":kind,"k":k,"C":C}
    entry={**conf,"candidate_id":digest(conf),"mean_AP":float(np.mean([v["AP"] for v in scores])),"mean_Brier":float(np.mean([v["Brier"] for v in scores])),"fold_metrics":scores}
    candidates.append(entry);fn="inner_"+entry["candidate_id"][:16]+".csv"
    pd.concat(pred).to_csv(d/fn,index=False);files.append(fn)
 winners={kind:choose([x for x in candidates if x["kind"]==kind]) for kind in ["Voting","Stacking"]}
 held=load_scope({"output":r["base_output"],"line":r["line"],"cohort":r["cohort"]},stage,branch)
 assert set(held.excel_row)==set(r["assessment_rows"]) and set(held.group_hash).isdisjoint(root_oof.group_hash)
 for kind,w in winners.items():
  pred,model,members=infer(root_oof,held,kind,w["k"],w["C"])
  fn=kind+"_outer.csv";q=held[["excel_row","group_hash","center","y"]].copy();q["prob_raw"]=pred;q.to_csv(d/fn,index=False);files.append(fn)
  fn=kind+"_private_meta.joblib";joblib.dump({"meta":model,"members":members,"base_output":r["base_output"]},d/fn,compress=3);files.append(fn)
  # Same runtime, reload and individual/reverse inference.
  restored=joblib.load(d/fn)
  if model is not None:
   pp=restored["meta"].predict_proba(held[members])[:,1]
   rev=restored["meta"].predict_proba(held[members].iloc[::-1])[:,1][::-1]
   solo=restored["meta"].predict_proba(held[members].iloc[:1])[:,1]
   np.testing.assert_allclose(pred,pp,atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(pred,rev,atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(pred[:1],solo,atol=1e-12,rtol=1e-12)
  w["outer_metrics"]=score(held.y,pred);w["members"]=members
 root_oof.to_csv(d/"honest_base_oof.csv",index=False);files.append("honest_base_oof.csv")
 atomic(d/"SELECTION.json",{"candidates":candidates,"winners":winners,"proof":proof});files.append("SELECTION.json")
 lineage={}
 for s in top+[sub for t in top for sub in children(t["id"])]:
  for f in FAMILIES:
   p=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/f/"COMPLETE.json";lineage[str(p)]=sha(p)
 for f in FAMILIES:
  p=Path(r["base_output"])/"outer_raw"/r["line"]/r["cohort"]/stage/branch/"outer_0"/f/"COMPLETE.json";lineage[str(p)]=sha(p)
 atomic(d/"UPSTREAM_LINEAGE.json",lineage);files.append("UPSTREAM_LINEAGE.json")
 atomic(d/"COMPLETE.json",{"signature":sig,"files":{f:sha(d/f) for f in files},"all_upstream_selection_crossfit":True,"first_layer_ensemble_predictions_only":True,"final_model_locked":False,"no_final_hospital_time_assessment_used":True})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("task");a=p.parse_args();t=json.loads(Path(a.task).read_text());run(t["root"],t["stage"],t["branch"])
