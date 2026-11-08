"""Raw nested ensembles and base-family calibration; no final-test access."""
from core import *
from ensemble_math import *
import argparse

MAN=json.loads((OUT/"SCOPE_MANIFEST.json").read_text())
SCOPES={s["id"]:s for s in MAN["scopes"]}
def children(sid):
 return sorted([s for s in MAN["scopes"] if s["parent"]==sid],key=lambda s:s["parent_fold"])
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
def destination(r,stage,branch):
 return OUT/"ensemble_results"/r["line"]/r["cohort"]/stage/branch/r["context"]
def run(r,stage,branch):
 BASE=Path(r["base_output"])
 d=destination(r,stage,branch);d.mkdir(parents=True,exist_ok=True)
 signature={"root":r,"stage":stage,"branch":branch,"worker":sha(__file__),"math":sha(CODE/"ensemble_math.py"),"source_manifest":sha(OUT/"SCOPE_MANIFEST.json")}
 if ready(d,signature):return
 top=children(r["id"]);root_oof=oof(top,stage,branch)
 assert set(root_oof.excel_row)==set(r["train_rows"])
 folds=[];proof=[]
 for s in top:
  train=oof(children(s["id"]),stage,branch);valid=load_scope(s,stage,branch)
  assert set(train.excel_row)==set(s["train_rows"])
  assert set(valid.excel_row)==set(s["assessment_rows"])
  assert set(train.group_hash).isdisjoint(valid.group_hash)
  folds.append((train,valid));proof.append({"scope":s["id"],"train_groups":digest(sorted(train.group_hash)),"held_groups":digest(sorted(valid.group_hash)),"members_ranked_on_training_only":True,"meta_training_OOF_from_deeper_refits":True})
 candidates=[]
 for kind in ["Voting","Stacking"]:
  for k in [3,5,9]:
   for C in ([None] if kind=="Voting" else [.01,.1,1.,10.]):
    scores=[];pred=[]
    for f,(tr,va) in enumerate(folds):
     p,_,members=infer(tr,va,kind,k,C)
     scores.append(score(va.y,p));q=va[["excel_row","group_hash","center","y"]].copy();q["fold"]=f;q["prob_raw"]=p;pred.append(q)
    conf={"kind":kind,"k":k,"C":C}
    entry={**conf,"candidate_id":digest(conf),"mean_AP":float(np.mean([v["AP"] for v in scores])),"mean_Brier":float(np.mean([v["Brier"] for v in scores])),"fold_metrics":scores}
    candidates.append(entry);pd.concat(pred).to_csv(d/("inner_"+entry["candidate_id"][:16]+".csv"),index=False)
 winners={}
 for kind in ["Voting","Stacking"]:
  rows=[c for c in candidates if c["kind"]==kind]
  ap=max(x["mean_AP"] for x in rows);rows=[x for x in rows if x["mean_AP"]>=ap-1e-12]
  br=min(x["mean_Brier"] for x in rows);w=min([x for x in rows if x["mean_Brier"]<=br+1e-12],key=lambda x:x["candidate_id"])
  winners[kind]=w
 files=[]
 if r["context"]!="final":
  basepaths={f:BASE/"outer_raw"/r["line"]/r["cohort"]/stage/branch/r["context"]/f for f in FAMILIES}
  valid=matrix({f:pd.read_csv(basepaths[f]/"predictions.csv",dtype={"group_hash":str}) for f in FAMILIES})
  assert set(valid.excel_row)==set(r["assessment_rows"])
  assert set(root_oof.group_hash).isdisjoint(valid.group_hash)
  for kind,w in winners.items():
   pred,model,members=infer(root_oof,valid,kind,w["k"],w["C"])
   fn=kind+"_outer.csv";q=valid[["excel_row","group_hash","center","y"]].copy();q["prob_raw"]=pred;q.to_csv(d/fn,index=False);files.append(fn)
   bundle={"kind":kind,"members":members,"meta":model,"base_bundles":{f:{"path":str(basepaths[f]/"raw_bundle.joblib"),"sha256":sha(basepaths[f]/"raw_bundle.joblib")} for f in members},"calibrator":None,"raw_internal_only":True}
   joblib.dump(bundle,d/(kind+"_private_bundle.joblib"),compress=3);files.append(kind+"_private_bundle.joblib")
   if model is not None:
    restored=joblib.load(d/(kind+"_private_bundle.joblib"));np.testing.assert_allclose(pred,restored["meta"].predict_proba(valid[members])[:,1],atol=1e-12,rtol=1e-12)
   winners[kind]["outer_metrics"]=score(valid.y,pred);winners[kind]["members"]=members
  # All nine candidates: independently rebuilt upstream selection for each calibration OOF row.
  for f in FAMILIES:
   tr=root_oof[["excel_row","group_hash","y",f]].rename(columns={f:"prob_raw"})
   model=calibrator(tr);pr=apply_calibration(model,valid[f])
   name=f+"_calibrated_outer.csv";q=valid[["excel_row","group_hash","center","y"]].copy();q["prob_raw"]=valid[f];q["prob_calibrated"]=pr;q.to_csv(d/name,index=False);files.append(name)
   joblib.dump(model,d/(f+"_calibrator.joblib"),compress=3);files.append(f+"_calibrator.joblib")
 root_oof.to_csv(d/"honest_base_oof.csv",index=False);files.append("honest_base_oof.csv")
 atomic(d/"SELECTION.json",{"candidates":candidates,"winners":winners,"proof":proof,"member_ranking":"mean fold AP, mean fold Brier, canonical family id"})
 files+=["SELECTION.json"]
 files += [x.name for x in d.glob("inner_*.csv")]
 lineage={}
 for s in top+[sub for t in top for sub in children(t["id"])]:
  for f in FAMILIES:
   rec=Path(s["output"])/"outer_raw"/s["line"]/s["cohort"]/stage/branch/"outer_0"/f/"COMPLETE.json"
   lineage[str(rec)]=sha(rec)
 atomic(d/"UPSTREAM_LINEAGE.json",lineage);files.append("UPSTREAM_LINEAGE.json")
 atomic(d/"COMPLETE.json",{"signature":signature,"files":{f:sha(d/f) for f in files},"raw_Voting_Stacking_internal_evaluation_complete":r["context"]!="final","strict_base_family_calibration_complete":r["context"]!="final","strict_ensemble_calibration_complete":False,"final_model_locked":False,"no_final_hospital_time_assessment_used":True})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("task");a=p.parse_args();t=json.loads(Path(a.task).read_text());run(t["root"],t["stage"],t["branch"])
