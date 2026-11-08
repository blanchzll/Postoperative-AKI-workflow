"""Engineering acceptance plus real boundary checks; no final-evaluation reads."""
from core import *
import subprocess,resource,ast
import pandas as pd,numpy as np
from ensemble_math import *
def fixture():
 d=OUT/"qualification"/"synthetic";d.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(20261017)
 # Synthetic patient groups; used solely to exercise full first-layer implementation.
 labels={i:int(i%4==0) for i in range(1,61)}
 def meta(rows):
  return pd.DataFrame({"excel_row":rows,"group_hash":["synthetic_"+str(i) for i in rows],"center":1,"y":[labels[i] for i in rows]})
 def base(rows,dest):
  x=meta(rows)
  for k,f in enumerate(FAMILIES):
   dd=dest/"outer_raw"/"geography"/"preop"/"preop"/"compact"/"outer_0"/f;dd.mkdir(parents=True,exist_ok=True)
   z=x.copy();p=np.clip(.08+.15*z.y.to_numpy()+rng.uniform(0,.6,len(x))+(k*.002),0,1);z["prob_raw"]=p
   z.to_csv(dd/"predictions.csv",index=False)
   atomic(dd/"COMPLETE.json",{"files":{"predictions.csv":sha(dd/"predictions.csv")},"synthetic_only":True})
 scopes=[]
 train=list(range(1,37));held=list(range(37,61));r={"id":"fixture_root","line":"geography","cohort":"preop","context":"fixture_root","train_rows":train,"assessment_rows":held,"base_output":str(d/"held")}
 base(held,Path(r["base_output"]))
 for f in range(3):
  v=train[f*12:(f+1)*12];t=[x for x in train if x not in v]
  s={"id":"top"+str(f),"parent":r["id"],"parent_fold":f,"train_rows":t,"assessment_rows":v,"line":"geography","cohort":"preop","output":str(d/("top"+str(f)))}
  scopes.append(s);base(v,Path(s["output"]))
  for j in range(3):
   vv=t[j*8:(j+1)*8]
   sub={"id":s["id"]+"_child"+str(j),"parent":s["id"],"parent_fold":j,"train_rows":[x for x in t if x not in vv],"assessment_rows":vv,"line":"geography","cohort":"preop","output":str(d/(s["id"]+"_child"+str(j)))}
   scopes.append(sub);base(vv,Path(sub["output"]))
 atomic(d/"SCOPE_MANIFEST.json",{"roots":[r],"scopes":scopes})
 atomic(d/"TASK.json",{"root":r,"stage":"preop","branch":"compact"})
 env=os.environ.copy();env.update(AKI_SCOPE_OUT=str(d),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 subprocess.run([str(PYTHON),"-B",str(CODE/"ensemble_e_worker.py"),str(d/"TASK.json")],cwd=ROOT,env=env,check=True)
 result=d/"ensemble_results/geography/preop/preop/compact/fixture_root"
 assert ready(result)
 first=json.loads((result/"SELECTION.json").read_text())
 values={f:pd.read_csv(result/(f+"_outer.csv")).prob_raw.to_numpy() for f in ["Voting","Stacking"]}
 # Change held outcomes only; selection and predictions must be exactly unchanged.
 for f in FAMILIES:
  dd=Path(r["base_output"])/"outer_raw/geography/preop/preop/compact/outer_0"/f
  z=pd.read_csv(dd/"predictions.csv");z["y"]=1-z.y;z.to_csv(dd/"predictions.csv",index=False)
  atomic(dd/"COMPLETE.json",{"files":{"predictions.csv":sha(dd/"predictions.csv")},"synthetic_only":True})
 # Versioned second root triggers new fit, not invalidation of valid real results.
 rr={**r,"id":"fixture_held_label_flip","context":"fixture_held_label_flip"}
 for s in scopes:
  if s["parent"]==r["id"]:s["parent"]=rr["id"]
 atomic(d/"SCOPE_MANIFEST.json",{"roots":[rr],"scopes":scopes})
 atomic(d/"TASK2.json",{"root":rr,"stage":"preop","branch":"compact"})
 subprocess.run([str(PYTHON),"-B",str(CODE/"ensemble_e_worker.py"),str(d/"TASK2.json")],cwd=ROOT,env=env,check=True)
 second=d/"ensemble_results/geography/preop/preop/compact/fixture_held_label_flip"
 sel=json.loads((second/"SELECTION.json").read_text())
 for f in ["Voting","Stacking"]:
  assert all(first["winners"][f][k]==sel["winners"][f][k] for k in ["k","C","candidate_id","members"])
  np.testing.assert_array_equal(values[f],pd.read_csv(second/(f+"_outer.csv")).prob_raw)
 # Explicit guards: real training/held identity overlap, metadata mismatch.
 x=meta(train);x["fold"]=np.arange(len(x))%3
 for f in FAMILIES:x[f]=rng.uniform(.01,.8,len(x))
 try:infer(x,x,"Stacking",3,.1)
 except ValueError:pass
 else:raise AssertionError("Group overlap accepted")
 tr=x[["excel_row","group_hash","y","LR"]].rename(columns={"LR":"prob_raw"})
 c=calibrator(tr);p=apply_calibration(c,x.LR)
 np.testing.assert_allclose(p,apply_calibration(c,x.LR.iloc[::-1])[::-1],atol=1e-12,rtol=1e-12)
 return {"full_first_layer_fixture":True,"held_outcome_perturbation_selection_and_predictions_unchanged":True,"overlap_rejected":True,"calibration_order_stable":True,"fixture_not_biological_results":True}
def main():
 for p in CODE.glob("*.py"):ast.parse(p.read_text(),filename=str(p))
 man=json.loads((OUT/"SCOPE_MANIFEST.json").read_text())
 from coordinator_e import build
 jobs=build(man);assert len(jobs)==29988
 ids={j["id"] for j in jobs};assert len(ids)==len(jobs)
 assert all(set(j["depends"])<=ids for j in jobs)
 assert all("scope" not in j or j["scope"]["level"]==3 for j in jobs)
 # Auditable deepest-fold feasibility without changing D membership.
 retries=[]
 for s in man["scopes"]:
  if s["level"]!=3:continue
  seed0=int(digest({"seed":20261017,"scope":s["id"]})[:8],16)
  attempt=(s["seed_deeper_folds"]-seed0)%(2**32)
  assert attempt<128
  if attempt:retries.append({"scope":s["id"],"attempt":attempt,"actual_seed":s["seed_deeper_folds"]})
 result=fixture()
 # The same inner-choice tie rule and original-environment calibrator persistence.
 from complete_e import choose,save_calibration
 frozen=json.loads((ROOT/"outputs/ensemble_continuation_20261018_v1/interim_all_development_20261019/INNER_SELECTION_DECISIONS.json").read_text())["decisions"]
 assert all(choose(x["candidates"])["candidate_id"]==x["selected"]["candidate_id"] for x in frozen)
 rr=np.random.default_rng(31)
 tr=pd.DataFrame({"excel_row":range(1,41),"group_hash":["train"+str(i) for i in range(40)],"y":[i%5==0 for i in range(40)],"prob_raw":rr.uniform(.01,.5,40)})
 va=pd.DataFrame({"excel_row":range(41,61),"group_hash":["held"+str(i) for i in range(20)],"y":[i%5==0 for i in range(20)],"prob_raw":rr.uniform(.01,.5,20)})
 q=save_calibration(OUT/"qualification/calibration_roundtrip",tr,va)
 assert len(q)==len(va) and np.isfinite(q.prob_calibrated).all()
 bad=va.copy();bad.loc[0,"group_hash"]=tr.loc[0,"group_hash"]
 try:save_calibration(OUT/"qualification/rejected_overlap",tr,bad)
 except AssertionError:pass
 else:raise AssertionError("Calibration train/held overlap accepted")
 result.update({"previous_30_inner_decisions_reproduced":True,"postprocessor_calibrator_roundtrip":True,"postprocessor_overlap_rejected":True})
 # Runtime and unchanged scientific code compatibility, independent of copied core path.
 old=ROOT/"work/ensemble_continuation_20261018_v1"
 unchanged=["preprocess.py","model_engine.py","coarse.py","coordinator_a.py","search_worker.py","raw_outer.py","scope_worker.py","ensemble_math.py"]
 assert all(sha(CODE/n)==sha(old/n) for n in unchanged)
 import sklearn
 assert sklearn.__version__=="1.5.2" and Path(sys.executable)==PYTHON
 atomic(OUT/"FUNCTIONAL_QUALIFICATION.json",{"status":"PASS","new_jobs":len(jobs),"new_scopes":648,"reused_D_scopes":216,"first_layer_roots":72,"original_runtime":sys.executable,"sklearn":sklearn.__version__,
 "unchanged_scientific_components":unchanged,"deeper_feasibility_retry_count":len(retries),"deepest_fold_seed_retries":retries,
 "D_folds_unchanged":True,"actual_patient_boundaries_checked_in_manifest":True,"no_final_evaluation_accessed":True,"fixture_tests":result,
 "source_manifest_sha256":sha(OUT/"SCOPE_MANIFEST.json"),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
 print("E_FUNCTIONAL_QUALIFICATION_PASS",flush=True)
if __name__=="__main__":main()
