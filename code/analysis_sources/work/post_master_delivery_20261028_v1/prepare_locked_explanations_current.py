"""Prepare the original locked risk-curve/SHAP methods on current corrected-source model locks.
Only method contracts are inherited. No old predictions, explanation values or acceptances are imported.
"""
from pathlib import Path
import hashlib,json,sys,subprocess,datetime,os
R=Path(__file__).resolve().parents[2];P=R.parents[1]
STAGES=["locked_risk_curves_20261024_v1","locked_pipeline_shap_20261024_v1"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def exact(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert read(p)==d,("Inspect changed prepared artifact",str(p))
 else:p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def map_dependency(value):
 value=str(value)
 if value.startswith(str(P)+"/"):return Path(value.replace(str(P)+"/",str(R)+"/",1))
 old="/data/codex-workspaces/aki-prediction-manuscript/"
 if value.startswith(old):
  relative=value[len(old):]
  if relative.startswith("outputs/"):relative=relative[len("outputs/"):]
  return R/"outputs"/relative
 raise ValueError("Unexpected historical dependency namespace: "+value)
def main():
 source=R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"
 assert sha(source)=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
 accpath=R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json"
 acc=read(accpath);assert acc["status"]=="PASS"
 routes={}
 for row in acc["models"]:
  if not row["primary_route_winner"]:continue
  assert sha(row["path"])==row["sha256"]
  m=read(row["path"]);assert m["primary_route_winner"]
  t=m["identity"];key="_".join(t[k] for k in ["line","cohort","stage","branch"])
  assert key not in routes
  routes[key]=row
 expected=read(P/"outputs/locked_risk_curves_20261024_v1/ROUTES.json")
 assert len(routes)==10 and set(routes)==set(expected)
 mapping=[]
 for stage in STAGES:
  out=R/"outputs"/stage;out.mkdir(parents=True,exist_ok=True)
  exact(out/"ROUTES.json",dict(sorted(routes.items())))
  contract=read(P/"outputs"/stage/"CONTRACT.json")
  exact(out/"CONTRACT.json",contract)
  exact(out/"CODE_LOCK.json",{p.name:sha(p) for p in sorted((R/"work"/stage).glob("*.py"))})
  deps={}
  for old,digest in read(P/"outputs"/stage/"DEPENDENCIES.json").items():
   new=map_dependency(old);assert new.is_file(),str(new)
   assert (new==source or str(new).startswith(str(R)+"/")),str(new)
   deps[str(new)]=sha(new)
   mapping.append({"stage":stage,"historical_dependency":old,"current_dependency":str(new),"current_sha256":deps[str(new)],"historical_hash_not_used_as_current_result":True})
  deps[str(accpath)]=sha(accpath);deps[str(source)]=sha(source)
  exact(out/"DEPENDENCIES.json",deps)
 sys.path.insert(0,str(R/"work/locked_risk_curves_20261024_v1"))
 import curves
 import numpy as np
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=1):
  a=np.array([.1,.2,.3]);b=np.array([.2,.3,.4]);x=curves.curve_stats(a,b)
  assert abs(x["mean_raw_risk"]-.2)<1e-14 and abs(x["mean_calibrated_risk"]-.3)<1e-14
  assert abs(x["reference_risk_median"]-.3)<1e-14
  assert abs(x["reference_risk_p10"]-.22)<1e-14 and abs(x["reference_risk_p90"]-.38)<1e-14
  try:curves.curve_stats(np.array([np.nan]),np.array([.2]))
  except AssertionError:pass
  else:raise AssertionError("Nonfinite curve values must fail")
  checked={}
  for route in routes:
   m,sp,meta,features,ref,c=curves.inputs(route)
   t=m["identity"]
   assert meta[t["line"]+"_partition_v2"].eq("development").all()
   assert len(meta)==m["training_n"] and ref.group_hash.nunique()==256
   assert set(ref.index)<=set(meta.index)
   assert list(features)==m["input_columns"]
   keys=curves.rt.keys(ref);assert len(keys)==len(set(keys))
   if t["branch"]=="compact":assert len(m["input_columns"])==9
   checked[route]={"development_rows":len(meta),"reference_rows":len(ref),"reference_groups":int(ref.group_hash.nunique()),"inputs":len(m["input_columns"]),"features":[f for f in c["clinical_features"] if f in sp["num"]],"model_lock_sha256":sha(read(R/"outputs/locked_risk_curves_20261024_v1/ROUTES.json")[route]["path"])}
  curves.base.gate()
  out=R/"outputs/locked_risk_curves_20261024_v1"
  exact(out/"QUALIFICATION.json",{"status":"PASS_preparation_and_synthetic_curve_math","routes":checked,"current_source_sha256":sha(source),"actual_curve_mean_and_quantiles_checked":True,"nonfinite_rejected":True,"all_reference_patients_from_current_development":True,"actual_pipeline_replay_pending_clinical_pilot":True,"old_fitted_state_or_predictions_imported":False,"method_contract_sha256":sha(out/"CONTRACT.json"),"time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()})
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
 subprocess.run([str(R/"envs/prediction_gpu/bin/python"),"-B",str(R/"work/locked_pipeline_shap_20261024_v1/qualify.py")],cwd=R,env=env,check=True)
 target=R/"outputs/post_master_delivery_20261028_v1/EXPLANATION_PREPARATION.json"
 exact(target,{"status":"PASS_CURRENT_SOURCE_ORIGINAL_RISK_CURVE_AND_SHAP_METHOD_PREPARATION_ONLY","source_sha256":sha(source),"current_model_lock_acceptance_sha256":sha(accpath),"route_count":len(routes),"method_contracts_unchanged":True,"dependency_mapping":mapping,"actual_full_model_clinical_pilots_still_required":True,"science_review_performed":False})
 print("PASS_CURRENT_SOURCE_EXPLANATION_PREPARATION_NOT_RESULTS",flush=True)
if __name__=="__main__":main()
