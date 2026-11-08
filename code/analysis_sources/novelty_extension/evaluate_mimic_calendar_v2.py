"""Evaluate original locked common6 RFs on qualified calendar-anchor MIMIC sensitivity."""
from pathlib import Path
import os,sys,json,hashlib,fcntl,time,resource
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot");O=R/"outputs";A=O/"novelty_strengthening_20261101_v1";D=A/"mimic_calendar_v2"
sys.path.insert(0,str(R/"work/formal_evaluation_20261023_v1"))
import evaluation_stats as stats
sys.path.insert(0,str(R/"work/common6_model_lock_20261024_v1"))
import runtime
sys.path.insert(0,str(R/"work/common6_public_evaluation_20261024_v1"))
from evaluate_public import threshold_intervals
SIX=["Age","PreopCr","PreopBUN","PreopAlb","PreopHb","Sex"]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(p,j):
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)
def inverse(model,target,w):
 n=int(w.sum());e=int(w@model.y)
 if not e:return None
 ys=model.y[model.order];ws=w[model.order];ct=np.cumsum(ws*ys);j=int(np.searchsorted(ct,math.ceil(e*target),side="left"))
 return float((ws[:j].sum()+math.ceil(e*target)-(ct[j-1] if j else 0))*1000/n)
import math

def main():
 with (D/"EVALUATION_RUNNER.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (D/"COMPUTATION_ACCEPTANCE.json").exists()
  iq=read(D/"INPUT_QUALIFICATION.json")
  vr=read(D/"INDEPENDENT_INPUT_ENDPOINT_REVIEW.json")
  assert vr["status"]=="PASS_INDEPENDENT_EXACT_RATIONAL_INPUT_AND_ENDPOINT_REPLAY" and vr["disagreement_records"]==0
  assert iq["events"]==59 and iq["n"]==759
  for n,h in iq["files"].items():assert sha(D/n)==h
  original=A/"mimic_calendar"
  assert sha(D/"PRIVATE_COMMON6_FEATURES.csv")==sha(original/"PRIVATE_COMMON6_FEATURES.csv")
  x=pd.read_csv(D/"PRIVATE_COMMON6_FEATURES.csv",dtype={"record_key":str,"group_hash":str,"Sex":"string"},float_precision="round_trip")
  meta=pd.read_csv(D/"PRIVATE_METADATA.csv",dtype={"record_key":str,"group_hash":str},float_precision="round_trip")
  x=x.set_index("record_key",drop=False).sort_index();meta=meta.set_index("record_key",drop=False).loc[x.index]
  y=meta.y_scr_calendar.to_numpy(int);g=x.group_hash.astype(str).to_numpy();r=x.record_key.astype(str).to_numpy();c=["MIMIC_BIDMC"]*len(y)
  rc=read(O/"common6_model_lock_20261024_v1/STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json");models={}
  for it in rc["models"]:
   assert sha(it["path"])==it["sha256"];m=read(it["path"]);models[m["identity"]["line"]]=m
  save(D/"PRE_SCORE_LOCK.json",{"status":"CORRECTED_TARGET_REEVALUATION_LOCK_NO_INFERENCE_REPEATED","protocol_sha256":sha(A/"ANALYSIS_PROTOCOL_LOCK.json"),"input_qualification_sha256":sha(D/"INPUT_QUALIFICATION.json"),"independent_rational_verification_sha256":sha(D/"INDEPENDENT_INPUT_ENDPOINT_REVIEW.json"),"remediation_sha256":sha(A/"DECIMAL_ENDPOINT_REMEDIATION.json"),"original_pre_score_lock_sha256":sha(original/"PRE_SCORE_LOCK.json"),"original_models":rc["models"],"code_sha256":sha(__file__),"n":len(y),"events":int(y.sum()),"features_unchanged":True,"no_new_model_calls":True,"clinical_endpoint_ratio_threshold_unchanged":True})
  pred={}
  for line in ["geography","temporal"]:
   src=original/line/"PRIVATE_PREDICTIONS.csv";oldrec=read(original/line/"PREDICTION_ACCEPTANCE.json");assert sha(src)==oldrec["prediction_sha256"]
   z=pd.read_csv(src,dtype={"record_key":str,"group_hash":str},float_precision="round_trip").set_index("record_key",drop=False).loc[r]
   assert z.group_hash.tolist()==list(g)
   raw=z.prob_raw.to_numpy();cal=z.prob_calibrated.to_numpy()
   dd=D/line;dd.mkdir(mode=0o700)
   new=pd.DataFrame({"record_key":r,"group_hash":g,"y_scr_calendar":y,"prob_raw":raw,"prob_calibrated":cal});new.to_csv(dd/"PRIVATE_PREDICTIONS.csv",index=False)
   reload=pd.read_csv(dd/"PRIVATE_PREDICTIONS.csv",float_precision="round_trip")
   np.testing.assert_array_equal(reload.prob_raw.to_numpy(),raw);np.testing.assert_array_equal(reload.prob_calibrated.to_numpy(),cal)
   save(dd/"PREDICTION_ACCEPTANCE.json",{"status":"PASS_IDENTICAL_SOURCE_PROBABILITIES_REUSED_TARGET_DECIMAL_CORRECTION_ONLY","original_prediction_sha256":sha(src),"prediction_sha256":sha(dd/"PRIVATE_PREDICTIONS.csv"),"raw_probability_bytes_sha256":hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest(),"calibrated_probability_bytes_sha256":hashlib.sha256(cal.astype("<f8").tobytes()).hexdigest(),"point_predictions_bitwise_unchanged":True,"original_label_independent_probe_receipt_sha256":sha(original/line/"PREDICTION_ACCEPTANCE.json"),"no_predictive_fit_or_rescoring":True})
   pred[line+"__raw"]=raw;pred[line+"__calibrated"]=cal
  def progress(k):save(D/"EVALUATION_PROGRESS.json",{"phase":"fixed_model_patient_bootstrap","done":k,"total":2000,"pid":os.getpid()})
  start=time.time();result=stats.evaluate(y,pred,g,r,c,{},[],replicates=2000,seed=20261017,progress=progress)
  save(D/"EVALUATION.json",result)
  pd.DataFrame([{"model":k,**v} for k,v in result["point"].items()]).to_csv(D/"POINT_ESTIMATES.csv",index=False);pd.DataFrame(result["intervals"]).to_csv(D/"INTERVALS.csv",index=False)
  burden,proof=threshold_intervals(y,pred,g,r,c,{k:v["training_thresholds"] for k,v in models.items()})
  assert proof==result["receipt"]["resample_weight_sha256"];pd.DataFrame(burden).to_csv(D/"SOURCE_THRESHOLD_BURDEN.csv",index=False)
  yy,gg,rr,cc,strata,fixed=stats._design(y,g,r,c,pred);names=list(fixed);rng=np.random.default_rng(20261017);proof=hashlib.sha256();weights=[];samples=np.full((2000,len(names),2),np.nan)
  for i in range(2000):
   w=stats.group_weights(strata,len(yy),rng);weights.append(w.astype(np.int16));proof.update(w.astype("<i8").tobytes())
   for j,n in enumerate(names):
    for k,t in enumerate([.5,.8]):v=inverse(fixed[n],t,w);samples[i,j,k]=np.nan if v is None else v
  assert proof.hexdigest()==result["receipt"]["resample_weight_sha256"]
  inv=[];curves=[]
  for j,n in enumerate(names):
   cum=np.r_[0,np.cumsum(fixed[n].y[fixed[n].order])].astype(int)
   curves.extend({"model":n,"reviewed_n":k,"captured_n":int(t),"n":len(yy),"events":int(yy.sum())} for k,t in enumerate(cum))
   for k,t in enumerate([.5,.8]):
    v=samples[:,j,k];v=v[np.isfinite(v)];ci=np.quantile(v,[.025,.975]).tolist() if len(v) else [None,None]
    inv.append({"model":n,"sensitivity_target":t,"required_reviews_per1000":inverse(fixed[n],t,np.ones(len(yy),int)),"lower95":ci[0],"upper95":ci[1],"valid_replicates":len(v),"requested_replicates":2000,"retrospective_not_source_threshold":True})
  pd.DataFrame(inv).to_csv(D/"INVERSE_COVERAGE.csv",index=False);pd.DataFrame(curves).to_csv(D/"PRIVATE_DENSE_CAPTURE_CURVES.csv",index=False)
  # Persist exact same patient stream for independent verification without fitting again.
  np.savez_compressed(D/"BOOTSTRAP_INPUT_PRIVATE.npz",weights=np.array(weights,dtype=np.int16),y=yy,groups=gg,records=rr,centers=cc,**{name:fixed[name].p for name in names},inverse_samples=samples,model_names=np.array(names))
  save(D/"COMPUTATION_ACCEPTANCE.json",{"status":"COMPUTED_NEW_INDEPENDENT_SOURCE_CALENDAR_SENSITIVITY_REQUIRES_INDEPENDENT_AND_SCIENTIFIC_REVIEW","n":len(y),"events":int(y.sum()),"models":4,"replicates":2000,"resample_weight_sha256":proof.hexdigest(),"no_predictive_fit_or_target_update":True,"calibration_diagnostics_only_not_deployment_update":True,"exact_incision_end7dayvalidation_claimed":False,"elapsed_seconds":time.time()-start,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"files":{n:sha(D/n) for n in ["PRE_SCORE_LOCK.json","EVALUATION.json","POINT_ESTIMATES.csv","INTERVALS.csv","SOURCE_THRESHOLD_BURDEN.csv","INVERSE_COVERAGE.csv","BOOTSTRAP_INPUT_PRIVATE.npz"]}})
  print(json.dumps({"status":"MIMIC_CALENDAR_COMPUTATION_FINISHED","n":len(y),"events":int(y.sum()),"views":4,"replicates":2000}),flush=True)

if __name__=="__main__":main()
