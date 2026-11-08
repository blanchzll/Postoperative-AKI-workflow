"""Independent MIMIC v2 numerical audit; no original evaluation imports or predictive fitting."""
from pathlib import Path
import os,json,hashlib,time,math,resource,warnings,fcntl
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot");A=R/"outputs/novelty_strengthening_20261101_v1";D=A/"mimic_calendar_v2"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 t=Path(str(p)+".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n");os.replace(t,p)
checks=0;maxerr={};calfail=0
def check(a,b,name,tol=1e-9):
 global checks
 if a is None or (isinstance(a,(float,np.floating)) and not np.isfinite(a)):assert pd.isna(b),(name,a,b);checks+=1;return
 assert pd.notna(b),(name,a,b)
 d=abs(float(a)-float(b));maxerr[name]=max(maxerr.get(name,0),d);assert d<=tol,(name,a,b,d,tol);checks+=1
def ci(v):
 v=sorted(x for x in v if x is not None and np.isfinite(x))
 def quant(q):
  if not v:return None
  k=(len(v)-1)*q;lo=math.floor(k);hi=math.ceil(k);return v[lo]+(k-lo)*(v[hi]-v[lo])
 return quant(.025),quant(.975),len(v)
def counts(y,a):
 n=len(y);e=int(y.sum());k=int(a.sum());tp=int(y[a].sum());fp=k-tp
 return {"reviewed_n":k,"captured_n":tp,"missed_n":e-tp,"false_alerts_n":fp,"sensitivity":tp/e if e else None,"PPV":tp/k if k else None,"specificity":1-fp/(n-e) if n>e else None,"captured_per1000":1000*tp/n,"missed_per1000":1000*(e-tp)/n,"reviewed_per1000":1000*k/n,"false_alerts_per1000":1000*fp/n}
def metrics(y,p,order):
 global calfail
 n=len(y);e=int(y.sum());z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
 s={"event_rate":e/n,"AUROC":roc_auc_score(y,p) if 0<e<n else None,"AP":average_precision_score(y,p) if e else None,"Brier":float(np.mean((p-y)**2)),"observed_expected":e/float(p.sum()) if p.sum()>0 else None}
 s.update(calibration_intercept=None,calibration_slope=None,calibration_joint_intercept=None,upgraded_per1000=None)
 if 0<e<n:
  with warnings.catch_warnings():
   warnings.simplefilter("error")
   fit=sm.GLM(y,np.ones((n,1)),offset=z,family=sm.families.Binomial()).fit(tol=1e-11,maxiter=100)
   s["calibration_intercept"]=float(fit.params[0])
   if np.std(z)>=1e-10 and not (min(z[y==1])>=max(z[y==0]) or min(z[y==0])>=max(z[y==1])):
    fit=sm.GLM(y,np.column_stack([np.ones(n),z]),family=sm.families.Binomial()).fit(tol=1e-11,maxiter=100)
    h=np.asarray(fit.cov_params())
    if np.isfinite(h).all() and np.linalg.cond(h)<1e10:
     s["calibration_joint_intercept"],s["calibration_slope"]=map(float,fit.params)
    else:calfail+=1
 for f in [.1,.2]:
  mask=np.zeros(n,dtype=bool);mask[order[:math.ceil(n*f)]]=True
  s.update({f"capacity{int(f*100)}_"+k:v for k,v in counts(y,mask).items()})
 for t in [.01,.02,.03,.05,.075,.1,.15,.2]:
  pos=p>=t;s["net_benefit_"+str(t)]=float(((y*pos).sum()-(pos*(1-y)).sum()*t/(1-t))/n)
 return s
def main():
 start=time.time()
 with (D/"INDEPENDENT_NUMERICAL_RUN.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  comp=read(D/"COMPUTATION_ACCEPTANCE.json")
  for n,h in comp["files"].items():assert sha(D/n)==h
  inp=np.load(D/"BOOTSTRAP_INPUT_PRIVATE.npz",allow_pickle=False);y=inp["y"].astype(int);g=inp["groups"].astype(str);r=inp["records"].astype(str);c=inp["centers"].astype(str);ws=inp["weights"].astype(int);names=list(inp["model_names"].astype(str))
  assert len(y)==759 and y.sum()==59 and len(set(g))==759 and ws.shape==(2000,759)
  proof=hashlib.sha256();rng=np.random.default_rng(20261017)
  for w in ws:
   expected=np.zeros(len(y),int)
   for cc in sorted(set(c)):
    gs=sorted(set(g[c==cc]));draw=rng.integers(0,len(gs),len(gs));ct=np.bincount(draw,minlength=len(gs))
    for gg,k in zip(gs,ct):expected[g==gg]=k
   np.testing.assert_array_equal(expected,w);proof.update(w.astype("<i8").tobytes())
  assert proof.hexdigest()==comp["resample_weight_sha256"]
  meta=pd.read_csv(D/"PRIVATE_METADATA.csv",dtype={"record_key":str,"group_hash":str},float_precision="round_trip").set_index("record_key")
  np.testing.assert_array_equal(meta.loc[r].y_scr_calendar.to_numpy(),y)
  for line in ["geography","temporal"]:
   new=pd.read_csv(D/line/"PRIVATE_PREDICTIONS.csv",dtype={"record_key":str,"group_hash":str},float_precision="round_trip").set_index("record_key").loc[r]
   old=pd.read_csv(A/"mimic_calendar"/line/"PRIVATE_PREDICTIONS.csv",dtype={"record_key":str,"group_hash":str},float_precision="round_trip").set_index("record_key").loc[r]
   for mode,col in [("raw","prob_raw"),("calibrated","prob_calibrated")]:
    np.testing.assert_array_equal(new[col],old[col]);np.testing.assert_array_equal(new[col],inp[line+"__"+mode])
  point=pd.read_csv(D/"POINT_ESTIMATES.csv").set_index("model")
  intervals=pd.read_csv(D/"INTERVALS.csv");inverse=pd.read_csv(D/"INVERSE_COVERAGE.csv");threshold=pd.read_csv(D/"SOURCE_THRESHOLD_BURDEN.csv")
  values={n:{f:[] for f in intervals.loc[intervals.model==n,"metric"]} for n in names}
  ivals={n:{.5:[],.8:[]} for n in names};cuts={}
  for (line,rule),block in threshold.groupby(["source_line","rule"]):
   t=float(block.threshold.iloc[0]);cuts[(line,rule)]=(t,{f:[] for f in block.metric})
  for rep,w in [(-1,np.ones(len(y),int))]+list(enumerate(ws)):
   idx=np.repeat(np.arange(len(y)),w);yy=y[idx]
   for n in names:
    p=inp[n][idx]
    order=np.array(sorted(range(len(idx)),key=lambda j:(-p[j],g[idx[j]],r[idx[j]],j)))
    mm=metrics(yy,p,order)
    if rep<0:
     for f,v in mm.items():check(v,point.loc[n,f],f,3e-5 if f.startswith("calibration") else 1e-9)
    else:
     for f,v in mm.items():values[n][f].append(v)
    for target in [.5,.8]:
     need=math.ceil(int(yy.sum())*target)
     v=float((np.flatnonzero(np.cumsum(yy[order])>=need)[0]+1)*1000/len(yy)) if need else None
     if rep<0:
      row=inverse[(inverse.model==n)&(inverse.sensitivity_target==target)].iloc[0];check(v,row.required_reviews_per1000,"inverse")
     else:ivals[n][target].append(v);check(v,inp["inverse_samples"][rep,names.index(n),[.5,.8].index(target)],"inverse_sample")
   for key,(t,cols) in cuts.items():
    line,rule=key;m=counts(yy,inp[line+"__calibrated"][idx]>=t)
    for f,v in m.items():
     if rep<0:
      row=threshold[(threshold.source_line==line)&(threshold.rule==rule)&(threshold.metric==f)].iloc[0];check(v,row.estimate,"threshold_"+f)
     else:cols[f].append(v)
   if rep>=0 and (rep+1)%200==0:save(D/"INDEPENDENT_NUMERICAL_PROGRESS.json",{"done":rep+1,"total":2000,"pid":os.getpid()})
  for row in intervals.itertuples():
   lo,hi,k=ci(values[row.model][row.metric]);tol=3e-5 if row.metric.startswith("calibration") else 1e-9
   check(lo,row.lower95,"CI_"+row.metric,tol);check(hi,row.upper95,"CI_"+row.metric,tol);check(k,row.valid_replicates,"valid_replicates",0)
  for row in inverse.itertuples():
   lo,hi,k=ci(ivals[row.model][row.sensitivity_target]);check(lo,row.lower95,"inverse_CI");check(hi,row.upper95,"inverse_CI");check(k,row.valid_replicates,"inverse_valid",0)
  for row in threshold.itertuples():
   lo,hi,k=ci(cuts[(row.source_line,row.rule)][1][row.metric]);check(lo,row.lower95,"threshold_CI");check(hi,row.upper95,"threshold_CI");check(k,row.valid_replicates,"threshold_valid",0)
  save(D/"INDEPENDENT_NUMERICAL_VERIFICATION.json",{"status":"PASS_INDEPENDENT_ALL2000_PATIENT_RESAMPLES_ALL_METRICS_INTERVALS_SOURCE_CUTS_INVERSE_COVERAGE","cases":759,"events":59,"models":names,"replicates_each":2000,"point_metric_rows":len(point)*len(intervals.metric.unique()),"interval_rows":len(intervals),"source_threshold_rows":len(threshold),"inverse_rows":len(inverse),"assertions":checks,"max_abs_error_by_metric":maxerr,"calibration_implementation":"Independent statsmodels Binomial IRLS offset and joint regression on every original and all2000resamples; original evaluator uses scipy brentq/BFGS. Diagnostics only; no risk update.","calibration_absolute_tolerance":3e-5,"other_absolute_tolerance":1e-9,"undefined_calibration_conditioned_covariance":calfail,"resample_weight_sha256":proof.hexdigest(),"probabilities_unchanged_from_original_candidate":True,"no_original_evaluation_code_imported":True,"files_verified":comp["files"],"verifier_sha256":sha(__file__),"elapsed_seconds":time.time()-start,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
  print(json.dumps({"status":"MIMIC_V2_INDEPENDENT_NUMERICAL_PASS","assertions":checks,"seconds":time.time()-start}),flush=True)
if __name__=="__main__":main()
