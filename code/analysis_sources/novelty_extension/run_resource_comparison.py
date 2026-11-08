"""Only new fixed-budget functional reductions of frozen predictions; never fit."""
from pathlib import Path
import os,sys,json,csv,hashlib,math,time,fcntl,resource,multiprocessing as mp
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,pandas as pd
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot");O=R/"outputs";A=O/"novelty_strengthening_20261101_v1";D=A/"resource_comparison"
sys.path.insert(0,str(R/"work/formal_evaluation_20261023_v1"))
import evaluation_stats as stats
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(p,j):
 p=Path(p);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)
def table(p):
 z=pd.read_csv(p,dtype={"group_hash":str,"center":str},float_precision="round_trip")
 z["record"]=z["record_key"].astype(str) if "record_key" in z else z.excel_row.astype(str)
 if "y_scr" in z:z["y"]=z.y_scr
 if "center" not in z:z["center"]="INSPIRE"
 assert z.record.is_unique
 return z
def prefix(risk,w):
 ys=risk.y[risk.order].astype(np.int64);ws=np.asarray(w,dtype=np.int64)[risk.order]
 return np.cumsum(ws),np.cumsum(ws*ys),ys
def tp_at(prefix,k):
 cn,cy,ys=prefix;k=np.asarray(k,dtype=np.int64);j=np.searchsorted(cn,k,side="right");j0=np.minimum(j,len(cn)-1)
 before=np.where(j0>0,cn[np.maximum(j0-1,0)],0);old=np.where(j0>0,cy[np.maximum(j0-1,0)],0)
 return np.where(j==len(cn),cy[-1],old+(k-before)*ys[j0])
def inverse_count(prefix,target):
 cn,cy,_=prefix;target=np.asarray(target,dtype=np.int64);j=np.searchsorted(cy,target,side="left");j=np.minimum(j,len(cy)-1)
 before=np.where(j>0,cn[np.maximum(j-1,0)],0);old=np.where(j>0,cy[np.maximum(j-1,0)],0)
 return np.where(target==0,0,before+target-old)
FIELDS=["extra_captured_per1000","base_captured_per1000","delta_captured_per1000","extra_missed_per1000","extra_false_alerts_per1000","extra_reviewed_per1000","extra_PPV","extra_sensitivity"]
def values(base,extra,w):
 n=int(np.sum(w));e=int(np.dot(w,base.y));p=np.array([5,10,15,20,25,30]);K=(p*n+99)//100;c=np.arange(31)
 available=100*K[:,None]-n*c[None,:];feasible=available>=0;ke=available//100;ke=np.maximum(ke,0)
 b=prefix(base,w);x=prefix(extra,w);bt=tp_at(b,K);xt=tp_at(x,ke)
 out=np.stack([xt*1000/n,np.broadcast_to(bt[:,None]*1000/n,xt.shape),(xt-bt[:,None])*1000/n,(e-xt)*1000/n,(ke-xt)*1000/n,ke*1000/n,np.divide(xt,ke,out=np.full(xt.shape,np.nan),where=ke>0),xt/e if e else np.full(xt.shape,np.nan)],axis=-1)
 out[~feasible]=np.nan
 minimal=inverse_count(x,bt);boundary=(K-minimal)*1000/n
 return out.reshape(-1,len(FIELDS)),boundary,{"n":n,"events":e,"K":K,"base_tp":bt,"extra_k":ke,"extra_tp":xt,"feasible":feasible,"matched_extra_reviews":minimal,"zero_reference_capture":bt==0}
def intervals(samples):
 finite=np.isfinite(samples);lo=np.full(samples.shape[1:],np.nan);hi=lo.copy();valid=finite.sum(axis=0)
 for k in np.ndindex(lo.shape):
  a=samples[(slice(None),)+k];a=a[np.isfinite(a)]
  if len(a):lo[k],hi[k]=np.quantile(a,[.025,.975],method="linear")
 return lo,hi,valid
def numeric(v):return None if not np.isfinite(v) else float(v)
def analytic_qualification():
 b=stats.FixedRisk([1,0,1,0],[.9,.8,.7,.6],["a","b","c","d"],["1","2","3","4"])
 x=stats.FixedRisk([1,0,1,0],[.8,.6,.9,.7],["a","b","c","d"],["1","2","3","4"])
 w=np.ones(4,int);out,bound,m=values(b,x,w)
 at=list([5,10,15,20,25,30]).index(25)
 assert int(m["K"][at])==1 and int(m["base_tp"][at])==1 and bound[at]==0
 k=np.arange(5);assert tp_at(prefix(x,w),k).tolist()==[0,1,2,2,2]
 assert inverse_count(prefix(x,w),[0,1,2]).tolist()==[0,1,2]
 # Literal duplicated weighted records verify exact partial multiplicity at a capacity boundary.
 w=np.array([2,0,1,1]);order=np.repeat(x.order,w[x.order]);yy=x.y[order];want=np.r_[0,np.cumsum(yy)]
 assert np.array_equal(tp_at(prefix(x,w),np.arange(5)),want)
 return {"status":"PASS_ANALYTIC_STAIRCASE_PARTIAL_MULTIPLICITY_ZERO_AND_MATCHED_TARGET_CHECK","assertions":5}
def prepare_contexts(protocol):
 contextmap={}
 for it in protocol["matrix"]:contextmap.setdefault(it["context"],[]).append(it)
 allsources={};receipts={}
 for ctx,items in contextmap.items():
  sources={}
  for it in items:
   for side in ["base","extra_information"]:
    j=it[side];p=j["prediction"];assert sha(p)==j["prediction_sha256"];sources[p]=table(p);allsources[p]=j["prediction_sha256"]
  ref=next(iter(sources.values()));keys=ref.record.tolist();yy=ref.y.to_numpy(int);gg=ref.group_hash.to_numpy(str);rr=ref.record.to_numpy(str);cc=ref.center.to_numpy(str);aligned={}
  for p,z in sources.items():
   z=z.set_index("record",drop=False).loc[keys]
   assert np.array_equal(z.y.to_numpy(int),yy) and np.array_equal(z.group_hash.to_numpy(str),gg) and np.array_equal(z.center.to_numpy(str),cc)
   for mode in ["raw","calibrated"]:aligned[p+"|"+mode]=z["prob_"+mode].to_numpy(float)
  y,g,r,c,strata,models=stats._design(yy,gg,rr,cc,aligned)
  rng=np.random.default_rng(20261017);weights=[];proof=hashlib.sha256()
  for i in range(2000):
   w=stats.group_weights(strata,len(y),rng);proof.update(w.astype("<i8").tobytes());weights.append(w.astype(np.int16))
  expected=read(O/"formal_evaluation_20261023_v1"/ctx/"EVALUATION.json")["receipt"]["resample_weight_sha256"] if ctx!="INSPIRE" else read(O/"translation_extension_20261031_v1/INSPIRE_common6_JOINT_ACCEPTANCE.json")["original_resampling_sha256_verified"]
  assert proof.hexdigest()==expected,(ctx,proof.hexdigest(),expected)
  np.savez_compressed(D/(ctx+"_BOOTSTRAP_WEIGHTS_PRIVATE.npz"),weights=np.array(weights,dtype=np.int16),groups=g,records=r,centers=c,y=y)
  # Order/cache only; no model state or patient data transmitted.
  payload={}
  for key,model in models.items():
   h=hashlib.sha256(key.encode()).hexdigest()[:20];payload[h]={"p":model.p,"order":model.order}
  predcache=D/(ctx+"_PREDICTION_CACHE_PRIVATE.npz")
  np.savez_compressed(predcache,**{h+"_"+k:v for h,z in payload.items() for k,v in z.items()})
  ids={key:hashlib.sha256(key.encode()).hexdigest()[:20] for key in models}
  receipts[ctx]={"n":len(y),"events":int(y.sum()),"groups":len(set(g)),"resampling_sha256":proof.hexdigest(),"original_stream_exact_match":True,"weight_file_sha256":sha(D/(ctx+"_BOOTSTRAP_WEIGHTS_PRIVATE.npz")),"prediction_cache_sha256":sha(predcache),"model_keys":ids}
 save(D/"CONTEXT_INPUT_AND_RESAMPLING_ACCEPTANCE.json",{"status":"PASS_ORIGINAL_PATIENT_STREAM_AND_ALIGNED_FROZEN_PREDICTIONS","contexts":receipts,"original_sources_sha256":allsources,"no_old_fit_or_metric_recomputed_as_independent_evidence":True})
 return receipts
def worker(job):
 it,ctx,mode=job;cid=it["comparison_id"];name=cid+"__"+mode
 wfile=np.load(D/(ctx+"_BOOTSTRAP_WEIGHTS_PRIVATE.npz"),allow_pickle=False)
 cache=np.load(D/(ctx+"_PREDICTION_CACHE_PRIVATE.npz"),allow_pickle=False)
 y=wfile["y"];g=wfile["groups"];r=wfile["records"];weights=wfile["weights"]
 def risk(side):
  key=it[side]["prediction"]+"|"+mode;h=hashlib.sha256(key.encode()).hexdigest()[:20]
  return stats.FixedRisk(y,cache[h+"_p"],g,r)
 base=risk("base");extra=risk("extra_information");point,boundary,meta=values(base,extra,np.ones(len(y),int))
 samples=np.full((2000,186,len(FIELDS)),np.nan);bounds=np.empty((2000,6));zero=np.empty((2000,6),bool)
 for i,w in enumerate(weights):
  samples[i],bounds[i],m=values(base,extra,w.astype(int));zero[i]=m["zero_reference_capture"]
  if i==24:save(D/(name+"_PILOT25.json"),{"status":"FIRST25_ACTUAL_REDUCTIONS_COMPLETE_AND_REUSED_IN_FINAL2000","comparison":cid,"mode":mode,"done":25,"finite_outputs":int(np.isfinite(samples[:25]).sum())})
 lo,hi,valid=intervals(samples);bl,bh,bv=intervals(bounds)
 rows=[];breaks=[]
 for qi,pct in enumerate([5,10,15,20,25,30]):
  for ci in range(31):
   k=qi*31+ci
   row={"comparison_id":cid,"context":ctx,"role":it["role"],"mode":mode,"budget_percent":pct,"extra_cost_per_person_review_equivalents":ci/100,"n":meta["n"],"events":meta["events"],"base_reviews_n":int(meta["K"][qi]),"base_captured_n":int(meta["base_tp"][qi]),"feasible_at_point":bool(meta["feasible"][qi,ci]),"extra_reviews_n":int(meta["extra_k"][qi,ci]) if meta["feasible"][qi,ci] else None,"extra_captured_n":int(meta["extra_tp"][qi,ci]) if meta["feasible"][qi,ci] else None,"requested_replicates":2000}
   for f in range(len(FIELDS)):
    row[FIELDS[f]]=numeric(point[k,f]);row[FIELDS[f]+"_lower95"]=numeric(lo[k,f]);row[FIELDS[f]+"_upper95"]=numeric(hi[k,f]);row[FIELDS[f]+"_valid_replicates"]=int(valid[k,f])
   rows.append(row)
  breaks.append({"comparison_id":cid,"context":ctx,"role":it["role"],"mode":mode,"budget_percent":pct,"n":meta["n"],"events":meta["events"],"reference_reviews_n":int(meta["K"][qi]),"reference_captured_n":int(meta["base_tp"][qi]),"minimum_extra_path_reviews_for_same_capture":int(meta["matched_extra_reviews"][qi]),"signed_allowable_extra_cost_per1000":float(boundary[qi]),"lower95":float(bl[qi]),"upper95":float(bh[qi]),"valid_replicates":int(bv[qi]),"zero_reference_capture_at_point":bool(meta["zero_reference_capture"][qi]),"zero_reference_capture_replicates":int(zero[:,qi].sum()),"interpretation":"Retrospective fixed-model equal-capture review-equivalent difference, not clinical staffing threshold; negative values retained."})
 pd.DataFrame(rows).to_csv(D/(name+"_GRID.csv"),index=False);pd.DataFrame(breaks).to_csv(D/(name+"_BOUNDARIES.csv"),index=False)
 np.savez_compressed(D/(name+"_NEW_FUNCTION_SAMPLES_PRIVATE.npz"),samples=samples,boundaries=bounds,zero_reference_capture=zero,fields=np.array(FIELDS))
 save(D/(name+"_COMPLETE.json"),{"status":"PASS_NEW_FUNCTION_COMPUTATION_ONLY","grid_rows":len(rows),"boundary_rows":len(breaks),"replicates":2000,"pilot25_included_not_restarted":True,"original_weight_stream_reused":True,"samples_sha256":sha(D/(name+"_NEW_FUNCTION_SAMPLES_PRIVATE.npz"))})
 return {"comparison_id":cid,"mode":mode,"n":meta["n"],"events":meta["events"],"grid":str(D/(name+"_GRID.csv")),"boundaries":str(D/(name+"_BOUNDARIES.csv"))}
def main():
 D.mkdir(mode=0o700,exist_ok=True)
 with (D/"RUNNER.lock").open("a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (D/"NUMERICAL_COMPUTATION_ACCEPTANCE.json").exists()
  protocol=read(A/"ANALYSIS_PROTOCOL_LOCK.json");assert sha(A/"ANALYSIS_PROTOCOL_LOCK.json")=="1afa3e481704e2ea993b2ea4bd44ba13866c13843e4218e2895a624f583edb10"
  save(D/"MATH_QUALIFICATION.json",analytic_qualification())
  contexts=prepare_contexts(protocol)
  jobs=[(it,it["context"],mode) for it in protocol["matrix"] for mode in ["raw","calibrated"]]
  start=time.time()
  with mp.Pool(processes=4) as pool:receipts=list(pool.imap(worker,jobs))
  pd.concat([pd.read_csv(it["grid"]) for it in receipts],ignore_index=True).to_csv(D/"RESOURCE_BUDGET_GRID.csv",index=False)
  pd.concat([pd.read_csv(it["boundaries"]) for it in receipts],ignore_index=True).to_csv(D/"BREAK_EVEN_BOUNDARIES.csv",index=False)
  save(D/"NUMERICAL_COMPUTATION_ACCEPTANCE.json",{"status":"NEW_BUDGET_FUNCTIONS_COMPUTED_REQUIRES_INDEPENDENT_AND_SCIENTIFIC_REVIEW","protocol_sha256":sha(A/"ANALYSIS_PROTOCOL_LOCK.json"),"contexts":len(contexts),"comparison_modes":len(jobs),"grid_rows":5208,"boundary_rows":168,"replicates":2000,"jobs":receipts,"predictive_fits":0,"new_clinical_threshold_selection":False,"old_replicate_indices_verified_and_reused":True,"elapsed_seconds":time.time()-start,"parent_peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
  print(json.dumps({"status":"NEW_RESOURCE_FUNCTIONS_COMPUTED","jobs":len(jobs),"grid_rows":5208,"boundary_rows":168}),flush=True)
if __name__=="__main__":main()
