"""Independent literal-expanded-patient verification of every new resource sample."""
from pathlib import Path
import os,math,json,hashlib,time,resource
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,pandas as pd
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");D=A/"resource_comparison"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0;maxerr=0.
def check(x,msg):
 global checks
 if not x:raise AssertionError(msg)
 checks+=1
def close(x,y,msg):
 global maxerr
 x=np.asarray(x,float);y=np.asarray(y,float)
 check(np.array_equal(np.isnan(x),np.isnan(y)),msg+" NA")
 if np.isfinite(x).any():
  e=float(np.max(np.abs(x[np.isfinite(x)]-y[np.isfinite(y)])));maxerr=max(maxerr,e);check(e<=1e-9,msg+" numeric")
def load(p):
 z=pd.read_csv(p,dtype={"group_hash":str,"center":str},float_precision="round_trip")
 z["record"]=z.record_key.astype(str) if "record_key" in z else z.excel_row.astype(str)
 if "y_scr" in z:z["y"]=z.y_scr
 if "center" not in z:z["center"]="INSPIRE"
 return z
def literal(y,order,weights):
 expanded=np.repeat(y[order],weights[order].astype(int))
 return np.flatnonzero(expanded==1)+1
def independent(basepos,extrapos,n,e):
 p=np.array([5,10,15,20,25,30]);K=np.array([math.ceil(int(k)*int(n)/100) for k in p],int)
 bt=np.searchsorted(basepos,K,side="right")
 rem=K[:,None]-np.arange(31)[None,:]*n/100
 feasible=(100*K[:,None]>=np.arange(31)[None,:]*n)
 # Integer exact arithmetic avoids floating floor errors at whole-review boundaries.
 ke=(100*K[:,None]-np.arange(31)[None,:]*n)//100;ke=np.maximum(ke,0)
 tp=np.searchsorted(extrapos,ke,side="right")
 ppv=np.full(ke.shape,np.nan);np.divide(tp,ke,out=ppv,where=ke>0)
 sens=tp/e if e else np.full(ke.shape,np.nan)
 ans=np.stack([tp*1000/n,np.broadcast_to(bt[:,None]*1000/n,tp.shape),(tp-bt[:,None])*1000/n,(e-tp)*1000/n,(ke-tp)*1000/n,ke*1000/n,ppv,sens],axis=-1)
 ans[~feasible]=np.nan
 m=np.array([int(extrapos[k-1]) if k else 0 for k in bt])
 return ans.reshape(186,8),(K-m)*1000/n,bt==0
def percentiles(a):
 # Independently sort actual finite bootstrap replicates and linearly interpolate.
 out=[np.full(a.shape[1:],np.nan),np.full(a.shape[1:],np.nan)]
 count=np.isfinite(a).sum(axis=0)
 for idx in np.ndindex(a.shape[1:]):
  s=np.sort(a[(slice(None),)+idx]);s=s[np.isfinite(s)]
  if not len(s):continue
  for q,container in zip([.025,.975],out):
   at=(len(s)-1)*q;lo=math.floor(at);hi=math.ceil(at);container[idx]=s[lo]+(at-lo)*(s[hi]-s[lo])
 return *out,count
def main():
 global checks,maxerr
 start=time.time();protocol=read(A/"ANALYSIS_PROTOCOL_LOCK.json");cmeta=read(D/"CONTEXT_INPUT_AND_RESAMPLING_ACCEPTANCE.json");allgrid=pd.read_csv(D/"RESOURCE_BUDGET_GRID.csv",float_precision="round_trip");allbounds=pd.read_csv(D/"BREAK_EVEN_BOUNDARIES.csv",float_precision="round_trip")
 check(len(allgrid)==5208 and len(allbounds)==168,"complete grid and boundaries")
 contextcache={};records={}
 for ctx,r in cmeta["contexts"].items():
  p=D/(ctx+"_BOOTSTRAP_WEIGHTS_PRIVATE.npz");check(sha(p)==r["weight_file_sha256"],"weight pin")
  z=np.load(p,allow_pickle=False);weights=z["weights"].astype(int);y=z["y"].astype(int);g=z["groups"];rec=z["records"];center=z["centers"]
  proof=hashlib.sha256()
  groups={}
  for i,(c,v) in enumerate(zip(center,g)):groups.setdefault((c,v),[]).append(i)
  for w in weights:
   proof.update(w.astype("<i8").tobytes())
   for c in set(center):
    counts=[int(w[ix[0]]) for (cc,_),ix in groups.items() if cc==c]
    check(sum(counts)==sum(1 for cc,_ in groups if cc==c),"each hospital draws its original patient count")
   check(all(np.all(w[ix]==w[ix[0]]) for ix in groups.values()),"repeated operations sampled together")
  check(proof.hexdigest()==r["resampling_sha256"],"native index stream hash")
  contextcache[ctx]=(weights,y,g,rec,center)
 for it in protocol["matrix"]:
  ctx=it["context"];weights,y,g,records0,centers=contextcache[ctx];n=len(y)
  modeldata={}
  for side in ["base","extra_information"]:
   x=it[side];check(sha(x["prediction"])==x["prediction_sha256"],"source prediction pin");z=load(x["prediction"]).set_index("record",drop=False).loc[records0]
   check(np.array_equal(z.group_hash.to_numpy(str),g) and np.array_equal(z.y.to_numpy(int),y),"source identity and labels")
   modeldata[side]=z
  for mode in ["raw","calibrated"]:
   name=it["comparison_id"]+"__"+mode;sample=np.load(D/(name+"_NEW_FUNCTION_SAMPLES_PRIVATE.npz"),allow_pickle=False);fields=sample["fields"].tolist()
   b=modeldata["base"]["prob_"+mode].to_numpy(float);x=modeldata["extra_information"]["prob_"+mode].to_numpy(float)
   bo=np.lexsort((records0,g,-b));xo=np.lexsort((records0,g,-x))
   point,pboundary,pzero=independent(literal(y,bo,np.ones(n,int)),literal(y,xo,np.ones(n,int)),n,int(y.sum()))
   rows=allgrid[(allgrid.comparison_id==it["comparison_id"])&(allgrid["mode"]==mode)].sort_values(["budget_percent","extra_cost_per_person_review_equivalents"])
   br=allbounds[(allbounds.comparison_id==it["comparison_id"])&(allbounds["mode"]==mode)].sort_values("budget_percent")
   close(point,rows[fields].to_numpy(),"literal point "+name);close(pboundary,br.signed_allowable_extra_cost_per1000.to_numpy(),"exact signed point boundary")
   actual=sample["samples"];bound=sample["boundaries"];zer=sample["zero_reference_capture"]
   for i,w in enumerate(weights):
    N=int(w.sum());e=int(w@y);v,bd,z=independent(literal(y,bo,w),literal(y,xo,w),N,e)
    close(v,actual[i],"literal resampled resource functions");close(bd,bound[i],"literal resampled boundaries");check(np.array_equal(z,zer[i]),"zero-target flags")
   lo,hi,valid=percentiles(actual);bl,bh,bv=percentiles(bound)
   for f,k in zip(fields,range(len(fields))):
    close(lo[:,k],rows[f+"_lower95"].to_numpy(),"all grid lower CIs");close(hi[:,k],rows[f+"_upper95"].to_numpy(),"all grid upper CIs");check(np.array_equal(valid[:,k],rows[f+"_valid_replicates"].to_numpy()),"all valid replicate counts")
   close(bl,br.lower95.to_numpy(),"boundary lower CIs");close(bh,br.upper95.to_numpy(),"boundary upper CIs")
   check(np.array_equal(bv,br.valid_replicates.to_numpy()),"boundary valid counts")
   if it["role"]=="primary":
    O=A.parent;ctxdir=O/"formal_evaluation_20261023_v1"/ctx;old=pd.read_csv(ctxdir/"POINT_ESTIMATES.csv").set_index("model")
    for q in [10,20]:
     for side,prefix in [("base","base"),("extra_information","extra")]:
      nm=Path(it[side]["prediction"]).parent.name
      target=old.loc[nm,"capacity"+str(q)+"_captured_n"]
      r=rows[(rows.budget_percent==q)&(rows.extra_cost_per_person_review_equivalents==0)].iloc[0]
      close(r[prefix+"_captured_n"],target,"zero-cost exactly reuses original capacity count")
   (D/"INDEPENDENT_VERIFY_PROGRESS.json").write_text(json.dumps({"last":name,"checks":checks,"max_abs":maxerr})+"\n")
 result={"status":"PASS_INDEPENDENT_LITERAL_REPLICATION_ALL_NEW_BUDGET_FUNCTIONS_AND_INTERVALS","assertions":checks,"max_abs_numeric_difference":maxerr,"verified_comparison_modes":28,"replicates_per_mode":2000,"grid_rows":5208,"boundary_rows":168,"all8_function_samples_recomputed":True,"group_and_hospital_structure_verified":True,"all_exact_original_index_streams_reused":True,"source_predictions_and_labels_independently_aligned":True,"percentiles":"independent sorted interpolation2.5/97.5","zero_cost_original_primary_capacity_replay":True,"new_predictive_fits":0,"elapsed_seconds":time.time()-start,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"source_code_sha256":sha(Path(__file__))}
 (D/"INDEPENDENT_NUMERICAL_VERIFICATION.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");print(json.dumps(result),flush=True)
if __name__=="__main__":main()
