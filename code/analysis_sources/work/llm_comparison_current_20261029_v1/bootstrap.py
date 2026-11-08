"""Finite CPU bootstrap after the closed 48-fit LLM benchmark. No training."""
from pathlib import Path
import argparse,concurrent.futures as cf,multiprocessing as mp,json,hashlib,os,resource,time,datetime,subprocess
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import comparison as c
R=c.R;O=c.O;GIB=1024**3
def check_code_lock():
 lock=c.read(Path(__file__).parent/"STATISTICS_CODE_LOCK.json")
 for p,h in lock["files"].items():assert c.sha(p)==h,p
 q=c.read(O/"ENGINEERING_STATISTICS_QUALIFICATION.json");assert q["code_sha256"]==c.sha(Path(c.__file__))
def prepared(line):
 d=O/line;a=c.read(d/"PREPARATION_ACCEPTANCE.json")
 for name,h in a["input_files_sha256"].items():assert c.sha(d/name)==h
 with np.load(d/"PRIVATE_FIXED_EVALUATION_INPUTS.npz",allow_pickle=False) as z:
  y=z["y"];g=z["group_hash"];r=z["records"];centers=z["centers"];weights=z["weights"]
  names=sorted(set(z.files)-{"y","group_hash","records","centers","weights"});models={n:c.FixedRisk(y,z[n],g,r) for n in names}
 assert len(models)==30 and weights.shape==(2000,len(y))
 proof=hashlib.sha256()
 for w in weights:proof.update(w.astype("<i8").tobytes())
 assert proof.hexdigest()==a["group_weights_sha256"]
 return a,models,weights,c.read(d/"SOURCE_THRESHOLDS.json"),c.read(d/"COMPARISONS.json")
def chunk_signature(line,start,stop,source_sha):
 return hashlib.sha256(json.dumps({"line":line,"replicates":[start,stop],"prepared_receipt_sha256":source_sha,"comparison_code_sha256":c.sha(Path(c.__file__)),"bootstrap_code_sha256":c.sha(__file__)},sort_keys=True).encode()).hexdigest()
def chunk_worker(task):
 line,start,stop=task;d=O/line;receipt=d/"bootstrap_chunks"/f"{start:04d}_{stop:04d}.json";out=receipt.with_suffix(".npz")
 source_sha=c.sha(d/"PREPARATION_ACCEPTANCE.json");sig=chunk_signature(line,start,stop,source_sha)
 if receipt.exists():
  a=c.read(receipt);assert a["signature"]==sig and c.sha(out)==a["sha256"];return a
 t=time.monotonic();a,models,weights,cuts,pairs=prepared(line);names=sorted(models);points={n:c.score(f,cuts=cuts.get(n)) for n,f in models.items()};fields=sorted(set().union(*(v.keys() for v in points.values())))
 values=np.full((stop-start,len(names),len(fields)),np.nan,dtype=float)
 with threadpool_limits(limits=1):
  for i,rep in enumerate(range(start,stop)):
   for j,name in enumerate(names):
    q=c.score(models[name],weights[rep],cuts.get(name))
    for k,field in enumerate(fields):
     v=q.get(field)
     if v is not None and np.isfinite(v):values[i,j,k]=float(v)
 out.parent.mkdir(mode=0o700,parents=True,exist_ok=True);tmp=out.with_suffix(".tmp")
 with tmp.open("wb") as f:np.savez_compressed(f,values=values,models=np.array(names,dtype=str),fields=np.array(fields,dtype=str),replicate=np.arange(start,stop))
 os.replace(tmp,out)
 rec={"status":"PASS_FIXED_PREDICTION_BOOTSTRAP_CHUNK","line":line,"start":start,"stop":stop,"signature":sig,"sha256":c.sha(out),"prepared_receipt_sha256":source_sha,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"seconds":time.monotonic()-t,"retraining":False}
 c.save(receipt,rec);return rec
def numeric(v):return v is not None and np.isfinite(v)
def contrast(point,terms,field):
 v=[point[n].get(field) for n in terms]
 return float(sum(point[n][field]*w for n,w in terms.items())) if all(numeric(x) for x in v) else None
def aggregate(line):
 d=O/line;a,models,weights,cuts,pairs=prepared(line);names=sorted(models);point={n:c.score(f,cuts=cuts.get(n)) for n,f in models.items()};fields=sorted(set().union(*(v.keys() for v in point.values())));mat=[];receipts={}
 for start in range(0,2000,25):
  stop=min(2000,start+25);rec=d/"bootstrap_chunks"/f"{start:04d}_{stop:04d}.json";r=c.read(rec)
  assert r["signature"]==chunk_signature(line,start,stop,c.sha(d/"PREPARATION_ACCEPTANCE.json"));p=rec.with_suffix(".npz");assert c.sha(p)==r["sha256"]
  with np.load(p,allow_pickle=False) as z:
   assert z["models"].tolist()==names and z["fields"].tolist()==fields and z["replicate"].tolist()==list(range(start,stop));mat.append(z["values"].copy())
  receipts[str(rec)]=c.sha(rec)
 v=np.concatenate(mat);assert v.shape==(2000,len(names),len(fields))
 def interval(samples):
  valid=samples[np.isfinite(samples)];ci=np.quantile(valid,[.025,.975],method="linear").tolist() if len(valid) else [None,None]
  return {"lower95":ci[0],"upper95":ci[1],"valid_replicates":len(valid),"requested_replicates":2000}
 model_intervals=[];paired=[]
 for j,name in enumerate(names):
  for k,field in enumerate(fields):
   model_intervals.append({"model":name,"metric":field,"estimate":point[name].get(field),**interval(v[:,j,k])})
 for name,terms in pairs.items():
  for k,field in enumerate(fields):
   samples=sum(v[:,names.index(n),k]*w for n,w in terms.items())
   paired.append({"comparison":name,"metric":field,"estimate":contrast(point,terms,field),**interval(samples)})
 # Independent sklearn point checks; no selection or updating from heldout labels.
 checks=0;worst=0.
 from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
 for name,f in models.items():
  for field,value in {"AUROC":roc_auc_score(f.y,f.p),"AP":average_precision_score(f.y,f.p),"Brier":brier_score_loss(f.y,f.p)}.items():
   delta=abs(point[name][field]-value);assert delta<1e-12;worst=max(worst,delta);checks+=1
 # The saved comparator probabilities must replay previously accepted main/TabPFN point metrics.
 original=c.read(R/"outputs/formal_evaluation_20261023_v1"/(line+"_paired")/"EVALUATION.json")
 tab=c.read(R/"outputs/tabpfn_benchmark_20261025_v1/statistics"/(line+"_paired")/"RESULTS.json")
 lock=c.read(R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json")
 for stage in ["preop","endop"]:
  ref=next(x for x in lock["models"] if x["primary_route_winner"] and (x["identity"]["line"],x["identity"]["cohort"],x["identity"]["stage"],x["identity"]["branch"])==(line,"paired",stage,"full"));i=ref["identity"];key="_".join(i[k] for k in ["stage","branch","role","family"])
  for field in ["AUROC","AP","Brier","capacity10_captured_per1000","capacity20_captured_per1000"]:
   assert abs(point["primary_"+stage+"_cal"][field]-original["point"][key][field])<1e-12
   assert abs(point["tabpfn_"+stage][field]-tab["point"]["tabpfn_"+stage+"_full"][field])<1e-12
 result={"status":"COMPLETE_NUMERICAL_LLM_COMPARISON_SCIENTIFIC_REVIEW_REQUIRED","time":c.now(),"line":line,"point":point,"intervals":model_intervals,"paired_differences":paired,"receipt":{"prepared_receipt_sha256":c.sha(d/"PREPARATION_ACCEPTANCE.json"),"bootstrap_chunks":receipts,"replicates":2000,"seed":20261017,"resample_weight_sha256":a["group_weights_sha256"],"hospital_stratified_patient_unit":True,"fixed_models_calibration_and_thresholds":True,"calibration_diagnostic_refit_does_not_update_predictions":True,"point_independent_checks":checks,"point_max_abs_discrepancy":worst,"existing_main_and_TabPFN_replay_pass":True,"pointwise_exploratory_CI_no_multiplicity_adjustment":True,"not_new_blind_validation":True,"final_scientific_review_complete":False}}
 c.save(d/"RESULTS.json",result);pd.DataFrame([{"model":n,**q} for n,q in point.items()]).to_csv(d/"POINT_ESTIMATES.csv",index=False)
 pd.DataFrame(model_intervals).to_csv(d/"INTERVALS.csv",index=False);pd.DataFrame(paired).to_csv(d/"PAIRED_DIFFERENCES.csv",index=False)
 c.save(d/"BOOTSTRAP_ACCEPTANCE.json",{"status":"PASS_ALL2000_FIXED_PREDICTION_PAIRED_BOOTSTRAPS_AND_EXISTING_COMPARATOR_REPLAY","files":{n:c.sha(d/n) for n in ["RESULTS.json","POINT_ESTIMATES.csv","INTERVALS.csv","PAIRED_DIFFERENCES.csv"]},"chunk_count":80,"clinical_retraining":False,"final_scientific_review_complete":False})
 return {"line":line,"receipt":str(d/"BOOTSTRAP_ACCEPTANCE.json"),"sha256":c.sha(d/"BOOTSTRAP_ACCEPTANCE.json")}
def admission(measured_peak):
 mem=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024;osreserve=max(4*GIB,mem//4)
 own=(Path("/sys/fs/cgroup")/Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/"));effective=mem
 for cg in [own,*own.parents]:
  if cg==Path("/sys/fs/cgroup"):break
  if (cg/"memory.max").exists():
   x=(cg/"memory.max").read_text().strip()
   if x!="max":effective=min(effective,max(0,int(x)-int((cg/"memory.current").read_text())))
 worker_budget=max(256*1024**2,int(measured_peak*1.5));parent_growth=512*1024**2
 slots=int(max(0,min(effective,mem-osreserve)-parent_growth)//worker_budget)
 workers=min(len(os.sched_getaffinity(0)),slots)
 assert workers>=1,"Insufficient actual RAM/cgroup after OS/parent growth"
 return {"MemAvailable":mem,"effective_cgroup_available":effective,"OS_reserve":osreserve,"parent_growth_reserve":parent_growth,"measured_worker_peak":measured_peak,"worker_budget":worker_budget,"workers":workers,"CPU_affinity_count":len(os.sched_getaffinity(0))}
def run():
 c.require_closed_formal();check_code_lock()
 existing=[(O/line/"PREPARATION_ACCEPTANCE.json").exists() for line in ["geography","temporal"]]
 if any(existing) and not all(existing):raise RuntimeError("Partial preparation needs evidence-preserving scoped continuation; never overwrite accepted source calibrators blindly")
 if all(existing):
  for line in ["geography","temporal"]:
   a=c.read(O/line/"PREPARATION_ACCEPTANCE.json");assert a["code_sha256"]==c.sha(Path(c.__file__))
   for f,h in a["input_files_sha256"].items():assert c.sha(O/line/f)==h
   for f,h in a["sources"].items():assert c.sha(f)==h
 else:c.prepare()
 tasks=[(line,start,start+25) for line in ["geography","temporal"] for start in range(0,2000,25)]
 pilots=[chunk_worker(tasks[0]),chunk_worker(tasks[80])];peak=max(x["peak_RSS_bytes"] for x in pilots);a=admission(peak);c.save(O/"BOOTSTRAP_RESOURCE_ADMISSION.json",a)
 remaining=[x for x in tasks if x not in [tasks[0],tasks[80]]]
 with cf.ProcessPoolExecutor(max_workers=a["workers"],mp_context=mp.get_context("spawn"),max_tasks_per_child=1) as pool:
  future={pool.submit(chunk_worker,t):t for t in remaining}
  completed=2
  for f in cf.as_completed(future):
   f.result();completed+=1;c.save(O/"BOOTSTRAP_PROGRESS.json",{"status":"RUNNING_FIXED_STATISTICS","completed":completed,"total":160,"time":c.now(),"formal_training_modified":False})
 receipts=[aggregate(line) for line in ["geography","temporal"]]
 c.save(O/"NUMERICAL_COMPARISON_ACCEPTANCE.json",{"status":"PASS_TWO_PAIRED_CONTEXTS24_CALIBRATED_LLM_ROUTES_RAW_DIAGNOSTICS_FIXED_MAIN_TABPFN_COMPARATORS","time":c.now(),"receipts":receipts,"formal_training_acceptance_sha256":c.sha(c.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"),"statistics_protocol_sha256":c.sha(O/"STATISTICS_PROTOCOL.json"),"code_lock_sha256":c.sha(Path(__file__).parent/"STATISTICS_CODE_LOCK.json"),"LLM_result_routes":24,"bootstrap_replicates_per_context":2000,"numerical_computation_only":True,"resource_reporting_interpretation_and_independent_recalculation_still_required":True,"COMPARISON_ACCEPTANCE_not_claimed":True,"final_scientific_review_complete":False})
 print("NUMERICAL_COMPARISON_COMPLETE_REVIEW_REQUIRED",flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--run",action="store_true");a=ap.parse_args()
 if not a.run:raise SystemExit("Explicit --run required after all48 fits and6 banks complete; no waiting resident process.")
 with threadpool_limits(limits=1):run()
