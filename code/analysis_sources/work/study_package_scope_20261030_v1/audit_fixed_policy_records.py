"""Bounded scalar audit of current private policy records. No model load or fit."""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, resource, time
from datetime import datetime
from zoneinfo import ZoneInfo

R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
BASE=R/"outputs/study_package_scope_20261030_v1"
INPUT=BASE/"private_evaluation_dependencies_v1"
OUT=BASE/"private_policy_scalar_review_v1"
SOURCE=R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"
PINS={
 SOURCE:"afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3",
 INPUT/"PRIVATE_PARTITION_REPLAY_CONTRACTS.json":"9e2c9db65417da8879abd0e6ac7b8b2571e6fcfcb9de465a121873d9a426ab14",
 INPUT/"PRIVATE_MAIN_EVALUATION_DEPENDENCIES.json":"19d2ad47e1c97f08bfe2b5daba88d931cd85e3389c2c065c2d5c187f25198efb",
 INPUT/"UNBOUND_POLICY_RISK_FILES_REQUIRING_REPLAY.json":"87f6d58b63d0103fd74927f99cfaa4ccfc7adc4c569a92adc8051a8302e6a56b"
}
TOL=1e-12

def now(): return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def save(p,x):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def rel(p): return str(Path(p).relative_to(R))
def path(value):
 p=Path(value)
 if not p.is_absolute(): p=R/p
 if p.is_relative_to(R): return p
 physical=(R/"outputs").resolve()
 if p.is_relative_to(physical): return R/"outputs"/p.relative_to(physical)
 raise ValueError("Out-of-scope dependency")
def read_csv(p):
 with Path(p).open(newline="") as f:
  rd=csv.DictReader(f); rows=list(rd)
  return rd.fieldnames,rows
def protection():
 files=read(R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json")["files"]
 assert len(files)==92 and all(sha(p)==v["sha256"] for p,v in files.items())
 return {p:v["sha256"] for p,v in files.items()}
def identities(rows):
 ids=[(r["group_hash"],r["excel_row"]) for r in rows]
 assert len(ids)==len(set(ids))
 return ids
def probabilities(rows,column):
 p=[float(r[column]) for r in rows]
 assert all(math.isfinite(x) and 0<=x<=1 for x in p)
 return p

def scalar_metrics(y,p,ids):
 n=len(y); e=sum(y); neg=n-e
 assert n and n==len(p)==len(ids)
 assert all(v in (0,1) for v in y) and len(set(ids))==n
 grouped={}
 for yi,pi in zip(y,p):
  pos,count=grouped.get(pi,(0,0)); grouped[pi]=(pos+yi,count+1)
 rank=0; positive_ranks=0.0
 for pi in sorted(grouped):
  pos,count=grouped[pi]
  positive_ranks+=pos*(rank+(count+1)/2)
  rank+=count
 auc=(positive_ranks-e*(e+1)/2)/(e*neg) if e and neg else None
 seen=tp=0; ap_terms=[]
 for pi in sorted(grouped,reverse=True):
  pos,count=grouped[pi];seen+=count;tp+=pos
  if e: ap_terms.append((pos/e)*(tp/seen))
 expected=math.fsum(p)
 out={"n":n,"events":e,"event_rate":e/n,"AUROC":auc,
      "AP":math.fsum(ap_terms) if e else None,
      "Brier":math.fsum((yi-pi)**2 for yi,pi in zip(y,p))/n,
      "observed_expected":e/expected if expected else None}
 order=sorted(range(n),key=lambda k:(-p[k],ids[k]))
 for frac in [.1,.2]:
  k=math.ceil(frac*n);caught=sum(y[i] for i in order[:k]);fp=k-caught
  values={"reviewed_n":k,"captured_n":caught,"missed_n":e-caught,"false_alerts_n":fp,
   "sensitivity":caught/e if e else None,"PPV":caught/k if k else None,
   "specificity":1-fp/neg if neg else None,
   "captured_per1000":1000*caught/n,"missed_per1000":1000*(e-caught)/n,
   "reviewed_per1000":1000*k/n,"false_alerts_per1000":1000*fp/n}
  out.update({f"capacity{int(100*frac)}_{key}":value for key,value in values.items()})
 for threshold in [.01,.02,.03,.05,.075,.1,.15,.2]:
  caught=sum(yi for yi,pi in zip(y,p) if pi>=threshold)
  fp=sum(1-yi for yi,pi in zip(y,p) if pi>=threshold)
  out[f"net_benefit_{threshold}"]=caught/n-fp/n*threshold/(1-threshold)
 return out

def selective_allocation(p,ids,gate,fraction,mode):
 assert mode in ("capacity","individual") and 0<=fraction<=1
 clip=float(gate["clip"]); threshold=float(gate["threshold"])
 assert 0<clip<.5 and 0<=threshold<=1
 def logit(value):
  z=min(1-clip,max(clip,value));return math.log(z)-math.log1p(-z)
 anchor=logit(threshold);distance=[abs(logit(v)-anchor) for v in p]
 if fraction in (0,1):return [bool(fraction)]*len(p)
 if mode=="individual":
  cutoff=float(gate["cutoffs"][str(fraction)])
  return [v<=cutoff for v in distance]
 chosen=set(sorted(range(len(p)),key=lambda i:(distance[i],ids[i]))[:math.ceil(len(p)*fraction)])
 return [i in chosen for i in range(len(p))]

def qualification():
 y=[1,0,1,0];p=[.9,.7,.7,.1];ids=[("g3","r1"),("g2","r1"),("g1","r1"),("g4","r1")]
 v=scalar_metrics(y,p,ids)
 assert abs(v["AUROC"]-.875)<1e-15 and abs(v["AP"]-5/6)<1e-15
 assert abs(v["Brier"]-.15)<1e-15
 assert v["capacity20_reviewed_n"]==1 and v["capacity20_captured_n"]==1
 assert scalar_metrics([0,0],[.2,.2],[("a","1"),("b","1")])["AUROC"] is None
 assert scalar_metrics([1,1],[.2,.2],[("a","1"),("b","1")])["capacity10_specificity"] is None
 assert scalar_metrics([0,0],[.2,.2],[("a","1"),("b","1")])["AP"] is None
 ii=[3,0,2,1];vp=scalar_metrics([y[i] for i in ii],[p[i] for i in ii],[ids[i] for i in ii])
 assert all((vp[k] is None and value is None) or (vp[k] is not None and value is not None and abs(vp[k]-value)<=1e-15) for k,value in v.items())
 gate={"clip":1e-6,"threshold":.5,"cutoffs":{"0.25":0,"0.5":0}}
 assert selective_allocation([.5,.5,.1,.9],ids,gate,.25,"capacity")==[False,True,False,False]
 assert selective_allocation([.5,.5,.1,.9],ids,gate,.5,"individual")==[True,True,False,False]
 assert selective_allocation(p,ids,gate,0,"capacity")==[False]*4
 assert selective_allocation(p,ids,gate,1,"capacity")==[True]*4
 return {"status":"PASS_manually_derived_scalar_operator_cases","scope":"Ties, one-class outcomes, permutation, ceiling capacity and selective allocation; no clinical performance evidence"}

def run():
 start=time.monotonic()
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
 resource.setrlimit(resource.RLIMIT_CPU,(180,180))
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[-1]})
 qual=qualification()
 for p,h in PINS.items():assert sha(p)==h,str(p)
 protected=protection()
 catalog=read(INPUT/"PRIVATE_MAIN_EVALUATION_DEPENDENCIES.json")
 snapshot={item["path"]:item["sha256"] for item in catalog}
 assert len(snapshot)==199
 for name,h in snapshot.items():assert sha(path(name))==h,name
 contracts=read(INPUT/"PRIVATE_PARTITION_REPLAY_CONTRACTS.json")
 unbound=read(INPUT/"UNBOUND_POLICY_RISK_FILES_REQUIRING_REPLAY.json")
 for row in unbound:
  assert sha(path(row["path"]))==row["sha256"]
  snapshot[row["path"]]=row["sha256"]
 assert len(unbound)==24 and len(contracts)==4
 OUT.mkdir(mode=0o700,parents=True,exist_ok=False)
 save(OUT/"OPERATOR_QUALIFICATION.json",qual)
 evidence=[];max_difference=0.;cells=0;vectors=0;selective_count=0;policy_count=0;raw_cells=0
 def compare(part,name,y,p,ids,truth,upgraded=None):
  nonlocal max_difference,cells,vectors
  result=scalar_metrics(y,p,ids)
  if upgraded is not None:result["upgraded_per1000"]=1000*sum(upgraded)/len(y)
  maximum=0.;different=[]
  for metric,value in result.items():
   target=truth[metric];cells+=1
   if value is None or target is None:
    if value is not None or target is not None: different.append(metric)
   else:
    error=abs(value-target);maximum=max(maximum,error)
    if error>TOL:different.append(metric)
  assert not different,(part,name,different)
  max_difference=max(max_difference,maximum);vectors+=1
  return {"partition":part,"route":name,"n":len(y),"events":sum(y),
    "metric_cells":len(result),"maximum_absolute_metric_difference":maximum}
 for contract in contracts:
  part=contract["partition"];dest=R/"outputs/formal_evaluation_20261023_v1"/part
  targets=read(dest/"EVALUATION.json")["point"]
  banks={};headers=None;canonical=None;meta_keys=None
  for item in contract["banks"]:
   header,rows=read_csv(path(item["file"]))
   keys=[k for k in header if k not in ("prob_raw","prob_calibrated")]
   values=[tuple(row[k] for k in keys) for row in rows]
   if canonical is None: canonical=values;meta_keys=keys;headers=header
   else: assert meta_keys==keys and values==canonical
   ids=identities(rows);y=[int(row["y"]) for row in rows]
   assert len(rows)==contract["n"] and sum(y)==contract["events"]
   risks=probabilities(rows,"prob_calibrated")
   banks[item["model_lock"]]={"rows":rows,"ids":ids,"y":y,"risk":risks}
   evidence.append(compare(part,item["route"],y,risks,ids,targets[item["route"]]))
  for item in contract["policies"]:
   policy_count+=1;lock=read(path(item["lock"]));identity=lock["identity"]
   base=banks[rel(path(lock["base_model"]["path"]))];upgrade=banks[rel(path(lock["upgrade_model"]["path"]))]
   name=item["name"];header,rows=read_csv(dest/(name+"_PRIVATE.csv"))
   assert set(header)==set(meta_keys+["prob_raw_mixture","prob_calibrated","upgraded"])
   assert [tuple(row[k] for k in meta_keys) for row in rows]==canonical
   assert all(row["upgraded"] in ("True","False") for row in rows)
   stored_mask=[row["upgraded"]=="True" for row in rows]
   mask=stored_mask
   if identity["kind"]=="selective":
    mask=selective_allocation(base["risk"],base["ids"],lock["gate"],identity["upgrade_fraction"],identity["mode"])
    assert mask==stored_mask,(part,name,"SELECTIVE_MASK_DISAGREEMENT")
    selective_count+=1
   else:
    assert identity["kind"]=="random" and identity["mode"]=="capacity"
    assert sum(stored_mask)==math.ceil(len(rows)*identity["upgrade_fraction"])
   raw=[u if choose else b for b,u,choose in zip(base["risk"],upgrade["risk"],mask)]
   stored_raw=probabilities(rows,"prob_raw_mixture")
   raw_error=max(abs(a-b) for a,b in zip(raw,stored_raw))
   assert raw_error==0,(part,name,"RAW_CONSTITUENT_MIXTURE_DIFFERENCE",raw_error)
   raw_cells+=len(rows)
   cal=probabilities(rows,"prob_calibrated")
   entry=compare(part,name,base["y"],cal,base["ids"],targets[name],mask)
   entry.update(selective_mask_independently_reconstructed=identity["kind"]=="selective",
     random_mask_independently_reconstructed=False,
     raw_mixture_reconstructed_from_signed_constituents=True,
     raw_mixture_conditional_on_stored_mask=identity["kind"]=="random",
     raw_mixture_rows=len(rows),maximum_raw_mixture_difference=raw_error,
     calibrator_state_not_loaded=True,calibrated_risk_reconstructed=False)
   evidence.append(entry)
   evidence.append(compare(part,name+"__raw_mixture",base["y"],raw,base["ids"],targets[name+"__raw_mixture"],mask))
 assert policy_count==24 and selective_count==16 and vectors==74
 for name,h in snapshot.items(): assert sha(path(name))==h,name
 assert protection()==protected
 save(OUT/"FIXED_POLICY_SCALAR_REVIEW_INTERNAL.json",evidence)
 receipt={"status":"PARTIAL_FIXED_RECORD_AND_AGGREGATE_CONSISTENCY_VERIFIED_FULL_POLICY_REPLAY_PENDING",
  "time":now(),"code_sha256":sha(__file__),"input_pins":{rel(p):h for p,h in PINS.items()},
  "original_files_verified_unchanged":len(snapshot),"protected_manuscript_files_verified_unchanged":92,
  "fixed_model_vectors_checked":26,"policy_variants_checked":24,"raw_and_calibrated_policy_metric_vectors":48,
  "metric_cells_compared":cells,"maximum_absolute_metric_difference":max_difference,
  "selective_masks_independently_reconstructed":16,"random_masks_independently_reconstructed":0,
  "raw_mixture_cells_compared":raw_cells,"maximum_raw_mixture_difference":0,
  "calibrated_risks_reconstructed_from_calibrator":0,
  "frozen_models_refitted":0,"new_bootstrap_fits":0,"bootstrap_intervals_replayed":False,
  "seconds":time.monotonic()-start,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  "CPU_affinity":sorted(os.sched_getaffinity(0)),"address_space_limit_bytes":128*1024**2,
  "bounded_existing_monitoring_work":True,"GPU_or_numerical_libraries_loaded":False,
  "follow_up":"Within admitted resources reconstruct eight random allocations using locked NumPy PCG64 seed 20261021, reload 24 trusted SHA-bound calibrators and independently reconstruct calibrated risks; fixed replay is not rerouted patient bootstrap replay.",
  "whole_package_or_scientific_acceptance":False,
  "output_hashes":{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
 save(OUT/"SCALAR_REVIEW_RECEIPT.json",receipt)
 print(json.dumps({k:receipt[k] for k in ["status","metric_cells_compared","maximum_absolute_metric_difference","selective_masks_independently_reconstructed","raw_mixture_cells_compared","seconds","peak_RSS_bytes"]}))

if __name__=="__main__":
 try: run()
 except Exception as error:
  if OUT.exists() and not (OUT/"SCALAR_REVIEW_RECEIPT.json").exists() and not (OUT/"FAILURE.json").exists():
   save(OUT/"FAILURE.json",{"time":now(),"error":type(error).__name__+": "+str(error),"automatic_retry_allowed":False,"original_files_untouched":True})
  raise
