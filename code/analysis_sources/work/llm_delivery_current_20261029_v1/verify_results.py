"""Independent numerical verification after all formal predictions and native statistics close."""
import argparse, csv, hashlib, math, json
from decimal import Decimal
from pathlib import Path
import common as c
from independent_metrics import metrics,close,quantile,logit,sigmoid,thresholds,joint_score_residual
SELECTED_REPLICATES=(0,1,24,25,63,127,255,511,1023,1535,1999)
def rowkey(x): return str(int(Decimal(str(x))))
def load_probability(path):
 import numpy as np
 with np.load(path,allow_pickle=False) as z:
  k=z["record_key"].astype(str).tolist();p=z["prob_raw"].tolist()
 assert len(k)==len(set(k));return dict(zip(k,p))
def check_source_calibration(line,arrays,meta,folds,cuts):
 """Independent row alignment, LR score equations, map replay and source cut calculation."""
 import numpy as np
 d=c.S/line; rows=list(csv.DictReader((d/"SOURCE_CALIBRATION.csv").open()))
 assert len(rows)==12 and len({(x["model"],x["stage"],x["condition"]) for x in rows})==12
 dev=[k for k,v in meta.items() if v[line+"_partition_v2"]=="development"]
 val=[k for k,v in meta.items() if v[line+"_partition_v2"]=="validation"]
 assert not({meta[k]["group_hash"] for k in dev}&{meta[k]["group_hash"] for k in val})
 expected_eval_keys=[rowkey(x) for x in arrays["records"]]
 assert set(val)==set(expected_eval_keys);out=[]
 for row in rows:
  model,stage,condition=(row[x] for x in ("model","stage","condition"))
  name=f"llm_{model}_{condition}_{stage}"
  if condition=="before":
   bank=load_probability(c.F/"before"/model/stage/"RAW_PREDICTIONS.npz");train={k:bank[k] for k in dev};prediction={k:bank[k] for k in val}
  else:
   train={}
   for fold in [x for x in folds if x["line"]==line]:
    tr=fold["train_record_keys"];ass=fold["assessment_record_keys"]
    assert set(tr)|set(ass)==set(dev) and not(set(tr)&set(ass))
    assert not({meta[k]["group_hash"] for k in tr}&{meta[k]["group_hash"] for k in ass})
    folder=c.F/"fits"/model/(line+"_"+stage)/str(fold["fold"])
    fields=c.read(folder/"SIGNATURE.json")["fields"]
    assert fields["train_record_keys"]==tr and fields["prediction_record_keys"]==ass
    p=load_probability(folder/"RAW_PREDICTIONS.npz");assert set(p)==set(ass) and not(set(p)&set(train));train.update(p)
   assert set(train)==set(dev)
   folder=c.F/"fits"/model/(line+"_"+stage)/"final"
   fields=c.read(folder/"SIGNATURE.json")["fields"]
   assert fields["train_record_keys"]==dev and fields["prediction_record_keys"]==val
   prediction=load_probability(folder/"RAW_PREDICTIONS.npz");assert set(prediction)==set(val)
  assert int(row["source_development_records"])==len(dev)
  yy=[int(meta[k]["y"]) for k in dev];raw=[train[k] for k in dev]
  assert int(row["source_development_events"])==sum(yy)
  intercept=float(row["intercept"]);slope=float(row["slope"])
  z=[logit(p) for p in raw];trans=[sigmoid(intercept+slope*x) for x in z]
  err=[p-y for p,y in zip(trans,yy)];n=len(yy)
  gradient_intercept=math.fsum(err)/n
  gradient_slope=math.fsum(e*x for e,x in zip(err,z))/n+slope/(1e6*n)
  # Native sklearn tol=1e-4 for the mean penalized logistic objective.
  assert max(abs(gradient_intercept),abs(gradient_slope))<=1.2e-4,(line,name,"source_calibration_score_equation",gradient_intercept,gradient_slope)
  expected=[sigmoid(intercept+slope*logit(prediction[k])) for k in expected_eval_keys]
  max_map_error=max(abs(float(a)-b) for a,b in zip(arrays[name+"_cal"],expected))
  max_raw_error=max(abs(float(a)-prediction[k]) for a,k in zip(arrays[name+"_raw"],expected_eval_keys))
  assert max_map_error<1e-12 and max_raw_error<1e-12
  cut=thresholds(yy,trans)
  for key,value in cut.items():assert close(cuts[name+"_cal"][key],value,1e-12),(name,key)
  assert row["heldout_labels_used_for_calibration"]=="False"
  out.append({"line":line,"model":model,"stage":stage,"condition":condition,"n_source_development":n,
    "events_source_development":sum(yy),"source_intercept":intercept,"source_slope":slope,
    "negative_slope_preserved":slope<0,"source_score_equation_intercept":gradient_intercept,
    "source_score_equation_slope":gradient_slope,"raw_alignment_max_error":max_raw_error,
    "calibration_map_max_error":max_map_error,"source_thresholds_independently_recomputed":True})
 return out
def check_weight_bank(arrays,expected_proof):
 """Regenerate each hospital's patient draw using a dictionary expansion, not native group_weights."""
 import numpy as np
 g=arrays["group_hash"].astype(str);center=arrays["centers"].astype(str);weights=arrays["weights"]
 assert weights.shape==(2000,len(g)) and np.issubdtype(weights.dtype,np.integer)
 patients={}
 for i,(hospital,patient) in enumerate(zip(center,g)):
  patients.setdefault(hospital,{}).setdefault(patient,[]).append(i)
 location={}
 for hospital,groups in patients.items():
  for patient in groups:assert patient not in location or location[patient]==hospital;location[patient]=hospital
 rng=np.random.default_rng(20261017);h=hashlib.sha256()
 for index,w in enumerate(weights):
  own=np.zeros(len(g),dtype=np.int64)
  for hospital in sorted(patients):
   keys=sorted(patients[hospital]);draw=rng.integers(len(keys),size=len(keys))
   drawn,counts=np.unique(draw,return_counts=True)
   for j,count in zip(drawn,counts):
    for row in patients[hospital][keys[int(j)]]:own[row]=int(count)
  assert np.array_equal(w,own),("patient_bootstrap_mismatch",index)
  h.update(own.astype("<i8").tobytes())
 assert h.hexdigest()==expected_proof
 return {"replicates":2000,"patient_groups":len(location),"hospitals":len(patients),"resample_weight_sha256":h.hexdigest()}
def expected_pairs():
 pairs={}
 for model in c.MODELS:
  for stage in c.STAGES:
   before=f"llm_{model}_before_{stage}_cal";after=f"llm_{model}_after_{stage}_cal"
   for reference in (before,"primary_"+stage+"_cal","tabpfn_"+stage):pairs[after+" MINUS "+reference]={after:1,reference:-1}
  for condition in ("before","after"):
   pre=f"llm_{model}_{condition}_preop_cal";end=f"llm_{model}_{condition}_endop_cal"
   pairs[end+" MINUS "+pre]={end:1,pre:-1}
  pairs[model+"_change_in_paired_increment"]={f"llm_{model}_after_endop_cal":1,f"llm_{model}_after_preop_cal":-1,f"llm_{model}_before_endop_cal":-1,f"llm_{model}_before_preop_cal":1}
 return pairs
def check_line(line,meta,folds):
 import numpy as np
 d=c.S/line;prep=c.read(d/"PREPARATION_ACCEPTANCE.json");result=c.read(d/"RESULTS.json")
 with np.load(d/"PRIVATE_FIXED_EVALUATION_INPUTS.npz",allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 cuts=c.read(d/"SOURCE_THRESHOLDS.json");pairs=c.read(d/"COMPARISONS.json")
 assert pairs==expected_pairs()
 for name in ("tabpfn_preop","tabpfn_endop"):assert name not in cuts
 names=sorted(set(arrays)-{"y","group_hash","records","centers","weights"})
 assert len(names)==30 and set(names)==set(result["point"])
 y=arrays["y"].tolist();g=arrays["group_hash"].astype(str).tolist();r=arrays["records"].astype(str).tolist()
 expected_counts={"geography":(1158,72),"temporal":(1011,36)}
 assert (len(y),sum(y))==expected_counts[line]
 for i,key in enumerate(r):
  m=meta[rowkey(key)];assert int(m["y"])==y[i] and m["group_hash"]==g[i] and m["center"]==str(arrays["centers"][i])
  assert m[line+"_partition_v2"]=="validation"
 weights=check_weight_bank(arrays,prep["group_weights_sha256"])
 calibration=check_source_calibration(line,arrays,meta,folds,cuts)
 point_checks=selected_checks=0;largest=0.;calibration_equations=[]
 for name in names:
  target=result["point"][name]
  dca=[float(k[len("net_benefit_"):]) for k in target if k.startswith("net_benefit_")]
  own=metrics(y,arrays[name],g,r,cuts=cuts.get(name),dca=dca,include_offset=True)
  for field,value in own.items():
   tol=1e-8 if field=="calibration_intercept" else 1e-10
   assert close(target[field],value,tol),(line,name,field,target[field],value)
   if value is not None:largest=max(largest,abs(target[field]-value))
   point_checks+=1
  resid=joint_score_residual(y,arrays[name],[1]*len(y),target.get("calibration_joint_intercept"),target.get("calibration_slope"))
  if resid is not None:assert resid<=1.2e-6,(line,name,"heldout_calibration_score_equation",resid)
  calibration_equations.append({"line":line,"model":name,"joint_score_residual":resid,"joint_diagnostic_defined":target.get("calibration_slope") is not None})
  # Per-metric NA is retained, including no invented source threshold for TabPFN.
 matrix=[];all_fields=None;chunk_checks=0
 for start in range(0,2000,25):
  p=d/"bootstrap_chunks"/f"{start:04d}_{start+25:04d}.npz";rec=c.read(p.with_suffix(".json"))
  assert c.sha(p)==rec["sha256"] and rec["prepared_receipt_sha256"]==c.sha(d/"PREPARATION_ACCEPTANCE.json")
  with np.load(p,allow_pickle=False) as z:
   assert z["models"].tolist()==names and z["replicate"].tolist()==list(range(start,start+25))
   fields=z["fields"].tolist();block=z["values"].copy()
  if all_fields is None:all_fields=fields
  assert fields==all_fields and block.shape==(25,30,len(fields));matrix.append(block);chunk_checks+=1
 values=np.concatenate(matrix,axis=0);del matrix
 for rep in SELECTED_REPLICATES:
  for j,name in enumerate(names):
   dca=[float(k[len("net_benefit_"):]) for k in result["point"][name] if k.startswith("net_benefit_")]
   own=metrics(y,arrays[name],g,r,arrays["weights"][rep],cuts.get(name),dca)
   for field,value in own.items():
    actual=float(values[rep,j,all_fields.index(field)])
    assert close(actual,value,1e-10),(line,name,rep,field,actual,value)
    if value is not None:largest=max(largest,abs(actual-value))
    selected_checks+=1
 def check_intervals(rows,kind):
  seen=set();count=0
  for row in rows:
   name=row[kind];field=row["metric"];identity=(name,field);assert identity not in seen;seen.add(identity)
   k=all_fields.index(field)
   if kind=="model":
    series=values[:,names.index(name),k].tolist();expected=result["point"][name].get(field)
   else:
    terms=pairs[name]
    series=[math.fsum(float(values[i,names.index(n),k])*w for n,w in terms.items()) for i in range(2000)]
    pts=[result["point"][n].get(field) for n in terms]
    expected=math.fsum(result["point"][n][field]*w for n,w in terms.items()) if all(v is not None and math.isfinite(v) for v in pts) else None
   valid=[x for x in series if math.isfinite(x)]
   assert row["requested_replicates"]==2000 and row["valid_replicates"]==len(valid)
   for field2,value in (("estimate",expected),("lower95",quantile(valid,.025)),("upper95",quantile(valid,.975))):
    assert close(row[field2],value,1e-10),(line,name,field,field2,row[field2],value);count+=1
  expected_names=names if kind=="model" else pairs
  assert seen=={(n,f) for n in expected_names for f in all_fields}
  return count
 interval_checks=check_intervals(result["intervals"],"model")+check_intervals(result["paired_differences"],"comparison")
 return {"line":line,"n":len(y),"events":sum(y),"weights":weights,"point_scalar_checks":point_checks,
   "selected_bootstrap_scalar_checks":selected_checks,"selected_replicates":list(SELECTED_REPLICATES),
   "bootstrap_chunks":chunk_checks,"all_interval_and_contrast_checks":interval_checks,
   "max_abs_scalar_discrepancy":largest,"heldout_joint_calibration_score_equations":calibration_equations},calibration
def run():
 c.check_code();c.numeric_guard()
 inputs={str(c.S/"NUMERICAL_COMPARISON_ACCEPTANCE.json"):c.sha(c.S/"NUMERICAL_COMPARISON_ACCEPTANCE.json")}
 with c.owned_lock("independent"):
  if c.existing_receipt("INDEPENDENT_RESULT_VERIFICATION.json",inputs):return
  meta={x["record_key"]:x for x in csv.DictReader((c.L/"data_preparation_v1/labels_and_partitions.csv").open())}
  assert len(meta)==2428 and sum(int(x["y"]) for x in meta.values())==138
  folds=c.read(c.L/"data_preparation_v1/ORIGINAL_OOF_FOLDS.json")
  details=[];cal=[]
  for line in c.LINES:
   a,b=check_line(line,meta,folds);details.append(a);cal.extend(b)
  c.save(c.O/"INDEPENDENT_NUMERIC_DETAILS.json",{"contexts":details,"source_calibration_rows":24,
    "methods":"Independent stdlib ascending-block concordance, threshold AP, explicit replicated record capacity, scalar probability replay, logistic score equations, regenerated hospital patient draws and manual linear percentile interpolation.",
    "limits":"All point metrics and all stored percentile intervals; 11 outcome-independent bootstrap replicates recalculated for every fixed predictor. Joint calibration diagnostics checked via native objective score equations, not a new clinical fit. This is numerical QC, not full scientific acceptance."})
  c.write_csv(c.O/"SOURCE_CALIBRATION_INDEPENDENT.csv",cal)
  c.save(c.O/"INDEPENDENT_RESULT_VERIFICATION.json",{"status":"PASS_INDEPENDENT_SOURCE_CALIBRATION_REPLAY_PATIENT_WEIGHTS_KEY_METRICS_AND_ALL_INTERVALS",
    "time":c.now(),"inputs_sha256":inputs,"code_lock_sha256":c.check_code(),
    "files":{str(c.O/n):c.sha(c.O/n) for n in ("INDEPENDENT_NUMERIC_DETAILS.json","SOURCE_CALIBRATION_INDEPENDENT.csv")},
    "LLM_routes":24,"all_performance_conditions_preserved":True,"final_scientific_review_complete":False})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");a=p.parse_args()
 if not a.run:raise SystemExit("Explicit --run after complete formal and numerical receipts required")
 run()
