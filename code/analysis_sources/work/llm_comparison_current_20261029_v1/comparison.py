"""Fixed downstream LLM comparison. No predictive training or model/prompt selection."""
from pathlib import Path
import argparse,ast,json,hashlib,math,sys,datetime,resource,os,subprocess
from decimal import Decimal
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
L=R/"outputs/llm_supplement_20261028_v1";D=L/"data_preparation_v1";F=L/"formal_v1";O=L/"statistics_v1"
MODELS=["biomedlm_2_7b","medgemma_4b_it","llama3_8b_instruct_ngc_v1"]
SOURCE_SHA="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
FORMAL_SHA="eb27122860f1500b848efeea0ce1a19846e7d94aa5d0de7adbabc2c8db2041de"
for rel in ["final_evaluation_engine_20261022_v1","formal_evaluation_20261023_v1","ensemble_calibration_20261020_v1"]:sys.path.insert(0,str(R/"work"/rel))
from fixed_prediction_metrics import FixedRisk,_design,group_weights
from evaluation_stats import calibration,expanded,DCA,summary
from ensemble_math import calibrator,apply_calibration
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for v in iter(lambda:f.read(1048576),b""):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,mode=0o700,exist_ok=True);tmp=p.with_name(p.name+".tmp."+str(os.getpid()))
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)
def key(x):
 v=Decimal(str(x));assert v==v.to_integral_value();return str(int(v))
def threshold_function():
 p=R/"work/model_lock_20261022_v1/runtime.py";tree=ast.parse(p.read_text());node=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=="thresholds")
 env={"np":np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec"),env);return env["thresholds"]
thresholds=threshold_function()
def protocol():
 return {"scope":"Exploratory fixed three-LLM within-checkpoint before/after-task-PEFT and paired information comparison, after prior primary evaluation exposure","current_source_sha256":SOURCE_SHA,"formal_training_contract_sha256":FORMAL_SHA,"source_calibration":"Original LR-logit C1e6/lbfgs/max_iter5000/random_state42 on pooled source-development raw OOF;clip1e-6. Before checkpoint has no clinical-task fitting, same bank supplies source-development scores; calibration still uses source labels.","before_not_label_free_zero_shot":True,"after_OOF":"Each original3 assessment fold scored by its complement training model;final source-development model scores heldout validation only","source_thresholds":"Unchanged native thresholds on calibrated source-development raw OOF;source sensitivity80 and higher empirical quantile capacity10/20, alert>=ties included;not unbiased calibrated-development metrics","evaluation":"Fixed calibrated/raw scores;geography1158/72,temporal1011/36 paired operations; hospital-stratified patient-group bootstrap2000 seed20261017, original weight SHA verified","capacity":"Ceil(sample operations*fraction), deterministic risk/group/record ties;same patients and multiplicities for every model and contrast","calibration_diagnostics":"Offset intercept, joint intercept/slope evaluated descriptively only;never update heldout predictions. Undefined metrics keep per-metric replicate count.","fixed_thresholds":"Source-fitted thresholds only for calibrated LLM and calibrated primary models;TabPFN has no authorized honest source-OOF threshold so only equal-capacity/DCA comparisons, no invented source thresholds","contrasts":["After minus before within each checkpoint and information set","Endop minus preop for each checkpoint and before/after condition","After calibrated minus same-information locked primary calibrated model","After calibrated minus same-information existing TabPFN","Within-model change in endop-minus-preop increment after versus before"],"all_families_retained":True,"intervals":"Pointwise2.5/97.5 percentile intervals, exploratory without multiplicity correction;no confirmatory significance or winner selection","uncertainty":"Conditional on fitted predictions/calibration/source threshold;no repeat training;hospital/time overlap retained and not independent replications","resources":"Pilot separately;fixed training and prediction walltime/host/GPU peaks retained;architecture/size/pretraining domain not isolated","no_early_or_partial_comparison":"Actual all48 fit and6 before receipts plus closed formal controller required before real metrics/calibration","main_winner_unchanged":True,"final_scientific_review_complete":False}
def verify_sources():
 assert sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")==SOURCE_SHA
 assert sha(F/"FORMAL_TRAINING_CONTRACT.json")==FORMAL_SHA
 recipe=read(F/"FORMAL_TRAINING_CONTRACT.json");assert sha(R/"work/llm_supplement_20261028_v1/formal_benchmark_v1.py")==recipe["worker_code_sha256"]
 prep=read(D/"DATA_PREPARATION_CHECK.json")
 for p,h in {**prep["input_sha256"],**prep["files"]}.items():assert sha(p)==h,p
 q=pd.read_csv(R/"outputs/v4_development_inputs_20261018/paired_metadata.csv",dtype={"group_hash":str},float_precision="round_trip");q["record_key"]=q.excel_row.map(key);assert q.record_key.is_unique
 side=pd.read_csv(D/"labels_and_partitions.csv",dtype=str).set_index("record_key")
 assert set(side.index)==set(q.record_key)
 q=q.set_index("record_key",drop=False)
 for col in ["group_hash","center","y","geography_partition_v2","temporal_partition_v2"]:assert q[col].astype(str).equals(side.loc[q.index,col].rename(col)),col
 return q.loc[side.index],read(D/"ORIGINAL_OOF_FOLDS.json")
def native_weight_bank(meta,line):
 val=meta[meta[line+"_partition_v2"].eq("validation")]
 yy,gg,rr,cc,strata,models=_design(val.y.to_numpy(),val.group_hash.to_numpy(),val.excel_row.astype(str).to_numpy(),val.center.astype(str).to_numpy(),{"placeholder":np.full(len(val),.1)})
 rng=np.random.default_rng(20261017);proof=hashlib.sha256();weights=[]
 for _ in range(2000):
  w=group_weights(strata,len(yy),rng);proof.update(w.astype("<i8").tobytes());weights.append(w)
 expected=read(R/"outputs/formal_evaluation_20261023_v1"/(line+"_paired")/"EVALUATION.json")["receipt"]["resample_weight_sha256"]
 assert proof.hexdigest()==expected,(line,proof.hexdigest(),expected)
 return np.stack(weights),proof.hexdigest()
def score(f,w=None,cuts=None):
 w=np.ones(len(f.y),dtype=np.int64) if w is None else np.asarray(w,dtype=np.int64);s=f.score(w)
 for frac in [.1,.2]:
  c=f.capacity(frac,w,s)
  for k in ["reviewed_n","captured_n","missed_n","false_alerts_n","sensitivity","PPV","specificity","captured_per1000","missed_per1000","reviewed_per1000","false_alerts_per1000"]:s[f"capacity{int(frac*100)}_"+k]=c[k]
 idx=np.repeat(np.arange(len(f.y)),w);s.update(calibration(f.y[idx],f.p[idx]))
 for t in DCA:
  z=f.net_benefit(t,w);s["net_benefit_"+str(t)]=z["net_benefit"]
  s["treat_all_net_benefit_"+str(t)]=z["treat_all_net_benefit"]
 if cuts:
  n=int(w.sum());events=float(w@f.y);neg=n-events
  for cut in ["training_sensitivity80","training_capacity10","training_capacity20"]:
   a=f.p>=cuts[cut];review=float(w@a);tp=float(w@(a*f.y));fp=review-tp
   for k,v in {"reviewed_n":review,"captured_n":tp,"missed_n":events-tp,"false_alerts_n":fp,"sensitivity":tp/events if events else None,"PPV":tp/review if review else None,"specificity":1-fp/neg if neg else None,"captured_per1000":1000*tp/n,"missed_per1000":1000*(events-tp)/n,"reviewed_per1000":1000*review/n,"false_alerts_per1000":1000*fp/n}.items():s[cut+"_"+k]=v
 return s
def qualify():
 meta,folds=verify_sources();weight_proofs={line:native_weight_bank(meta,line)[1] for line in ["geography","temporal"]}
 rng=np.random.default_rng(31029);groups=np.array([f"g{i//2:03d}" for i in range(40)]);records=np.array([str(i) for i in range(40)]);centers=np.array(["A" if i<20 else "B" for i in range(40)])
 y=np.array([(i%7 in [0,1]) for i in range(40)],dtype=int);p=np.round(rng.uniform(.02,.9,40),1);p[[0,1]]=0.;p[[2,3]]=1.
 yy,gg,rr,cc,strata,models=_design(y,groups,records,centers,{"fixture":p});f=models["fixture"];worst=0.;checks=0;grouptests=0
 for _ in range(24):
  w=group_weights(strata,len(yy),rng)
  for g in set(gg):assert len(set(w[gg==g]))==1;grouptests+=1
  idx,eg,er=expanded(w,gg,rr);ref=summary(yy[idx],f.p[idx],eg,er);out=score(f,w)
  independent={"AUROC":roc_auc_score(yy,f.p,sample_weight=w) if 0<w@yy<w.sum() else None,"AP":average_precision_score(yy,f.p,sample_weight=w) if w@yy else None,"Brier":brier_score_loss(yy,f.p,sample_weight=w)}
  for k,v in independent.items():
   if v is not None:assert abs(out[k]-v)<1e-12;worst=max(worst,abs(out[k]-v));checks+=1
  for k,v in ref.items():
   if v is None:assert out.get(k) is None
   elif k in out:assert abs(out[k]-v)<1e-10,(k,out[k],v);worst=max(worst,abs(out[k]-v));checks+=1
 # Capacity ceilings and included ties distinct from a source numeric threshold.
 cuts=thresholds(np.array([1,1,1,1,1,0]),np.array([.9,.8,.7,.7,.1,.6]));assert cuts["training_sensitivity80"]==.7 and cuts["training_capacity20"]==.8
 before=O/"STATISTICS_PROTOCOL.json"
 if before.exists():assert read(before)==protocol()
 else:save(before,protocol())
 save(O/"ENGINEERING_STATISTICS_QUALIFICATION.json",{"status":"PASS_independent_weighted_metrics_expanded_tied_capacity_and_current_patient_boundaries_only","time":now(),"fixture":"Synthetic operator fixture only;not new clinical cohort or model performance evidence","checks":checks,"patient_multiplicity_checks":grouptests,"max_abs_discrepancy":worst,"current_original_weight_proofs":weight_proofs,"metadata_records":len(meta),"original_folds":len(folds),"real_LLM_performance_computed":False,"formal_training_modified":False,"all37_component_complete":False,"code_sha256":sha(__file__)})
 print("QUALIFIED_ENGINEERING_ONLY",checks,grouptests,worst,flush=True)
def require_closed_formal():
 props=subprocess.run(["systemctl","--user","show","aki-postslot-llm-formal-20261029-v1.service","-p","ActiveState","-p","ExecMainStatus","-p","Result"],capture_output=True,text=True,check=True).stdout
 state=dict(line.split("=",1) for line in props.splitlines() if "=" in line)
 assert state.get("ActiveState")=="inactive" and state.get("ExecMainStatus")=="0" and state.get("Result")=="success","Formal controller must finish successfully before real comparison"
 p=F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"
 if not p.exists():raise RuntimeError("FORMAL_48_FITS_6_BANKS_NOT_READY_NO_REAL_STATISTICS_ALLOWED")
 a=read(p);assert a["formal_fits"]==48 and a["original_checkpoint_banks"]==6 and a["contract_sha256"]==FORMAL_SHA
 assert read(F/"STATUS.json")["status"]=="COMPLETE_TRAINING_AND_PREDICTION_BANKS_STATISTICAL_COMPARISON_REQUIRED"
 expected={(m,stage,line,str(fit)) for m in MODELS for stage in ["preop","endop"] for line in ["geography","temporal"] for fit in [0,1,2,"final"]}
 actual=[(v["model"],v["stage"],v["line"],str(v["fit"])) for v in a["tasks"] if v["kind"]=="fit"]
 assert len(actual)==len(set(actual))==48 and set(actual)==expected
 assert {(v["model"],v["stage"]) for v in a["tasks"] if v["kind"]=="before"}=={(m,s) for m in MODELS for s in ["preop","endop"]}
 for r in a["tasks"]:
  assert sha(r["receipt"])==r["sha256"];d=read(r["receipt"])
  for name,h in d["files"].items():assert sha(name)==h,name
 return a
def load_bank(dest,expected_pred,expected_train,kind):
 rec=read(dest/"COMPLETE.json");sig=read(dest/"SIGNATURE.json");fields=sig["fields"]
 assert rec["signature"]==sig["signature"]==hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
 assert fields["contract_sha256"]==FORMAL_SHA and fields["kind"]==kind
 assert fields["code_sha256"]==read(F/"FORMAL_TRAINING_CONTRACT.json")["worker_code_sha256"]
 if kind=="before":assert fields["model"]==dest.parent.name and fields["stage"]==dest.name and fields["line"]=="all"
 else:assert fields["model"]==dest.parents[1].name and fields["line"]+"_"+fields["stage"]==dest.parent.name and str(fields["fit"])==dest.name
 assert fields["train_record_keys"]==list(expected_train) and fields["prediction_record_keys"]==list(expected_pred)
 for name,h in rec["files"].items():assert sha(name)==h
 p=dest/"RAW_PREDICTIONS.npz"
 with np.load(p,allow_pickle=False) as z:keys=z["record_key"].astype(str).tolist();prob=z["prob_raw"].copy();odds=z["binary_log_odds"].copy()
 assert keys==list(expected_pred) and len(keys)==len(set(keys)) and np.isfinite(prob).all() and np.isfinite(odds).all() and ((prob>=0)&(prob<=1)).all()
 return pd.Series(prob,index=keys,dtype=float),{str(dest/"COMPLETE.json"):sha(dest/"COMPLETE.json"),str(p):sha(p)}
def prepare():
 require_closed_formal();meta,folds=verify_sources();assert read(O/"STATISTICS_PROTOCOL.json")==protocol();sources={}
 allkeys=meta.index.tolist();selection=read(R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json")["models"]
 for line in ["geography","temporal"]:
  dest=O/line;dest.mkdir(mode=0o700,parents=True,exist_ok=True)
  dev=meta[meta[line+"_partition_v2"].eq("development")];val=meta[meta[line+"_partition_v2"].eq("validation")]
  assert not(set(dev.group_hash)&set(val.group_hash));pred={};cuts={};calrows=[];resources=[];before_sources={}
  for model in MODELS:
   for stage in ["preop","endop"]:
    bank,refs=load_bank(F/"before"/model/stage,allkeys,[],"before");sources.update(refs)
    for condition in ["before","after"]:
     if condition=="before":trainprob=bank.loc[dev.index].to_numpy();valprob=bank.loc[val.index].to_numpy()
     else:
      oof=pd.Series(np.nan,index=dev.index,dtype=float)
      for fold in [x for x in folds if x["line"]==line]:
       tr=fold["train_record_keys"];ass=fold["assessment_record_keys"]
       assert not(set(meta.loc[tr].group_hash)&set(meta.loc[ass].group_hash))
       z,refs=load_bank(F/"fits"/model/(line+"_"+stage)/str(fold["fold"]),ass,tr,"fit");sources.update(refs)
       assert oof.loc[ass].isna().all();oof.loc[ass]=z
      assert oof.notna().all()
      z,refs=load_bank(F/"fits"/model/(line+"_"+stage)/"final",val.index.tolist(),dev.index.tolist(),"fit");sources.update(refs)
      trainprob=oof.to_numpy();valprob=z.to_numpy()
     name="llm_"+model+"_"+condition+"_"+stage;train=pd.DataFrame({"y":dev.y.to_numpy(),"prob_raw":trainprob})
     cal=calibrator(train);calpath=dest/(name+"_source_calibrator.joblib");joblib.dump(cal,calpath)
     traincal=apply_calibration(cal,trainprob);pred[name+"_raw"]=valprob;pred[name+"_cal"]=apply_calibration(cal,valprob);cuts[name+"_cal"]=thresholds(dev.y.to_numpy(),traincal)
     calrows.append({"model":model,"condition":condition,"stage":stage,"source_development_records":len(dev),"source_development_events":int(dev.y.sum()),"intercept":float(cal.intercept_[0]),"slope":float(cal.coef_[0,0]),"calibrator_path":str(calpath),"calibrator_sha256":sha(calpath),"heldout_labels_used_for_calibration":False,"before_calibration_uses_source_labels":True,"negative_slope_retained_without_target_based_flip":bool(cal.coef_[0,0]<0)})
  for ref in selection:
   i=ref["identity"]
   if not ref["primary_route_winner"] or (i["line"],i["cohort"],i["branch"])!=(line,"paired","full"):continue
   assert sha(ref["path"])==ref["sha256"];lock=read(ref["path"]);name="_".join(i[k] for k in ["stage","branch","role","family"])
   pdest=R/"outputs/formal_evaluation_20261023_v1"/(line+"_paired")/"private_predictions"/name;p=pdest/"PREDICTIONS.csv"
   assert sha(p)==read(pdest/"COMPLETE.json")["sha256"]
   z=pd.read_csv(p,dtype={"group_hash":str},float_precision="round_trip");z["record_key"]=z.excel_row.map(key);z=z.set_index("record_key").loc[val.index]
   for c in ["group_hash","center","y"]:assert np.array_equal(z[c].astype(str),val[c].astype(str))
   pred["primary_"+i["stage"]+"_raw"]=z.prob_raw.to_numpy();pred["primary_"+i["stage"]+"_cal"]=z.prob_calibrated.to_numpy();cuts["primary_"+i["stage"]+"_cal"]=lock["training_thresholds"];sources[str(p)]=sha(p)
  for stage in ["preop","endop"]:
   d=L.parents[0]/"tabpfn_benchmark_20261025_v1"/"routes"/(line+"_paired_"+stage+"_full");p=d/"predictions.csv";assert sha(p)==read(d/"COMPLETE.json")["predictions_sha256"]
   z=pd.read_csv(p,dtype={"group_hash":str},float_precision="round_trip");z["record_key"]=z.excel_row.map(key);z=z.set_index("record_key").loc[val.index]
   for c in ["group_hash","center","y"]:assert np.array_equal(z[c].astype(str),val[c].astype(str))
   pred["tabpfn_"+stage]=z.prob_tabpfn.to_numpy();sources[str(p)]=sha(p)
  assert len(pred)==30
  yy,gg,rr,cc,strata,models=_design(val.y.to_numpy(),val.group_hash.to_numpy(),val.excel_row.astype(str).to_numpy(),val.center.astype(str).to_numpy(),pred)
  weights,proof=native_weight_bank(meta,line);arrays={name:m.p for name,m in models.items()}
  with (dest/"PRIVATE_FIXED_EVALUATION_INPUTS.npz").open("wb") as f:np.savez_compressed(f,y=yy,group_hash=gg,records=rr,centers=cc,weights=weights,**arrays)
  pairs={}
  for model in MODELS:
   for stage in ["preop","endop"]:
    base=f"llm_{model}_before_{stage}_cal";after=f"llm_{model}_after_{stage}_cal"
    pairs[after+" MINUS "+base]={after:1,base:-1}
    pairs[after+" MINUS primary_"+stage+"_cal"]={after:1,"primary_"+stage+"_cal":-1}
    pairs[after+" MINUS tabpfn_"+stage]={after:1,"tabpfn_"+stage:-1}
   for condition in ["before","after"]:
    pre=f"llm_{model}_{condition}_preop_cal";end=f"llm_{model}_{condition}_endop_cal";pairs[end+" MINUS "+pre]={end:1,pre:-1}
   pairs[model+"_change_in_paired_increment"]={f"llm_{model}_after_endop_cal":1,f"llm_{model}_after_preop_cal":-1,f"llm_{model}_before_endop_cal":-1,f"llm_{model}_before_preop_cal":1}
  save(dest/"COMPARISONS.json",pairs);save(dest/"SOURCE_THRESHOLDS.json",cuts);pd.DataFrame(calrows).to_csv(dest/"SOURCE_CALIBRATION.csv",index=False)
  save(dest/"PREPARATION_ACCEPTANCE.json",{"status":"PASS_SOURCE_ONLY_LLM_CALIBRATION_ALIGNED_FIXED_COMPARATORS_AND_ORIGINAL_WEIGHTS","sources":sources,"n":len(val),"events":int(val.y.sum()),"LLM_calibrated_routes":12,"raw_diagnostics_and_comparators":18,"group_weights_sha256":proof,"source_thresholds_not_capacity_guarantees_after_transfer":True,"code_sha256":sha(__file__),"input_files_sha256":{f:sha(dest/f) for f in ["PRIVATE_FIXED_EVALUATION_INPUTS.npz","SOURCE_THRESHOLDS.json","COMPARISONS.json","SOURCE_CALIBRATION.csv"]}})
 print("PREPARED_NO_MODEL_SELECTION",flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--qualify",action="store_true");ap.add_argument("--prepare",action="store_true");a=ap.parse_args()
 with threadpool_limits(limits=1):
  if a.qualify:qualify()
  elif a.prepare:prepare()
  else:raise SystemExit("Choose --qualify or --prepare; full bootstrap controller is a separate finite stage.")
if __name__=="__main__":main()
