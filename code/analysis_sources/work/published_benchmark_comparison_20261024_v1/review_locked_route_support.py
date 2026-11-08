from pathlib import Path
import json,hashlib,fcntl,datetime,os
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
import evaluation_stats as es
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
B=R/"outputs/published_benchmark_comparison_20261024_v1";O=B/"locked_route_support"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((O/"ACCEPTANCE.json").read_text())
for f,h in a["files_sha256"].items():assert sha(O/f)==h,f
owner=a["owner"];st=Path(f"/proc/{owner['pid']}/stat")
assert not st.exists() or st.read_text().split(") ",1)[1].split()[19]!=owner["start_ticks"]
with open(O/".run.lock","a") as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=json.loads((O/"SOURCE_MODELS.json").read_text())
for s in source:assert sha(s["source_lock"])==s["source_lock_sha256"]
for f,h in json.loads((O/"ORIGINAL_PREDICTION_RECEIPTS.json").read_text()).items():assert sha(R/f)==h["sha256"]
pt=pd.read_csv(O/"POINT_ESTIMATES.csv");it=pd.read_csv(O/"INTERVALS.csv");dt=pd.read_csv(O/"PAIRED_DIFFERENCES.csv")
br=json.loads((O/"BOOTSTRAP_RECEIPTS.json").read_text())
source_winners=json.loads((R/"outputs/model_lock_20261022_v1/PRIVATE_MODEL_SELECTION_AND_JOBS.json").read_text())["route_winners"]
ncp=0
for context in br:
 kind,line=context.split("_",1)
 d=pd.read_csv(B/f"private_{kind}_{line}_{'predictions' if kind=='SIM' else 'scores'}.csv")
 d["rkey"]=d.excel_row.astype(str);d["ckey"]=d.center.astype(str)
 d=d.sort_values(["ckey","group_hash","rkey"]).reset_index(drop=True)
 y=d.y.to_numpy();g=d.group_hash.to_numpy();r=d.rkey.to_numpy()
 predictions={}
 for s in [s for s in source if s["context"]==context]:
  lock=json.loads(Path(s["source_lock"]).read_text());i=lock["identity"]
  pred=R/f"outputs/formal_evaluation_20261023_v1/{line}_preop/private_predictions/preop_{i['branch']}_champion_{i['family']}/PREDICTIONS.csv"
  z=pd.read_csv(pred).set_index("excel_row").loc[d.excel_row]
  assert z.group_hash.to_list()==d.group_hash.to_list() and z.y.to_list()==d.y.to_list()
  v=z.prob_calibrated.to_numpy();predictions[s["model"]]=v
  pp=pt.loc[(pt.context==context)&(pt.model==s["model"])].iloc[0]
  np.testing.assert_allclose(pp.AUROC,roc_auc_score(y,v),atol=1e-14)
  np.testing.assert_allclose(pp.AP,average_precision_score(y,v),atol=1e-14)
  order=sorted(range(len(d)),key=lambda j:(-v[j],g[j],r[j]))
  for cap in [.1,.2]:
   k=int(np.ceil(cap*len(d)));assert pp[f"capacity{int(100*cap)}_captured_n"]==y[order[:k]].sum()
 _,_,_,_,strata,_=es._design(y,g,r,d.ckey,{"identity":np.full(len(d),.5)})
 rng=np.random.default_rng(20261017);proof=hashlib.sha256();stream=hashlib.sha256();vv={};dd={}
 pairs=dt.loc[dt.context==context,"comparison"].unique()
 for rep in range(2000):
  p=O/"checkpoints"/context/f"{rep:04d}.json";q=json.loads(p.read_text());w=es.group_weights(strata,len(d),rng)
  proof.update(w.astype("<i8").tobytes());stream.update(p.read_bytes())
  assert q["signature"]==a["signature"] and q["weight_sha256"]==hashlib.sha256(w.astype("<i8").tobytes()).hexdigest()
  if rep in [0,1,99,1999] and 0<(w@y)<w.sum():
   for model,v in predictions.items():
    np.testing.assert_allclose(q["scores"][model]["AUROC"],roc_auc_score(y,v,sample_weight=w),atol=1e-13)
    np.testing.assert_allclose(q["scores"][model]["AP"],average_precision_score(y,v,sample_weight=w),atol=1e-13)
  for model,mm in q["scores"].items():
   for k,v in mm.items():
    if k not in ["n","events"] and v is not None and np.isfinite(v):vv.setdefault((model,k),[]).append(v)
  for pair in pairs:
   x,z=pair.split(" MINUS ")
   for k,v in q["scores"][x].items():
    b=q["scores"][z].get(k)
    if k not in ["n","events"] and v is not None and b is not None and np.isfinite(v) and np.isfinite(b):dd.setdefault((pair,k),[]).append(v-b)
  ncp+=1
 assert proof.hexdigest()==br[context]["resample_weight_sha256"]==json.loads((B/"BOOTSTRAP_RECEIPTS.json").read_text())[context]["resample_weight_sha256"]
 assert stream.hexdigest()==br[context]["checkpoint_stream_sha256"]
 for table,values,col in [(it,vv,"model"),(dt,dd,"comparison")]:
  for rr in table.loc[table.context==context].itertuples():
   v=values[(getattr(rr,col),rr.metric)];assert len(v)==rr.valid_replicates
   np.testing.assert_allclose([rr.lower95,rr.upper95],np.quantile(v,[.025,.975]),atol=2e-12)
rec={"status":"PASS_all_locked_route_support_source_selection_predictions_and_intervals_reviewed",
 "reviewed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 "checkpoints_reverified":ncp,"source_models_reverified":len(source),"original_group_resampling_identical":True,
 "original_main38_comparison_unchanged":True,"validation_winner_selection":False,
 "post_primary_result_support_addition_disclosed":True,"actual_selected_input_counts_disclosed":True,
 "code_sha256":sha(__file__),"acceptance_sha256":sha(O/"ACCEPTANCE.json"),"whole_project_complete":False}
(O/"REVIEW_ACCEPTANCE.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(rec))
