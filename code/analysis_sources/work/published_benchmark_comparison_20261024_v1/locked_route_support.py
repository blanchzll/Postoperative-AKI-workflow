"""All already-selected preoperative branches, additional fixed-route support only."""
from pathlib import Path
import sys,os,json,hashlib,datetime,fcntl,time,multiprocessing as mp
import numpy as np,pandas as pd
import run as parent
from core import canonical_context,context_scores
R=parent.ROOT;BASE=R/"outputs/published_benchmark_comparison_20261024_v1"
O=BASE/"locked_route_support";O.mkdir(mode=0o700,exist_ok=True)
sha=parent.sha
def main():
 start=time.monotonic();lock=open(O/".run.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 parent.OUT=O;parent.CONTEXTS={};parent.WEIGHTS={};parent.PRED_RECEIPTS={}
 selected=json.loads((R/"outputs/model_lock_20261022_v1/PRIVATE_MODEL_SELECTION_AND_JOBS.json").read_text())
 contract=json.loads((BASE/"LOCKED_ROUTE_SUPPORT_CONTRACT.json").read_text())
 assert sha(R/"outputs/model_lock_20261022_v1/PRIVATE_MODEL_SELECTION_AND_JOBS.json")==contract["source_selection_sha256"]
 owner={"pid":os.getpid(),"start_ticks":Path(f"/proc/{os.getpid()}/stat").read_text().split(") ",1)[1].split()[19]}
 parent.write_json(O/"OWNER.json",owner)
 source_info=[];signatures=[sha(__file__),sha(BASE/"LOCKED_ROUTE_SUPPORT_CONTRACT.json"),sha(Path(parent.__file__))]
 for line in ["geography","temporal"]:
  branches={x["branch"]:x for x in selected["route_winners"] if x["line"]==line and x["cohort"]=="preop"}
  for kind in ["SIM","Chen"]:
   d=pd.read_csv(BASE/f"private_{kind}_{line}_{'predictions' if kind=='SIM' else 'scores'}.csv")
   if kind=="SIM":
    predictions={k:d[k].to_numpy() for k in ["our_endop_cal","SIM_original_2009","SIM_development_recalibrated_2009"]}
    ref="SIM_original_2009";newbranches=["full","compact","selected"];mode="prob"
   else:
    predictions={"Chen_Table4_ordinal_score":d.Chen_Table4_ordinal_score.to_numpy()}
    ref="Chen_Table4_ordinal_score";newbranches=["selected"];mode="rank"
   names=[]
   for branch in newbranches:
    family=branches[branch]["family"];route=f"preop_{branch}_champion_{family}"
    pred=parent.frozen_prediction(line,"preop",route,d)
    name=f"whole_source_preop_{branch}_{family}_carryforward"
    predictions[name]=pred.prob_calibrated.to_numpy();names.append(name)
    lp=R/f"outputs/model_lock_20261022_v1/locked_routes/{line}/preop/preop/{branch}/champion/{family}/MODEL_LOCK.json"
    m=json.loads(lp.read_text());assert m["primary_route_winner"] and m["family"]==family
    source_info.append({"context":kind+"_"+line,"model":name,"training_n":m["training_n"],"actual_input_count":m["actual_input_count"],
                       "source_lock":str(lp),"source_lock_sha256":sha(lp),"was_overall_preop_choice":selected["overall_preop_branch_choice"][line]["branch"]==branch})
   c=canonical_context(d,predictions,mode);c["comparisons"]=[(name,ref) for name in names]
   if kind=="SIM":c["comparisons"] +=[(name,"SIM_development_recalibrated_2009") for name in names]
   c["metadata"]={"line":line,"n":len(d),"events":int(d.y.sum()),"support_only":True,"no_champion_reselected":True}
   name=kind+"_"+line;parent.CONTEXTS[name]=c
   rng=np.random.default_rng(20261017)
   parent.WEIGHTS[name]=np.stack([parent.es.group_weights(c["strata"],len(c["y"]),rng) for _ in range(2000)]).astype("<i8")
   proof=hashlib.sha256(parent.WEIGHTS[name].tobytes()).hexdigest()
   old=json.loads((BASE/"BOOTSTRAP_RECEIPTS.json").read_text())[name]
   assert proof==old["resample_weight_sha256"],"Changed groups/weights"
 parent.write_json(O/"ORIGINAL_PREDICTION_RECEIPTS.json",parent.PRED_RECEIPTS)
 parent.write_json(O/"SOURCE_MODELS.json",source_info)
 parent.SIGNATURE=hashlib.sha256("".join(signatures+[sha(O/"ORIGINAL_PREDICTION_RECEIPTS.json"),sha(O/"SOURCE_MODELS.json")]).encode()).hexdigest()
 point={n:context_scores(c) for n,c in parent.CONTEXTS.items()}
 for n in parent.CONTEXTS:parent.evaluate_repeat((n,0))
 avail,cpus,limits=parent.cgroup_resources();workers=max(1,min(cpus,int(max(0,avail-3*1024**3)//(320*1024**2))))
 parent.write_json(O/"STATUS.json",{"state":"running_existing_locked_route_support","total":8000,"completed":4,"workers":workers})
 tasks=[(n,i) for n in parent.CONTEXTS for i in range(2000)];done=0
 with mp.get_context("fork").Pool(workers) as pool:
  for _ in pool.imap_unordered(parent.evaluate_repeat,tasks,chunksize=1):
   done+=1
   if done%400==0:parent.write_json(O/"STATUS.json",{"state":"running_existing_locked_route_support","total":8000,"completed":done,"workers":workers})
 parent.aggregate(point)
 oldp=pd.read_csv(BASE/"POINT_ESTIMATES.csv");newp=pd.read_csv(O/"POINT_ESTIMATES.csv")
 for name in parent.CONTEXTS:
  br=json.loads((O/"BOOTSTRAP_RECEIPTS.json").read_text())[name]
  oldbr=json.loads((BASE/"BOOTSTRAP_RECEIPTS.json").read_text())[name]
  assert br["resample_weight_sha256"]==oldbr["resample_weight_sha256"]
  for model in set(oldp.loc[oldp.context==name,"model"])&set(newp.loc[newp.context==name,"model"]):
   a=oldp.loc[(oldp.context==name)&(oldp.model==model)].iloc[0]
   b=newp.loc[(newp.context==name)&(newp.model==model)].iloc[0]
   for k in ["n","events","AUROC","AP","capacity10_captured_n","capacity20_captured_n"]:np.testing.assert_allclose(a[k],b[k],atol=1e-14)
 files=["SOURCE_MODELS.json","POINT_ESTIMATES.csv","INTERVALS.csv","PAIRED_DIFFERENCES.csv","BOOTSTRAP_RECEIPTS.json","ORIGINAL_PREDICTION_RECEIPTS.json"]
 parent.write_json(O/"ACCEPTANCE.json",{"status":"PASS_all_existing_locked_routes_support_computation","source_contract_sha256":sha(BASE/"LOCKED_ROUTE_SUPPORT_CONTRACT.json"),
   "signature":parent.SIGNATURE,"files_sha256":{f:sha(O/f) for f in files},"source_cases_weights_and_reused_reference_points_identical":True,
   "main38_comparison_replaced":False,"new_fitting":False,"new_winner_selection":False,"seconds":time.monotonic()-start,"owner":owner,"workers":workers})
 parent.write_json(O/"STATUS.json",{"state":"computed_review_required","total":8000,"completed":8000,"workers":0})
 print("DONE_SUPPORT",time.monotonic()-start,flush=True)
if __name__=="__main__":
 try:main()
 except Exception:
  import traceback
  parent.write_json(O/"FAILURE.json",{"error":traceback.format_exc()});raise
