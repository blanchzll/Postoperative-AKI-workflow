"""Raw base-model outer predictions only; no calibration or final-test access."""
from core import *
from model_engine import *
import argparse,traceback
def outer_result(t):return OUT/"outer_raw"/t["line"]/t["cohort"]/t["stage"]/t["branch"]/t["context"]/t["family"]
def raw_fit_pair(t,p,params,threads,d):
 d=Path(d);d.mkdir(parents=True,exist_ok=True)
 if t["branch"]=="selected":ct=ensure_subset(t,p,int(params["selector_k"]),threads)
 else:
  ct=task_for(t,p);impute(ct,threads)
 tx,tm,vx,vm,sp=read_cache(ct)
 pipe,pr,notes,fallback=one_fit(tx,tm,vx,sp,t["family"],params,threads)
 # Save and re-read in the same original sklearn environment.
 bundle={"imputer":joblib.load(cache_dir(ct)/"imputer.joblib"),"pipeline":pipe,"input_spec":sp,"input_columns":list(tx),"method":t["method"],"params":params,"raw_internal_model_only":True,"calibrator":None}
 tmp=d/"raw_bundle.joblib.tmp";joblib.dump(bundle,tmp,compress=3);os.replace(tmp,d/"raw_bundle.joblib")
 loaded=joblib.load(d/"raw_bundle.joblib")
 raw_spec=spec_for(t["stage"],list(tx));_,_,raw_valid,raw_vm=frame_and_meta(ct,raw_spec)
 xx=loaded["imputer"].transform(raw_valid,keys(raw_vm))
 pp=loaded["pipeline"].predict_proba(xx)[:,1]
 np.testing.assert_allclose(pr,pp,rtol=1e-12,atol=1e-12)
 pred=vm[["excel_row","group_hash","center","y"]].copy();pred["prob_raw"]=pr
 pred.to_csv(d/"predictions.csv",index=False)
 return {"AP":float(average_precision_score(vm.y,pr)),"Brier":float(brier_score_loss(vm.y,pr)),"AUROC":float(roc_auc_score(vm.y,pr)),"assessment_n":len(vm),"assessment_AKI":int(vm.y.sum()),"training_n":len(tm),"warnings":notes,"spline_fallbacks":fallback,"input_columns":list(tx),"input_receipt_sha256":sha(cache_dir(ct)/"COMPLETE.json"),"roundtrip_full_raw_pipeline_verified":True}
def run(t,threads):
 assert t["context"].startswith("outer_")
 i=int(t["context"].split("_")[1])
 d=outer_result(t)
 source=OUT/"search_v2"/t["line"]/t["cohort"]/t["stage"]/t["branch"]/t["context"]/t["family"]
 assert ready(source),str(source)
 selected=json.loads((source/"COMPLETE.json").read_text())["selected"]
 assert sha(selected["prediction_file"])==selected["prediction_sha256"]
 methodfile=OUT/"selection"/t["line"]/t["cohort"]/t["context"]/"METHOD_SELECTION.json"
 assert json.loads(methodfile.read_text())["selected_method"]==t["method"]
 p=next(p for p in json.loads((INPUT/(t["line"]+"_"+t["cohort"]+"_resampling.json")).read_text()) if p["scheme"]=="outer" and p["fold"]==i)
 signature={"task":t,"search_receipt_sha256":sha(source/"COMPLETE.json"),"method_selection_sha256":sha(methodfile),"outer_train":p["train_excel_rows"],"outer_assessment":p["assessment_excel_rows"],"code_sha256":{n:sha(CODE/n) for n in ["model_engine.py","raw_outer.py"]}}
 if ready(d,signature):return
 start=time.time()
 with threadpool_limits(limits=1):rec=raw_fit_pair(t,p,selected["params"],threads,d)
 atomic(d/"COMPLETE.json",{"signature":signature,"files":{f:sha(d/f) for f in ["predictions.csv","raw_bundle.joblib"]},"metrics":rec,"seconds":time.time()-start,"no_final_hospital_time_evaluation_access":True,"calibrated_or_final_clinical_model_complete":False,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
def main():
 ap=argparse.ArgumentParser();ap.add_argument("task");ap.add_argument("--threads",type=int,default=1);a=ap.parse_args()
 run(json.loads(Path(a.task).read_text()),a.threads)
if __name__=="__main__":main()
