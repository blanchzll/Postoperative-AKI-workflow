"""Nested grouped six-imputation/nine-family development comparison.
Local model factory; Codex integration of scientific training/calibration boundaries.
"""
from pathlib import Path
import argparse,gc,hashlib,json,os,time,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,roc_curve
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from threadpoolctl import threadpool_limits
from model_candidates import model_candidates
from run_imputation_jobs import ROOT,BASE,CODE,atomic,sha,resource_snapshot,signature,check_outputs,MIB
from train_preprocess import NearZeroVariance
def read_pair(pair,method,spec):
    out=BASE/"imputed"/method/pair["name"]
    while True:
        if (out/"provenance.json").exists() and (out/"SUCCESS.json").exists():
            if json.loads((out/"provenance.json").read_text())==signature(pair,method):
                check_outputs(out,pair);break
        print("WAIT_CACHE",method,pair["name"],flush=True);time.sleep(30)
    frames=[]
    for side in ["train","valid"]:
        x=pd.read_csv(out/(side+".csv"),dtype={k:str for k in spec["cat"]})
        meta=pd.read_csv(ROOT/pair["files"][side+"_meta"]["path"],dtype={"group_hash":str})
        assert len(x)==len(meta)==pair[side+"_n"] and not x.isna().any().any()
        frames.extend([x,meta])
    return frames
def make_pipeline(est,spec,cols):
    nums=[k for k in spec["num"] if k in cols];cats=[k for k in spec["cat"] if k in cols]
    tr=ColumnTransformer([("numeric",StandardScaler(),nums),("categorical",OneHotEncoder(categories=[spec["cat_levels"][k] for k in cats],handle_unknown="ignore",sparse_output=False),cats)])
    return Pipeline([("encode",tr),("nzv",NearZeroVariance()),("smote",SMOTE(random_state=42,k_neighbors=5)),("model",clone(est))])
def logit(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def fit_predict(est,spec,a,ma,b,threads):
    while resource_snapshot()["effective_free"]<2304*MIB:time.sleep(30)
    pipe=make_pipeline(est,spec,a.columns)
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always")
        pipe.fit(a,ma.y.to_numpy());pred=pipe.predict_proba(b)[:,1]
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    notes=list(dict.fromkeys(str(w.message)[:250] for w in ww))
    return pipe,pred,notes
def threshold_at_80(y,p):
    _,tpr,threshold=roc_curve(y,p,drop_intermediate=False)
    ok=np.isfinite(threshold)&(tpr>=.8);return float(np.max(threshold[ok]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--methods",default="mean_mode,random_forest,pmm,bayes,cart,knn")
    ap.add_argument("--families",default="LR,DT,RF,KNN,SVM,NB,XGB,SGBT,NNET");ap.add_argument("--threads",type=int,default=len(os.sched_getaffinity(0)))
    ap.add_argument("--outer",type=int);ap.add_argument("--protocol-only",action="store_true");a=ap.parse_args()
    manifest=json.loads((BASE/"fold_inputs/fold_manifest.json").read_text());spec=json.loads((BASE/"fold_inputs/feature_spec.json").read_text())
    assert manifest["source_cohort_sha256"]==sha(BASE/"cohort.csv")
    models=model_candidates(42,a.threads,True);families=a.families.split(",");assert set(families)<=set(models)
    pairs=manifest["pairs"];outers=[p for p in pairs if p["role"]=="development_outer" and (a.outer is None or p["outer"]==a.outer)]
    methods=a.methods.split(",");dest=BASE/"six_by_nine";dest.mkdir(exist_ok=True)
    protocol={"source_cohort_sha256":sha(BASE/"cohort.csv"),"code_sha256":sha(Path(__file__)),"factory_sha256":sha(CODE/"model_candidates.py"),"preprocessing_sha256":sha(CODE/"train_preprocess.py"),
              "tuning":"3 inner grouped folds within each of5 outer grouped folds, pooled inner AP",
              "features":"27 preoperative candidates; original near-zero-variance/SMOTE branch",
              "preprocessing":"train-only numeric scaling and fixed-codebook one-hot; train-only NZV; SMOTE on training only",
              "sampling":"SMOTE1:1 seed42, k5; validation retains natural prevalence",
              "calibration":"logistic recalibration using selected-candidate inner OOF raw probabilities, no outer validation labels",
              "imputation_selection":"after all54 method-family OOF estimates, development-only mean AP rank across nine families; ties mean Brier rank",
              "parameters":{k:[{"name":n,"params":e.get_params()} for n,e in models[k]] for k in models}}
    protocol=json.loads(json.dumps(protocol),parse_constant=lambda value:None)
    protocol_path=dest/"protocol.json"
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:atomic(protocol_path,protocol)
    if a.protocol_only:print("MODEL_PROTOCOL_READY",flush=True);return
    for method in methods:
        for outer in outers:
            inners=[p for p in pairs if p["role"]=="development_inner" and p["outer"]==outer["outer"]];assert len(inners)==3
            data=[read_pair(p,method,spec) for p in inners]
            train,tm,valid,vm=read_pair(outer,method,spec)
            assert not set(tm.group_hash)&set(vm.group_hash)
            for family in families:
                out=dest/method/outer["name"]/family;out.mkdir(parents=True,exist_ok=True)
                sig={"protocol_sha256":sha(protocol_path),"outer_signature":signature(outer,method),"family":family}
                if (out/"SUCCESS.json").exists() and json.loads((out/"SUCCESS.json").read_text())==sig:continue
                start=time.time();scores=[];best=None;best_ap=-np.inf
                print("START_MODEL",method,outer["name"],family,flush=True)
                for candidate,est in models[family]:
                    parts=[];notes=[]
                    for j,(tr,trm,va,vam) in enumerate(data):
                        pipe,p,w=fit_predict(est,spec,tr,trm,va,a.threads)
                        part=vam[["excel_row","group_hash","y"]].copy();part["prob"]=p;part["inner"]=j;parts.append(part);notes.extend(w)
                        del pipe;gc.collect()
                    pred=pd.concat(parts,ignore_index=True)
                    assert pred.excel_row.is_unique and set(pred.excel_row)==set(tm.excel_row)
                    score=float(average_precision_score(pred.y,pred.prob))
                    scores.append({"candidate":candidate,"inner_ap":score,"warnings":list(dict.fromkeys(notes))})
                    if score>best_ap:
                        best_ap=score;best=(candidate,est,pred.copy())
                candidate,est,ip=best
                cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=42)
                with threadpool_limits(limits=1):cal.fit(logit(ip.prob),ip.y)
                cp=cal.predict_proba(logit(ip.prob))[:,1];threshold=threshold_at_80(ip.y,cp)
                pipe,rawp,notes=fit_predict(est,spec,train,tm,valid,a.threads)
                calibrated=cal.predict_proba(logit(rawp))[:,1]
                results=vm.copy();results["method"]=method;results["family"]=family;results["outer"]=outer["outer"]
                results["prob_raw"]=rawp;results["prob_calibrated"]=calibrated
                results["threshold80"]=threshold;results["alert80"]=calibrated>=threshold;results["candidate"]=candidate
                results.to_csv(out/"predictions.csv",index=False)
                ip.to_csv(out/"selected_inner_oof.csv",index=False)
                atomic(out/"tuning.json",{"scores":scores,"selected":candidate,"selected_inner_ap":best_ap,"warnings_outer":notes,"calibrator_slope":float(cal.coef_[0,0]),"calibrator_intercept":float(cal.intercept_[0]),"threshold80":threshold,"retained_encoded_features":int(pipe.named_steps["nzv"].keep_.sum()),"elapsed_seconds":time.time()-start})
                joblib.dump({"pipeline":pipe,"calibrator":cal,"feature_columns":list(train.columns),"threshold80":threshold},out/"model.joblib",compress=3)
                atomic(out/"SUCCESS.json",sig);print("DONE_MODEL",method,outer["name"],family,"seconds",round(time.time()-start,1),flush=True)
                del pipe,cal;gc.collect()
    atomic(dest/("task_complete_"+a.methods.replace(",","_")+"_"+str(a.outer)+"_"+a.families.replace(",","_")+".json"),{"methods":methods,"families":families,"outers":[x["outer"] for x in outers],"source_cohort_sha256":sha(BASE/"cohort.csv"),"time":time.time()})
if __name__=="__main__":main()
