"""Original-runtime functional checks on real development-fold data."""
from core import *
from model_engine import *
from sklearn.base import clone
import time
start=time.time()
p=parts("geography","preop","final")[0]
t=pair_task("geography","preop","preop","final",p,"bayes","compact")
tx,tm,vx,vm,sp=read_cache(t)
params={
 "LR":{"variant":"linear","C":0.1},
 "DT":{"max_depth":3,"min_samples_leaf":5,"min_samples_split":2,"ccp_alpha":0},
 "RF":{"n_estimators":300,"max_depth":3,"min_samples_leaf":5,"max_features":"sqrt"},
 "KNN":{"n_neighbors":15,"weights":"distance","p":2},
 "SVM":{"C":1,"gamma":"scale"},
 "NB":{"var_smoothing":1e-9},
 "XGB":{"n_estimators":200,"max_depth":2,"learning_rate":0.03,"min_child_weight":3,"subsample":0.8,"colsample_bytree":0.75,"gamma":0,"reg_alpha":0,"reg_lambda":1},
 "SGBT":{"n_estimators":100,"max_depth":1,"learning_rate":0.03,"min_samples_leaf":5,"subsample":0.8},
 "NNET":{"hidden_layer_sizes":[16],"alpha":0.1,"learning_rate_init":0.001}
}
checks={};warnings_by_family={}
for f,par in params.items():
 pipe,pr,notes,_=one_fit(tx,tm,vx,sp,f,par,2)
 reverse=pipe.predict_proba(vx.iloc[::-1])[:,1][::-1]
 np.testing.assert_allclose(pr,reverse,rtol=1e-10,atol=1e-12)
 dest=OUT/"qualification_models";dest.mkdir(exist_ok=True)
 joblib.dump(pipe,dest/(f+".joblib"),compress=3)
 np.testing.assert_allclose(pr,joblib.load(dest/(f+".joblib")).predict_proba(vx)[:,1],rtol=1e-12,atol=1e-12)
 checks[f]=True;warnings_by_family[f]=notes
 del pipe;gc.collect()
for nk in [3,4]:
 pipe,pr,notes,fall=one_fit(tx,tm,vx,sp,"LR",{"variant":"spline","n_knots":nk,"C":0.1},2)
 assert len(pr)==len(vx)
 checks["spline_"+str(nk)]=True
study={"line":"geography","cohort":"preop","stage":"preop","context":"final","branch":"selected","method":"bayes","family":"LR"}
t=ensure_subset(study,p,9,2)
tr,_,va,_,ss=read_cache(t)
assert len(tr.columns)==len(va.columns)==9
assert set(tr)==set(ss["num"]+ss["cat"])
saved=joblib.load(cache_dir(t)/"imputer.joblib")
assert len(saved.cols)==9 and saved.original_.shape[1]==9
checks["real_RFE_selected9_refit_imputer_only9"]=True
atomic(OUT/"MODEL_ENGINE_FUNCTIONAL_QUALIFICATION.json",{"status":"PASS","checks":checks,"warnings":warnings_by_family,"scope":"actual real-input engineering checks; not comparative performance evidence","seconds":time.time()-start,"code_sha256":{n:sha(CODE/n) for n in ["model_engine.py","search_worker.py"]}})
print(canonical({"status":"PASS","checks":checks,"seconds":time.time()-start}))
