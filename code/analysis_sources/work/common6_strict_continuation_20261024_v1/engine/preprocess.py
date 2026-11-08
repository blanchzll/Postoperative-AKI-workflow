"""Finite owned imputation/RF-RFE tasks. Outcome excluded from every imputer."""
from core import *
import argparse,resource,warnings,shutil
import pandas as pd,numpy as np,joblib
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/"work/v2_stable_imputation_20261017"))
from stable_imputer import StableMixedImputer
def imported_cache(t,d,sig):
 # common6 is a new input boundary: never import any historical 9/27/38 fitted state.
 return None
def impute(t,threads):
 sig=cache_signature(t);d=cache_dir(t);d.mkdir(parents=True,exist_ok=True)
 with (d/"task.lock").open("w") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if ready(d,sig):return {"reused":True,"directory":str(d)}
  start=time.time();extra=imported_cache(t,d,sig)
  if extra is None:
   sp=sig["spec"];tx,tm,vx,vm=frame_and_meta(t,sp)
   with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):
    warnings.simplefilter("always")
    imp=StableMixedImputer(sp,t["method"],seed=42,iterations=5,trees=200,neighbors=5,threads=threads)
    ft=imp.fit_transform(tx,keys(tm));fv=imp.transform(vx,keys(vm))
   for raw,filled in [(tx,ft),(vx,fv)]:
    assert not filled.isna().any().any()
    for c in sp["num"]+sp["cat"]:
     mask=raw[c].notna()
     if c in sp["num"]:np.testing.assert_array_equal(raw.loc[mask,c],filled.loc[mask,c])
     else:assert raw.loc[mask,c].astype(str).tolist()==filled.loc[mask,c].astype(str).tolist()
   # Real missing cases: batch, reverse, singleton and persisted inference.
   check=vx.loc[vx.isna().any(axis=1)].head(3);cm=vm.loc[check.index]
   joblib.dump(imp,d/"imputer.joblib",compress=3)
   if len(check):
    batch=imp.transform(check,keys(cm));other=imp.transform(check.iloc[::-1],keys(cm.iloc[::-1])).iloc[::-1]
    solo=pd.concat([imp.transform(check.iloc[i:i+1],keys(cm.iloc[i:i+1])) for i in range(len(check))])
    loaded=joblib.load(d/"imputer.joblib").transform(check,keys(cm))
    for z in [other,solo,loaded]:
     for c in sp["num"]:np.testing.assert_allclose(batch[c],z[c],rtol=1e-12,atol=1e-12)
     for c in sp["cat"]:assert batch[c].tolist()==z[c].tolist()
   for name,df in [("train.csv",ft),("assessment.csv",fv)]:df.to_csv(d/name,index_label="excel_row")
   extra={"observed_values_unchanged":True,"case_stability_checked":True,"input_columns":list(tx),"warnings":sorted(set(str(w.message)[:500] for w in ww)),"all_missing_training_fields":imp.all_missing_}
  receipt={"signature":sig,"files":{f:sha(d/f) for f in ["train.csv","assessment.csv","imputer.joblib"]},"task":{k:v for k,v in t.items() if not k.endswith("_rows")},"train_n":len(t["train_rows"]),"assessment_n":len(t["assessment_rows"]),"seconds":time.time()-start,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"no_outcome_in_imputer":True,"no_final_evaluation_used":True,**extra}
  atomic(d/"COMPLETE.json",receipt)
  return {"directory":str(d),"seconds":receipt["seconds"],"peak_rss_bytes":receipt["peak_rss_bytes"]}
def rank(t,threads):
 # Supervised RF-RFE fits only this pair's training patients. Validation y never read.
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import OneHotEncoder,StandardScaler
 from sklearn.feature_selection import VarianceThreshold
 tx,tm,_,_,sp=read_cache(t)
 dest=cache_dir(t)/"RFE";dest.mkdir(exist_ok=True)
 sig={"imputation_receipt":sha(cache_dir(t)/"COMPLETE.json"),"training_y":digest(tm.y.tolist()),"training_groups":digest(tm.group_hash.tolist()),"selector_code":sha(__file__),"rf":{"trees":200,"leaf":5,"max_features":"sqrt","seed":42}}
 if ready(dest,sig):return dest
 cols=list(tx);paths={};start=time.time()
 with threadpool_limits(limits=1):
  while True:
   paths[str(len(cols))]=list(cols)
   if len(cols)==3:break
   nums=[c for c in sp["num"] if c in cols];cats=[c for c in sp["cat"] if c in cols]
   enc=ColumnTransformer([("num",StandardScaler(),nums),("cat",OneHotEncoder(categories=[sp["cat_levels"][c] for c in cats],handle_unknown="ignore",sparse_output=False),cats)])
   xx=enc.fit_transform(tx[cols]);keep=VarianceThreshold().fit(xx)
   model=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features="sqrt",random_state=42,n_jobs=threads)
   model.fit(keep.transform(xx),tm.y)
   owner=np.asarray(nums+[c for c in cats for _ in sp["cat_levels"][c]])[keep.get_support()]
   importance={c:float(model.feature_importances_[owner==c].sum()) for c in cols}
   cols.remove(min(cols,key=lambda c:(importance[c],c)))
 atomic(dest/"path.json",{"features_by_count":paths,"training_only":True})
 atomic(dest/"COMPLETE.json",{"signature":sig,"files":{"path.json":sha(dest/"path.json")},"seconds":time.time()-start,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
 return dest
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("task");a.add_argument("--threads",type=int,default=1);args=a.parse_args()
 t=json.loads(Path(args.task).read_text())
 with threadpool_limits(limits=1):
  result=impute(t,args.threads) if t["kind"]=="impute" else {"ranking":str(rank(t,args.threads))}
 print(canonical(result),flush=True)
