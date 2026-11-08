"""Original-runtime fold-local model fits; serialisable request/score interface."""
from core import *
from preprocess import impute,rank
import numpy as np,pandas as pd,joblib,warnings,gc,resource
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
def task_for(study,p,columns=None):
 t=pair_task(study["line"],study["cohort"],study["stage"],study["context"],p,study["method"],"compact" if study["branch"]=="compact" else "full")
 if columns is not None:t.update(branch="selected",columns=columns)
 return t
def ensure_subset(study,p,k,threads):
 full=task_for(study,p);impute(full,threads)
 dest=cache_dir(full)/"RFE";dest.mkdir(exist_ok=True)
 with (dest/"selector.lock").open("w") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);rank(full,threads)
 paths=json.loads((dest/"path.json").read_text())["features_by_count"]
 columns=paths[str(k)]
 t=task_for(study,p,columns);impute(t,threads)
 return t
def encoder(sp,frame,params):
 nums=sp["num"];cats=sp["cat"];trs=[];spline=params.get("variant")=="spline";fallback=[]
 if not spline:
  trs.append(("num",StandardScaler(),nums))
 else:
  nk=int(params["n_knots"])
  for i,col in enumerate(nums):
   # Duplicate quantile locations are handled deterministically from training only.
   q=np.quantile(frame[col].to_numpy(),np.linspace(0,1,nk))
   if len(np.unique(q))!=nk:
    trans=StandardScaler();fallback.append(col)
   else:
    trans=Pipeline([("spline",SplineTransformer(n_knots=nk,degree=3,knots="quantile",extrapolation="linear",include_bias=False)),("scale",StandardScaler())])
   trs.append(("num"+str(i),trans,[col]))
 trs.append(("cat",OneHotEncoder(categories=[sp["cat_levels"][c] for c in cats],handle_unknown="ignore",sparse_output=False),cats))
 return ColumnTransformer(trs,remainder="drop"),fallback
def estimator(family,params,threads):
 p={k:v for k,v in params.items() if k not in ["variant","n_knots","selector_k"]}
 if family=="LR":return LogisticRegression(**p,penalty="l2",solver="lbfgs" if params.get("variant")=="spline" else "liblinear",max_iter=5000,random_state=42)
 if family=="DT":return DecisionTreeClassifier(**p,random_state=42)
 if family=="RF":return RandomForestClassifier(**p,bootstrap=True,n_jobs=threads,random_state=42)
 if family=="KNN":return KNeighborsClassifier(**p,n_jobs=threads)
 if family=="SVM":return SVC(**p,kernel="rbf",probability=True,random_state=42)
 if family=="NB":return GaussianNB(**p)
 if family=="XGB":return XGBClassifier(**p,eval_metric="logloss",tree_method="hist",device="cpu",n_jobs=threads,random_state=42)
 if family=="SGBT":return GradientBoostingClassifier(**p,random_state=42)
 if family=="NNET":return MLPClassifier(**p,activation="relu",solver="adam",max_iter=2000,early_stopping=False,random_state=42)
 raise ValueError(family)
def one_fit(train,tm,valid,sp,family,params,threads):
 enc,fallback=encoder(sp,train,params)
 pipe=Pipeline([("encode",enc),("constant",VarianceThreshold(0)),("model",estimator(family,params,threads))])
 with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):
  warnings.simplefilter("always");pipe.fit(train,tm.y);pr=pipe.predict_proba(valid)[:,1]
 if not np.isfinite(pr).all() or not np.all((pr>=0)&(pr<=1)):raise ValueError("Invalid probability")
 return pipe,pr,sorted(set(str(w.message)[:600] for w in ww)),fallback
class StudyEngine:
 def __init__(self,study,threads):
  self.study=study;self.threads=threads;self.cached={}
 def fold_data(self,params):
  k=params.get("selector_k");key=str(k)
  if key in self.cached:return self.cached[key]
  folds=[]
  for p in parts(self.study["line"],self.study["cohort"],self.study["context"]):
   t=ensure_subset(self.study,p,int(k),self.threads) if self.study["branch"]=="selected" else task_for(self.study,p)
   impute(t,self.threads)
   tx,tm,vx,vm,sp=read_cache(t)
   folds.append((tx,tm,vx,vm,sp,sha(cache_dir(t)/"COMPLETE.json")))
  self.cached[key]=folds
  return folds
 def evaluate(self,params,trial_number,dest):
  folds=self.fold_data(params);metrics=[];allmeta=[];allprob=[];notes=[];inputhash=[];fallbacks={}
  for f,(tx,tm,vx,vm,sp,h) in enumerate(folds):
   pipe,pr,warn,fallback=one_fit(tx,tm,vx,sp,self.study["family"],params,self.threads)
   metrics.append({"fold":f,"AP":float(average_precision_score(vm.y,pr)),"Brier":float(brier_score_loss(vm.y,pr)),"AUROC":float(roc_auc_score(vm.y,pr))})
   allmeta.append(vm.assign(inner_fold=f));allprob.append(pr);notes.extend(warn);inputhash.append(h);fallbacks[str(f)]=fallback
   del pipe;gc.collect()
  m=pd.concat(allmeta);prob=np.concatenate(allprob)
  assert m.excel_row.is_unique
  d=Path(dest)/"trial_predictions";d.mkdir(parents=True,exist_ok=True)
  fn=d/("trial_"+str(trial_number)+".npz")
  np.savez_compressed(fn,excel_row=m.excel_row.to_numpy(),group_hash=m.group_hash.to_numpy(dtype=str),y=m.y.to_numpy(),fold=m.inner_fold.to_numpy(),prob_raw=prob)
  return {"mean_AP":float(np.mean([v["AP"] for v in metrics])),"mean_Brier":float(np.mean([v["Brier"] for v in metrics])),"mean_AUROC":float(np.mean([v["AUROC"] for v in metrics])),"fold_metrics":metrics,"warnings":sorted(set(notes)),"spline_fallbacks":fallbacks,"fold_input_receipt_sha256":inputhash,"prediction_file":str(fn),"prediction_sha256":sha(fn),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def server():
 study=json.loads(Path(sys.argv[1]).read_text());threads=int(sys.argv[2]);dest=Path(sys.argv[3])
 engine=StudyEngine(study,threads)
 # Startup response allows orchestration to identify actual model runtime.
 import sklearn
 print(canonical({"ready":True,"sklearn":sklearn.__version__,"runtime":sys.executable}),flush=True)
 for line in sys.stdin:
  request=json.loads(line)
  if request.get("action")=="close":break
  try:
   response=engine.evaluate(request["params"],request["trial"],dest)
   response.update(ok=True,trial=request["trial"])
  except Exception:
   import traceback
   response={"ok":False,"trial":request.get("trial"),"error":traceback.format_exc()}
  print(canonical(response),flush=True)
if __name__=="__main__":server()
