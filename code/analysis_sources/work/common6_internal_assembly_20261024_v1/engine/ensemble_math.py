"""Index-safe ensemble estimation, independent of clinical data loading."""
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
FAMILIES=["LR","DT","RF","KNN","SVM","NB","XGB","SGBT","NNET"]
def score(y,p):
 if not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):raise ValueError("Invalid predictions")
 return {"AP":float(average_precision_score(y,p)),"Brier":float(brier_score_loss(y,p)),"AUROC":float(roc_auc_score(y,p))}
def matrix(frames):
 result=None;ids=None
 for f in FAMILIES:
  x=frames[f].set_index("excel_row",drop=False)
  if not x.index.is_unique:raise ValueError("Duplicate patient row")
  if ids is None:ids=x.index.tolist();result=x[["excel_row","group_hash","center","y"]].copy()
  if set(x.index)!=set(ids):raise ValueError("Different prediction populations")
  x=x.loc[ids]
  if not np.array_equal(result.y.to_numpy(),x.y.to_numpy()) or not np.array_equal(result.group_hash.to_numpy(),x.group_hash.to_numpy()):raise ValueError("Metadata mismatch")
  result[f]=x.prob_raw
 return result.reset_index(drop=True)
def rank_families(m):
 rows=[]
 for f in FAMILIES:
  scores=[score(g.y,g[f]) for _,g in m.groupby("fold",sort=True)]
  rows.append({"family":f,"AP":float(np.mean([r["AP"] for r in scores])),"Brier":float(np.mean([r["Brier"] for r in scores]))})
 # Canonical family id resolves ties; same AP/Brier tolerance as searches.
 order=[]
 while rows:
  ap=max(r["AP"] for r in rows);eligible=[r for r in rows if r["AP"]>=ap-1e-12]
  br=min(r["Brier"] for r in eligible);winner=min([r for r in eligible if r["Brier"]<=br+1e-12],key=lambda r:r["family"])
  order.append(winner["family"]);rows.remove(winner)
 return order
def infer(train,valid,kind,k,C=None,ranking=None):
 if set(train.group_hash)&set(valid.group_hash):raise ValueError("Meta learner patient leakage")
 members=(ranking or rank_families(train))[:k]
 if kind=="Voting":return valid[members].mean(axis=1).to_numpy(),None,members
 with threadpool_limits(limits=1):
  model=LogisticRegression(C=C,penalty="l2",solver="lbfgs",max_iter=5000,random_state=42)
  model.fit(train[members],train.y)
  p=model.predict_proba(valid[members])[:,1]
 return p,model,members
def calibrator(train):
 p=np.clip(train.prob_raw.to_numpy(),1e-6,1-1e-6)
 model=LogisticRegression(C=1e6,penalty="l2",solver="lbfgs",max_iter=5000,random_state=42)
 with threadpool_limits(limits=1):model.fit(np.log(p/(1-p)).reshape(-1,1),train.y)
 return model
def apply_calibration(model,p):
 p=np.clip(np.asarray(p),1e-6,1-1e-6)
 return model.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
