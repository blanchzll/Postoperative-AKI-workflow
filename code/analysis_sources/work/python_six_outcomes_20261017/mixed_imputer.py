"""Python mixed-type conditional imputation. Outcome never enters the imputer."""
import numpy as np,pandas as pd
from sklearn.linear_model import BayesianRidge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from scipy.stats import truncnorm
from donor_helpers import pmm_donors,mixed_knn_transform
METHODS=["mean_mode","knn","cart","random_forest","bayes","pmm"]
class MixedImputer:
    def __init__(self,spec,method,seed=42,iterations=5,trees=200,neighbors=5,threads=1):
        if method not in METHODS:raise ValueError(method)
        self.spec=spec;self.method=method;self.seed=seed;self.iterations=iterations
        self.trees=trees;self.neighbors=neighbors;self.threads=threads
        self.cols=spec["num"]+spec["cat"];self.nnum=len(spec["num"])
    def _array(self,frame):
        if list(frame.columns)!=self.cols:frame=frame[self.cols]
        a=np.empty((len(frame),len(self.cols)))
        for j,c in enumerate(self.cols):
            if j<self.nnum:a[:,j]=pd.to_numeric(frame[c],errors="raise")
            else:
                mapping={v:i for i,v in enumerate(self.spec["cat_levels"][c])}
                s=frame[c].astype("string")
                if not set(s.dropna())<=set(mapping):raise ValueError("Invalid category:"+c)
                a[:,j]=s.map(mapping).to_numpy(dtype=float,na_value=np.nan)
        if np.isinf(a).any():raise ValueError("Infinite predictor")
        return a
    def _frame(self,a,index):
        out=pd.DataFrame(index=index)
        for j,c in enumerate(self.cols):
            if j<self.nnum:out[c]=a[:,j]
            else:
                levels=self.spec["cat_levels"][c];v=a[:,j]
                if not np.all(v==np.floor(v)) or np.any(v<0) or np.any(v>=len(levels)):raise ValueError("Invalid imputed category:"+c)
                out[c]=[levels[int(x)] for x in v]
        return out
    def _initialize(self,a):
        out=a.copy();r,c=np.where(np.isnan(out));out[r,c]=self.fill_[c]
        return out
    def _design(self,a,target):
        columns=[]
        for j,c in enumerate(self.cols):
            if j==target:continue
            if j<self.nnum:columns.append((a[:,j]-self.center_[j])/self.scale_[j])
            else:
                # Static clinical codebook, independent of validation frequencies.
                for k in range(1,len(self.spec["cat_levels"][c])):columns.append((a[:,j]==k).astype(float))
        return np.column_stack(columns) if columns else np.ones((len(a),1))
    def _fit_conditional(self,z,raw,j,step):
        mask=~np.isnan(raw[:,j]);y=raw[mask,j];cat=j>=self.nnum
        if not len(y):return {"kind":"constant","value":self.fill_[j]}
        if len(np.unique(y))==1:return {"kind":"constant","value":float(y[0])}
        x=self._design(z,j);xx=x[mask];seed=self.seed+1009*step+37*j
        if self.method in ["bayes","pmm"]:
            est=LogisticRegression(C=1,max_iter=2000,random_state=seed) if cat else BayesianRidge()
        elif self.method=="random_forest":
            cls=RandomForestClassifier if cat else RandomForestRegressor
            est=cls(n_estimators=self.trees,min_samples_leaf=2,max_features=1.0,n_jobs=self.threads,random_state=seed)
        elif self.method=="cart":
            cls=DecisionTreeClassifier if cat else DecisionTreeRegressor
            est=cls(min_samples_leaf=min(5,max(1,len(y)//2)),random_state=seed)
        else:raise ValueError(self.method)
        est.fit(xx,y);record={"kind":"model","model":est,"categorical":cat}
        if self.method=="pmm" and not cat:
            record.update(donor_prediction=est.predict(xx),donor_values=y.copy(),xmean=xx.mean(axis=0),ymean=float(y.mean()))
        if self.method=="cart":
            leaves=est.apply(xx)
            record["leaf_donors"]={int(k):y[leaves==k].copy() for k in np.unique(leaves)}
        return record
    def _predict(self,record,x,j,rng):
        if record["kind"]=="constant":return np.full(len(x),record["value"])
        if not len(x):return np.empty(0)
        model=record["model"]
        if self.method=="cart":
            return np.asarray([rng.choice(record["leaf_donors"][int(k)]) for k in model.apply(x)])
        if record["categorical"]:
            if self.method in ["bayes","pmm"]:
                probs=model.predict_proba(x)
                return np.asarray([rng.choice(model.classes_,p=p/p.sum()) for p in probs])
            return model.predict(x)
        if self.method=="pmm":
            beta=rng.multivariate_normal(model.coef_,model.sigma_,check_valid="raise",tol=1e-7)
            intercept=record["ymean"]-record["xmean"]@beta
            predicted=x@beta+intercept
            return pmm_donors(record["donor_prediction"],predicted,record["donor_values"],self.neighbors,rng)
        if self.method=="bayes":
            mean,std=model.predict(x,return_std=True);std=np.maximum(std,1e-10)
            if self.cols[j]=="WeightLoss":return rng.normal(mean,std)
            # Numeric age, size, lab and hospital-day variables have nonnegative support.
            return truncnorm.rvs((0-mean)/std,np.inf,loc=mean,scale=std,random_state=rng)
        return model.predict(x)
    def fit(self,frame):
        raw=self._array(frame);self.original_=raw.copy();p=raw.shape[1]
        self.fill_=np.zeros(p);self.center_=np.zeros(p);self.scale_=np.ones(p)
        self.all_missing_=[]
        for j,c in enumerate(self.cols):
            y=raw[~np.isnan(raw[:,j]),j]
            if not len(y):self.all_missing_.append(c);continue
            if j<self.nnum:
                self.fill_[j]=y.mean();self.center_[j]=y.mean();self.scale_[j]=y.std() or 1
            else:
                values,counts=np.unique(y,return_counts=True);self.fill_[j]=values[np.argmax(counts)]
        z=self._initialize(raw);self.initial_filled_=z.copy();self.models_={};self.history_=[]
        if self.method=="knn":
            z=mixed_knn_transform(raw,self.initial_filled_,raw,list(range(self.nnum)),list(range(self.nnum,p)),self.neighbors)
        elif self.method!="mean_mode":
            rng=np.random.default_rng(self.seed)
            for step in range(self.iterations):
                before=z.copy()
                for j in range(p):
                    record=self._fit_conditional(z,raw,j,step);self.models_[j]=record
                    missing=np.isnan(raw[:,j])
                    if missing.any():z[missing,j]=self._predict(record,self._design(z,j)[missing],j,rng)
                mn=np.isnan(raw[:,:self.nnum]);mc=np.isnan(raw[:,self.nnum:])
                delta=np.abs((z[:,:self.nnum]-before[:,:self.nnum])/self.scale_[:self.nnum])
                self.history_.append({"sweep":step+1,"mean_absolute_standardized_numeric_change":float(delta[mn].mean()) if mn.any() else 0.,"categorical_changed_fraction":float((z[:,self.nnum:]!=before[:,self.nnum:])[mc].mean()) if mc.any() else 0.})
        if not np.isfinite(z).all():raise ValueError("Nonfinite training completion")
        self.train_filled_=z
        return self
    def fit_transform(self,frame):
        self.fit(frame);return self._frame(self.train_filled_,frame.index)
    def transform(self,frame):
        raw=self._array(frame);z=self._initialize(raw)
        if self.method=="knn":
            z=mixed_knn_transform(self.original_,self.initial_filled_,raw,list(range(self.nnum)),list(range(self.nnum,len(self.cols))),self.neighbors)
        elif self.method!="mean_mode":
            rng=np.random.default_rng(self.seed+1000003)
            for step in range(self.iterations):
                for j in range(len(self.cols)):
                    missing=np.isnan(raw[:,j])
                    if missing.any():z[missing,j]=self._predict(self.models_[j],self._design(z,j)[missing],j,rng)
        if not np.isfinite(z).all():raise ValueError("Nonfinite validation completion")
        return self._frame(z,frame.index)

