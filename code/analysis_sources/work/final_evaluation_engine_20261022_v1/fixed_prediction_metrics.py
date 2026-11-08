"""Fixed-risk evaluation only. No file access, selection, fitting, or clinical scoring entrypoint."""
import math, hashlib, json
from collections import defaultdict
import numpy as np

def _ids(values):
    result=[str(v) for v in values]
    if any(v is None or not s.strip() or s.strip().lower() in {'nan','none','<na>'} for v,s in zip(values,result)):
        raise ValueError('Missing patient, record, or hospital identity')
    return result

def _weights(w,n):
    w=np.ones(n,dtype=np.int64) if w is None else np.asarray(w)
    if w.shape!=(n,) or not np.isfinite(w).all() or (w<0).any() or (w!=np.floor(w)).any():
        raise ValueError('Non-negative integer record multiplicities required')
    w=w.astype(np.int64)
    if not w.sum():raise ValueError('Empty resampled population')
    return w

class FixedRisk:
    def __init__(self,y,p,groups,records):
        self.y=np.asarray(y,dtype=float);self.p=np.asarray(p,dtype=float)
        n=len(self.y)
        if self.y.shape!=(n,) or self.p.shape!=(n,) or not n:
            raise ValueError('Nonempty aligned one-dimensional predictions required')
        if not np.isfinite(self.y).all() or not np.isin(self.y,[0,1]).all():
            raise ValueError('Binary observed outcomes required')
        if not np.isfinite(self.p).all() or ((self.p<0)|(self.p>1)).any():
            raise ValueError('Invalid probabilities')
        if len(groups)!=n or len(records)!=n:raise ValueError('Identity length mismatch')
        self.ids=list(zip(_ids(groups),_ids(records)))
        if len(set(self.ids))!=n:raise ValueError('Duplicate patient/record identity')
        self.order=np.array(sorted(range(n),key=lambda i:(-self.p[i],self.ids[i])),dtype=int)
        s=self.p[self.order]
        self.starts=np.flatnonzero(np.r_[True,s[1:]!=s[:-1]])
    def review_counts(self,fraction,weights=None):
        f=float(fraction)
        if not math.isfinite(f) or not 0<=f<=1:raise ValueError('Invalid review fraction')
        w=_weights(weights,len(self.y));k=math.ceil(int(w.sum())*f)
        ws=w[self.order];before=np.r_[0,np.cumsum(ws)[:-1]]
        chosen=np.minimum(ws,np.maximum(0,k-before))
        out=np.empty_like(chosen);out[self.order]=chosen
        if int(out.sum())!=k:raise AssertionError('Capacity allocation mismatch')
        return out
    def score(self,weights=None):
        w=_weights(weights,len(self.y));n=int(w.sum());e=float(w@self.y);neg=n-e
        ws=w[self.order];ys=self.y[self.order]
        pos=np.add.reduceat(ws*ys,self.starts)
        non=np.add.reduceat(ws*(1-ys),self.starts)
        tp=np.cumsum(pos);fp=np.cumsum(non)
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
        ap=float(np.sum(precision*pos)/e) if e else None
        previous_tp=np.r_[0,tp[:-1]]
        auc=float(np.sum((previous_tp+tp)*non)*.5/(e*neg)) if e and neg else None
        expected=float(w@self.p)
        return {'n':n,'events':e,'event_rate':e/n,'AUROC':auc,'AP':ap,
                'Brier':float(w@((self.y-self.p)**2)/n),
                'observed_expected':e/expected if expected>0 else None}
    def capacity(self,fraction,weights=None,summary=None):
        w=_weights(weights,len(self.y));r=self.score(w) if summary is None else dict(summary)
        a=self.review_counts(fraction,w);reviewed=int(a.sum());tp=float(a@self.y)
        e=r['events'];neg=r['n']-e;fp=reviewed-tp;fn=e-tp;n=r['n']
        r.update({'review_fraction':float(fraction),'reviewed_n':reviewed,'captured_n':tp,
                  'false_alerts_n':fp,'missed_n':fn,
                  'sensitivity':tp/e if e else None,'specificity':1-fp/neg if neg else None,
                  'PPV':tp/reviewed if reviewed else None,
                  'reviewed_per1000':1000*reviewed/n,'captured_per1000':1000*tp/n,
                  'missed_per1000':1000*fn/n,'false_alerts_per1000':1000*fp/n})
        return r
    def net_benefit(self,threshold,weights=None):
        t=float(threshold)
        if not math.isfinite(t) or not 0<t<1:raise ValueError('Threshold must be in (0,1)')
        w=_weights(weights,len(self.y));n=int(w.sum());a=self.p>=t
        tp=float(w@(a*self.y));fp=float(w@(a*(1-self.y)));e=float(w@self.y)
        return {'risk_threshold':t,'net_benefit':tp/n-fp/n*t/(1-t),
                'treat_all_net_benefit':e/n-(n-e)/n*t/(1-t),'treat_none_net_benefit':0.}

def _design(y,groups,records,centers,predictions):
    n=len(y)
    if any(len(a)!=n for a in [groups,records,centers]):raise ValueError('Metadata lengths differ')
    if not predictions:raise ValueError('No predictions')
    if any(np.asarray(p).shape!=(n,) for p in predictions.values()):
        raise ValueError('Prediction lengths or dimensions differ from metadata')
    groups=_ids(groups);records=_ids(records);centers=_ids(centers)
    order=np.array(sorted(range(n),key=lambda i:(centers[i],groups[i],records[i])),dtype=int)
    yy=np.asarray(y)[order];gg=np.asarray(groups,dtype=str)[order]
    rr=np.asarray(records,dtype=str)[order];cc=np.asarray(centers,dtype=str)[order]
    all_centers=defaultdict(set)
    for g,c in zip(gg,cc):all_centers[g].add(c)
    if any(len(cs)!=1 for cs in all_centers.values()):
        raise ValueError('One patient group occurs across hospital strata')
    strata=[]
    for c in sorted(set(cc)):
        groups_here=sorted(set(gg[cc==c]))
        strata.append([np.flatnonzero((cc==c)&(gg==g)) for g in groups_here])
    models={name:FixedRisk(yy,np.asarray(p)[order],gg,rr) for name,p in predictions.items()}
    return yy,gg,rr,cc,strata,models

def group_weights(strata,n,rng):
    w=np.zeros(n,dtype=np.int64)
    for groups in strata:
        counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
        for idx,k in zip(groups,counts):w[idx]=k
    return w

def bootstrap_fixed_predictions(y,predictions,groups,records,centers,reference,
                                fractions=(.1,.2),replicates=2000,seed=20261017):
    """Paired hospital-stratified patient-group bootstrap conditional on fixed risk scores.
    Batch-selective upgrade strategies must reapply their locked routing per resample
    in a separately qualified adapter; passing a frozen mixed-risk vector estimates
    a different conditional-allocation target and must not be called strategy validation.
    """
    if reference not in predictions:raise ValueError('Reference absent')
    if not isinstance(replicates,int) or replicates<1:raise ValueError('Positive replicate count required')
    fractions=tuple(float(f) for f in fractions)
    if not fractions or any(not math.isfinite(f) or not 0<=f<=1 for f in fractions):
        raise ValueError('Nonempty valid review fractions required')
    if len(set(fractions))!=len(fractions):raise ValueError('Duplicate fractions')
    yy,gg,rr,cc,strata,models=_design(y,groups,records,centers,predictions)
    point={(name,float(f)):m.capacity(f) for name,m in models.items() for f in fractions}
    fields=['AUROC','AP','Brier','observed_expected','sensitivity','specificity','PPV',
            'captured_per1000','missed_per1000','false_alerts_per1000','reviewed_per1000']
    samples={key:{metric:[] for metric in fields} for key in point}
    deltas={key:{metric:[] for metric in fields} for key in point if key[0]!=reference}
    rng=np.random.default_rng(seed);zero=one=0;proof=hashlib.sha256()
    for _ in range(replicates):
        w=group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes())
        events=float(w@yy);zero+=events==0;one+=events==w.sum()
        scores={name:m.score(w) for name,m in models.items()}
        current={(name,float(f)):m.capacity(f,w,scores[name]) for name,m in models.items() for f in fractions}
        for key,r in current.items():
            for metric in fields:
                v=r[metric]
                if v is not None and np.isfinite(v):samples[key][metric].append(float(v))
                if key[0]!=reference:
                    b=current[(reference,key[1])][metric]
                    if v is not None and b is not None and np.isfinite(v) and np.isfinite(b):
                        deltas[key][metric].append(float(v-b))
    def intervals(items,difference=False):
        out=[]
        for (name,f),by_metric in items.items():
            for metric,values in by_metric.items():
                ci=np.quantile(values,[.025,.975],method='linear').tolist() if values else [None,None]
                v=point[(name,f)][metric]
                if difference:
                    b=point[(reference,f)][metric]
                    v=v-b if v is not None and b is not None else None
                out.append({'model':name,'reference':reference if difference else None,
                    'review_fraction':f,'metric':metric,'estimate':v,
                    'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(values),
                    'requested_replicates':replicates})
        return out
    return {'point':[{'model':k[0],**v} for k,v in point.items()],
            'intervals':intervals(samples),'paired_differences':intervals(deltas,True),
            'receipt':{'requested_replicates':replicates,'seed':seed,
                'unit':'patient-group sampled within hospital, all its records share multiplicity',
                'resample_weight_sha256':proof.hexdigest(),'zero_event_replicates':int(zero),
                'all_event_replicates':int(one),'per_metric_missingness_retained':True,
                'intervals_conditional_on_metric_defined':True,
                'conditioned_on_fixed_predictions':True,'independent_retraining':False,
                'batch_selective_policy_adapter_qualified':False}}
