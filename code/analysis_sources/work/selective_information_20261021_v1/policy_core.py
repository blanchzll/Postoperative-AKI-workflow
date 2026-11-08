"""Prespecified outcome-free gate and prediction routing; no model fitting."""
import math
import numpy as np

def _prob(values,empty=True):
 p=np.asarray(values,dtype=float)
 if p.ndim!=1 or (not empty and not len(p)):raise ValueError("One-dimensional probabilities required")
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("Invalid probabilities")
 return p
def _fraction(f):
 f=float(f)
 if not math.isfinite(f) or not 0<=f<=1:raise ValueError("Invalid fraction")
 return f
def _identity(groups,keys,n):
 if len(groups)!=n or len(keys)!=n:raise ValueError("Identity length mismatch")
 ids=[(str(g),str(k)) for g,k in zip(groups,keys)]
 if len(set(ids))!=n:raise ValueError("Duplicate patient-group/record identity")
 return ids
def _logit(p,clip=1e-6):
 q=np.clip(p,clip,1-clip)
 return np.log(q)-np.log1p(-q)
def _distance(p,gate):
 c=float(gate["clip"]);t=float(gate["threshold"])
 if not math.isfinite(c) or not 0<c<.5:raise ValueError("Invalid clipping")
 _prob([t])
 return np.abs(_logit(p,c)-_logit(t,c))
def fit_gate(train_probs,fractions=(.25,.5)):
 p=_prob(train_probs,empty=False)
 gate={"threshold":float(np.quantile(p,.8,method="linear")),"clip":1e-6,"cutoffs":{}}
 d=_distance(p,gate)
 for f in fractions:
  f=_fraction(f);gate["cutoffs"][str(f)]=float(np.quantile(d,f,method="linear"))
 return gate
def upgrade_mask(probs,groups,keys,gate,fraction,mode="capacity",random=False):
 p=_prob(probs);ids=_identity(groups,keys,len(p));f=_fraction(fraction)
 if mode not in {"capacity","individual"}:raise ValueError("Unknown routing mode")
 if random and mode!="capacity":raise ValueError("Random individual routing undefined")
 d=_distance(p,gate)
 if f in (0.,1.):return np.full(len(p),bool(f),dtype=bool)
 if mode=="individual":
  cutoff=float(gate["cutoffs"][str(f)])
  if not math.isfinite(cutoff) or cutoff<0:raise ValueError("Invalid individual cutoff")
  return d<=cutoff
 n=len(p);k=math.ceil(n*f);mask=np.zeros(n,dtype=bool)
 if random:
  canonical=np.array(sorted(range(n),key=lambda i:ids[i]),dtype=int)
  order=canonical[np.random.default_rng(20261021).choice(n,k,replace=False)]
 else:order=np.array(sorted(range(n),key=lambda i:(d[i],ids[i])),dtype=int)[:k]
 mask[order]=True
 return mask
def mix_predictions(base,upgraded,mask):
 p=_prob(base);u=np.asarray(upgraded,dtype=float);m=np.asarray(mask)
 if u.shape!=p.shape or m.shape!=p.shape or m.dtype!=np.dtype(bool):
  raise ValueError("Prediction/mask shape or dtype mismatch")
 _prob(u[m])
 q=p.copy();q[m]=u[m]
 return q
def alert_mask(probs,groups,keys,fraction):
 p=_prob(probs);ids=_identity(groups,keys,len(p));f=_fraction(fraction)
 order=sorted(range(len(p)),key=lambda i:(-p[i],ids[i]))
 mask=np.zeros(len(p),dtype=bool);mask[order[:math.ceil(len(p)*f)]]=True
 return mask
def assert_disjoint(training_groups,assessment_groups):
 if set(map(str,training_groups))&set(map(str,assessment_groups)):
  raise ValueError("Patient-group overlap")
