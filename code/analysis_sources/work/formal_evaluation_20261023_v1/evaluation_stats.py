"""Held-out fixed model and locked adaptive-policy evaluation, no predictive fitting."""
from pathlib import Path
import sys,math,hashlib
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq,minimize
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
sys.path.insert(0,str(ROOT/'work/final_evaluation_engine_20261022_v1'))
from fixed_prediction_metrics import FixedRisk,_design,group_weights
sys.path.insert(0,str(ROOT/'work/selective_information_20261021_v1'))
import policy_core
DCA=(.01,.02,.03,.05,.075,.1,.15,.2)
def calibration(y,p):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
 if not len(y) or y.min()==y.max():return {'calibration_intercept':None,'calibration_slope':None,'calibration_joint_intercept':None}
 z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
 offset=float(brentq(lambda a:np.mean(expit(z+a))-np.mean(y),-60,60))
 out={'calibration_intercept':offset,'calibration_slope':None,'calibration_joint_intercept':None}
 if z.std()<1e-10 or z[y==1].min()>=z[y==0].max() or z[y==0].min()>=z[y==1].max():return out
 mu=z.mean();sd=z.std();x=(z-mu)/sd
 def objective(b):
  eta=b[0]+b[1]*x
  return np.mean(np.logaddexp(0,eta)-y*eta)
 def jac(b):
  e=expit(b[0]+b[1]*x)-y
  return np.array([e.mean(),np.mean(e*x)])
 fit=minimize(objective,[np.log(y.mean()/(1-y.mean())),0.],jac=jac,method='BFGS',options={'gtol':1e-8,'maxiter':150})
 if np.linalg.norm(jac(fit.x),np.inf)>1e-6 or not np.isfinite(fit.x).all():return out
 q=expit(fit.x[0]+fit.x[1]*x);v=q*(1-q)
 h=np.array([[v.mean(),np.mean(v*x)],[np.mean(v*x),np.mean(v*x*x)]])
 if np.linalg.cond(h)>1e10:return out
 slope=float(fit.x[1]/sd)
 out.update(calibration_slope=slope,calibration_joint_intercept=float(fit.x[0]-slope*mu))
 return out
def counts(y,mask):
 y=np.asarray(y);a=np.asarray(mask,dtype=bool);n=len(y);e=int(y.sum());r=int(a.sum());tp=int(y[a].sum())
 return {'n':n,'events':e,'reviewed_n':r,'captured_n':tp,'missed_n':e-tp,'false_alerts_n':r-tp,
  'sensitivity':tp/e if e else None,'PPV':tp/r if r else None,
  'specificity':1-(r-tp)/(n-e) if n>e else None,
  'captured_per1000':1000*tp/n,'missed_per1000':1000*(e-tp)/n,'reviewed_per1000':1000*r/n,
  'false_alerts_per1000':1000*(r-tp)/n}
def apply_calibration(cal,p):
 q=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
 return cal.predict_proba(np.log(q/(1-q)).reshape(-1,1))[:,1]
def expanded(weights,groups,records):
 """Replicate a sampled group as a unit; suffix tie identity only, never imputation key."""
 weights=np.asarray(weights,dtype=int)
 if (weights<0).any():raise ValueError('negative multiplicity')
 g=np.asarray(groups,dtype=str);r=np.asarray(records,dtype=str)
 grank={v:i for i,v in enumerate(sorted(set(g)))}
 rrank={v:i for i,v in enumerate(sorted(set(r)))}
 idx=np.repeat(np.arange(len(g)),weights)
 gg=[];rr=[]
 for i in range(len(g)):
  for copy in range(weights[i]):
   gg.append(f'g{grank[g[i]]:08d}')
   rr.append(f'r{rrank[r[i]]:08d}|copy{copy:08d}')
 return idx,gg,rr
def route(item,base,upgrade,g,r):
 lock=item['lock'];i=lock['identity']
 mask=policy_core.upgrade_mask(base,g,r,lock['gate'],i['upgrade_fraction'],i['mode'],i['kind']=='random')
 mixed=policy_core.mix_predictions(base,upgrade,mask)
 return mixed,apply_calibration(item['calibrator'],mixed),mask
def summary(y,p,g,r,upgraded=None,with_calibration=True):
 f=FixedRisk(y,p,g,r);q=f.score()
 if with_calibration:q.update(calibration(y,p))
 q['upgraded_per1000']=None if upgraded is None else 1000*float(np.mean(upgraded))
 for fraction in [.1,.2]:
  a=f.review_counts(fraction).astype(bool);c=counts(y,a)
  for k,v in c.items():
   if k not in ['n','events']:q[f'capacity{int(100*fraction)}_{k}']=v
 for t in DCA:q[f'net_benefit_{t}']=f.net_benefit(t)['net_benefit']
 return q
def evaluate(y,predictions,groups,records,centers,policies,comparisons,replicates=2000,seed=20261017,progress=None):
 yy,gg,rr,cc,strata,models=_design(y,groups,records,centers,predictions)
 pred={k:m.p for k,m in models.items()}
 point={k:summary(yy,v,gg,rr) for k,v in pred.items()}
 for name,it in policies.items():
  raw,cal,mask=route(it,pred[it['base']],pred[it['upgrade']],gg,rr)
  point[name]=summary(yy,cal,gg,rr,mask)
  point[name+'__raw_mixture']=summary(yy,raw,gg,rr,mask)
 fields=[k for k in next(iter(point.values())) if k not in ['n','events']]
 values={name:{f:[] for f in fields} for name in point}
 differences={a+' MINUS '+b:{f:[] for f in fields} for a,b in comparisons}
 rng=np.random.default_rng(seed);proof=hashlib.sha256();zero=allpos=0
 for rep in range(replicates):
  w=group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes())
  idx,g,r=expanded(w,gg,rr);yrep=yy[idx]
  zero+=yrep.sum()==0;allpos+=yrep.sum()==len(yrep)
  scores={name:summary(yrep,p[idx],g,r) for name,p in pred.items()}
  for name,it in policies.items():
   raw,cal,mask=route(it,pred[it['base']][idx],pred[it['upgrade']][idx],g,r)
   scores[name]=summary(yrep,cal,g,r,mask)
   scores[name+'__raw_mixture']=summary(yrep,raw,g,r,mask)
  for name,q in scores.items():
   for f in fields:
    v=q.get(f)
    if v is not None and np.isfinite(v):values[name][f].append(float(v))
  for a,b in comparisons:
   for f in fields:
    va=scores[a].get(f);vb=scores[b].get(f)
    if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):differences[a+' MINUS '+b][f].append(float(va-vb))
  if progress and (rep+1)%100==0:progress(rep+1)
 def intervals(data,diff=False):
  result=[]
  for name,cols in data.items():
   for field,v in cols.items():
    if diff:
     a,b=name.split(' MINUS ');va=point[a].get(field);vb=point[b].get(field);est=va-vb if va is not None and vb is not None else None
    else:est=point[name].get(field)
    ci=np.quantile(v,[.025,.975],method='linear').tolist() if v else [None,None]
    result.append({'model':name,'metric':field,'estimate':est,'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(v),'requested_replicates':replicates})
  return result
 return {'point':point,'intervals':intervals(values),'paired_differences':intervals(differences,True),
  'receipt':{'replicates':replicates,'seed':seed,'resample_weight_sha256':proof.hexdigest(),
   'zero_event_replicates':int(zero),'all_event_replicates':int(allpos),'group_unit_within_hospital':True,
   'policy_rerouted_every_resample':True,'original_imputation_identity_unchanged':True,
   'fixed_models_no_retraining':True,'calibration_coefficients_for_evaluation_only_never_update_predictions':True,
   'calibration_intercept_definition':'offset logit risk with slope fixed at one; joint intercept/slope separately',
   'undefined_calibration_reported_per_metric':True}}
