"""Prespecified subgroup intervals, inherited allocation and severe-event capture."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/stratified_evaluation_20261023_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
from evaluation_stats import calibration,counts,route,expanded,FixedRisk,_design,group_weights,policy_core
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
KINDS=('hospital','age','sex','diabetes','tumor_site','egfr')
def egfr2021(age,sex,scr):
 age=np.asarray(age,float);scr=np.asarray(scr,float);sex=np.asarray(sex,dtype=str)
 out=np.full(len(age),np.nan);ok=np.isfinite(age)&np.isfinite(scr)&(age>=18)&(scr>0)&np.isin(sex,['0','1'])
 female=sex[ok]=='0';k=np.where(female,.7,.9);alpha=np.where(female,-.241,-.302);r=(scr[ok]/88.4)/k
 out[ok]=142*np.minimum(r,1)**alpha*np.maximum(r,1)**(-1.2)*.9938**age[ok]*np.where(female,1.012,1.)
 return out
def labels(kind,meta,x):
 n=len(meta)
 if kind=='hospital':return meta.center.astype(str).to_numpy()
 if kind=='age':
  a=pd.to_numeric(x.Age,errors='coerce').to_numpy();return np.where(np.isnan(a),'unknown',np.where(a<65,'age_lt65','age_ge65'))
 if kind in ['sex','diabetes','tumor_site']:
  col={'sex':'Sex','diabetes':'Diabetes','tumor_site':'TumorSite'}[kind]
  a=x[col].astype('string').fillna('unknown').to_numpy(dtype=str)
  allowed={'sex':{'0','1','unknown'},'diabetes':{'0','1','unknown'},'tumor_site':{'0','1','2','unknown'}}[kind]
  assert set(a)<=allowed
  mapping={'sex':{'0':'female','1':'male'},'diabetes':{'0':'no_diabetes','1':'diabetes'},'tumor_site':{str(i):'source_code_'+str(i) for i in range(3)}}[kind]
  return np.array([mapping.get(v,'unknown') for v in a])
 if kind=='egfr':
  a=egfr2021(pd.to_numeric(x.Age,errors='coerce'),x.Sex.astype('string').fillna('unknown'),pd.to_numeric(x.PreopCr,errors='coerce'))
  return np.where(np.isnan(a),'unknown',np.where(a<60,'egfr_lt60','egfr_ge60'))
 raise ValueError(kind)
def partition_score(y,p,groups,records,select,alerts,severe,upgraded):
 mask=np.asarray(select,dtype=bool);n=int(mask.sum());yy=y[mask]
 if not n:return {'n':0,'events':0,'severe_events':0}
 f=FixedRisk(yy,p[mask],np.asarray(groups)[mask],np.asarray(records)[mask])
 q={**f.score(),**calibration(yy,p[mask]),'severe_events':int(severe[mask].sum()),'upgraded_per1000':None if upgraded is None else float(1000*np.mean(upgraded[mask]))}
 for label,amask in alerts.items():
  a=amask[mask];r=counts(yy,a)
  for k,v in r.items():
   if k not in ['n','events']:q[label+'_'+k]=v
  sv=int(severe[mask].sum());cap=int((a&severe[mask]).sum())
  q.update({label+'_severe_captured_n':cap,label+'_severe_missed_n':sv-cap,
   label+'_severe_sensitivity':cap/sv if sv else None,
   label+'_severe_PPV':cap/int(a.sum()) if a.sum() else None,
   label+'_severe_captured_per1000':1000*cap/n})
 return q
def locked_alerts(y,p,g,r,thresholds):
 f=FixedRisk(y,p,g,r)
 out={f'capacity{int(frac*100)}':f.review_counts(frac).astype(bool) for frac in [.1,.2]}
 for name,t in thresholds.items():
  if name.startswith('training_') and isinstance(t,(float,int)):out[name]=p>=t
 return out
def evaluate(y,predictions,groups,records,centers,policies,thresholds,level,severe,comparisons,replicates=2000):
 # Canonical order shared by all models, strata and patient-group draws.
 order=np.array(sorted(range(len(y)),key=lambda i:(str(centers[i]),str(groups[i]),str(records[i]))))
 yy,gg,rr,cc,strata,models=_design(y,groups,records,centers,predictions)
 levels=np.asarray(level)[order];sv=np.asarray(severe,dtype=bool)[order];pp={k:v.p for k,v in models.items()}
 if np.any(sv&(yy==0)):raise ValueError('Severe stage without clinical AKI')
 levels_present=sorted(set(levels));points={};samples={};deltas={};rng=np.random.default_rng(20261017);proof=hashlib.sha256()
 def scores(idx,g,r,lev):
  probs={k:v[idx] for k,v in pp.items()};masks={k:None for k in pp}
  for nm,item in policies.items():
   _,probs[nm],masks[nm]=route(item,pp[item['base']][idx],pp[item['upgrade']][idx],g,r)
  out={}
  for nm,pr in probs.items():
   alert=locked_alerts(yy[idx],pr,g,r,thresholds[nm])
   for lab in levels_present:
    out[(nm,lab)]=partition_score(yy[idx],pr,g,r,lev==lab,alert,sv[idx],masks[nm])
  return out
 points=scores(np.arange(len(yy)),gg,rr,levels)
 fields=sorted(set().union(*(q.keys() for q in points.values()))-{'n','events','severe_events'})
 samples={k:{f:[] for f in fields} for k in points}
 for a,b in comparisons:
  for lab in levels_present:deltas[(a+' MINUS '+b,lab)]={f:[] for f in fields}
 zero_subgroup={lab:0 for lab in levels_present}
 for rep in range(replicates):
  w=group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes())
  idx,g,r=expanded(w,gg,rr);lev=levels[idx];current=scores(idx,g,r,lev)
  for lab in levels_present:zero_subgroup[lab]+=int(not np.any(lev==lab))
  for key,q in current.items():
   for f in fields:
    v=q.get(f)
    if v is not None and np.isfinite(v):samples[key][f].append(float(v))
  for a,b in comparisons:
   for lab in levels_present:
    for f in fields:
     va=current[(a,lab)].get(f);vb=current[(b,lab)].get(f)
     if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
      deltas[(a+' MINUS '+b,lab)][f].append(float(va-vb))
 def intervals(data,diff):
  rows=[]
  for (nm,lab),cols in data.items():
   for f,v in cols.items():
    if diff:
     a,b=nm.split(' MINUS ');va=points[(a,lab)].get(f);vb=points[(b,lab)].get(f);estimate=va-vb if va is not None and vb is not None else None
    else:estimate=points[(nm,lab)].get(f)
    ci=np.quantile(v,[.025,.975],method='linear').tolist() if v else [None,None]
    rows.append({'model':nm,'level':lab,'metric':f,'estimate':estimate,'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(v),'requested_replicates':replicates})
  return rows
 return {'point':[{'model':nm,'level':lab,**q} for (nm,lab),q in points.items()],
  'intervals':intervals(samples,False),'paired_differences':intervals(deltas,True),
  'receipt':{'replicates':replicates,'seed':20261017,'resample_weight_sha256':proof.hexdigest(),
   'empty_subgroup_replicates':zero_subgroup,'all_hospitals_sampled_before_inherited_allocation':True,
   'policy_and_review_allocation_rerun_on_full_sample':True,'no_subgroup_retraining':True,
   'clinical_AKI_risk_used_for_severe_capture':True,'no_heterogeneity_claim_from_different_significance':True}}
def load(line,cohort):
 a,b=base.gate()
 assert base.read(base.OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json')['status']=='PASS_four_heldout_partitions_fixed_models_and_locked_policies'
 dest=base.OUT/(line+'_'+cohort);acc=base.read(dest/'COMPLETE.json')
 assert acc['signature']['code_lock']==base.read(OUT/'EXECUTION_CONTRACT.json')['dependencies']['formal_engine_code_lock_sha256']
 for fn,h in acc['files'].items():assert base.sha(dest/fn)==h
 meta=pd.read_csv(base.SOURCE/(cohort+'_metadata.csv'),dtype={'group_hash':str}).set_index('excel_row',drop=False)
 meta=meta.loc[meta[line+'_partition_v2']=='validation']
 pred={};th={};lookup={};primary={};modelrefs=[]
 for ref in a['models']:
  if not ref['primary_route_winner']:continue
  lock=base.read(ref['path']);ident=lock['identity']
  if (ident['line'],ident['cohort'])!=(line,cohort):continue
  nm='_'.join(ident[k] for k in ['stage','branch','role','family'])
  folder=dest/'private_predictions'/nm;receipt=base.read(folder/'COMPLETE.json')
  assert receipt['signature']['model_lock']==ref['sha256']
  assert base.sha(folder/'PREDICTIONS.csv')==receipt['sha256']
  d=pd.read_csv(folder/'PREDICTIONS.csv',dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False)
  assert d.index.tolist()==meta.index.tolist()
  for col in ['group_hash','center','y','stage']:assert d[col].tolist()==meta[col].tolist()
  pred[nm]=d.prob_calibrated.to_numpy();th[nm]=lock['training_thresholds'];lookup[ref['path']]=nm
  primary[(ident['stage'],ident['branch'])]=nm;modelrefs.append(ref)
 pol={}
 for ref in b['policy_manifests']:
  lock=base.read(ref['path']);ident=lock['identity']
  if (ident['line'],ident['cohort'])!=(line,cohort):continue
  nm=f"policy_{ident['upgrade_fraction']}_{ident['mode']}_{ident['kind']}"
  assert base.sha(lock['calibrator']['path'])==lock['calibrator']['sha256']
  pol[nm]={'lock':lock,'calibrator':joblib.load(lock['calibrator']['path']),'base':lookup[lock['base_model']['path']],'upgrade':lookup[lock['upgrade_model']['path']]}
  th[nm]=lock['training_thresholds']
 xb=pd.read_csv(base.SOURCE/(cohort+'_preop_features.csv'),dtype={k:'string' for k in ['Sex','Diabetes','TumorSite']},float_precision='round_trip').set_index('excel_row').loc[meta.index]
 base_name=primary[('preop','compact' if cohort=='preop' else 'full')]
 up=primary[('preop' if cohort=='preop' else 'endop','full')]
 comps=[(up,base_name)]
 for nm,item in pol.items():
  comps.extend([(nm,base_name),(nm,up)])
  i=item['lock']['identity']
  if i['mode']=='capacity' and i['kind']=='selective':comps.append((nm,f"policy_{i['upgrade_fraction']}_capacity_random"))
 return meta,xb,pred,th,pol,comps
def run(line,cohort,kind):
 assert kind in KINDS
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_subgroup_allocation_and_severe_capture'
 dest=OUT/(line+'_'+cohort)/kind;dest.mkdir(parents=True,exist_ok=True)
 signature={'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),
  'final_evaluation_acceptance':base.sha(base.OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json'),'source':base.sha(base.SOURCE/'SOURCE_LOCK.json'),'line':line,'cohort':cohort,'kind':kind}
 if (dest/'COMPLETE.json').exists():
  old=base.read(dest/'COMPLETE.json');assert old['signature']==signature
  for f,h in old['files'].items():assert base.sha(dest/f)==h
  return
 m,x,p,th,pol,comparisons=load(line,cohort)
 stage=pd.to_numeric(m.stage,errors='coerce')
 if stage.isna().any():raise ValueError('Missing clinical stage requires separate severe-outcome handling')
 res=evaluate(m.y.to_numpy(),p,m.group_hash.tolist(),m.excel_row.astype(str).tolist(),m.center.astype(str).tolist(),
  pol,th,labels(kind,m,x),stage.ge(2).to_numpy(),comparisons)
 for key,fn in [('point','POINT.csv'),('intervals','INTERVALS.csv'),('paired_differences','PAIRED_DIFFERENCES.csv')]:
  pd.DataFrame(res[key]).to_csv(dest/fn,index=False)
 base.atomic(dest/'RESAMPLING_RECEIPT.json',res['receipt'])
 base.atomic(dest/'COMPLETE.json',{'status':'PASS_prespecified_stratum_intervals_and_severe_capture','signature':signature,
  'files':{f:base.sha(dest/f) for f in ['POINT.csv','INTERVALS.csv','PAIRED_DIFFERENCES.csv','RESAMPLING_RECEIPT.json']},
  'n':len(m),'events':int(m.y.sum()),'severe_events':int(stage.ge(2).sum()),'observed_subgroup_missing_separate':True})
if __name__=='__main__':
 with threadpool_limits(limits=1):run(*sys.argv[1:4])
