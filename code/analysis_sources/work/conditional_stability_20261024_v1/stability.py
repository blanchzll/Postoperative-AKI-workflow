"""Conditional fixed-choice patient-group bootstrap; development reference only."""
from pathlib import Path
import sys,os,json,hashlib,datetime,gc,time,resource,warnings
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/conditional_stability_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
os.environ['AKI_SCOPE_OUT']=str(ROOT/'outputs/ensemble_calibration_20261020_v1')
rt=base.module('stability_original_runtime',ROOT/'work/model_lock_20261022_v1/runtime.py')
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from ensemble_math import calibrator,apply_calibration
class InvalidSample(Exception):pass
def draw_indices(meta,rep):
 groups=meta.group_hash.astype(str).to_numpy();u=np.array(sorted(set(groups)));rng=np.random.default_rng(20261023)
 for _ in range(rep+1):draw=rng.choice(u,len(u),replace=True)
 mapping={g:np.flatnonzero(groups==g) for g in u}
 idx=np.concatenate([mapping[g] for g in draw])
 return idx,draw
def grouped_folds(meta):
 if meta.y.nunique()!=2:raise InvalidSample('single_class_bootstrap')
 for label in [0,1]:
  if meta.loc[meta.y==label,'group_hash'].nunique()<3:raise InvalidSample('fewer_than_three_groups_for_a_class')
 result=[]
 with warnings.catch_warnings(record=True):
  for tr,va in StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(np.zeros(len(meta)),meta.y,meta.group_hash):
   if meta.iloc[tr].y.nunique()!=2 or meta.iloc[va].y.nunique()!=2:raise InvalidSample('single_class_inner_fold')
   assert set(meta.iloc[tr].group_hash).isdisjoint(meta.iloc[va].group_hash)
   result.append((tr,va))
 coverage=np.concatenate([v for t,v in result]);assert sorted(coverage)==list(range(len(meta)))
 return result
def fill_training(raw,meta,method,sp):
 # Fit with every sampled copy; transformation once per original encounter preserves
 # identical stochastic completion for repeated copies. Copy IDs never enter predictors/folds.
 original=rt.keys(meta);count={};fitkeys=[]
 for k in original:
  j=count.get(k,0);count[k]=j+1;fitkeys.append(k+'|bootstrap_copy_'+str(j).zfill(6))
 imp=rt.StableMixedImputer(sp,method,seed=42,iterations=5,trees=200,neighbors=5,threads=1)
 imp.fit(raw,fitkeys)
 first={};inverse=[]
 for i,k in enumerate(original):
  if k not in first:first[k]=len(first)
  inverse.append(first[k])
 unique_pos=np.flatnonzero(~pd.Series(original).duplicated().to_numpy())
 unique=raw.iloc[unique_pos];filled_unique=imp.transform(unique,[original[i] for i in unique_pos])
 filled=filled_unique.iloc[inverse].copy();filled.index=raw.index
 assert len(filled)==len(raw) and not filled.isna().any().any()
 return imp,filled
def source_data(lock):
 ref=next(iter(lock['base_components'].values()))
 t=base.read(Path(ref['path']).parent/'COMPLETE.json')['signature']['task']
 sp=rt.source_spec(lock['identity']['stage'],lock['input_columns'])
 x,m=rt.data(t,sp)
 order=sorted(range(len(m)),key=lambda i:(str(m.group_hash.iloc[i]),str(m.excel_row.iloc[i])))
 return x.iloc[order].reset_index(drop=True),m.iloc[order].reset_index(drop=True)
def definitions(lock):
 out={}
 for fam in lock['members']:
  rec=lock['base_components'][fam];assert base.sha(rec['path'])==rec['sha256']
  bundle=joblib.load(rec['path'])
  out[fam]={'method':bundle['method'],'params':bundle['params'],'columns':bundle['input_columns'],'spec':bundle['input_spec']}
 assert lock['family'] in rt.FAMILIES+['Voting'], 'A newly selected Stacking route requires the same full upstream meta-refitting contract before this stage can run'
 return out
def fit_flow(lock,defs,x,m,valid,vm,dest,signature,save_bundle=False):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():
  c=base.read(dest/'COMPLETE.json');assert c['signature']==signature
  for f,h in c['files'].items():assert base.sha(dest/f)==h
  return np.load(dest/'raw.npy')
 filled={};preds={};bundles={};notes={}
 for fam in lock['members']:
  d=defs[fam];cols=d['columns'];sp=d['spec']
  key=rt.digest({'method':d['method'],'columns':cols,'spec':sp})
  if key not in filled:
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always')
    imp,tx=fill_training(x[cols],m,d['method'],sp)
    vx=imp.transform(valid[cols],rt.keys(vm))
   filled[key]=(imp,tx,vx);notes['imputer_'+key]=sorted(set(str(w.message)[:400] for w in ww))
  imp,tx,vx=filled[key]
  pipe,p,w,fb=rt.one_fit(tx,m,vx,sp,fam,d['params'],1)
  preds[fam]=np.asarray(p,dtype=np.float64);notes[fam]={'warnings':w,'spline_fallback':fb}
  if save_bundle:bundles[fam]={'imputer':imp,'pipeline':pipe,'columns':cols,'spec':sp}
  else:del pipe
 raw=np.column_stack([preds[f] for f in lock['members']]).mean(axis=1) if lock['family']=='Voting' else preds[lock['family']]
 assert np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1))
 np.save(dest/'raw.npy',raw);files={'raw.npy':base.sha(dest/'raw.npy')}
 if save_bundle:
  joblib.dump(bundles,dest/'private_components.joblib',compress=3);files['private_components.joblib']=base.sha(dest/'private_components.joblib')
  loaded=joblib.load(dest/'private_components.joblib');probe=valid.iloc[:3];pm=vm.iloc[:3]
  def infer_probe(frame,metadata):
   pp=[]
   for fam in lock['members']:
    b=loaded[fam];ff=b['imputer'].transform(frame[b['columns']],rt.keys(metadata))
    pp.append(b['pipeline'].predict_proba(ff)[:,1])
   return np.column_stack(pp).mean(axis=1) if lock['family']=='Voting' else np.asarray(pp[0])
  pp=infer_probe(probe,pm);rr=infer_probe(probe.iloc[::-1],pm.iloc[::-1])[::-1]
  solo=np.concatenate([infer_probe(probe.iloc[i:i+1],pm.iloc[i:i+1]) for i in range(len(probe))])
  strict=all(np.allclose(z,raw[:len(probe)],rtol=1e-12,atol=1e-12) for z in [pp,rr,solo])
  # Do not relax the original tolerance or erase its failure. Bootstrap artifacts
  # are batch-reference diagnostics, not newly qualified deployment models.
  allp=[];reloaded_filled={}
  for fam in lock['members']:
   b=loaded[fam];key=id(b['imputer'])
   if key not in reloaded_filled:reloaded_filled[key]=b['imputer'].transform(valid[b['columns']],rt.keys(vm))
   allp.append(b['pipeline'].predict_proba(reloaded_filled[key])[:,1])
  batch=np.column_stack(allp).mean(axis=1) if lock['family']=='Voting' else np.asarray(allp[0])
  np.testing.assert_allclose(batch,raw,rtol=1e-12,atol=1e-12)
  base.atomic(dest/'INFERENCE_DIAGNOSTIC.json',{'strict_single_batch_reorder_tolerance_pass':strict,'rtol':1e-12,'atol':1e-12,'max_probe_abs_difference':float(max(np.max(abs(z-raw[:len(probe)])) for z in [pp,rr,solo])),'same_full_batch_reload_pass':True,'not_qualified_for_individual_deployment':not strict,'diagnostic_reference_predictions_unchanged':True})
  files['INFERENCE_DIAGNOSTIC.json']=base.sha(dest/'INFERENCE_DIAGNOSTIC.json')
  del loaded,reloaded_filled
 base.atomic(dest/'COMPLETE.json',{'signature':signature,'files':files,'notes':notes,'input_boundaries':{f:d['columns'] for f,d in defs.items()}})
 del bundles,filled;gc.collect()
 return raw
def top(p,m,f):
 order=sorted(range(len(p)),key=lambda i:(-float(p[i]),str(m.group_hash.iloc[i]),str(m.excel_row.iloc[i])))
 return set(order[:int(np.ceil(f*len(p)))])
def reference(route,lock,x,m):
 d=OUT/'references'/route;d.mkdir(parents=True,exist_ok=True)
 import fcntl
 with (d/'reference.lock').open('a') as h:
  fcntl.flock(h,fcntl.LOCK_EX)
  if not (d/'COMPLETE.json').exists():
   raw,cal=rt.route_prediction(lock,x,rt.keys(m))
   np.savez_compressed(d/'predictions.npz',raw=np.asarray(raw,dtype=float),cal=cal,excel_row=m.excel_row.to_numpy())
   base.atomic(d/'COMPLETE.json',{'model_sha256':rt.digest(lock),'file_sha256':base.sha(d/'predictions.npz'),'development_only':True})
  c=base.read(d/'COMPLETE.json');assert c['model_sha256']==rt.digest(lock) and c['file_sha256']==base.sha(d/'predictions.npz')
  z=np.load(d/'predictions.npz');assert np.array_equal(z['excel_row'],m.excel_row)
  return z['raw'],z['cal']
def run(route,rep):
 base.gate();manifest=base.read(OUT/'ROUTES.json');ref=manifest[route]
 assert base.sha(ref['path'])==ref['sha256']
 for p,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(p)==h
 code=base.read(OUT/'CODE_LOCK.json')
 for n,h in code.items():assert base.sha(CODE/n)==h
 lock=base.read(ref['path']);x,m=source_data(lock);defs=definitions(lock)
 signature={'route_lock':ref['sha256'],'code_lock':base.sha(OUT/'FIT_IMPLEMENTATION_CODE_LOCK.json'),'dependencies':base.sha(OUT/'DEPENDENCIES.json'),'replicate':rep,'bootstrap_seed':20261023}
 dest=OUT/'replicates'/route/f'{rep:03d}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():
  c=base.read(dest/'COMPLETE.json');assert c['signature']==signature
  for f,h in c.get('files',{}).items():assert base.sha(dest/f)==h
  return
 t0=time.time();idx,draw=draw_indices(m,rep);bm=m.iloc[idx].reset_index(drop=True);bx=x.iloc[idx].reset_index(drop=True)
 np.savez_compressed(dest/'DRAW.npz',indices=idx,draw_groups=draw)
 try:folds=grouped_folds(bm)
 except InvalidSample as e:
  base.atomic(dest/'COMPLETE.json',{'status':'INVALID_planned_sample_retained_no_replacement','reason':str(e),'signature':signature,'files':{'DRAW.npz':base.sha(dest/'DRAW.npz')}});return
 oof=np.full(len(bm),np.nan);fold_id=np.full(len(bm),-1);audit=[]
 for f,(tr,va) in enumerate(folds):
  trainm=bm.iloc[tr].reset_index(drop=True);validm=bm.iloc[va].reset_index(drop=True)
  # Heldout duplicates are predicted once per original encounter then expanded.
  vkeys=rt.keys(validm);pos=np.flatnonzero(~pd.Series(vkeys).duplicated().to_numpy())
  mapv={k:j for j,k in enumerate([vkeys[i] for i in pos])};back=[mapv[k] for k in vkeys]
  p=fit_flow(lock,defs,bx.iloc[tr].reset_index(drop=True),trainm,bx.iloc[va].iloc[pos].reset_index(drop=True),validm.iloc[pos].reset_index(drop=True),dest/f'fold_{f}',
   {**signature,'train_indices':rt.digest(tr.tolist()),'valid_indices':rt.digest(va.tolist()),'fold':f})
  oof[va]=p[back];fold_id[va]=f
  audit.append({'fold':f,'train_n':len(tr),'heldout_n':len(va),'patient_overlap':0,'train_groups_sha256':rt.digest(sorted(set(trainm.group_hash))),'heldout_groups_sha256':rt.digest(sorted(set(validm.group_hash)))})
 assert np.isfinite(oof).all()
 assert pd.DataFrame({'g':bm.group_hash,'fold':fold_id}).groupby('g')['fold'].nunique().eq(1).all()
 cal=calibrator(pd.DataFrame({'y':bm.y,'prob_raw':oof}))
 joblib.dump(cal,dest/'calibrator.joblib',compress=3)
 refraw,refcal=reference(route,lock,x,m)
 raw=fit_flow(lock,defs,bx,bm,x,m,dest/'full',{**signature,'scope':'full_bootstrap_predict_development_reference'},True)
 p=apply_calibration(cal,raw);traincal=apply_calibration(cal,oof)
 cuts=rt.thresholds(bm.y,traincal)
 metrics={'raw_MAD_vs_V1':float(np.mean(abs(raw-refraw))),'calibrated_MAD_vs_V1':float(np.mean(abs(p-refcal)))}
 for k,cut in lock['training_thresholds'].items():
  if k.startswith('training_') and isinstance(cut,(int,float)):
   metrics[k+'_fixed_cut_flip']=float(np.mean((p>=cut)!=(refcal>=cut)))
   metrics[k+'_refit_cut_flip']=float(np.mean((p>=cuts[k])!=(refcal>=cut)))
 for f in [.1,.2]:
  a,b=top(p,m,f),top(refcal,m,f);metrics[f'top{int(f*100)}_overlap_fraction']=len(a&b)/len(b)
 np.savez_compressed(dest/'PREDICTIONS_PRIVATE.npz',excel_row=m.excel_row.to_numpy(),raw=raw,calibrated=p,OOF_raw=oof,OOF_fold=fold_id,bootstrap_excel_row=bm.excel_row.to_numpy())
 base.atomic(dest/'RESULT.json',{'metrics':metrics,'refit_training_thresholds':cuts,'audit':audit,'bootstrap_records':len(bm),'bootstrap_events':int(bm.y.sum()),'conditional_fixed_choices_only':True,'reference_overlaps_training_not_unbiased_performance':True})
 files=['DRAW.npz','PREDICTIONS_PRIVATE.npz','RESULT.json','calibrator.joblib']
 base.atomic(dest/'COMPLETE.json',{'status':'PASS_conditional_fixed_flow_refit','signature':signature,'files':{f:base.sha(dest/f) for f in files},
  'fold_receipts':{str(f):base.sha(dest/f'fold_{f}'/'COMPLETE.json') for f in range(3)},'full_receipt_sha256':base.sha(dest/'full/COMPLETE.json'),
  'execution_code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'seconds':time.time()-t0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]))
