"""Permutation Shapley estimates for complete frozen risk pipelines, raw feature units."""
from pathlib import Path
import os,sys,json,time,hashlib,resource
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/locked_pipeline_shap_20261024_v1'
sys.path.insert(0,str(ROOT/'work/locked_risk_curves_20261024_v1'))
import curves
base=curves.base
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def evaluate_pair(predict,target,background,keys,order):
 """Mean of forward/reverse path marginals, holding each encounter key fixed."""
 cols=list(target);n,p=target.shape;order=list(order)
 if sorted(order)!=list(range(p)) or len(background)!=1 or len(keys)!=n:raise ValueError('Invalid Shapley panel')
 if list(background)!=cols:raise ValueError('Feature mismatch')
 values=[];baselines=[];endpoints=[]
 for perm in [order,order[::-1]]:
  current=background.iloc[np.zeros(n,dtype=int)].copy();current.index=target.index
  last=np.column_stack(predict(current,keys));baseline=last.copy()
  contrib=np.zeros((n,p,2),dtype=float)
  for j in perm:
   current[cols[j]]=target[cols[j]].copy()
   nxt=np.column_stack(predict(current,keys))
   assert nxt.shape==(n,2) and np.isfinite(nxt).all()
   contrib[:,j]=nxt-last;last=nxt
  np.testing.assert_allclose(contrib.sum(axis=1)+baseline,last,atol=1e-10,rtol=1e-10)
  values.append(contrib);baselines.append(baseline);endpoints.append(last)
 np.testing.assert_allclose(endpoints[0],endpoints[1],atol=1e-10,rtol=1e-10)
 np.testing.assert_allclose(baselines[0],baselines[1],atol=1e-10,rtol=1e-10)
 return np.mean(values,axis=0),baselines[0],endpoints[0]
def prepare_inputs(route):
 m,sp,meta,x,ref,c=curves.inputs(route);contract=base.read(OUT/'CONTRACT.json')
 # Same hashed reference as the risk curves; use first 128 groups.
 groups=sorted(ref.group_hash.unique(),key=lambda g:hashlib.sha256((str(contract['seed'])+'|'+str(g)).encode()).hexdigest())[:contract['reference_groups']]
 r=ref[ref.group_hash.isin(groups)].copy()
 bg_groups=sorted(meta.group_hash.unique(),key=lambda g:hashlib.sha256((str(contract['seed'])+'|background|'+str(g)).encode()).hexdigest())[:contract['background_groups']]
 b=meta[meta.group_hash.isin(bg_groups)].reset_index(drop=True).sort_values(['group_hash','excel_row']).drop_duplicates('group_hash').set_index('excel_row',drop=False)
 return m,sp,x,r,b
def execute(route):
 from pair_summary import summarize_pairs
 check();dest=OUT/'routes'/route;dest.mkdir(parents=True,exist_ok=True);contract=base.read(OUT/'CONTRACT.json')
 if (dest/'COMPLETE.json').exists():
  for f,h in base.read(dest/'COMPLETE.json')['files'].items():assert base.sha(dest/f)==h
  return
 m,sp,x,r,b=prepare_inputs(route);target=x.loc[r.index];background=x.loc[b.index];keys=curves.rt.keys(r)
 pred=curves.Predictor(m);n,p=target.shape;B=len(background);R=contract['independent_permutations_per_background']
 sig={'route':base.read(OUT/'ROUTES.json')[route],'reference_rows':r.excel_row.tolist(),'background_rows':b.excel_row.tolist(),
  'contract_sha256':base.sha(OUT/'CONTRACT.json'),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')}
 base.atomic(dest/'EXPLANATION_LOCK.json',sig)
 # Verify cached adapter against frozen original and unchanged source identity.
 pk=keys[:4];probe=target.head(4);a=curves.rt.route_prediction(m,probe,pk);z=pred(probe,pk)
 for aa,zz in zip(a,z):np.testing.assert_allclose(aa,zz,atol=1e-12,rtol=1e-12)
 expected=np.column_stack(pred(target,keys));values=np.zeros((B,R,n,p,2));bases=np.zeros((B,R,n,2))
 started=time.monotonic();done=0;receipts={}
 for bi in range(B):
  for ri in range(R):
   path=dest/f'background_{bi:02d}_draw_{ri:02d}.npz';rec=path.with_suffix('.COMPLETE.json')
   seed=int.from_bytes(hashlib.sha256((str(contract['seed'])+'|'+route+'|'+str(bi)+'|'+str(ri)).encode()).digest()[:8],'little')
   order=np.random.default_rng(seed).permutation(p)
   if rec.exists():
    q=base.read(rec);assert q['signature']==sig and q['permutation']==order.tolist() and base.sha(path)==q['sha256']
    with np.load(path) as saved:v=saved['values'];bb=saved['baseline'];end=saved['endpoint']
   else:
    v,bb,end=evaluate_pair(pred,target,background.iloc[bi:bi+1],keys,order)
    np.testing.assert_allclose(end,expected,atol=1e-10,rtol=1e-10)
    np.savez_compressed(path,values=v,baseline=bb,endpoint=end)
    base.atomic(rec,{'signature':sig,'permutation':order.tolist(),'sha256':base.sha(path)})
   values[bi,ri]=v;bases[bi,ri]=bb
   receipts[rec.name]=base.sha(rec);receipts[path.name]=base.sha(path);done+=1
   base.atomic(dest/'progress.json',{'completed_pairs':done,'total_pairs':B*R,'path_count':2*done,
    'reference_rows':n,'elapsed_seconds':time.monotonic()-started,'pid':os.getpid(),'time_UTC':base.stamp()})
 summary=summarize_pairs(values);means=summary['mean'];mcse=summary['mc_se'];basevals=bases.mean(axis=(0,1))
 np.testing.assert_allclose(means.sum(axis=1)+basevals,expected,atol=1e-10,rtol=1e-10)
 np.savez_compressed(dest/'PRIVATE_SHAP_VALUES.npz',values=means,mc_se=mcse,baseline=basevals,prediction=expected,
  excel_row=r.excel_row.to_numpy(),features=np.asarray(target.columns,dtype=str))
 importance=[]
 for j,col in enumerate(target):
  for channel,scale in enumerate(['raw_probability','calibrated_probability']):
   importance.append({'feature':col,'scale':scale,'mean_absolute_SHAP':float(np.abs(means[:,j,channel]).mean()),
    'mean_signed_SHAP':float(means[:,j,channel].mean()),'mean_MC_SE':float(mcse[:,j,channel].mean()),
    'reference_rows':n,'reference_patient_groups':int(r.group_hash.nunique()),'background_groups':B,'independent_draws_per_background':R})
 pd.DataFrame(importance).to_csv(dest/'FEATURE_IMPORTANCE.csv',index=False)
 pd.DataFrame({'excel_row':r.excel_row,'group_hash':r.group_hash}).to_csv(dest/'PRIVATE_REFERENCE_IDENTITY.csv',index=False)
 base.atomic(dest/'EXPLANATION_DIAGNOSTICS.json',{'additivity_max_abs':float(np.max(np.abs(means.sum(axis=1)+basevals-expected))),
 'definition':'Interventional raw-feature permutation Shapley Monte Carlo estimate for fixed encounter key',
 'case_specific_baseline_due_to_keyed_missing_completion':True,'original_adapter_agrees':True,
 'MC_SE':'Within-background independent antithetic-pair variation; algorithmic Monte Carlo error, not a clinical interval',
 'causal_effects':False,'clinical_cutoff_optimized':False,'heldout_used':False,'models_refitted':False,
 'background_values_include_original_missingness':True,'correlated_hybrid_input_warning':True,
 'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
 files=list(dest.glob('*'));allhash={z.name:base.sha(z) for z in files if z.is_file() and z.name!='COMPLETE.json'}
 base.atomic(dest/'COMPLETE.json',{'status':'PASS_locked_pipeline_permutation_SHAP_estimates','route':route,
 'files':allhash,'pairs':B*R,'paths':2*B*R,'reference_groups':int(r.group_hash.nunique()),'whole_project_complete':False})
if __name__=='__main__':
 with threadpool_limits(limits=1):execute(sys.argv[1])
