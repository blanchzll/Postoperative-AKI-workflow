"""Synthetic six-input pipeline parity; no public or clinical outcome scoring."""
from pathlib import Path
import sys,os,json
import runtime as r
import lock_models as control
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
def main():
 control.check()
 d=control.OUT/'synthetic_qualification';d.mkdir(exist_ok=True)
 spec=r.source_spec('preop');cols=spec['num']+spec['cat'];assert set(cols)==control.SIX
 for line in ['geography','temporal']:
  sp=control.read(control.AB/'inputs'/line/'FEATURE_SPECS.json')['preop']
  assert sp==spec
  header=pd.read_csv(control.AB/'inputs'/line/'preop_preop_features.csv',nrows=0).columns.tolist()
  assert set(header)==control.SIX|{'excel_row'}
 rng=np.random.default_rng(20261024);n=48
 x=pd.DataFrame({v:rng.normal(10,2,n) for v in spec['num']})
 x['Sex']=pd.Series([str(i%2) for i in range(n)],dtype='string')
 for j,v in enumerate(spec['num']):x.loc[(np.arange(n)+j)%9==0,v]=np.nan
 y=np.array([0,0,0,1]*12);ids=['synthetic-'+str(i) for i in range(n)]
 imp=r.StableMixedImputer(spec,'mean_mode',seed=42,iterations=5,trees=200,neighbors=5,threads=1)
 xx=imp.fit_transform(x,ids);joblib.dump(imp,d/'imputer.joblib')
 components={};pred={}
 for family,params in [('LR',{'C':1.,'class_weight':None}),('RF',{'n_estimators':10,'min_samples_leaf':2})]:
  pipe,pr,_,_=r.one_fit(xx,pd.DataFrame({'y':y}),xx,spec,family,params,1)
  bundle={'imputer':imp,'pipeline':pipe,'input_spec':spec,'input_columns':cols,'imputer_sha256':r.sha(d/'imputer.joblib')}
  p=d/(family+'.joblib');joblib.dump(bundle,p)
  components[family]={'path':str(p),'sha256':r.sha(p)};pred[family]=pr
 cal=r.calibrator(pd.DataFrame({'y':y,'prob_raw':pred['LR']}));joblib.dump(cal,d/'calibrator.joblib')
 meta=r.LogisticRegression(C=1,random_state=42,max_iter=5000).fit(pd.DataFrame(pred),y);joblib.dump(meta,d/'meta.joblib')
 for family in ['LR','RF','Voting','Stacking']:
  members=[family] if family in ['LR','RF'] else ['LR','RF']
  m={'family':family,'members':members,'base_components':{f:components[f] for f in members},
     'calibrator':{'path':str(d/'calibrator.joblib'),'sha256':r.sha(d/'calibrator.joblib')},
     'meta':{'path':str(d/'meta.joblib'),'sha256':r.sha(d/'meta.joblib')} if family=='Stacking' else None}
  batch=r.route_prediction(m,x,ids);reverse=r.route_prediction(m,x.iloc[::-1],ids[::-1])
  singles=[r.route_prediction(m,x.iloc[i:i+1],ids[i:i+1]) for i in range(n)]
  control.atomic(d/(family+'_manifest.json'),m)
  loaded=r.route_prediction(control.read(d/(family+'_manifest.json')),x,ids)
  for k in [0,1]:
   np.testing.assert_allclose(batch[k],reverse[k][::-1],atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(batch[k],np.concatenate([z[k] for z in singles]),atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(batch[k],loaded[k],atol=1e-12,rtol=1e-12)
  cuts=r.thresholds(y,batch[1])
  for name in ['training_sensitivity80','training_capacity10','training_capacity20']:
   np.testing.assert_array_equal(batch[1]>=cuts[name],reverse[1][::-1]>=cuts[name])
 rows=[{'line':'x','cohort':'preop','stage':'preop','branch':'full','family':f,'AP_mean':.2,'Brier_mean':.1} for f in ['RF','LR']]
 assert control.pick(rows)['family']=='LR'
 control.check()
 control.atomic(control.OUT/'QUALIFICATION.json',{'status':'PASS_common6_lock_engineering_only',
  'six_only_source_header':True,'synthetic_families':['LR','RF','Voting','Stacking'],
  'batch_single_reorder_reload_tolerance':1e-12,'training_threshold_parity':True,
  'candidate_selection_inherits_main_V1':True,'calibration_uses_inner_selected_procedure_oof':True,
  'code_lock_sha256':control.sha(control.OUT/'CODE_LOCK.json'),
  'clinical_model_fitted':False,'public_scores_accessed':False})
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
