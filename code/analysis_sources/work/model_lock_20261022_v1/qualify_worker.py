"""Synthetic model-bundle engineering tests; no clinical fitting or final-score access."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,json,unittest,tempfile,resource,time,os
import numpy as np,pandas as pd,joblib
import runtime as r
import lock_models as c
ROOT=r.ROOT;OUT=r.OUT;CODE=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
 def test_calibration_input_precision_is_float64(self):
  import ast
  tree=ast.parse((CODE/'runtime.py').read_text())
  fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='route_prediction')
  returns=[x for x in ast.walk(fn) if isinstance(x,ast.Return)]
  expr=ast.unparse(returns[-1])
  self.assertIn('dtype=np.float64',expr)
  p=np.array([.019876543,.1254321,.754321],dtype=np.float32)
  train=pd.DataFrame({'prob_raw':p.astype(np.float64),'y':[0,0,1]})
  cal=r.calibrator(train)
  expected=r.apply_calibration(cal,train.prob_raw)
  np.testing.assert_array_equal(expected,r.apply_calibration(cal,np.asarray(p,dtype=np.float64)))
 def test_selection_AP_then_Brier_then_canonical(self):
  a={'line':'g','cohort':'p','stage':'p','branch':'full','family':'RF','AP_mean':.2,'Brier_mean':.04}
  b={**a,'family':'LR','Brier_mean':.03}
  self.assertEqual(c.pick([a,b])['family'],'LR')
  self.assertEqual(c.pick([{**a,'AP_mean':.3},b])['family'],'RF')
  self.assertEqual(c.pick([a,{**a,'family':'DT'}])['family'],'DT')
 def test_thresholds_manual_and_ties(self):
  y=np.array([1,1,1,1,1,0]);p=np.array([.9,.8,.7,.7,.1,.6])
  t=r.thresholds(y,p)
  self.assertEqual(t['training_sensitivity80'],.7)
  self.assertEqual(t['training_capacity20'],np.quantile(p,.8,method='higher'))
  self.assertGreaterEqual(np.mean(p[y==1]>=t['training_sensitivity80']),.8)
  with self.assertRaises(ValueError):r.thresholds(y,np.ones(6)*2)
 def test_no_final_fit_before_E_acceptance(self):
  old=r.E
  try:
   with tempfile.TemporaryDirectory(dir=OUT) as td:
    r.E=Path(td)
    with self.assertRaisesRegex(RuntimeError,'E acceptance'):r.fit_base({})
  finally:r.E=old
 def test_grouped_RFE_zero_importance_tie_rule(self):
  x=pd.DataFrame({'Age':np.arange(60),'AZero':np.ones(60),'BZero':np.ones(60),
   'CZero':np.ones(60),'Sex':pd.Series(['0','1']*30,dtype='string')})
  sp={'num':['Age','AZero','BZero','CZero'],'cat':['Sex'],'cat_levels':{'Sex':['0','1']}}
  path=r.rfe_path(x,np.array([0]*30+[1]*30),sp)
  self.assertEqual(path['4'],['Age','BZero','CZero','Sex'])
  self.assertEqual(path['3'],['Age','CZero','Sex'])
 def test_bundle_nine_input_and_calibrated_ensemble(self):
  rng=np.random.default_rng(911);n=48
  sp={'num':['Age','PreopCr','PreopBUN','PreopAlb','PreopHb'],
      'cat':['Sex','Diabetes','Hypertension','HeartDisease'],
      'cat_levels':{k:['0','1'] for k in ['Sex','Diabetes','Hypertension','HeartDisease']}}
  x=pd.DataFrame({k:rng.normal(60,10,n) for k in sp['num']})
  for k in sp['cat']:x[k]=pd.Series(rng.integers(0,2,n).astype(str),dtype='string')
  x.loc[[0,4,7],'PreopAlb']=np.nan;ids=[f'synthetic{i}' for i in range(n)]
  y=np.array([0,1]*24);meta=pd.DataFrame({'y':y})
  imp=r.StableMixedImputer(sp,'bayes',seed=42,iterations=5,trees=200,neighbors=5,threads=1)
  filled=imp.fit_transform(x,ids)
  with tempfile.TemporaryDirectory(dir=OUT) as td:
   td=Path(td);members={};original={};ih='same-synthetic-imputer'
   configurations={'LR':{'variant':'linear','C':.1},'NB':{'var_smoothing':1e-9},'DT':{'max_depth':3,'min_samples_leaf':3}}
   for fam,params in configurations.items():
    pipe,p,_,_=r.one_fit(filled,meta,filled.iloc[:5],sp,fam,params,1)
    b={'imputer':imp,'pipeline':pipe,'input_columns':list(filled),'imputer_sha256':ih}
    fp=td/(fam+'.joblib');joblib.dump(b,fp)
    members[fam]={'path':str(fp),'sha256':r.sha(fp)}
    expected=r.predict_raw(b,x.iloc[:5],ids[:5]);original[fam]=expected
    extra=x.iloc[:5].assign(outcome=999,hospital='unseen')
    np.testing.assert_allclose(expected,r.predict_raw(joblib.load(fp),extra,ids[:5]),atol=1e-12,rtol=1e-12)
   raw=np.column_stack(list(original.values())).mean(axis=1)
   cal=r.calibrator(pd.DataFrame({'prob_raw':raw,'y':y[:5]}))
   cp=td/'cal.joblib';joblib.dump(cal,cp)
   man={'family':'Voting','members':list(members),'base_components':members,
    'calibrator':{'path':str(cp),'sha256':r.sha(cp)},'meta':None}
   p,pc=r.route_prediction(man,x.iloc[:5],ids[:5])
   np.testing.assert_allclose(raw,p,atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(r.apply_calibration(cal,raw),pc,atol=1e-12,rtol=1e-12)
   _,reverse=r.route_prediction(man,x.iloc[:5].iloc[::-1],ids[:5][::-1])
   single=np.concatenate([r.route_prediction(man,x.iloc[i:i+1],ids[i:i+1])[1] for i in range(5)])
   np.testing.assert_allclose(pc,reverse[::-1],atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(pc,single,atol=1e-12,rtol=1e-12)
   matrix=pd.DataFrame(original)
   meta_model=r.LogisticRegression(C=.1,penalty='l2',solver='lbfgs',max_iter=5000,random_state=42)
   meta_model.fit(matrix,y[:5]);mp=td/'meta.joblib';joblib.dump(meta_model,mp)
   man.update(family='Stacking',meta={'path':str(mp),'sha256':r.sha(mp)})
   stack,sc=r.route_prediction(man,x.iloc[:5],ids[:5])
   np.testing.assert_allclose(stack,meta_model.predict_proba(matrix)[:,1],atol=1e-12,rtol=1e-12)
   _,sr=r.route_prediction(man,x.iloc[:5].iloc[::-1],ids[:5][::-1])
   np.testing.assert_allclose(sc,sr[::-1],atol=1e-12,rtol=1e-12)
 def test_role_preserving_branch_family_OOF_adapter(self):
  import types
  fake=types.SimpleNamespace()
  children=[{'id':str(i),'parent':'root','parent_fold':i,'train_rows':[j for j in range(6) if j//2!=i],'assessment_rows':[2*i,2*i+1]} for i in range(3)]
  fake.DM={'scopes':children}
  calls=[]
  def pool(child,stage,branches,lineage):
   self.assertEqual(branches,['compact']);calls.append(child['id'])
   return [{'family':'LR','candidate_id':str(j)} for j in range(11)]
  fake.training_pool=pool;fake.choose=lambda rows:rows[0]
  def source(child,stage,branch,family):
   rr=child['assessment_rows']
   return pd.DataFrame({'excel_row':rr,'group_hash':[str(i) for i in rr],'y':[0,1],'prob_raw':[.1,.8]}),Path(__file__)
  fake.esource=source
  frame,proof=c.route_selection_oof({'root':'root','stage':'preop','branch':'compact','train_rows':list(range(6))},fake)
  self.assertEqual(set(frame.excel_row),set(range(6)));self.assertEqual(len(calls),3)
  self.assertTrue(proof['scored_patients_excluded_from_child_selection_and_fitting'])

def main():
 start=time.time()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 if not result.wasSuccessful():raise SystemExit(1)
 code={p.name:r.sha(p) for p in CODE.glob('*.py')}
 contract=json.loads((OUT/'EXECUTION_CONTRACT.json').read_text())
 q={'status':'PASS_model_lock_engineering_only','time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
  'tests_run':result.testsRun,'seconds':time.time()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  'code_sha256':code,'contract_sha256':r.sha(OUT/'EXECUTION_CONTRACT.json'),
  'reference_frozen_model_engine_sha256':r.sha(ROOT/'work/ensemble_calibration_20261020_v1/model_engine.py'),
  'reference_frozen_preprocess_sha256':r.sha(ROOT/'work/ensemble_calibration_20261020_v1/preprocess.py'),
  'single_batch_reorder_reload':True,'nine_input_fixture_only':True,
  'clinical_models_fitted':False,'final_performance_accessed':False,'final_model_locked':False,
  'independent_review_claimed':False}
 r.atomic(OUT/'FUNCTIONAL_QUALIFICATION.json',q)
 print(json.dumps(q,ensure_ascii=False))
if __name__=='__main__':main()
