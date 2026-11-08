"""Focused engineering qualification before new secondary fits."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import engine as e
def main():
 e.check();e.init()
 assert e.digest({'b':2,'a':'白'})==hashlib.sha256('{"a":"白","b":2}'.encode()).hexdigest()
 assert e.digest({'a':1,'b':2})==e.digest({'b':2,'a':1})
 try:e.digest(float('nan'));raise AssertionError('NaN accepted')
 except ValueError:pass
 sp={'num':['a','b'],'cat':['sex'],'cat_levels':{'sex':['0','1']}}
 raw=pd.DataFrame({'a':[1.,np.nan,3,4,5,6,7,8,9,10,11,12],'b':np.arange(12.)+1,'sex':['0','1']*6})
 for method in ['mean_mode','cart']:
  imp=e.rt.StableMixedImputer(sp,method,seed=42,iterations=5,threads=1)
  filled=imp.fit_transform(raw,[str(i) for i in range(12)]);e.safe_observed(raw,filled,sp)
  kk=['case'+str(i) for i in range(12)];p=imp.transform(raw,kk)
  rev=imp.transform(raw.iloc[::-1],kk[::-1]).iloc[::-1]
  assert p.reset_index(drop=True).equals(rev.reset_index(drop=True))
  for i in range(12):assert imp.transform(raw.iloc[i:i+1],[kk[i]]).iloc[0].equals(p.iloc[i])
 est=dict(e.candidates.model_candidates(42,1,True)['XGB'])['XGB_md2_lr0.03']
 rng=np.random.default_rng(42);x=rng.normal(size=(120,5));y=np.arange(120)%2
 est.fit(x,y);q=est.predict_proba(x[:7])[:,1]
 device=json.loads(est.get_booster().save_config())['learner']['generic_param']['device']
 assert device.startswith('cuda') and np.isfinite(q).all(),device
 tasks=e.base.read(e.OUT/'TASKS.json')
 for t in tasks:
  tr=e.canonical(t['train_rows']);va=e.canonical(t['valid_rows'])
  assert set(e.DATA.loc[tr,'group_hash']).isdisjoint(e.DATA.loc[va,'group_hash'])
  for a,b in e.folds(tr):
   if t['family']=='Stacking':
    for aa,bb in e.folds(a):assert set(e.DATA.loc[aa,'group_hash']).isdisjoint(e.DATA.loc[bb,'group_hash'])
 # Trace a real development-only LR and stacked pilot. Every nested fit must
 # exclude both its target groups and the target groups of its parent request.
 original=e.raw_predict;ancestors=[];trace=[]
 def traced(task,family,train,valid,save_model=False):
  training=set(e.DATA.loc[train,'group_hash']);target=set(e.DATA.loc[valid,'group_hash'])
  assert training.isdisjoint(target)
  assert all(training.isdisjoint(parent) for parent in ancestors)
  ancestors.append(target)
  try:return original(task,family,train,valid,save_model)
  finally:
   ancestors.pop();trace.append({'family':family,'train_n':len(train),'valid_n':len(valid),'parent_targets_excluded':True})
 e.raw_predict=traced
 lr=next(i for i,t in enumerate(tasks) if t['family']=='LR' and t['role']=='conditional_internal')
 stack=next(i for i,t in enumerate(tasks) if t['family']=='Stacking' and t['role']=='conditional_internal')
 e.run(lr,1);e.run(stack,1);e.raw_predict=original
 assert any(r['family']=='XGB' for r in trace) and any(r['family']=='Stacking' for r in trace)
 e.base.atomic(e.OUT/'PILOT_ACCEPTANCE.json',{'status':'PASS_real_development_LR_and_nested_stack_pilots',
  'tasks':[lr,stack],'actual_nested_raw_requests':len(trace),'all_parent_and_child_target_groups_excluded':True,
  'heldout_final_population_not_scored':True,'reusable_complete_receipts':{str(i):e.base.sha(e.OUT/'tasks'/str(i).zfill(3)/'COMPLETE.json') for i in [lr,stack]}})
 e.base.atomic(e.OUT/'QUALIFICATION.json',{'status':'PASS_POD1_current_inputs_fixed_choices_nested_group_boundaries_and_CUDA',
  'contexts':len(tasks),'observed_values_preserved':True,'cart_single_reorder_keys_pass':True,
  'nested_stacking_group_boundaries_checked':True,'no_global_OOF_recuts':True,'actual_XGB_device':device,
  'qualification_scope':'Executed source identity, hash behavior, patient boundaries, nested development pilots and original CUDA checks',
  'code_lock_sha256':e.base.sha(e.OUT/'CODE_LOCK.json'),'heldout_metrics_viewed':False,'whole_project_complete':False})
 print('POD1_QUALIFIED',len(tasks),'CUDA',device,flush=True)
if __name__=='__main__':main()
