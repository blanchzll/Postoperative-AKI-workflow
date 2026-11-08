from fit import *
def main():
 check();base.gate();routes=base.read(OUT/'ROUTES.json');proof=[]
 high=pd.read_csv(ROOT/'outputs/cohort_v2_rrt_days_20261017/UNVERIFIED_HIGH_BASELINE_SCR.csv')
 assert set(high.excel_row)==HIGH
 for route in routes:
  bas=None;cache={}
  for scenario in SCENARIOS:
   lock,defs,x,m,fold,vx,vm,sig,key=blueprint(route,scenario)
   if bas is None:bas=(set(m.excel_row),{int(row):int(f) for row,f in zip(m.excel_row,fold)})
   assert all(bas[1][int(row)]==int(f) for row,f in zip(m.excel_row,fold))
   assert set(m.group_hash).isdisjoint(vm.group_hash)
   if scenario=='exclude_high_baseline_Cr':assert set(m.excel_row)==bas[0]-HIGH
   else:assert set(m.excel_row)==bas[0]
   cols=set(v for d in defs.values() for v in d['columns'])
   if scenario=='omit_ALT':assert 'PreopALT' not in cols
   if lock['identity']['branch']=='compact':assert len(cols)==9
   cache[scenario]=key
   proof.append({'route':route,'scenario':scenario,'train_n':len(m),'train_events':int(m.y.sum()),
    'original_heldout_n':len(vm),'retained_assessment_n':len(vm)-int(vm.excel_row.isin(HIGH).sum()) if scenario=='exclude_high_baseline_Cr' else len(vm),
    'input_union_count':len(cols),'patient_train_assessment_disjoint':True,'fold_membership_preserved':True,'cache_key':key})
  if lock['identity']['branch']=='compact':assert cache['matched_control']==cache['omit_ALT']
 # Synthetic missing-data and complete fitted-bundle round trip.
 n=60;rng=np.random.default_rng(42)
 x=pd.DataFrame({'Age':rng.normal(60,10,n),'PreopALT':rng.normal(30,5,n),'Sex':pd.Series([str(i%2) for i in range(n)],dtype='string')})
 x.loc[::7,'PreopALT']=np.nan
 m=pd.DataFrame({'excel_row':np.arange(n),'group_hash':['synthetic'+str(i//2) for i in range(n)],'y':[int(i%5==0) for i in range(n)]})
 sp={'num':['Age','PreopALT'],'cat':['Sex'],'cat_levels':{'Sex':['0','1']}}
 defs={'LR':{'method':'mean_mode','params':{'C':1.,'class_weight':None},'columns':['Age','PreopALT','Sex'],'spec':sp}}
 lock={'family':'LR','members':['LR']};d=OUT/'synthetic_qualification'
 for scenario in ['matched_control','omit_ALT']:
  dd=adjust(defs,scenario)
  p=st.fit_flow(lock,dd,x.iloc[:40].reset_index(drop=True),m.iloc[:40].reset_index(drop=True),x.iloc[40:].reset_index(drop=True),m.iloc[40:].reset_index(drop=True),
   d/scenario,{'synthetic':True,'scenario':scenario,'code_lock':base.sha(OUT/'CODE_LOCK.json')},True)
  assert len(p)==20 and np.isfinite(p).all()
  imp=joblib.load(d/scenario/'private_components.joblib')['LR'];assert imp['columns']==dd['LR']['columns']
  if scenario=='omit_ALT':assert 'PreopALT' not in imp['columns']
 check()
 base.atomic(OUT/'QUALIFICATION.json',{'status':'PASS_fixed_flow_sensitivity_preparation','routes':10,'scenario_contracts':30,'proof':proof,
  'synthetic_fit_imputation_reload_pass':True,'matched_control_required_for_interpretation':True,
  'no_clinical_refit_started':True,'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
