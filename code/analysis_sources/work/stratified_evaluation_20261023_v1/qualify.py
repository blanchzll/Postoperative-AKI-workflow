from stratified import *
from sklearn.linear_model import LogisticRegression
def main():
 start=time.time();rng=np.random.default_rng(20261017);n=96
 age=np.array([40,40,17,50]);sex=np.array(['0','1','0','unknown']);cr=np.array([.7,.9,1,1])*88.4
 e=egfr2021(age,sex,cr)
 np.testing.assert_allclose(e[:2],np.array([142*.9938**40*1.012,142*.9938**40]),atol=1e-10)
 assert np.isnan(e[2:]).all()
 example=pd.DataFrame({'Age':[64,65,None],'Sex':['0','1',None],'Diabetes':['1','0',None],'TumorSite':['0','2',None],'PreopCr':[90,100,80]})
 mm=pd.DataFrame({'center':['1','2','3']})
 assert labels('age',mm,example).tolist()==['age_lt65','age_ge65','unknown']
 assert labels('sex',mm,example).tolist()==['female','male','unknown']
 assert labels('diabetes',mm,example).tolist()==['diabetes','no_diabetes','unknown']
 assert labels('egfr',mm,example)[2]=='unknown'
 y=np.array([0,1,1,0]);risk=np.array([.9,.8,.2,.1]);g=['a','b','c','d'];r=['1','2','3','4']
 alerts={'capacity25':FixedRisk(y,risk,g,r).review_counts(.25).astype(bool)}
 q=partition_score(y,risk,g,r,np.array([0,0,1,1],bool),alerts,np.array([0,0,1,0],bool),None)
 assert q['capacity25_reviewed_n']==0 and q['capacity25_severe_missed_n']==1 and q['capacity25_severe_sensitivity']==0
 assert q['upgraded_per1000'] is None
 g=[f'p{i//2:03d}' for i in range(n)];r=list(map(str,range(n)));c=[str((i//2)%3) for i in range(n)]
 p0=rng.uniform(.03,.5,n);up=np.clip(p0+rng.normal(0,.1,n),.001,.999);y=rng.binomial(1,p0)
 severe=(y==1)&(np.arange(n)%3==0);level=np.where(np.arange(n)%3==0,'a','b');level[-1]='u'
 cal=LogisticRegression(C=1e6).fit(np.log(p0/(1-p0)).reshape(-1,1),y);pol={};ths={'base':{'training_capacity20':.3},'same':{'training_capacity20':.3},'up':{'training_capacity20':.4}}
 gate0=policy_core.fit_gate(p0)
 for fraction in [.25,.5]:
  for mode,kind in [('capacity','selective'),('individual','selective'),('capacity','random')]:
   nm=f'{fraction}_{mode}_{kind}';pol[nm]={'lock':{'identity':{'upgrade_fraction':fraction,'mode':mode,'kind':kind},'gate':gate0},'calibrator':cal,'base':'base','upgrade':'up'};ths[nm]={'training_capacity20':.3}
 res=evaluate(y,{'base':p0,'same':p0.copy(),'up':up},g,r,c,pol,ths,level,severe,[('same','base'),('up','base'),('0.5_capacity_selective','base')],replicates=2000)
 for row in res['paired_differences']:
  if row['model']=='same MINUS base' and row['estimate'] is not None:assert row['estimate']==row['lower95']==row['upper95']==0
 for row in res['point']:
  assert row['capacity10_severe_captured_n']<=row['capacity10_captured_n']
  assert row['capacity20_severe_captured_n']+row['capacity20_severe_missed_n']==row['severe_events']
  assert row['capacity20_reviewed_n']<=row['n']
 assert res['receipt']['empty_subgroup_replicates']['u']>0
 assert any(r['valid_replicates']<2000 for r in res['intervals'])
 source=base.read(OUT/'EGFR_SOURCE_VERIFICATION.json')
 assert all(v in source['excerpt'] for v in ['142','0.9938','1.012','0.241','0.302'])
 base.atomic(OUT/'QUALIFICATION.json',{'status':'PASS_subgroup_allocation_and_severe_capture','synthetic_n':n,'replicates':2000,
  'seconds':time.time()-start,'known_egfr_formula_and_missing_sex_verified':True,
  'subgroup_inherits_population_alerts_no_local_reranking':True,'missing_subgroup_retained':True,
  'identical_model_delta_zero':True,'empty_subgroup_and_undefined_metric_repeats_counted':True,
  'severe_capture_uses_any_AKI_model':True,'no_clinical_final_scores_accessed':True,
  'source_hashes':{p.name:base.sha(p) for p in CODE.glob('*.py')},'egfr_source_sha256':base.sha(OUT/'EGFR_SOURCE_VERIFICATION.json')})
 print('PASS subgroup and severe capture',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
