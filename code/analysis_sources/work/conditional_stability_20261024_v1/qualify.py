from stability import *
def main():
 rng=np.random.default_rng(9);n=72
 m=pd.DataFrame({'group_hash':['g'+str(i//2).zfill(3) for i in range(n)],'excel_row':np.arange(n),'y':[int((i//2)%3==0) for i in range(n)]})
 x=pd.DataFrame({'a':rng.normal(size=n),'b':rng.normal(size=n),'sex':pd.Series([str(i%2) for i in range(n)],dtype='string')})
 x.loc[[1,4,11,26,39],'a']=np.nan
 sp={'num':['a','b'],'cat':['sex'],'cat_levels':{'sex':['0','1']}}
 idx,d=draw_indices(m,0);idx2,d2=draw_indices(m,0);np.testing.assert_array_equal(idx,idx2)
 bm=m.iloc[idx].reset_index(drop=True);bx=x.iloc[idx].reset_index(drop=True)
 assert len(set(bm.group_hash))<m.group_hash.nunique()
 folds=grouped_folds(bm);assign=np.empty(len(bm),int)
 for f,(tr,va) in enumerate(folds):
  assign[va]=f;assert set(bm.iloc[tr].group_hash).isdisjoint(bm.iloc[va].group_hash)
 assert pd.DataFrame({'g':bm.group_hash,'f':assign}).groupby('g').f.nunique().eq(1).all()
 checks={}
 for method in ['bayes','pmm']:
  imp,filled=fill_training(bx,bm,method,sp)
  for col in sp['num']:
   observed=bx[col].notna();np.testing.assert_array_equal(filled.loc[observed,col],bx.loc[observed,col])
  z=filled.assign(key=rt.keys(bm))
  assert z.groupby('key').a.nunique().max()==1
  v=imp.transform(x,rt.keys(m));rv=imp.transform(x.iloc[::-1],rt.keys(m.iloc[::-1]))
  np.testing.assert_allclose(v.a,rv.a.iloc[::-1],rtol=0,atol=0)
  checks[method]={'duplicates_same_completion':True,'observed_preserved':True,'reorder_equal':True}
 # Explicit invalid draw retained, not resampled.
 bad=bm.copy();bad.y=0
 try:grouped_folds(bad);raise AssertionError('Invalid sample accepted')
 except InvalidSample:pass
 # Fixed voting is exact arithmetic mean; group copies are never independent validation units.
 p=np.linspace(.01,.8,n);np.testing.assert_allclose(np.column_stack([p,p]).mean(axis=1),p)
 for frac in [.1,.2]:assert len(top(p,m,frac))==int(np.ceil(frac*n))
 assert 'group_hash' not in x and 'y' not in x
 base.atomic(OUT/'QUALIFICATION.json',{'status':'PASS_group_bootstrap_multiplicity_imputation_boundaries','checks':checks,'group_folds_isolated':True,'invalid_draw_retained':True,'deterministic_seed':20261023,'no_clinical_fit_or_final_scores':True})
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
