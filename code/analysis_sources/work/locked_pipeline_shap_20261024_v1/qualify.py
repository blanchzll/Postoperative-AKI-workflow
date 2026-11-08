from explain import *
from pair_summary import summarize_pairs
import itertools,math
v=np.arange(2*3*2*4*2,dtype=float).reshape(2,3,2,4,2)/1000
s=summarize_pairs(v)
for i in range(2):
 for j in range(4):
  for k in range(2):
   means=[sum(v[b,r,i,j,k] for r in range(3))/3 for b in range(2)]
   expected=sum(means)/2
   var=[sum((v[b,r,i,j,k]-means[b])**2 for r in range(3))/2 for b in range(2)]
   se=math.sqrt(sum(t/3 for t in var)/4)
   assert abs(s['mean'][i,j,k]-expected)<1e-14
   assert abs(s['mc_se'][i,j,k]-se)<1e-14
assert s['mean'].shape==(2,4,2) and s['mean_absolute_importance'].shape==(4,2)
assert np.max(summarize_pairs(np.ones((2,2,3,4,2)))['mc_se'])==0
for invalid in [np.zeros((2,2,3,4)),np.zeros((2,1,3,4,2)),np.zeros((2,2,0,4,2)),np.full((2,2,3,4,2),np.nan)]:
 try:summarize_pairs(invalid)
 except ValueError:pass
 else:raise AssertionError('invalid accepted')
target=pd.DataFrame([[.1,.2,.3],[.5,.3,.2]],columns=['a','b','c'])
bg=pd.DataFrame([[.2,.1,.4]],columns=target.columns);keys=['case_1','case_2']
def f(x,keys):
 z=x.astype(float).fillna(.15).to_numpy()
 raw=.05+.1*z[:,0]+.2*z[:,1]+.3*z[:,2]+.4*z[:,0]*z[:,1]+.2*z[:,0]*z[:,1]*z[:,2]
 return raw,raw**2
perms=list(itertools.permutations(range(3)))
estimated=np.mean([evaluate_pair(f,target,bg,keys,p)[0] for p in perms],axis=0)
exact=np.zeros_like(estimated)
for j in range(3):
 others=[k for k in range(3) if k!=j]
 for length in range(3):
  for subset in itertools.combinations(others,length):
   a=bg.iloc[np.zeros(len(target),dtype=int)].copy();a.index=target.index
   for k in subset:a[target.columns[k]]=target.iloc[:,k].copy()
   before=np.column_stack(f(a,keys));a[target.columns[j]]=target.iloc[:,j].copy()
   after=np.column_stack(f(a,keys));w=math.factorial(length)*math.factorial(2-length)/math.factorial(3)
   exact[:,j]+=w*(after-before)
np.testing.assert_allclose(estimated,exact,atol=1e-13,rtol=1e-13)
missing=target.copy();missing.iloc[0,1]=np.nan
v,bb,end=evaluate_pair(f,missing,bg,keys,[2,1,0])
np.testing.assert_allclose(v.sum(axis=1)+bb,np.column_stack(f(missing,keys)),atol=1e-13,rtol=1e-13)
try:evaluate_pair(f,target,bg,keys,[0,0,2])
except ValueError:pass
else:raise AssertionError('Duplicate permutation accepted')
routes=base.read(OUT/'ROUTES.json');groups={}
for route in routes:
 m,sp,x,r,b=prepare_inputs(route)
 assert r.group_hash.nunique()==128 and b.group_hash.nunique()==16 and len(b)==16
 assert list(x)==m['input_columns'] and len(curves.rt.keys(r))==len(set(curves.rt.keys(r)))
 t=m['identity']
 if t['branch']=='compact':assert len(x.columns)==9
 key=(t['line'],t['cohort'])
 if key in groups:assert groups[key]==r.excel_row.tolist()
 else:groups[key]=r.excel_row.tolist()
base.gate()
base.atomic(OUT/'QUALIFICATION.json',{'status':'PASS_SHAP_math_and_source_boundary_qualification',
 'exact_nonlinear_subset_formula_agrees':True,'raw_and_calibrated_functions_explained_directly':True,
 'missing_target_additivity':True,'MC_SE_independent_loop_check':True,'source_boundary_10_routes':True,
 'reference_groups_each':128,'background_groups_each':16,'real_original_pipeline_probe_pending_pilot':True,
 'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
print('QUALIFIED_SHAP: exact nonlinear subset + independent MC SE + source boundary')
