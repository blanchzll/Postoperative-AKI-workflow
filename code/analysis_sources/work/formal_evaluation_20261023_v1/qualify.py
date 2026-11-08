"""Synthetic statistical checks and source-independent prerequisite tests."""
from common import *
from evaluation_stats import *
import time
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.linear_model import LogisticRegression
def main():
 start=time.time();rng=np.random.default_rng(20261017)
 n=96;g=[f'p{i//2:03d}' for i in range(n)];r=[str(i) for i in range(n)];c=[str((i//2)%3) for i in range(n)]
 p=rng.uniform(.01,.5,n);u=np.clip(p+rng.normal(0,.12,n),.001,.999);y=rng.binomial(1,p)
 w=np.repeat(rng.integers(0,4,n//2),2)
 idx,gg,rr=expanded(w,g,r)
 assert len(set(zip(gg,rr)))==len(idx)
 f=FixedRisk(y,p,g,r);sc=f.score(w)
 np.testing.assert_allclose(sc['AUROC'],roc_auc_score(y[idx],p[idx]),atol=1e-12)
 np.testing.assert_allclose(sc['AP'],average_precision_score(y[idx],p[idx]),atol=1e-12)
 np.testing.assert_allclose(sc['Brier'],brier_score_loss(y[idx],p[idx]),atol=1e-12)
 for frac in [.1,.2]:
  expanded_alert=FixedRisk(y[idx],p[idx],gg,rr).review_counts(frac)
  original_counts=f.review_counts(frac,w)
  np.testing.assert_array_equal(np.bincount(idx,weights=expanded_alert,minlength=n),original_counts)
 # Exact logistic population proportions: repeated outcomes yield intercept zero and slope one.
 yp=np.concatenate([np.r_[np.ones(k),np.zeros(10-k)] for k in [2,4,6,8]])
 pp=np.repeat([.2,.4,.6,.8],10);q=calibration(yp,pp)
 np.testing.assert_allclose([q['calibration_intercept'],q['calibration_joint_intercept'],q['calibration_slope']],[0,0,1],atol=2e-6)
 assert calibration(np.zeros(4),[.1,.2,.3,.4])['calibration_slope'] is None
 assert calibration([0,0,1,1],[.1,.2,.3,.4])['calibration_slope'] is None
 cal=LogisticRegression(C=1e6).fit(np.log(p/(1-p)).reshape(-1,1),y)
 policies={};gate0=policy_core.fit_gate(p)
 for fraction in [.25,.5]:
  for mode,kind in [('capacity','selective'),('individual','selective'),('capacity','random')]:
   nm=f'{fraction}_{mode}_{kind}'
   lock={'identity':{'upgrade_fraction':fraction,'mode':mode,'kind':kind},'gate':gate0}
   policies[nm]={'lock':lock,'calibrator':cal,'base':'base','upgrade':'up'}
   raw,pc,mask=route(policies[nm],p,u,g,r)
   rev=route(policies[nm],p[::-1],u[::-1],g[::-1],r[::-1])
   np.testing.assert_array_equal(mask,rev[2][::-1]);np.testing.assert_allclose(pc,rev[1][::-1])
   oneidx,onegg,onerr=expanded(np.ones(n,int),g,r)
   another=route(policies[nm],p,u,onegg,onerr)
   np.testing.assert_array_equal(mask,another[2])
   expanded_mask=route(policies[nm],p[idx],u[idx],gg,rr)[2]
   if mode=='capacity':assert expanded_mask.sum()==math.ceil(len(idx)*fraction)
 # An actual duplicate-group bootstrap must reroute: at least one sample has different allocations.
 item=policies['0.5_capacity_selective'];orig=route(item,p,u,g,r)[2]
 changed=False
 for j in range(12):
  ww=np.repeat(rng.integers(0,4,n//2),2);ii,gr,re=expanded(ww,g,r)
  changed |= not np.array_equal(route(item,p[ii],u[ii],gr,re)[2],orig[ii])
 assert changed
 bad=False
 try:FixedRisk(y,p,g,['same']*n)
 except ValueError:bad=True
 assert bad
 # Entire paired resampling engine. Two identical risks must have exact zero paired deltas.
 allpred={'base':p,'identical':p.copy(),'up':u}
 result=evaluate(y,allpred,g,r,c,policies,[('identical','base'),('up','base'),('0.5_capacity_selective','base')],replicates=2000)
 for row in result['paired_differences']:
  if row['model']=='identical MINUS base' and row['estimate'] is not None:
   assert row['estimate']==0 and row['lower95']==0 and row['upper95']==0
 assert result['receipt']['replicates']==2000 and result['receipt']['policy_rerouted_every_resample']
 # Verify a blocked final gate before policy acceptance. No clinical final files opened.
 if not (POLICIES/'STAGE_POLICY_LOCK_ACCEPTANCE.json').exists():
  blocked=False
  try:gate()
  except FileNotFoundError:blocked=True
  assert blocked
 atomic(OUT/'FUNCTIONAL_QUALIFICATION.json',{'status':'PASS_synthetic_metrics_calibration_policy_bootstrap',
  'synthetic_records':n,'replicates':2000,'policy_variants':6,'seconds':time.time()-start,
  'weighted_metrics_match_sklearn':True,'group_duplicate_capacity_and_identity_verified':True,
  'calibration_known_solution_and_undefined_cases_verified':True,'identical_model_paired_difference_zero':True,
  'policy_rerouting_changes_allocation_as_expected':True,'source_code_hashes':{p.name:sha(p) for p in CODE.glob('*.py')},
  'final_scores_accessed':False})
 print('PASS synthetic evaluation and policy bootstrap',flush=True)
if __name__=='__main__':main()
