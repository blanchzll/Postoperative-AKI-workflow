"""Synthetic interface checks; no clinical calibration fit or final model scoring."""
from pathlib import Path
import json,sys
import build_policy as b
import runtime as pr
import numpy as np,pandas as pd,joblib
from ensemble_math import calibrator,apply_calibration
def main():
 rng=np.random.default_rng(20261023)
 p=np.linspace(.001,.4,120)
 y=(rng.random(120)<p).astype(int);assert len(np.unique(y))==2
 cal=calibrator(pd.DataFrame({'prob_raw':p,'y':y}))
 train=apply_calibration(cal,p);gate=b.policy.fit_gate(train)
 groups=[str(i//2) for i in range(120)];keys=[str(i) for i in range(120)]
 upgraded=np.clip(p*.6+np.linspace(0,.2,120),0,1)
 checks=[]
 for fraction in [.25,.5]:
  for mode,random in [('capacity',False),('individual',False),('capacity',True)]:
   lock={'identity':{'upgrade_fraction':fraction,'mode':mode,'kind':'random' if random else 'selective'},'gate':gate,'training_thresholds':b.thresholds(y,train)}
   z=pr.risk(lock,train,upgraded,groups,keys,cal)
   mask=b.policy.upgrade_mask(train,groups,keys,gate,fraction,mode,random)
   raw=b.policy.mix_predictions(train,upgraded,mask)
   np.testing.assert_array_equal(z['upgraded'],mask)
   np.testing.assert_allclose(z['prob_calibrated'],apply_calibration(cal,raw),atol=0,rtol=0)
   reverse=pr.risk(lock,train[::-1],upgraded[::-1],groups[::-1],keys[::-1],cal)
   for field in ['prob_raw_mixture','prob_calibrated','upgraded']:
    np.testing.assert_array_equal(z[field],reverse[field][::-1])
   if mode=='capacity':assert mask.sum()==int(np.ceil(len(p)*fraction))
   else:
    solo=np.concatenate([pr.risk(lock,train[i:i+1],upgraded[i:i+1],groups[i:i+1],keys[i:i+1],cal)['prob_calibrated'] for i in range(len(p))])
    np.testing.assert_allclose(solo,z['prob_calibrated'],atol=1e-12,rtol=1e-12)
   u=upgraded.copy();u[~mask]=np.nan
   partial=pr.risk(lock,train,u,groups,keys,cal)
   np.testing.assert_array_equal(partial['prob_calibrated'],z['prob_calibrated'])
   checks.append({'fraction':fraction,'mode':mode,'random':random,'alignment_routing_reorder_precision_passed':True})
 # Identity and group overlap must fail rather than silently route or calibrate.
 try:b.policy.upgrade_mask(p,['a']*120,['x']*120,gate,.5)
 except ValueError:pass
 else:raise AssertionError('duplicate identity accepted')
 a=pd.DataFrame({'excel_row':[1],'group_hash':['same'],'y':[0]})
 c=pd.DataFrame({'excel_row':[2],'group_hash':['same'],'y':[1]})
 try:b.bounded(a,c,'geography','preop',[1],[2])
 except AssertionError:pass
 else:raise AssertionError('patient overlap accepted')
 x=pd.DataFrame({'excel_row':[1,2],'group_hash':['a','b'],'y':[0,1],'prob_raw':[.1,.2]})
 xx=x.iloc[::-1].copy()
 _,aligned=b.align(x,xx);assert aligned.excel_row.tolist()==[1,2]
 bad=xx.copy();bad.loc[bad.excel_row==1,'group_hash']='wrong'
 try:b.align(x,bad)
 except AssertionError:pass
 else:raise AssertionError('identity mismatch accepted')
 # All object-code sources are pinned before this synthetic receipt.
 expected=b.read(b.OUT/'CODE_LOCK.json')
 assert expected=={p.name:b.sha(p) for p in b.CODE.glob('*.py')}
 b.atomic(b.OUT/'FUNCTIONAL_QUALIFICATION.json',{'status':'PASS_synthetic_policy_assembly_not_clinical_fit','source_code_lock_sha256':b.sha(b.OUT/'CODE_LOCK.json'),'variant_checks':checks,'duplicate_identity_rejected':True,'calibration_group_overlap_rejected':True,'row_alignment_and_identity_checked':True,'calibration_arithmetic_matches_frozen':True,'clinical_fits_started':False,'policy_locked':False,'final_scores_accessed':False})
 print('PASS_synthetic_policy_assembly_not_clinical_fit',flush=True)
if __name__=='__main__':main()
