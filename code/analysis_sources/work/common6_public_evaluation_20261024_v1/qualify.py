"""Verify real input contract and synthetic metrics; never load a target prediction model."""
from pathlib import Path
import numpy as np
import evaluate_public as e
from threadpoolctl import threadpool_limits
def main():
 e.check();x,m=e.source()
 n=60;g=np.array(['g%03d'%(i//2) for i in range(n)]);r=np.array(['r%03d'%i for i in range(n)])
 y=np.array([int(i%5==0) for i in range(n)]);p=np.linspace(.01,.3,n);c=['synthetic']*n
 pred={'geography__raw':p,'geography__calibrated':p*.9,'temporal__raw':p[::-1],'temporal__calibrated':p[::-1]*.8}
 cuts={'geography':{'training_capacity10':.2,'training_capacity20':.1,'training_sensitivity80':.05},
       'temporal':{'training_capacity10':.15,'training_capacity20':.08,'training_sensitivity80':.03}}
 result=e.stats.evaluate(y,pred,g,r,c,{},[],replicates=50,seed=20261017)
 table,proof=e.threshold_intervals(y,pred,g,r,c,cuts,replicates=50)
 assert proof==result['receipt']['resample_weight_sha256']
 for row in table:
  z=e.stats.counts(y,pred[row['source_line']+'__calibrated']>=row['threshold'])
  assert row['estimate']==z[row['metric']]
  assert row['valid_replicates']<=50
 ix=np.arange(n)[::-1]
 rev=e.stats.evaluate(y[ix],{k:v[ix] for k,v in pred.items()},g[ix],r[ix],c,{},[],replicates=50,seed=20261017)
 assert rev==result
 # Explicit single-class outcome handling, and ties consume exactly ceil(n*capacity).
 z=e.stats.summary(np.zeros(n),np.full(n,.1),g,r)
 assert z['AUROC'] is None and z['calibration_slope'] is None and z['capacity20_reviewed_n']==12
 assert e.stats.FixedRisk(y,np.full(n,.1),g,r).review_counts(.1).sum()==6
 e.check()
 e.base.atomic(e.OUT/'QUALIFICATION.json',{'status':'PASS_public_evaluation_preparation_only',
  'source_n':len(m),'source_events':int(m.y_scr.sum()),'six_only_source_verified':True,
  'frozen_source_file_hashes_verified':True,'subject_to_group_identity_consistent':True,
  'synthetic_bootstrap_replicates':50,'source_threshold_bootstrap_draws_match_metrics':True,
  'reverse_order_identical_results':True,'zero_event_and_tied_capacity_checks':True,
  'model_scores_computed':False,'code_lock_sha256':e.base.sha(e.OUT/'CODE_LOCK.json')})
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
