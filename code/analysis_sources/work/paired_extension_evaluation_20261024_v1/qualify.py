from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate as e
e.check()
x=e.pd.DataFrame({'excel_row':[10,11,12,13],'group_hash':['a','b','c','d'],'y':[0,1,0,1],
 'center':[2,2,4,4],'prob_raw':[.1,.7,.2,.8],'prob_calibrated':[.1,.7,.2,.8]})
ref=x.iloc[[2,0]].reset_index(drop=True)
e.pd.testing.assert_frame_equal(e.align_to_reference(x.iloc[::-1].reset_index(drop=True),ref),ref)
bad=ref.copy();bad.loc[0,'y']=1
try:e.align_to_reference(x,bad)
except AssertionError:pass
else:raise AssertionError('Changed outcome accepted')
r=e.stats.evaluate(x.y.to_numpy(),{'same1':x.prob_raw.to_numpy(),'same2':x.prob_raw.to_numpy()},x.group_hash.tolist(),
 x.excel_row.astype(str).tolist(),x.center.astype(str).tolist(),{},[('same2','same1')],replicates=20,seed=20261017)
for d in r['paired_differences']:
 for k in ['estimate','lower95','upper95']:
  if d[k] is not None:assert abs(d[k])<1e-14
assert len(e.task_list())==10
e.base.atomic(e.OUT/'QUALIFICATION.json',{'status':'PASS_cohort_reference_alignment_and_paired_stats_qualification',
 'outcome_identity_mismatch_rejected':True,'reordering_restores_exact_reference':True,'identical_model_pair_zero':True,
 'planned_contexts':10,'replicates_per_clinical_task':2000,'no_clinical_scores_generated':True,
 'code_lock_sha256':e.base.sha(e.OUT/'CODE_LOCK.json'),'time_UTC':e.base.stamp()})
print('QUALIFIED EVALUATION')
