import unittest
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from fixed_prediction_metrics import FixedRisk,bootstrap_fixed_predictions,group_weights
class MetricTests(unittest.TestCase):
 def test_alignment_and_missing_metadata_rejected(self):
  with self.assertRaisesRegex(ValueError,'Prediction lengths'):
   bootstrap_fixed_predictions([0,1],{'base':[.1,.2,.3]},['a','b'],['1','2'],['h','h'],'base',replicates=3)
  with self.assertRaisesRegex(ValueError,'Missing'):
   FixedRisk([0,1],[.1,.2],[None,'a'],['1','2'])
  with self.assertRaisesRegex(ValueError,'Missing'):
   bootstrap_fixed_predictions([0,1],{'base':[.1,.2]},['a','b'],['1','2'],[float('nan'),'h'],'base',replicates=3)
 def test_manual_ranking(self):
  m=FixedRisk([1,0,1,0],[.9,.8,.7,.6],list('abcd'),list('1234'))
  r=m.capacity(.5)
  self.assertAlmostEqual(r['AUROC'],.75);self.assertAlmostEqual(r['AP'],5/6)
  self.assertEqual((r['reviewed_n'],r['captured_n'],r['missed_n']),(2,1,1))
  self.assertEqual((r['PPV'],r['captured_per1000'],r['missed_per1000']),(.5,250,250))
 def test_ties_and_ceil(self):
  m=FixedRisk([1,0,0,1,0,0,0],[.2]*7,['g']*7,['7','6','5','4','3','2','1'])
  self.assertEqual(np.flatnonzero(m.review_counts(.25)).tolist(),[5,6])
  self.assertAlmostEqual(m.score()['AP'],2/7);self.assertAlmostEqual(m.score()['AUROC'],.5)
 def test_weighted_scores_against_sklearn(self):
  rng=np.random.default_rng(74)
  for i in range(35):
   y=rng.integers(0,2,31);p=np.round(rng.random(31),1);w=rng.integers(0,5,31)
   m=FixedRisk(y,p,[str(j) for j in range(31)],[str(j) for j in range(31)])
   r=m.score(w)
   self.assertAlmostEqual(r['AUROC'],roc_auc_score(y,p,sample_weight=w),places=12)
   self.assertAlmostEqual(r['AP'],average_precision_score(y,p,sample_weight=w),places=12)
   self.assertAlmostEqual(r['Brier'],brier_score_loss(y,p,sample_weight=w),places=12)
 def test_weighted_capacity_matches_expanded_records(self):
  y=np.array([0,1,0,1]);p=np.array([.8,.8,.2,.1]);w=np.array([3,2,0,1])
  m=FixedRisk(y,p,list('abcd'),list('1234'));idx=np.repeat(np.arange(4),w)
  expanded=FixedRisk(y[idx],p[idx],[str(j) for j in idx],
    [str(j)+'copy'+str(k) for k,j in enumerate(idx)])
  for f in [0,.1,.2,.5,1]:
   a=m.capacity(f,w);b=expanded.capacity(f)
   for metric in ['reviewed_n','captured_n','PPV','missed_per1000','AUROC','AP','Brier']:
    self.assertEqual(a[metric],b[metric])
 def test_zero_events_retained(self):
  m=FixedRisk([0,0,0],[.1,.2,.3],list('abc'),list('123'))
  r=m.capacity(.2)
  self.assertIsNone(r['AUROC']);self.assertIsNone(r['AP']);self.assertIsNone(r['sensitivity'])
  self.assertEqual((r['PPV'],r['captured_n'],r['missed_n']),(0,0,0))
 def test_all_events_and_empty_review(self):
  m=FixedRisk([1,1],[.1,.9],['a','b'],['1','2']);r=m.capacity(0)
  self.assertIsNone(r['AUROC']);self.assertEqual(r['AP'],1)
  self.assertIsNone(r['PPV']);self.assertEqual(r['missed_n'],2)
 def test_cluster_multiplicities(self):
  strata=[[np.array([0,1]),np.array([2])],[np.array([3,4]),np.array([5])]]
  for _ in range(5):
   w=group_weights(strata,6,np.random.default_rng(_))
   self.assertEqual(w[0],w[1]);self.assertEqual(w[3],w[4])
   self.assertEqual(w[0]+w[2],2);self.assertEqual(w[3]+w[5],2)
 def test_pairing_identity_and_reorder(self):
  y=np.array([1,0,0,1,0,0]);p=np.array([.8,.6,.2,.7,.1,.4])
  g=np.array(['b','b','c','d','e','f']);r=np.array(list('123456'));c=np.array([1,1,1,2,2,2])
  a=bootstrap_fixed_predictions(y,{'base':p,'same':p},g,r,c,'base',replicates=40)
  for row in a['paired_differences']:
   if row['valid_replicates']:self.assertEqual((row['estimate'],row['lower95'],row['upper95']),(0,0,0))
  ix=np.array([3,1,5,0,4,2])
  b=bootstrap_fixed_predictions(y[ix],{'base':p[ix],'same':p[ix]},g[ix],r[ix],c[ix],'base',replicates=40)
  self.assertEqual(a,b)
 def test_zero_event_replicates_not_dropped_globally(self):
  a=bootstrap_fixed_predictions([1,0],{'base':[.7,.1]},['a','b'],['1','2'],['h','h'],'base',replicates=80)
  self.assertGreater(a['receipt']['zero_event_replicates'],0)
  rows={r['metric']:r for r in a['intervals'] if r['review_fraction']==.1}
  self.assertEqual(rows['Brier']['valid_replicates'],80)
  self.assertLess(rows['AP']['valid_replicates'],80);self.assertLess(rows['AUROC']['valid_replicates'],rows['AP']['valid_replicates'])
 def test_net_benefit_manual(self):
  m=FixedRisk([1,0,1,0],[.8,.6,.3,.2],list('abcd'),list('1234'))
  d=m.net_benefit(.5)
  self.assertEqual(d['net_benefit'],0);self.assertEqual(d['treat_all_net_benefit'],0)
 def test_invalid_predictions_and_cross_hospital_group(self):
  with self.assertRaisesRegex(ValueError,'probabilities'):FixedRisk([0,1],[.1,float('nan')],['a','b'],['1','2'])
  with self.assertRaisesRegex(ValueError,'Duplicate'):FixedRisk([0,1],[.1,.2],['a','a'],['1','1'])
  with self.assertRaisesRegex(ValueError,'multiplicities'):FixedRisk([0,1],[.1,.2],['a','b'],['1','2']).score([1,.5])
  with self.assertRaisesRegex(ValueError,'hospital'):bootstrap_fixed_predictions([0,1],{'base':[.1,.2]},['a','a'],['1','2'],['h1','h2'],'base',replicates=3)
if __name__=='__main__':unittest.main()
