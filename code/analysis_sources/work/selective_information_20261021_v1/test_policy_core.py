"""Functional contracts for a prespecified outcome-free policy; synthetic only."""
import inspect,json,tempfile,unittest
from pathlib import Path
import numpy as np
from policy_core import fit_gate,upgrade_mask,mix_predictions,alert_mask,assert_disjoint

class PolicyContracts(unittest.TestCase):
 def setUp(self):
  self.p=np.array([.01,.1,.2,.29,.3,.4,.9])
  self.g=np.array(["g"+str(i) for i in range(7)])
  self.k=np.array(["r"+str(i) for i in range(7)])
  self.gate={"threshold":.3,"clip":1e-6,"cutoffs":{"0.25":.15,"0.5":.5}}
 def test_fitted_gate_and_json_reload(self):
  gate=fit_gate([.1,.2,.3,.4,.5])
  self.assertAlmostEqual(gate["threshold"],.42)
  reloaded=json.loads(json.dumps(gate))
  np.testing.assert_array_equal(upgrade_mask(self.p,self.g,self.k,gate,.5),
                                upgrade_mask(self.p,self.g,self.k,reloaded,.5))
 def test_nearest_and_capacity_rounding(self):
  m=upgrade_mask(self.p,self.g,self.k,self.gate,.5)
  self.assertEqual(set(np.flatnonzero(m)),{2,3,4,5})
  self.assertEqual(int(upgrade_mask(self.p,self.g,self.k,self.gate,.25).sum()),2)
 def test_permutation_and_random_invariance(self):
  ix=np.array([5,0,6,3,2,1,4])
  for random in [False,True]:
   a=upgrade_mask(self.p,self.g,self.k,self.gate,.5,random=random)
   b=upgrade_mask(self.p[ix],self.g[ix],self.k[ix],self.gate,.5,random=random)
   np.testing.assert_array_equal(a[ix],b)
   self.assertEqual(int(a.sum()),4)
 def test_lexicographic_ties(self):
  m=upgrade_mask([.3,.3,.3],["b","a","a"],["2","2","1"],self.gate,.5)
  np.testing.assert_array_equal(m,[False,True,True])
 def test_zero_one_and_empty(self):
  for f in [0.,1.]:
   m=upgrade_mask(self.p,self.g,self.k,self.gate,f)
   np.testing.assert_array_equal(m,np.full(7,bool(f)))
  self.assertEqual(len(upgrade_mask([],[],[],self.gate,.5)),0)
 def test_individual_batch_single_consistency(self):
  m=upgrade_mask(self.p,self.g,self.k,self.gate,.5,mode="individual")
  singles=[bool(upgrade_mask([v],[g],[k],self.gate,.5,mode="individual")[0])
           for v,g,k in zip(self.p,self.g,self.k)]
  np.testing.assert_array_equal(m,singles)
  with self.assertRaises(ValueError):
   upgrade_mask(self.p,self.g,self.k,self.gate,.5,mode="individual",random=True)
 def test_only_selected_upgrade_inputs_required(self):
  m=np.array([False,True,False])
  np.testing.assert_allclose(mix_predictions([.1,.2,.3],[np.nan,.8,np.nan],m),[.1,.8,.3])
  with self.assertRaises(ValueError):mix_predictions([.1,.2,.3],[np.nan,np.nan,np.nan],m)
 def test_alert_capacity_and_ranking(self):
  self.assertEqual(set(np.flatnonzero(alert_mask(self.p,self.g,self.k,.2))),{5,6})
 def test_invalid_probabilities_lengths_keys(self):
  for p in [[np.nan],[np.inf],[-.1],[1.1]]:
   with self.assertRaises(ValueError):fit_gate(p)
  with self.assertRaises(ValueError):fit_gate([])
  with self.assertRaises(ValueError):upgrade_mask([.1,.2],["x","x"],["1","1"],self.gate,.5)
  with self.assertRaises(ValueError):upgrade_mask([.1],["a","b"],["1"],self.gate,.5)
  with self.assertRaises(ValueError):upgrade_mask(self.p,self.g,self.k,self.gate,1.2)
 def test_boundary_groups_and_outcome_free_api(self):
  assert_disjoint(["patient1","patient2"],["patient3"])
  with self.assertRaises(ValueError):assert_disjoint(["patient1"],["patient1"])
  for f in [fit_gate,upgrade_mask,alert_mask]:
   self.assertFalse(set(inspect.signature(f).parameters)&{"y","outcome","labels","aki"})
 def test_extreme_finite_and_repeatability(self):
  gate=fit_gate([0,.2,.8,1])
  for _ in range(2):
   m=upgrade_mask([0,1],["a","b"],["x","y"],gate,.5)
   self.assertEqual(m.dtype,np.dtype(bool));self.assertEqual(int(m.sum()),1)

if __name__=="__main__":unittest.main(verbosity=2)
