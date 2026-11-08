"""Qualify exact source-model plus policy inference on development-only probes."""
from pathlib import Path
import importlib.util
import build_policy as b
import runtime as pr
import numpy as np,pandas as pd,joblib
def main():
 accepted=b.read(b.OUT/'POLICY_FITTING_ACCEPTANCE.json')
 assert accepted['status']=='PASS_policy_fitting_and_manifest_only_inference_acceptance_pending'
 spec=importlib.util.spec_from_file_location('aki_source_model_runtime',b.ROOT/'work/model_lock_20261022_v1/runtime.py')
 source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
 cache={};checks=[]
 for ref in accepted['policies']:
  assert b.sha(ref['path'])==ref['sha256']
  lock=b.read(ref['path']);ident=lock['identity'];pair=(ident['line'],ident['cohort'])
  if pair not in cache:
   base=b.read(lock['base_model']['path']);up=b.read(lock['upgrade_model']['path'])
   roots=[r for r in b.frozen.DM['roots'] if r['context']=='final' and (r['line'],r['cohort'])==pair]
   assert len(roots)==1
   rows=roots[0]['train_rows']
   t={'line':pair[0],'cohort':pair[1],'stage':base['identity']['stage'],'train_rows':rows}
   xb,m=source.data(t,source.source_spec(t['stage'],base['input_columns']))
   t['stage']=up['identity']['stage'];xu,_=source.data(t,source.source_spec(t['stage'],up['input_columns']))
   # Source-data engineering probes only, never the final hospital/time population.
   probes=xb.index[xb.isna().any(axis=1)].tolist()[:4]
   probes+= [i for i in xb.index if i not in probes][:max(0,4-len(probes))]
   xb=xb.loc[probes];xu=xu.loc[probes];m=m.loc[probes]
   assert m[pair[0]+'_partition_v2'].eq('development').all()
   ids=source.keys(m)
   _,pb=source.route_prediction(base,xb,ids);_,pu=source.route_prediction(up,xu,ids)
   cache[pair]=(pb,pu,m.group_hash.astype(str).tolist(),m.excel_row.astype(str).tolist())
  pb,pu,groups,keys=cache[pair]
  one=pr.risk(lock,pb,pu,groups,keys)
  reverse=pr.risk(lock,pb[::-1],pu[::-1],groups[::-1],keys[::-1])
  cal=joblib.load(lock['calibrator']['path'])
  again=pr.risk(lock,pb,pu,groups,keys,cal)
  for field in ['prob_raw_mixture','prob_calibrated','upgraded']:
   np.testing.assert_allclose(one[field],reverse[field][::-1],atol=1e-12,rtol=1e-12)
   np.testing.assert_allclose(one[field],again[field],atol=0,rtol=0)
  if ident['mode']=='individual':
   solo=np.concatenate([pr.risk(lock,pb[i:i+1],pu[i:i+1],groups[i:i+1],keys[i:i+1])['prob_calibrated'] for i in range(len(pb))])
   np.testing.assert_allclose(one['prob_calibrated'],solo,atol=1e-12,rtol=1e-12)
  else:assert one['upgraded'].sum()==int(np.ceil(len(pb)*ident['upgrade_fraction']))
  checks.append({'identity':ident,'source_locked_components_and_policy_reload_order_pass':True,'single_case_agreement':True if ident['mode']=='individual' else 'batch_capacity_rule_not_single_case_rule','probe_n':len(pb)})
 assert len(checks)==24 and len(cache)==4
 b.atomic(b.OUT/'CLINICAL_INFERENCE_QUALIFICATION.json',{'status':'PASS_development_only_exact_source_policy_inference','checks':checks,'final_scores_accessed':False})
 b.atomic(b.OUT/'STAGE_POLICY_LOCK_ACCEPTANCE.json',{'status':'PASS_two_selective_strategies_and_prespecified_variants_locked','source_models_accepted':True,'policy_manifests':accepted['policies'],'policy_variant_count':24,'two_strategy_four_development_roots':True,'clinical_inference_qualification_sha256':b.sha(b.OUT/'CLINICAL_INFERENCE_QUALIFICATION.json'),'code_lock_sha256':b.sha(b.OUT/'CODE_LOCK.json'),'whole_policy_training_calibrated_metrics_not_unbiased_validation':True,'final_hospital_time_scoring_permitted':True,'final_scores_accessed':False,'entire_research_complete':False,'next_required':'Frozen final hospital/time evaluation and same-patient comparisons; public/common6, subgroup and supplements per current plan.'})
 print('PASS_two_selective_strategies_and_prespecified_variants_locked',flush=True)
if __name__=='__main__':main()
