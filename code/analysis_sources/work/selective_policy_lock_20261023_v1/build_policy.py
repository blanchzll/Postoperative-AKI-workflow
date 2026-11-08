"""Honest constituent calibration, prespecified mixtures and private policy locks."""
from pathlib import Path
import os,sys,json,fcntl,datetime
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/selective_policy_lock_20261023_v1'
D=ROOT/'outputs/ensemble_continuation_20261018_v1'
E=ROOT/'outputs/ensemble_calibration_20261020_v1'
F=ROOT/'outputs/selective_upstream_20261023_v1'
SEARCH=ROOT/'outputs/selective_search_20261023_v1'
SOURCE=ROOT/'outputs/v4_development_inputs_20261018'
os.environ['AKI_SCOPE_OUT']=str(E)
os.environ['AKI_SCOPE_INPUT']=str(SOURCE)
sys.path.insert(0,str(ROOT/'work/ensemble_calibration_20261020_v1'))
import complete_e as frozen
from core import sha,atomic,digest,canonical,ready,FAMILIES
from ensemble_math import calibrator,apply_calibration
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'work/selective_information_20261021_v1'))
import policy_core as policy
def read(p):return json.loads(Path(p).read_text())
def route_key(line,cohort,stage,branch):return '/'.join([line,cohort,stage,branch])
def routes(cohort):
 return [('preop','compact'),('preop','full')] if cohort=='preop' else [('preop','full'),('endop','full')]
def pool(s,stage,branch,ensemble_out,lineage):
 result=[]
 for family in FAMILIES+['Voting','Stacking']:
  if family in FAMILIES:
   d=Path(s['output'])/'search_v2'/s['line']/s['cohort']/stage/branch/'outer_0'/family
   assert ready(d)
   p=d/'COMPLETE.json';r=read(p);best=r['selected']
   assert r['signature']['task']['scope_id']==s['id']
   assert r['signature']['task']['scope_sha256']==sha(D/'scopes'/s['id']/'SCOPE.json')
   assert sha(best['prediction_file'])==best['prediction_sha256']
   with np.load(best['prediction_file'],allow_pickle=False) as z:
    assert set(z['excel_row'])==set(s['train_rows'])
    m=frozen.meta(s['cohort']).loc[z['excel_row']]
    assert m.y.tolist()==z['y'].tolist() and m.group_hash.tolist()==z['group_hash'].tolist()
  else:
   d=ensemble_out/'ensemble_results'/s['line']/s['cohort']/stage/branch/s['id']
   assert ready(d);p=d/'COMPLETE.json';r=read(p)
   best=read(d/'SELECTION.json')['winners'][family]
   frozen.read_prediction(d/('inner_'+best['candidate_id'][:16]+'.csv'),r,s['train_rows'],s['line'],s['cohort'])
  lineage[str(p)]=sha(p)
  result.append({'branch':branch,'family':family,'inner_AP':float(best['mean_AP']),'inner_Brier':float(best['mean_Brier']),'candidate_id':canonical({'branch':branch,'family':family,'parameter_candidate':best['candidate_id']}),'source':str(p)})
 return result
def prediction(s,stage,branch,family,ensemble_out,lineage):
 if family in FAMILIES:
  d=Path(s['output'])/'outer_raw'/s['line']/s['cohort']/stage/branch/'outer_0'/family;name='predictions.csv'
 else:
  d=ensemble_out/'ensemble_results'/s['line']/s['cohort']/stage/branch/s['id'];name=family+'_outer.csv'
 assert ready(d);p=d/'COMPLETE.json';rec=read(p);lineage[str(p)]=sha(p)
 return frozen.read_prediction(d/name,rec,s['assessment_rows'],s['line'],s['cohort'])
def bounded(train,held,line,cohort,rows_train,rows_held):
 assert train.excel_row.is_unique and held.excel_row.is_unique
 assert set(train.excel_row)==set(rows_train) and set(held.excel_row)==set(rows_held)
 assert set(train.group_hash).isdisjoint(held.group_hash)
 for x in [train,held]:
  m=frozen.meta(cohort).loc[x.excel_row]
  assert m[line+'_partition_v2'].eq('development').all()
  assert m.y.tolist()==x.y.tolist() and m.group_hash.tolist()==x.group_hash.tolist()
def child_context(first,stage,branch):
 dest=OUT/'child_calibration'/first['line']/first['cohort']/stage/branch/first['id']
 sig={'scope':sha(D/'scopes'/first['id']/'SCOPE.json'),'code':sha(__file__),'F_manifest':sha(F/'SCOPE_MANIFEST.json'),'E_acceptance':sha(E/'STAGE_E_ACCEPTANCE.json')}
 if ready(dest,sig):
  return pd.read_csv(dest/'PRIVATE_TRAINING.csv',dtype={'group_hash':str},float_precision='round_trip'),pd.read_csv(dest/'PRIVATE_HELD.csv',dtype={'group_hash':str},float_precision='round_trip')
 lineage={};chosen=[]
 # This decision sees only first.train. The assessment is read after the decision.
 winner=frozen.choose(pool(first,stage,branch,E,lineage))
 held=prediction(first,stage,branch,winner['family'],E,lineage)
 second=sorted([s for s in frozen.DM['scopes'] if s['parent']==first['id']],key=lambda s:s['parent_fold'])
 assert len(second)==3
 parts=[]
 for s in second:
  w=frozen.choose(pool(s,stage,branch,F,lineage))
  z=prediction(s,stage,branch,w['family'],F,lineage)
  parts.append(z.assign(fold=s['parent_fold'],selected_family=w['family']))
  chosen.append({'scope':s['id'],'winner':w})
 train=pd.concat(parts,ignore_index=True)
 bounded(train,held,first['line'],first['cohort'],first['train_rows'],first['assessment_rows'])
 cal=calibrator(train)
 train['prob_calibrated']=apply_calibration(cal,np.asarray(train.prob_raw,dtype=np.float64))
 held['prob_calibrated']=apply_calibration(cal,np.asarray(held.prob_raw,dtype=np.float64))
 dest.mkdir(parents=True,exist_ok=True)
 joblib.dump(cal,dest/'calibrator.joblib',compress=3)
 loaded=joblib.load(dest/'calibrator.joblib')
 np.testing.assert_allclose(held.prob_calibrated,apply_calibration(loaded,np.asarray(held.prob_raw,dtype=np.float64)),rtol=1e-12,atol=1e-12)
 train.to_csv(dest/'PRIVATE_TRAINING.csv',index=False);held.to_csv(dest/'PRIVATE_HELD.csv',index=False)
 atomic(dest/'SELECTION_AND_LINEAGE.json',{'first_training_choice':winner,'second_training_choices':chosen,'source_receipts':lineage,'all_first_held_groups_excluded_from_selection_fitting_and_calibration':True})
 atomic(dest/'COMPLETE.json',{'signature':sig,'files':{f:sha(dest/f) for f in ['PRIVATE_TRAINING.csv','PRIVATE_HELD.csv','calibrator.joblib','SELECTION_AND_LINEAGE.json']},'train_n':len(train),'held_n':len(held),'final_assessment_used':False})
 return train,held
def align(base,upgraded):
 assert base.excel_row.is_unique and upgraded.excel_row.is_unique
 assert set(base.excel_row)==set(upgraded.excel_row)
 up=upgraded.set_index('excel_row').loc[base.excel_row].reset_index()
 assert base.group_hash.tolist()==up.group_hash.tolist() and base.y.tolist()==up.y.tolist()
 return base.reset_index(drop=True),up
def cases(frame):return frame.group_hash.astype(str).tolist(),frame.excel_row.astype(str).tolist()
def thresholds(y,p):
 y=np.asarray(y);p=np.asarray(p,dtype=float)
 assert y.sum()>0 and len(y)==len(p) and np.isfinite(p).all()
 positives=np.sort(p[y==1])[::-1]
 return {'training_sensitivity80':float(positives[int(np.ceil(.8*len(positives)))-1]),'training_capacity10':float(np.quantile(p,.9,method='higher')),'training_capacity20':float(np.quantile(p,.8,method='higher')),'alert_comparison':'>=','ties_included':True,'interpretation':'Training thresholds; fitted final calibrator applied to raw honest mixture OOF, not unbiased calibrated internal performance.'}
def route_models():
 result={}
 acc=read(ROOT/'outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json')
 assert acc['status']=='PASS' and acc['final_model_locked']
 for rec in acc['models']:
  if not rec['primary_route_winner']:continue
  assert sha(rec['path'])==rec['sha256']
  lock=read(rec['path']);identity=lock['identity']
  k=route_key(*(identity[x] for x in ['line','cohort','stage','branch']))
  result[k]=(rec,lock)
 assert len(result)==10
 return result
def main():
 assert read(SEARCH/'STAGE_F_SEARCH_FIT_ACCEPTANCE.json')['status']=='PASS_necessary_upstream_search_fit_ensemble_only'
 assert read(OUT/'FUNCTIONAL_QUALIFICATION.json')['status']=='PASS_synthetic_policy_assembly_not_clinical_fit'
 assert read(OUT/'CODE_LOCK.json')=={p.name:sha(p) for p in CODE.glob('*.py')}
 assert read(F/'E_FROZEN_CODE_LOCK.json')=={p.name:sha(p) for p in (ROOT/'work/ensemble_calibration_20261020_v1').glob('*.py')}
 models=route_models();roots=[r for r in frozen.DM['roots'] if r['context']=='final'];assert len(roots)==4
 locks=[]
 for root in roots:
  line,cohort=root['line'],root['cohort']
  base_route,up_route=routes(cohort)
  firsts=sorted([s for s in frozen.DM['scopes'] if s['parent']==root['id']],key=lambda s:s['parent_fold']);assert len(firsts)==3
  buckets={}
  for first in firsts:
   trb,hb=child_context(first,*base_route);tru,hu=child_context(first,*up_route)
   trb,tru=align(trb,tru);hb,hu=align(hb,hu)
   gate=policy.fit_gate(trb.prob_calibrated)
   g,k=cases(hb)
   for fraction in [.25,.5]:
    for mode,random in [('capacity',False),('individual',False),('capacity',True)]:
     name=f"{fraction}/{mode}/"+('random' if random else 'selective')
     mask=policy.upgrade_mask(hb.prob_calibrated,g,k,gate,fraction,mode,random)
     mixed=policy.mix_predictions(hb.prob_calibrated,hu.prob_calibrated,mask)
     q=hb[['excel_row','group_hash','center','y']].copy()
     q['prob_raw']=mixed;q['upgraded']=mask;q['fold']=first['parent_fold']
     buckets.setdefault(name,[]).append(q)
  base_ref,base_lock=models[route_key(line,cohort,*base_route)]
  up_ref,up_lock=models[route_key(line,cohort,*up_route)]
  p=base_lock['calibration_training_scores']['path'];assert sha(p)==base_lock['calibration_training_scores']['sha256']
  training_reference=pd.read_csv(p,dtype={'group_hash':str},float_precision='round_trip')
  assert set(training_reference.excel_row)==set(root['train_rows']) and training_reference.excel_row.is_unique
  gate=policy.fit_gate(training_reference.prob_calibrated)
  for name,frames in buckets.items():
   train=pd.concat(frames,ignore_index=True)
   assert train.excel_row.is_unique and set(train.excel_row)==set(root['train_rows'])
   m=frozen.meta(cohort).loc[train.excel_row]
   assert m[line+'_partition_v2'].eq('development').all()
   fraction,mode,kind=name.split('/');fraction=float(fraction)
   dest=OUT/'locked_policies'/line/cohort/name;dest.mkdir(parents=True,exist_ok=True)
   sig={'root':root['id'],'kind':kind,'fraction':fraction,'mode':mode,'base_model_lock':base_ref['sha256'],'upgrade_model_lock':up_ref['sha256'],'code':sha(__file__),'policy_core':sha(ROOT/'work/selective_information_20261021_v1/policy_core.py'),'F_compute_acceptance':sha(SEARCH/'STAGE_F_SEARCH_FIT_ACCEPTANCE.json')}
   if ready(dest,sig):
    locks.append({'path':str(dest/'POLICY_LOCK.json'),'sha256':sha(dest/'POLICY_LOCK.json')});continue
   cal=calibrator(train);calibrated=apply_calibration(cal,train.prob_raw)
   joblib.dump(cal,dest/'calibrator.joblib',compress=3)
   train.to_csv(dest/'PRIVATE_HONEST_MIXTURE_OOF.csv',index=False)
   manifest={'version':'selective_policy_private_v1','identity':{'line':line,'cohort':cohort,'upgrade_fraction':fraction,'mode':mode,'kind':kind},'base_model':base_ref,'upgrade_model':up_ref,'gate':gate,'calibrator':{'path':str(dest/'calibrator.joblib'),'sha256':sha(dest/'calibrator.joblib')},'training_thresholds':thresholds(train.y,calibrated),'raw_mixture_oof':{'path':str(dest/'PRIVATE_HONEST_MIXTURE_OOF.csv'),'sha256':sha(dest/'PRIVATE_HONEST_MIXTURE_OOF.csv')},'training_rows_sha256':digest(sorted(train.excel_row.tolist())),'training_n':len(train),'training_events':int(train.y.sum()),'calibration_training_scope':'Each raw mixture OOF row excludes its group from constituent selection/fitting/calibration and gate training. Final mixture calibrator fits all these OOF rows.','internal_metric_interpretation':'Do not call same-row final-calibrated OOF metrics unbiased complete-policy validation. Current final evaluation remains separate.','capacity_is_batch_relative':mode=='capacity','single_case_rule_supported':mode=='individual','input_burden':'Base inputs for all; upgraded model inputs for chosen records','inference_code_sha256':sha(CODE/'runtime.py'),'policy_core_sha256':sha(ROOT/'work/selective_information_20261021_v1/policy_core.py'),'contains_private_development_scores':True,'final_scores_accessed':False}
   atomic(dest/'POLICY_LOCK.json',manifest)
   atomic(dest/'COMPLETE.json',{'signature':sig,'files':{f:sha(dest/f) for f in ['POLICY_LOCK.json','calibrator.joblib','PRIVATE_HONEST_MIXTURE_OOF.csv']},'policy_fit_and_identity_saved':True,'clinical_inference_acceptance_pending':True,'final_scores_accessed':False})
   locks.append({'path':str(dest/'POLICY_LOCK.json'),'sha256':sha(dest/'POLICY_LOCK.json')})
 assert len(locks)==24
 # The assembler alone does not open final scoring; end-to-end clinical inference qualification follows.
 atomic(OUT/'POLICY_FITTING_ACCEPTANCE.json',{'status':'PASS_policy_fitting_and_manifest_only_inference_acceptance_pending','policies':locks,'constituent_contexts':24,'mixed_policy_calibrators':24,'task_owned_workers_remaining':0,'final_scoring_permitted':False,'final_scores_accessed':False,'entire_research_complete':False})
if __name__=='__main__':
 with (OUT/'ASSEMBLY.lock').open('a') as guard,threadpool_limits(limits=1):
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
