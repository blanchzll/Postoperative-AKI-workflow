"""Bridge preserved POD1 eligibility to latest frozen predictors, without imputing diagnosis."""
from pathlib import Path
import os,sys,json,hashlib,copy
import pandas as pd,numpy as np,joblib
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pod1_corrected_fixed_update_20261024_v1';OLD=ROOT/'outputs/pod1_update_20261017'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert os.path.ismount('/data')
 lock=read(ROOT/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json')
 for f,h in lock['input_sha256'].items():assert sha(ROOT/'outputs/v4_development_inputs_20261018'/f)==h
 proto=read(OLD/'protocol.json');src=ROOT/proto['source'];assert sha(src)==proto['source_sha256']
 old=pd.read_csv(src,dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False)
 elig=old.loc[old.strict_POD1_secondary_cohort.eq(True)].copy()
 assert len(elig)==2947 and int(elig.pod23_relative_AKI.sum())==39
 assert elig[['base','pod1','pod23']].notna().all().all()
 assert (elig.pod1_ratio<1.5).all() and (elig.pod1_delta_baseline<26.5).all()
 assert np.array_equal(elig.pod23_relative_AKI.astype(int),(elig.pod23/elig.base>=1.5).astype(int))
 assert 983 in elig.index and not bool(elig.loc[983,"pod23_relative_AKI"])
 latest=pd.read_csv(ROOT/'outputs/v4_development_inputs_20261018/full_source_endop_features.csv',float_precision='round_trip').set_index('excel_row',drop=False)
 meta=pd.read_csv(ROOT/'outputs/v4_development_inputs_20261018/preop_metadata.csv',dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False)
 paired=pd.read_csv(ROOT/'outputs/v4_development_inputs_20261018/paired_metadata.csv',dtype={'group_hash':str},float_precision='round_trip')
 assert set(elig.index)<=set(latest.index) and set(elig.index)<=set(meta.index)
 assert elig.group_hash.equals(meta.loc[elig.index,'group_hash'])
 np.testing.assert_allclose(elig.base,latest.loc[elig.index,'PreopCr'],rtol=0,atol=0)
 specs=copy.deepcopy(proto['branches'])
 current_specs=read(ROOT/'outputs/v4_development_inputs_20261018/FEATURE_SPECS.json')
 for branch,sp in specs.items():
  for c in sp['cat']:sp['cat_levels'][c]=current_specs['endop']['cat_levels'][c]
 def category(v):
  if pd.isna(v):return None
  try:
   n=float(v)
   assert n.is_integer()
   return str(int(n))
  except ValueError:return str(v).strip()
 allcols=specs['matched_pod1']['num']+specs['matched_pod1']['cat']
 cats=set(specs['matched_pod1']['cat']);changes=[]
 for c in allcols:
  if c in ['pod1','pod1_ratio']:continue
  before=elig[c].copy();after=latest.loc[elig.index,c].copy()
  if c in cats:
   before=before.map(category);after=after.map(category)
  same=(before.eq(after)|(before.isna()&after.isna()))
  changes.append({'feature':c,'changed_records':int((~same).sum()),'missingness_changed':int((before.isna()!=after.isna()).sum()),'changed_excel_rows':elig.index[~same].astype(int).tolist()})
  elig[c]=after
 # Endpoint quantities and original source chronology stay observed.
 elig['clinical_y']=meta.loc[elig.index,'y'].astype(int)
 elig['y']=elig.pod23_relative_AKI.astype(int)
 old_matched=set(elig.loc[elig.anesthesia_complete.eq(True)].index);current_matched=set(paired.excel_row)&set(elig.index)
 save(OUT/'MATCHED_POPULATION_AUDIT.json',{'old_matched_n':len(old_matched),'current_main_paired_overlap_n':len(current_matched),
  'old_only_rows':sorted(old_matched-current_matched),'current_only_rows':sorted(current_matched-old_matched),
  'execution_membership':'Use freshly prepared corrected-source POD1 membership and rebuilt group-safe split lists'})
 for c in cats:
  elig[c]=elig[c].map(category)
  assert set(elig[c].dropna())<=set(specs['matched_pod1']['cat_levels'][c])
 # Current source partitions are only audited, never used to reassign old POD1 membership.
 jobs=[];counts=[];sources={str(src):sha(src),str(OLD/'protocol.json'):sha(OLD/'protocol.json')}
 for line in ['geography','temporal']:
  manifest=read(OLD/line/'fold_manifest.json')
  sources[str(OLD/line/'fold_manifest.json')]=sha(OLD/line/'fold_manifest.json')
  final=read(OLD/line/'MODEL_LOCK.json')
  sources[str(OLD/line/'MODEL_LOCK.json')]=sha(OLD/line/'MODEL_LOCK.json')
  for p in manifest['pairs']:
   if p['role']!='development_outer' and not p['role'].startswith('heldout'):continue
   path=OLD/line/'pairs'/p['branch']/(p['name']+'.joblib');sources[str(path)]=sha(path)
   z=joblib.load(path);rows={}
   for side in ['train','valid']:
    m=z[side+'_meta'];rr=m.excel_row.astype(int).tolist()
    assert set(rr)<=set(elig.index) and len(rr)==len(set(rr))
    assert np.array_equal(elig.loc[rr,'y'],m.y)
    assert elig.loc[rr,'group_hash'].tolist()==m.group_hash.tolist()
    rows[side]=rr
   assert set(elig.loc[rows['train'],'group_hash']).isdisjoint(elig.loc[rows['valid'],'group_hash'])
   fams=proto['families'][p['branch']]
   for family in fams:
    jobs.append({'line':line,'branch':p['branch'],'family':family,'scope':p['name'],
     'role':'final' if p['role'].startswith('heldout') else 'conditional_internal',
     'train_rows':rows['train'],'valid_rows':rows['valid'],
     'parameters':final['final_candidates'][p['branch']]})
   counts.append({'line':line,'branch':p['branch'],'scope':p['name'],'train_n':len(rows['train']),'train_events':int(elig.loc[rows['train'],'y'].sum()),
    'valid_n':len(rows['valid']),'valid_events':int(elig.loc[rows['valid'],'y'].sum())})
 assert len(jobs)==132 and sum(j['role']=='final' for j in jobs)==22
 pd.DataFrame(changes).to_csv(OUT/'OBSERVED_INPUT_CHANGE_COUNTS.csv',index=False)
 bridge=read(OLD/"PREDICTOR_SOURCE_BRIDGE.json")
 assert bridge["status"]=="PASS_only_source_documented_surgical_text_recoveries"
 assert bridge["source_lock_sha256"]==sha(ROOT/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")
 assert bridge["latest_features_sha256"]==sha(ROOT/"outputs/v4_development_inputs_20261018/full_source_endop_features.csv")
 ledger=ROOT/"outputs/v4_development_inputs_20261018/SURGICAL_TEXT_RECOVERY_LEDGER.csv"
 assert bridge["recovery_ledger_sha256"]==sha(ledger)
 observed_changes=[c for c in changes if c["changed_records"] or c["missingness_changed"]]
 assert len(observed_changes)==1 and observed_changes[0]["feature"]=="SurgicalApproach"
 change=observed_changes[0]
 assert change["changed_records"]==change["missingness_changed"]==bridge["changed_n"]==11
 assert set(change["changed_excel_rows"])==set(bridge["eligible_recovery_rows"])
 sources[str(OLD/"PREDICTOR_SOURCE_BRIDGE.json")]=sha(OLD/"PREDICTOR_SOURCE_BRIDGE.json")
 sources[str(ledger)]=sha(ledger)
 elig.reset_index(drop=True).to_csv(OUT/'PRIVATE_CURRENT_POD1_COHORT.csv',index=False)
 save(OUT/'TASKS.json',jobs);save(OUT/'FEATURE_SPECS.json',specs)
 pd.DataFrame(counts).to_csv(OUT/'PARTITION_COUNTS.csv',index=False)
 contract={'version':'corrected_input_fixed_choice_POD1_secondary_v1','eligibility':2947,'events':39,
  'source_protocol_sha256':sha(OLD/'protocol.json'),'primary':proto['primary'],'target':proto['target'],
  'fixed_choices':'Use freshly rerun corrected-data POD1 development-selected candidates and current source-line imputation; final refitting does not reselect on heldout results',
  'internal_scope':'Conditional on previously selected fixed parameters, five rebuilt corrected-source outer partitions with all required inner fitting rebuilt; not an unbiased entire-selection validation',
  'stacking':'For every calibration fold, rebuild base OOF inside that fold training and train the meta learner there; never split a globally constructed base OOF matrix',
  'imputation':proto['imputation'],'implementation':'Current qualified StableMixedImputer with fixed encounter keys, five sweeps seed42; observed values preserved; differs from old order-dependent random stream',
  'calibration':'Group-isolated three-fold raw OOF; L2 logistic C=1e6, clip1e-6, seed42; training thresholds only',
  'meta':'StandardScaler and L2 logistic C=1, same fixed LR/RF/CUDA-XGB members as original',
  'device':'Original POD1 XGB CUDA hist; LR/RF/preprocessing on CPU; no change to main V1 backend',
  'heldout':'Previously viewed secondary holdouts retained, no new independent confirmation claim',
  'diagnostic_values_imputed':False,'clinical_onset_fabricated':False,'main_endpoint_is_current_corrected_composite':True,
  '132_contexts':True,'previous_evaluation_exposure_retained':True,'current_heldout_metrics_used_for_selection':False}
 save(OUT/'EXECUTION_CONTRACT.json',contract);save(OUT/'SOURCE_FILES.json',sources)
 save(OUT/'INPUT_ACCEPTANCE.json',{'status':'PASS_POD1_2947_39_corrected_source_and_rebuilt_group_partitions',
  'eligible_n':2947,'events':39,'predictor_changes':[c for c in changes if c['changed_records']],'missingness_change':str(bridge['changed_n'])+' explicit source-documented surgical text recoveries; no other missingness changed',
  'all_132_contexts_checked':True,'patient_overlap':0,'no_fitting_performed':True,
  'files':{f:sha(OUT/f) for f in ['PRIVATE_CURRENT_POD1_COHORT.csv','TASKS.json','FEATURE_SPECS.json','PARTITION_COUNTS.csv','OBSERVED_INPUT_CHANGE_COUNTS.csv','EXECUTION_CONTRACT.json','SOURCE_FILES.json']}})
 print('POD1_INPUTS_ACCEPTED',len(jobs),'contexts, 2947/39; all source-verified predictor changes audited')
if __name__=='__main__':main()
