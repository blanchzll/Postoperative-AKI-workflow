"""Add only the fourth-level scopes needed for final selective-policy calibration."""
from pathlib import Path
import os,sys,json,hashlib,datetime,fcntl
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/selective_upstream_20261023_v1'
os.environ['AKI_SCOPE_OUT']=str(OUT)
os.environ['AKI_SCOPE_INPUT']=str(ROOT/'outputs/v4_development_inputs_20261018')
sys.path.insert(0,str(ROOT/'work/ensemble_calibration_20261020_v1'))
from core import atomic,sha,digest,FAMILIES
import prepare_e as original
import pandas as pd
def read(p):return json.loads(Path(p).read_text())
def main():
 with (OUT/'PREPARATION.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (OUT/'SCOPE_MANIFEST.json').exists()
  assert read(ROOT/'outputs/ensemble_calibration_20261020_v1/STAGE_E_ACCEPTANCE.json')['status']=='PASS'
  assert read(ROOT/'outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json')['final_model_locked']
  dm=read(ROOT/'outputs/ensemble_continuation_20261018_v1/SCOPE_MANIFEST.json')
  em=read(ROOT/'outputs/ensemble_calibration_20261020_v1/SCOPE_MANIFEST.json')
  er=ROOT/'outputs/ensemble_calibration_20261020_v1'
  frozen=read(er/'CODE_LOCK.json')
  assert frozen=={p.name:sha(p) for p in (ROOT/'work/ensemble_calibration_20261020_v1').glob('*.py')}
  roots0=[x for x in dm['roots'] if x['context']=='final'];assert len(roots0)==4
  first=[x for x in dm['scopes'] if x['parent'] in {r['id'] for r in roots0}]
  second=[x for x in dm['scopes'] if x['parent'] in {r['id'] for r in first}]
  third=[x for x in em['scopes'] if x['level']==3 and x['parent'] in {r['id'] for r in second}]
  assert (len(first),len(second),len(third))==(12,36,108)
  metas={c:pd.read_csv(ROOT/'outputs/v4_development_inputs_20261018'/ (c+'_metadata.csv'),dtype={'group_hash':str}).set_index('excel_row',drop=False) for c in ['preop','paired']}
  fourth=[];upstream={}
  for s in third:
   for part in s['inner']:
    z=original.make(s,part,metas[s['cohort']]);z['level']=4
    z['new_selective_scope']=z.pop('new_E_scope');z['parent_E_scope_sha256']=sha(er/'scopes'/s['id']/'SCOPE.json')
    atomic(OUT/'scopes'/z['id']/'SCOPE.json',z);fourth.append(z)
   upstream[str(er/'scopes'/s['id']/'SCOPE.json')]=sha(er/'scopes'/s['id']/'SCOPE.json')
  assert len(fourth)==324 and len({s['id'] for s in fourth})==324
  all_scopes=third+fourth
  for parent in third:
   ch=[s for s in fourth if s['parent']==parent['id']]
   assert sorted(v for s in ch for v in s['assessment_rows'])==sorted(parent['train_rows'])
   m=metas[parent['cohort']]
   forbidden=set(m.loc[parent['assessment_rows'],'group_hash'])
   for s in ch:
    a=m.loc[s['train_rows']];b=m.loc[s['assessment_rows']]
    assert set(a.group_hash).isdisjoint(b.group_hash)
    assert (set(a.group_hash)|set(b.group_hash)).isdisjoint(forbidden)
    assert set(a.excel_row)|set(b.excel_row)==set(parent['train_rows'])
    assert a[parent['line']+'_partition_v2'].eq('development').all() and b[parent['line']+'_partition_v2'].eq('development').all()
    assert a.y.nunique()==2 and b.y.nunique()==2
  roots=[{'id':s['id'],'line':s['line'],'cohort':s['cohort'],'context':s['id'],'base_output':s['output'],'train_rows':s['train_rows'],'assessment_rows':s['assessment_rows'],'children':[z['id'] for z in third if z['parent']==s['id']]} for s in second]
  atomic(OUT/'SCOPE_MANIFEST.json',{'roots':roots,'scopes':all_scopes,'first_scopes':first,'new_scopes':324,'reused_third_scopes':108,'purpose':'Final selective-policy calibration only; original outer/internal models remain immutable.','source_E_manifest_sha256':sha(er/'SCOPE_MANIFEST.json'),'source_D_manifest_sha256':sha(ROOT/'outputs/ensemble_continuation_20261018_v1/SCOPE_MANIFEST.json'),'no_final_assessment_in_training':True})
  atomic(OUT/'UPSTREAM_SCOPE_LINEAGE.json',upstream)
  atomic(OUT/'E_FROZEN_CODE_LOCK.json',frozen)
  atomic(OUT/'BOUNDARY_QUALIFICATION.json',{'status':'PASS_real_development_partition_preparation_only','new_scopes':324,'second_layer_ensemble_roots':36,'needed_input_routes_per_root':2,'min_train_events':min(s['train_AKI'] for s in fourth),'min_assessment_events':min(s['assessment_AKI'] for s in fourth),'source_fold_algorithm_sha256':sha(ROOT/'work/ensemble_calibration_20261020_v1/prepare_e.py'),'parent_heldout_groups_excluded':True,'group_disjoint':True,'development_only':True,'scope_manifest_sha256':sha(OUT/'SCOPE_MANIFEST.json'),'clinical_new_fits_started':False,'policy_locked':False,'final_scores_accessed':False})
  print(json.dumps({'status':'PASS_scope_preparation','new_scopes':324,'next':'coarse selection only'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
