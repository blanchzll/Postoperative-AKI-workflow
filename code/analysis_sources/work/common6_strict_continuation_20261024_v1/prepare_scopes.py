"""Rebind accepted D/E patient-split topology to isolated six-input data; no old fitted states."""
from pathlib import Path
import sys,json,hashlib,copy
import pandas as pd
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/common6_strict_continuation_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
def main():
    D=ROOT/'outputs/ensemble_continuation_20261018_v1'
    E=ROOT/'outputs/ensemble_calibration_20261020_v1'
    A=ROOT/'outputs/common6_development_20261024_v1'
    d=base.read(D/'SCOPE_MANIFEST.json');e=base.read(E/'SCOPE_MANIFEST.json')
    assert (D/'STAGE_D_ACCEPTANCE.json').exists() and (E/'STAGE_E_ACCEPTANCE.json').exists()
    old=[s for s in d['scopes'] if s['cohort']=='preop']+[s for s in e['scopes'] if s['cohort']=='preop' and s['level']==3]
    assert len(old)==468 and len({s['id'] for s in old})==468
    scopes=[];sources={str(D/'SCOPE_MANIFEST.json'):base.sha(D/'SCOPE_MANIFEST.json'),str(E/'SCOPE_MANIFEST.json'):base.sha(E/'SCOPE_MANIFEST.json')}
    meta={line:pd.read_csv(A/'inputs'/line/'preop_metadata.csv',dtype={'group_hash':str}).set_index('excel_row',drop=False) for line in ['geography','temporal']}
    for s0 in old:
        s=copy.deepcopy(s0);line=s['line'];m=meta[line];inp=OUT/'scopes'/s['id']/'inputs'
        inp.mkdir(parents=True,exist_ok=True)
        source=A/'inputs'/line
        assert set(m[line+'_partition_v2'])=={'development'}
        features=pd.read_csv(source/'preop_preop_features.csv',nrows=0).columns.tolist()
        assert set(features)=={'excel_row','Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb'}
        for name in ['FEATURE_SPECS.json','SOURCE_LOCK.json','preop_metadata.csv','preop_preop_features.csv']:
            target=source/name;sources[str(target)]=base.sha(target)
            p=inp/name
            if p.exists():assert p.resolve()==target.resolve()
            else:p.symlink_to(target)
        for tr,te in [(s['train_rows'],s['assessment_rows'])]+[(x['train_excel_rows'],x['assessment_excel_rows']) for x in s['inner']]:
            a=m.loc[tr];b=m.loc[te]
            assert set(a.group_hash).isdisjoint(b.group_hash)
            assert a.y.nunique()==2 and b.y.nunique()==2
        train=m.loc[s['train_rows']];held=m.loc[s['assessment_rows']]
        assert len(train)==s['train_n'] and int(train.y.sum())==s['train_AKI']
        assert len(held)==s['assessment_n'] and int(held.y.sum())==s['assessment_AKI']
        outer={'scheme':'outer','parent_outer':None,'fold':0,'train_excel_rows':s['train_rows'],'assessment_excel_rows':s['assessment_rows']}
        parts=s['inner']+[outer];p=inp/(line+'_preop_resampling.json')
        if p.exists():assert base.read(p)==parts
        else:base.atomic(p,parts)
        s['input']=str(inp);s['output']=str(inp.parent/'results')
        s['route']='common6_measured_preop_v1';s['old_prediction_or_model_reuse_permitted']=False
        base.atomic(inp.parent/'SCOPE.json',s);scopes.append(s)
    by={s['id']:s for s in scopes}
    for s in scopes:
        child=[t for t in scopes if t['parent']==s['id']]
        if s['level']<3:
            assert len(child)==3
            assert sorted(v for t in child for v in t['assessment_rows'])==sorted(s['train_rows'])
            for t in child:assert set(t['train_rows'])|set(t['assessment_rows'])==set(s['train_rows'])
    roots=[copy.deepcopy(r) for r in d['roots'] if r['cohort']=='preop']
    first=[copy.deepcopy(r) for r in e['roots'] if r['cohort']=='preop']
    for r in roots:
        assert len(r['children'])==3
        assert sorted(v for sid in r['children'] for v in by[sid]['assessment_rows'])==sorted(r['train_rows'])
        r['base_output']=str(A/'runs'/r['line'])
        r['route']='common6_measured_preop_v1'
    for r in first:r['base_output']=by[r['id']]['output']
    base.atomic(OUT/'SCOPE_MANIFEST.json',{'roots':roots,'ensemble_calibration_roots':first,'scopes':scopes,
      'old_topology_only_no_old_fitted_state':True,'source_receipts':sources,'counts':{'roots':12,'level1':36,'level2':108,'level3':324}})
    base.atomic(OUT/'BOUNDARY_QUALIFICATION.json',{'status':'PASS_468_six_input_scopes_exact_prior_patient_topology',
      'source_hashes':sources,'scope_manifest_sha256':base.sha(OUT/'SCOPE_MANIFEST.json'),
      'min_training_events':min(s['train_AKI'] for s in scopes),'min_assessment_events':min(s['assessment_AKI'] for s in scopes),
      'all_group_boundaries_disjoint':True,'chinese_development_only':True,'public_scores_accessed':False,
      'clinical_fits_started':False,'workers_or_scheduler_implemented':False,
      'required_next':'Implement isolated scope coarse/search/refit coordinator and original ensemble/calibration assembly; qualify before launch.',
      'planned_scope_coarse_batches':468,'planned_scope_search_studies':4212,'planned_scope_base_fits':4212})
if __name__=='__main__':main()
