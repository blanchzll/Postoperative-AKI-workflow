"""Source-only preparation and actual patient-group boundary audit, without model scores."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, json, hashlib, os, resource, time

ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
SOURCE=ROOT/'outputs/v4_development_inputs_20261018'
DEST=ROOT/'outputs/next_stage_preparation_20261022_v1'
D=ROOT/'outputs/ensemble_continuation_20261018_v1'
E=ROOT/'outputs/ensemble_calibration_20261020_v1'
SIX=['Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb']
NUM=['Age','PreopCr','PreopBUN','PreopAlb','PreopHb']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with q.open('w') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    q.replace(p)
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def savecsv(p, fields, rows):
    p=Path(p);p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with q.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in fields})
        f.flush();os.fsync(f.fileno())
    q.replace(p)
def boundary(train,held,meta,line):
    if not train or len(set(train))!=len(train) or len(set(held))!=len(held):
        raise ValueError('Empty training or duplicate records')
    if set(train)&set(held): raise ValueError('Record overlap')
    if not set(train+held)<=set(meta): raise ValueError('Unknown record')
    tg={meta[x]['group_hash'] for x in train}
    hg={meta[x]['group_hash'] for x in held}
    if tg&hg: raise ValueError('Patient-group overlap')
    if any(meta[x][line+'_partition_v2']!='development' for x in train+held):
        raise ValueError('Final evaluation record in training boundary')
    return {'train_n':len(train),'assessment_n':len(held),
            'train_groups':len(tg),'assessment_groups':len(hg)}
def children_cover(parent,children,meta,line):
    if len(children)!=3:raise ValueError('Three child folds required')
    held=[]
    for c in children:
        boundary(c['train_rows'],c['assessment_rows'],meta,line)
        if set(c['train_rows'])|set(c['assessment_rows'])!=set(parent['train_rows']):
            raise ValueError('Child does not partition parent training population')
        held.extend(c['assessment_rows'])
    if Counter(held)!=Counter(parent['train_rows']):
        raise ValueError('OOF does not cover each parent training record exactly once')

def main():
    started=time.time();DEST.mkdir(mode=0o700,exist_ok=True)
    inputs=[SOURCE/x for x in ['SOURCE_LOCK.json','INPUT_ACCEPTANCE.json','FEATURE_SPECS.json',
       'preop_metadata.csv','paired_metadata.csv','preop_preop_features.csv',
       'geography_preop_resampling.json','temporal_preop_resampling.json']]
    inputs += [D/'SCOPE_MANIFEST.json',D/'STAGE_D_ACCEPTANCE.json',E/'SCOPE_MANIFEST.json',E/'CODE_LOCK.json',
       ROOT/'outputs/CURRENT_RESEARCH_PLAN.json',
       ROOT/'outputs/public_eligibility_refinement_20261022_v1/COMMON6_INPUT_CONTRACT.json']
    pointer=read(ROOT/'outputs/CURRENT_RESEARCH_PLAN.json')
    inputs += [Path(pointer['current_plan']),Path(pointer['original_frozen_method_plan'])]
    hashes={str(p):sha(p) for p in inputs}
    lock=read(E/'CODE_LOCK.json')
    assert all(sha(ROOT/'work/ensemble_calibration_20261020_v1'/f)==h for f,h in lock.items())
    if (DEST/'ACCEPTANCE.json').exists():
        a=read(DEST/'ACCEPTANCE.json')
        assert a['inputs']==hashes
        assert all(sha(DEST/f)==h for f,h in a['files'].items())
        print(json.dumps({'status':'existing_preparation_verified','acceptance':str(DEST/'ACCEPTANCE.json')}));return
    metas={c:{int(r['excel_row']):r for r in csvrows(SOURCE/(c+'_metadata.csv'))} for c in ['preop','paired']}
    assert (len(metas['preop']),sum(int(r['y']) for r in metas['preop'].values()))==(3710,171)
    assert (len(metas['paired']),sum(int(r['y']) for r in metas['paired'].values()))==(2428,122)
    features=csvrows(SOURCE/'preop_preop_features.csv')
    fx={int(r['excel_row']):r for r in features}
    assert len(fx)==len(features)==len(metas['preop'])
    assert set(fx)==set(metas['preop'])
    spec=read(SOURCE/'FEATURE_SPECS.json')['preop']
    public=read(ROOT/'outputs/public_eligibility_refinement_20261022_v1/COMMON6_INPUT_CONTRACT.json')
    assert public['features']==SIX and public['num']==NUM and public['cat']==['Sex']
    assert set(SIX)<=set(spec['num']+spec['cat']) and spec['cat_levels']['Sex']==['0','1']
    sixspec={'num':NUM,'cat':['Sex'],'cat_levels':{'Sex':['0','1']}}
    counts=[]
    for line,expected in [('geography',(2551,107)),('temporal',(2024,125))]:
        selected=[r for r in metas['preop'].values() if r[line+'_partition_v2']=='development']
        rows=[int(r['excel_row']) for r in selected]
        assert (len(rows),sum(int(r['y']) for r in selected))==expected
        parts=read(SOURCE/(line+'_preop_resampling.json'))
        for p in parts:
            boundary(p['train_excel_rows'],p['assessment_excel_rows'],metas['preop'],line)
            assert set(p['train_excel_rows']+p['assessment_excel_rows'])<=set(rows)
        for scheme,parent in [('final_development_inner',None)]+[('inner',i) for i in range(5)]:
            folds=[p for p in parts if p['scheme']==scheme and p['parent_outer']==parent]
            assert len(folds)==3
            parentrows=rows if parent is None else next(p['train_excel_rows'] for p in parts if p['scheme']=='outer' and p['fold']==parent)
            children_cover({'train_rows':parentrows},[{'train_rows':p['train_excel_rows'],'assessment_rows':p['assessment_excel_rows']} for p in folds],metas['preop'],line)
        out=DEST/'common6'/line
        savecsv(out/'preop_preop_features.csv',['excel_row']+NUM+['Sex'],[fx[i] for i in rows])
        savecsv(out/'preop_metadata.csv',list(selected[0]),selected)
        atomic(out/'FEATURE_SPECS.json',{'preop':sixspec})
        atomic(out/(line+'_preop_resampling.json'),parts)
        # Verify text-level equality, preserving missing cells and all observed values.
        reread=csvrows(out/'preop_preop_features.csv')
        assert reread==[{k:fx[i][k] for k in ['excel_row']+NUM+['Sex']} for i in rows]
        assert csvrows(out/'preop_metadata.csv')==selected
        assert all(r['Sex'] in ('','0','1') for r in reread)
        missing={k:sum(r[k]=='' for r in reread) for k in SIX}
        counts.append({'line':line,'n':len(rows),'events':expected[1],
            'patient_groups':len({r['group_hash'] for r in selected}),'ordinary_missing':missing,
            'resampling_records':len(parts),'CSV_source_value_equality':True})
    atomic(DEST/'COMMON6_SOURCE_INPUT_CONTRACT.json',{
      'status':'PASS_source_development_input_preparation_only',
      'model_route':'common6_measured_preop_v1','inputs':SIX,'spec':sixspec,'development':counts,
      'units':{'Age':'years; source exact age, public released five-year bin center',
          'PreopCr':'umol/L','PreopBUN':'mmol/L','PreopAlb':'g/L','PreopHb':'g/L'},
      'forbidden_predictors':['excel_row','group_hash','center','surgery_date','y','stage',
          'Diabetes','Hypertension','HeartDisease'],
      'model_fitted':False,'imputation_fitted':False,'public_performance_accessed':False,
      'imputer_and_prediction_must_use_only_six':True,
      'future_family_search_scope':'Resolve using Chinese development evidence only after E acceptance; no public score driven choice.',
      'engineering_caution':'Do not invoke frozen coarse.py final-preop old-result import for this new route. Do not borrow a 9/27-input imputer or model. New source hashes and route-specific states required.',
      'interpretation':'Separate common-variable transport model, not validation of original9/27/38 or of selective-upgrade international benefit.'})
    dm=read(D/'SCOPE_MANIFEST.json');em=read(E/'SCOPE_MANIFEST.json')
    nodes={r['id']:r for r in dm['roots']}
    for s in dm['scopes']+em['scopes']:
        if s['id'] in nodes:assert nodes[s['id']]==s
        nodes[s['id']]=s
    assert len(dm['roots'])==24 and len(nodes)==960
    children=defaultdict(list)
    proof=[]
    for n in nodes.values():
        m=metas[n['cohort']]
        b=boundary(n['train_rows'],n['assessment_rows'],m,n['line'])
        if 'parent' in n:
            parent=nodes[n['parent']]
            assert parent['line']==n['line'] and parent['cohort']==n['cohort']
            assert set(n['train_rows']+n['assessment_rows'])==set(parent['train_rows'])
            children[n['parent']].append(n)
            scopefile=(D if n['level']<3 else E)/'scopes'/n['id']/'SCOPE.json'
            assert read(scopefile)==n
        proof.append({'id':n['id'],'line':n['line'],'cohort':n['cohort'],
                      'level':n.get('level',0),**b})
    for parent,cc in children.items():children_cover(nodes[parent],cc,metas[nodes[parent]['cohort']],nodes[parent]['line'])
    # The paired routes share the exact row/group identity and outcome by construction.
    assert all(metas['preop'][i]['group_hash']==r['group_hash'] and metas['preop'][i]['y']==r['y'] for i,r in metas['paired'].items())
    atomic(DEST/'PATIENT_SCOPE_AUDIT.json',{
      'status':'PASS_existing_partition_topology_only',
      'counts_by_level':dict(Counter(n.get('level',0) for n in nodes.values())),
      'source_manifests':{str(D/'SCOPE_MANIFEST.json'):hashes[str(D/'SCOPE_MANIFEST.json')],str(E/'SCOPE_MANIFEST.json'):hashes[str(E/'SCOPE_MANIFEST.json')]},
      'nodes':proof,'all_patient_groups_disjoint':True,'all_nodes_development_only':True,
      'all_child_folds_exactly_cover_parent_training':True,'paired_identity_and_outcomes_match_main':True,
      'does_not_certify_prediction_artifact_or_policy_fit':True,
      'must_not_reuse':'Global OOF predictions split after upstream selection; a calibrator trained on the scored patient; ensemble inner-candidate predictions as child-selected honest outer predictions.',
      'reuse_requires':['same source and feature whitelist','exact train and assessment groups',
         'upstream selection excludes assessed groups','fitted preprocessing and estimator hashes',
         'calibration and gate training excludes assessed groups','verified completion receipt'],
      'pending':['E acceptance and route model locks','per-route selected-family artifact lineage audit',
         'fit only missing compatible scopes','whole-policy honest raw mixture and logistic calibration',
         'real single/batch/reorder/reload checks','policy lock before final evaluation']})
    assert hashes=={str(p):sha(p) for p in inputs}
    assert all(sha(ROOT/'work/ensemble_calibration_20261020_v1'/f)==h for f,h in lock.items())
    atomic(DEST/'AUTHORSHIP.json',{'author':'Codex authorized fallback; scientific boundary logic reviewed by Codex',
        'local_generation':'Two bounded CPU14B requests previously timed out; no unchanged third request.',
        'diagnostic_receipt':'outputs/public_eligibility_refinement_20261022_v1/LOCAL_MODEL_DIAGNOSIS.json',
        'active_code_or_jobs_modified':False,'models_or_public_scores_computed':False})
    files={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file() and p.name!='ACCEPTANCE.json'}
    a={'status':'PASS_next_stage_inputs_and_partition_audit_only',
       'created_at_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
       'inputs':hashes,'files':files,'source_sha256':sha(__file__),
       'seconds':time.time()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       'common6_source_preparation':True,'partition_nodes_verified':len(nodes),
       'clinical_model_fitted':False,'clinical_policy_fitted':False,'final_evaluation_accessed':False,
       'final_model_locked':False,'entire_project_complete':False}
    atomic(DEST/'ACCEPTANCE.json',a)
    print(json.dumps({k:v for k,v in a.items() if k not in ['inputs','files']},ensure_ascii=False))
if __name__=='__main__':main()
