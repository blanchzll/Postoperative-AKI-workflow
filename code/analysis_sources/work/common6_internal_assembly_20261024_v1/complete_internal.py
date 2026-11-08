"""Assemble strict common6 calibrated candidates and complete inner-selected procedure."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/common6_internal_assembly_20261024_v1'
BANK=ROOT/'outputs/common6_strict_continuation_20261024_v1';AB=ROOT/'outputs/common6_development_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'));import common as base
sys.path.insert(0,str(CODE/'engine'));from ensemble_math import *
ALL=FAMILIES+['Voting','Stacking']
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def ready(d):
    q=base.read(Path(d)/'COMPLETE.json')
    for f,h in q['files'].items():assert base.sha(Path(d)/f)==h
    return q
def read_prediction(d,name,rows,meta):
    rec=ready(d);p=Path(d)/name;assert base.sha(p)==rec['files'][name]
    x=pd.read_csv(p,dtype={'group_hash':str},float_precision='round_trip')
    assert x.excel_row.is_unique and set(x.excel_row)==set(rows)
    m=meta.loc[x.excel_row]
    assert x.y.tolist()==m.y.tolist() and x.group_hash.tolist()==m.group_hash.tolist()
    return x
def choose(pool):
    ap=max(x['inner_AP'] for x in pool);pool=[x for x in pool if x['inner_AP']>=ap-1e-12]
    br=min(x['inner_Brier'] for x in pool)
    return min([x for x in pool if x['inner_Brier']<=br+1e-12],key=lambda x:x['candidate_id'])
def pool(item,child,lineage,meta):
    result=[]
    if child:
        root=Path(item['output']);context='outer_0'
        ensemble=BANK/'ensemble_results'/item['line']/'preop/preop/full'/item['id']
    else:
        root=AB/'runs'/item['line'];context=item['context']
        ensemble=OUT/'ensemble_results'/item['line']/'preop/preop/full'/context
    for family in ALL:
        if family in FAMILIES:
            d=root/'search_v2'/item['line']/'preop/preop/full'/context/family;rec=ready(d);best=rec['selected']
            assert base.sha(best['prediction_file'])==best['prediction_sha256']
            if child:assert rec['signature']['task']['scope_sha256']==base.sha(BANK/'scopes'/item['id']/'SCOPE.json')
            with np.load(best['prediction_file'],allow_pickle=False) as z:
                assert set(z['excel_row'])==set(item['train_rows'])
                assert meta.loc[z['excel_row']].y.tolist()==z['y'].tolist()
                assert meta.loc[z['excel_row']].group_hash.tolist()==z['group_hash'].tolist()
        else:
            d=ensemble;rec=ready(d);best=base.read(d/'SELECTION.json')['winners'][family]
            read_prediction(d,'inner_'+best['candidate_id'][:16]+'.csv',item['train_rows'],meta)
        lineage[str(d/'COMPLETE.json')]=base.sha(d/'COMPLETE.json')
        result.append({'family':family,'branch':'full','inner_AP':float(best['mean_AP']),'inner_Brier':float(best['mean_Brier']),
          'candidate_id':canonical({'branch':'full','family':family,'parameter_candidate':best['candidate_id']}),'source':str(d/'COMPLETE.json')})
    return result
def child_prediction(s,family,meta):
    if family in FAMILIES:d=Path(s['output'])/'outer_raw'/s['line']/'preop/preop/full/outer_0'/family;name='predictions.csv'
    else:d=BANK/'ensemble_results'/s['line']/'preop/preop/full'/s['id'];name=family+'_outer.csv'
    return read_prediction(d,name,s['assessment_rows'],meta),d/'COMPLETE.json'
def held_prediction(root,family,meta):
    if family in FAMILIES:d=Path(root['base_output'])/'outer_raw'/root['line']/'preop/preop/full'/root['context']/family;name='predictions.csv'
    else:d=OUT/'ensemble_results'/root['line']/'preop/preop/full'/root['context'];name=family+'_outer.csv'
    return read_prediction(d,name,root['assessment_rows'],meta),d/'COMPLETE.json'
def save(d,train,held,lineage):
    d.mkdir(parents=True,exist_ok=True)
    assert train.excel_row.is_unique and train.y.nunique()==2
    if (d/'COMPLETE.json').exists():
        ready(d);assert base.read(d/'LINEAGE.json')==lineage
        if held is None:return None
        q=pd.read_csv(d/'PRIVATE_CALIBRATED_OUTER.csv',dtype={'group_hash':str},float_precision='round_trip')
        assert q.excel_row.tolist()==held.excel_row.tolist()
        return q
    cal=calibrator(train);joblib.dump(cal,d/'calibrator.joblib',compress=3)
    train.to_csv(d/'PRIVATE_HONEST_CALIBRATION_OOF.csv',index=False)
    if held is not None:
        assert set(train.group_hash).isdisjoint(held.group_hash)
        q=held.copy();q['prob_calibrated']=apply_calibration(cal,q.prob_raw)
        restored=joblib.load(d/'calibrator.joblib')
        np.testing.assert_allclose(q.prob_calibrated,apply_calibration(restored,q.prob_raw),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(q.prob_calibrated,apply_calibration(restored,q.prob_raw.iloc[::-1])[::-1],rtol=1e-12,atol=1e-12)
        q.to_csv(d/'PRIVATE_CALIBRATED_OUTER.csv',index=False)
    else:q=None
    base.atomic(d/'LINEAGE.json',lineage)
    base.atomic(d/'COMPLETE.json',{'status':'PASS_training_boundary_calibration','files':{p.name:base.sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'},'calibration_not_fitted_on_assessment':True})
    return q
def main():
    assert base.read(OUT/'ASSEMBLY_COMPUTE_ACCEPTANCE.json')['status']=='PASS_90_outer_base_fits_12_root_ensembles'
    man=base.read(OUT/'SCOPE_MANIFEST.json')
    meta={line:pd.read_csv(AB/'inputs'/line/'preop_metadata.csv',dtype={'group_hash':str}).set_index('excel_row',drop=False) for line in ['geography','temporal']}
    for line,m in meta.items():assert m[line+'_partition_v2'].eq('development').all()
    dest=OUT/'calibrated_internal';dest.mkdir(exist_ok=True)
    # Decisions use inner training scores only; save before root held-out predictions.
    decisions=[];lineage={}
    for root in man['roots']:
        children=sorted([s for s in man['scopes'] if s['parent']==root['id']],key=lambda s:s['parent_fold'])
        assert len(children)==3
        rp=pool(root,False,lineage,meta[root['line']])
        childdec=[]
        for s in children:
            cp=pool(s,True,lineage,meta[s['line']]);childdec.append({'scope':s['id'],'selected':choose(cp),'candidates':cp})
        decisions.append({'root':root['id'],'line':root['line'],'context':root['context'],'selected':choose(rp),'candidates':rp,'children':childdec})
    base.atomic(dest/'TRAINING_ONLY_SELECTION_DECISIONS.json',{'decisions':decisions,'public_scores_accessed':False})
    allmetrics=[];procmetrics=[];procframes=[];count=0
    for root in man['roots']:
        line,context=root['line'],root['context'];m=meta[line]
        children=sorted([s for s in man['scopes'] if s['parent']==root['id']],key=lambda s:s['parent_fold'])
        dec=next(x for x in decisions if x['root']==root['id']);childwin={x['scope']:x['selected']['family'] for x in dec['children']}
        for family in ALL+['inner_selected_procedure']:
            tr=[];sources={}
            for child in children:
                f=family if family!='inner_selected_procedure' else childwin[child['id']]
                x,rec=child_prediction(child,f,m);tr.append(x.assign(fold=child['parent_fold'],selected_family=f));sources[str(rec)]=base.sha(rec)
            train=pd.concat(tr,ignore_index=True);assert set(train.excel_row)==set(root['train_rows'])
            held=None;selected=dec['selected']['family'] if family=='inner_selected_procedure' else family
            if context!='final':
                held,rec=held_prediction(root,selected,m);sources[str(rec)]=base.sha(rec)
            path=dest/line/context/family;q=save(path,train,held,sources);count+=1
            if q is not None:
                v={'line':line,'cohort':'preop','stage':'preop','branch':'full','family':family,'selected_family':selected,'outer_fold':int(context.split('_')[1]),'n':len(q),'events':int(q.y.sum()),**score(q.y,q.prob_calibrated)}
                if family=='inner_selected_procedure':procmetrics.append(v);procframes.append(q.assign(line=line,outer_fold=v['outer_fold'],selected_family=selected))
                else:allmetrics.append(v)
    assert count==144 and len(allmetrics)==110 and len(procmetrics)==10
    df=pd.DataFrame(allmetrics);df.to_csv(dest/'CALIBRATED_ELEVEN_FAMILY_FOLD_METRICS.csv',index=False)
    summary=df.groupby(['line','family'])[['AP','AUROC','Brier']].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(dest/'CALIBRATED_ELEVEN_FAMILY_SUMMARY.csv',index=False)
    pd.DataFrame(procmetrics).to_csv(dest/'INNER_SELECTED_PROCEDURE_FOLD_METRICS.csv',index=False)
    pd.concat(procframes).to_csv(dest/'PRIVATE_INNER_SELECTED_PROCEDURE_PREDICTIONS.csv',index=False)
    base.atomic(dest/'SOURCE_LINEAGE.json',lineage)
    base.atomic(dest/'ACCEPTANCE.json',{'status':'PASS_common6_strict_calibrated_candidate_and_selected_procedure_internal_evaluation','candidate_fold_metrics':110,'procedure_fold_metrics':10,'calibration_contexts':144,
      'files':{str(p.relative_to(dest)):base.sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='ACCEPTANCE.json'},
      'model_locked':False,'public_scores_accessed':False,'whole_project_complete':False})
    base.atomic(OUT/'STAGE_INTERNAL_ACCEPTANCE.json',{'status':'PASS_common6_strict_internal_evaluation','detail_sha256':base.sha(dest/'ACCEPTANCE.json'),'candidate_fold_metrics':110,'procedure_fold_metrics':10,
      'next_required':'Fit and qualify final selected source-only common6 pipelines and training thresholds; lock before INSPIRE scoring.'})
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
