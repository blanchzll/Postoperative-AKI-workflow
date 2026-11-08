from pathlib import Path
import sys,json,hashlib
from complete_internal import *
assembly_supervisor=base.module('common6_assembly_supervisor_qualification',CODE/'supervisor.py')
build=assembly_supervisor.build
def main():
    man=base.read(OUT/'SCOPE_MANIFEST.json')
    original=base.read(BANK/'SCOPE_MANIFEST.json');assert man['scopes']==original['scopes']
    assert len(man['roots'])==12
    jobs=build();assert sum(j['kind']=='fit' for j in jobs)==90 and sum(j['kind']=='ensemble' for j in jobs)==12
    for j in jobs:
        assert Path(j['dest']).is_relative_to(OUT)
        if j['kind']=='fit':
            t=j['task'];d=AB/'runs'/j['line']/'search_v2'/t['line']/'preop/preop/full'/t['context']/t['family']
            rec=ready(d);assert rec['signature']['task']['method']==t['method']
    for line in ['geography','temporal']:
        spec=base.read(AB/'inputs'/line/'FEATURE_SPECS.json')['preop']
        assert set(spec['num']+spec['cat'])=={'Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb'}
        for n in ['search_v2','selection']:assert (OUT/'root_runs'/line/n).resolve()==(AB/'runs'/line/n).resolve()
    text=(CODE/'engine/root_ensemble.py').read_text()
    assert 'outputs/formal_development_20261018_v1' not in text and 'BASE=Path(r["base_output"])' in text
    assert 'old=ROOT/' not in (CODE/'engine/preprocess.py').read_text()
    for p in CODE.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    # Exact frozen calibration, persistence and independently held groups.
    rng=np.random.default_rng(42)
    tr=pd.DataFrame({'excel_row':np.arange(90),'group_hash':['t'+str(i) for i in range(90)],'y':np.arange(90)%5==0,'prob_raw':rng.uniform(.01,.4,90)})
    va=pd.DataFrame({'excel_row':np.arange(90,120),'group_hash':['v'+str(i) for i in range(30)],'y':np.arange(30)%5==0,'prob_raw':rng.uniform(.01,.4,30)})
    d=OUT/'synthetic_qualification';q=save(d,tr,va,{'synthetic':True})
    saved=base.sha(d/'COMPLETE.json');again=save(d,tr,va,{'synthetic':True})
    assert saved==base.sha(d/'COMPLETE.json');np.testing.assert_allclose(q.prob_calibrated,again.prob_calibrated,rtol=1e-12,atol=1e-12)
    assert choose([{'inner_AP':.2,'inner_Brier':.1,'candidate_id':'a'},{'inner_AP':.2,'inner_Brier':.1,'candidate_id':'b'}])['candidate_id']=='a'
    base.atomic(OUT/'QUALIFICATION.json',{'status':'PASS_common6_internal_assembly_contract','root_base_fits':90,'root_ensembles':12,
        'planned_calibration_contexts':144,'planned_candidate_fold_metrics':110,'planned_procedure_fold_metrics':10,
        'same_six_input_source_searches_only':True,'synthetic_calibration_reloaded_and_resume_checked':True,
        'clinical_assembly_not_started':True,'requires_nested_bank_acceptance':True,'model_locked':False,'public_scores_accessed':False})
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
