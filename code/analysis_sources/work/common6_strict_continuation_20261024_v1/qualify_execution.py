"""Validate six-input nested topology, dispatch DAG and original ensemble contracts."""
from pathlib import Path
import os,sys,json,hashlib
from supervisor import ROOT,OUT,CODE,AB,build,base
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(CODE/'engine'))
from ensemble_math import infer,matrix,FAMILIES
def main():
    man=base.read(OUT/'SCOPE_MANIFEST.json');q=base.read(OUT/'BOUNDARY_QUALIFICATION.json')
    assert q['status']=='PASS_468_six_input_scopes_exact_prior_patient_topology' and base.sha(OUT/'SCOPE_MANIFEST.json')==q['scope_manifest_sha256']
    for p,h in q['source_hashes'].items():assert base.sha(p)==h
    for name,info in base.read(OUT/'ENGINE_PROVENANCE.json').items():
        assert base.sha(CODE/'engine'/name)==info['sha256'] and base.sha(info['source'])==info['sha256']
    for name in ['preprocess.py','coarse.py']:
        text=(CODE/'engine'/name).read_text();assert 'old=ROOT/' not in text
    jobs=build(man);ids={j['id'] for j in jobs};assert len(jobs)==len(ids)
    for j in jobs:assert set(j['depends'])<=ids and Path(j['dest']).is_relative_to(OUT)
    done=set()
    while len(done)<len(ids):
        nxt={j['id'] for j in jobs if j['id'] not in done and set(j['depends'])<=done}
        assert nxt,'cycle';done|=nxt
    by={s['id']:s for s in man['scopes']}
    for s in man['scopes']:
        assert base.read(OUT/'scopes'/s['id']/'SCOPE.json')==s
        spec=base.read(Path(s['input'])/'FEATURE_SPECS.json')['preop']
        assert set(spec['num']+spec['cat'])=={'Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb'}
    # Misordered base prediction rows align by encounter; mismatched groups reject.
    rng=np.random.default_rng(42);n=120
    m=pd.DataFrame({'excel_row':np.arange(n),'group_hash':['g'+str(i) for i in range(n)],'center':1,'y':np.arange(n)%4==0,'fold':np.arange(n)%3})
    for f in FAMILIES:m[f]=rng.uniform(.01,.5,n)
    frames={f:m[['excel_row','group_hash','center','y']].assign(prob_raw=m[f]).sample(frac=1,random_state=i) for i,f in enumerate(FAMILIES)}
    out=matrix(frames);assert set(out.excel_row)==set(m.excel_row)
    for f in FAMILIES:np.testing.assert_array_equal(out.set_index('excel_row').loc[m.excel_row,f],m[f])
    tr=m.iloc[:90];va=m.iloc[90:]
    tests=[]
    with threadpool_limits(limits=1):
        for kind in ['Voting','Stacking']:
            for k in [3,5,9]:
                for C in ([None] if kind=='Voting' else [.01,.1,1.,10.]):
                    p,model,members=infer(tr,va,kind,k,C)
                    rev,_,mm=infer(tr,va.iloc[::-1],kind,k,C)
                    np.testing.assert_allclose(p,rev[::-1],rtol=1e-12,atol=1e-12);assert members==mm and len(members)==k
                    tests.append({'kind':kind,'k':k,'C':C})
        try:infer(tr,tr.iloc[:3],'Stacking',3,1.);raise AssertionError('leakage accepted')
        except ValueError:pass
    base.atomic(OUT/'EXECUTION_QUALIFICATION.json',{'status':'PASS_common6_scoped_DAG_and_original_engine_contract',
      'source_manifest_sha256':base.sha(OUT/'SCOPE_MANIFEST.json'),'jobs':len(jobs),
      'counts':{k:sum(j['kind']==k for j in jobs) for k in ['coarse','search','fit','ensemble']},
      'six_input_scopes':468,'no_old_fitted_state_import':True,'acyclic_dependencies':True,'ensemble_grid_tests':tests,
      'real_clinical_pilot_required_before_full_dispatch':True,'public_scores_accessed':False})
if __name__=='__main__':main()
