from pathlib import Path
import sys,json,time,hashlib,pickle,os
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
from stable_imputer import StableMixedImputer,METHODS
OUT=ROOT/"outputs/v2_stable_imputation_20261017";OUT.mkdir(mode=0o700,exist_ok=True)
rng=np.random.default_rng(20261017)
n=95
complete=pd.DataFrame({"Age":rng.uniform(25,85,n),"Lab":rng.uniform(10,150,n),"WeightLoss":rng.normal(1,2,n),"Sex":np.where(rng.random(n)>.5,"1","0")})
frame=complete.copy()
for col in frame:
    frame.loc[rng.random(n)<.22,col]=np.nan
tr=frame.iloc[:70].copy();va=frame.iloc[70:].copy()
trkeys=["fit-encounter-"+str(i) for i in tr.index]
vakeys=["eval-encounter-"+str(i) for i in va.index]
spec={"num":["Age","Lab","WeightLoss"],"cat":["Sex"],"cat_levels":{"Sex":["0","1"]}}
checks=[]
def same(a,b):
    for col in spec["num"]:np.testing.assert_allclose(a[col],b[col],rtol=1e-12,atol=1e-12)
    for col in spec["cat"]:assert a[col].astype(str).tolist()==b[col].astype(str).tolist()
for method in METHODS:
    start=time.time()
    m=StableMixedImputer(spec,method,seed=42,iterations=2,trees=12,neighbors=3,threads=1)
    training=m.fit_transform(tr,trkeys)
    same(training,m.transform(tr,trkeys))
    bat=m.transform(va,vakeys)
    reverse=m.transform(va.iloc[::-1],vakeys[::-1]).iloc[::-1]
    same(bat,reverse)
    single=pd.concat([m.transform(va.iloc[i:i+1],[vakeys[i]]) for i in range(len(va))])
    same(bat,single)
    same(bat,pickle.loads(pickle.dumps(m)).transform(va,vakeys))
    shuffled=StableMixedImputer(spec,method,seed=42,iterations=2,trees=12,neighbors=3,threads=1)
    shuffled.fit(tr.iloc[::-1],trkeys[::-1]);same(bat,shuffled.transform(va,vakeys))
    for col in spec["num"]:
        obs=va[col].notna();np.testing.assert_array_equal(bat.loc[obs,col],va.loc[obs,col])
    obs=va.Sex.notna();assert bat.loc[obs,"Sex"].tolist()==va.loc[obs,"Sex"].tolist()
    for invalid in [None,["same"]*len(va),vakeys[:-1]]:
        try:m.transform(va,invalid)
        except ValueError:pass
        else:raise AssertionError("Invalid encounter-key input accepted")
    checks.append({"method":method,"batch_reverse":True,"single_case_vs_batch":True,"canonical_training_order":True,"fit_transform_equals_fit_then_transform":True,"observed_values_preserved":True,"serialization_roundtrip":True,"invalid_case_keys_rejected":True,"seconds":time.time()-start})
    print(json.dumps(checks[-1]),flush=True)
receipt={"state":"functional_qualification_passed_scientific_comparison_pending","six_methods":checks,"real_cohort_models_trained":False,"legacy_states_or_results_modified":False,"method_version_change":"Retain conditional estimator families and original iterative fitting; canonical fit order, fixed final-conditionals completion for both train and inference, one independent keyed RNG stream per encounter. PMM posterior parameter draws now encounter-specific, not batch-shared; do not claim identical joint imputations to the old implementation.","scope":"Single predictive completion, not Rubin multiple imputation. Refit and compare all six methods on new development data before selecting a winner.","encounter_key_not_predictor":True,"base_source_sha256":hashlib.sha256((ROOT/"work/python_six_outcomes_20261017/mixed_imputer.py").read_bytes()).hexdigest(),"new_source_sha256":hashlib.sha256((Path(__file__).parent/"stable_imputer.py").read_bytes()).hexdigest(),"owned_process_exits_after_persisting":True}
(OUT/"FUNCTIONAL_QUALIFICATION.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"state":receipt["state"],"methods_passed":len(checks)}),flush=True)
