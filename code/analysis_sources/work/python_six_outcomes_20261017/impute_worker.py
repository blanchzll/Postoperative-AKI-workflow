"""Fit solely on one training split and transform its paired validation predictors."""
from run_imputation_jobs import *
import argparse,time,warnings,resource
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from mixed_imputer import MixedImputer
def main():
    ap=argparse.ArgumentParser();ap.add_argument("method",choices=METHODS);ap.add_argument("pair");ap.add_argument("--threads",type=int,required=True);a=ap.parse_args()
    manifest=json.loads((BASE/"fold_inputs/fold_manifest.json").read_text());spec=json.loads((BASE/"fold_inputs/feature_spec.json").read_text())
    p=next(p for p in manifest["pairs"] if p["name"]==a.pair)
    assert p["role"].startswith("development_") and sha(BASE/"cohort.csv")==manifest["source_cohort_sha256"]
    dest=BASE/"imputed"/a.method/p["name"];dest.mkdir(parents=True,exist_ok=True)
    sig=signature(p,a.method)
    if (dest/"provenance.json").exists() and (dest/"SUCCESS.json").exists():
        assert json.loads((dest/"provenance.json").read_text())==sig;check_outputs(dest,p);return
    frames=[pd.read_csv(ROOT/p["files"][s+"_features"]["path"],dtype={k:str for k in spec["cat"]},float_precision="round_trip") for s in ["train","valid"]]
    m=MixedImputer(spec,a.method,seed=42,iterations=5,trees=200,neighbors=5,threads=a.threads)
    start=time.time()
    with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):
        warnings.simplefilter("always");train=m.fit_transform(frames[0]);valid=m.transform(frames[1])
    observed=0
    for before,after,side in zip(frames,[train,valid],["train","valid"]):
        assert list(before)==list(after) and not after.isna().any().any()
        for c in before:
            mask=before[c].notna()
            if c in spec["cat"]:
                assert (before.loc[mask,c].to_numpy()==after.loc[mask,c].to_numpy()).all()
                assert set(after[c])<=set(spec["cat_levels"][c])
            else:
                assert np.allclose(before.loc[mask,c],after.loc[mask,c],rtol=1e-12,atol=1e-12)
                if c!="WeightLoss":assert not after.loc[~mask,c].lt(0).any()
            observed+=int(mask.sum())
        after.to_csv(dest/(side+".csv"),index=False)
    joblib.dump(m,dest/"state.joblib",compress=3)
    import sklearn,scipy
    atomic(dest/"metadata.json",{"engine":"Python","method":a.method,"training_only":True,"seed":42,"iterations":5,"forest_trees":200,"neighbors":5,"threads":a.threads,"all_missing_training_columns":m.all_missing_,"training_update_history":m.history_,"features":list(train),"observed_cells_verified":observed,"warnings":list(dict.fromkeys(str(w.message)[:500] for w in ww)),"elapsed_seconds":time.time()-start,"peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"packages":{"sklearn":sklearn.__version__,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__}})
    atomic(dest/"SUCCESS.json",{"status":"SUCCESS","method":a.method,"nfeatures":len(train.columns),"engine":"Python"})
    check_outputs(dest,p);atomic(dest/"provenance.json",sig)
    print("PYTHON_IMPUTATION_DONE",a.method,a.pair,round(time.time()-start,2),flush=True)
if __name__=="__main__":main()
