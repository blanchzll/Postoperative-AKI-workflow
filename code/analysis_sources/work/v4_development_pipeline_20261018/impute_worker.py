"""Finite imputation task: development-inner training only, immutable alignment keys."""
from pathlib import Path
import os,sys,json,time,hashlib,resource
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"work/v2_stable_imputation_20261017"))
from stable_imputer import StableMixedImputer,METHODS
import pandas as pd,numpy as np,joblib
INPUT=ROOT/"outputs/v4_development_inputs_20261018"
OUT=ROOT/"outputs/v4_development_pipeline_20261018"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
    temp=p.with_suffix(p.suffix+".tmp");temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n");os.replace(temp,p)
def main():
    line,fold,method,threads=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
    assert line in ["geography","temporal"] and method in METHODS
    dest=OUT/line/("inner_"+str(fold))/method;dest.mkdir(mode=0o700,parents=True,exist_ok=True)
    if (dest/"COMPLETE.json").exists():
        saved=json.loads((dest/"COMPLETE.json").read_text())
        for f,h in saved["outputs_sha256"].items():assert sha(dest/f)==h
        print(json.dumps({"reused":True,"task":[line,fold,method]}));return
    start=time.time();spec=json.loads((INPUT/"FEATURE_SPECS.json").read_text())["preop"]
    cols=spec["num"]+spec["cat"]
    assert len(cols)==27 and not set(cols)&{"y","stage","source_RRT","pod1","pod23","excel_row","group_hash"}
    meta=pd.read_csv(INPUT/"preop_metadata.csv",dtype={"group_hash":str}).set_index("excel_row",drop=False)
    features=pd.read_csv(INPUT/"preop_preop_features.csv",dtype={c:"string" for c in spec["cat"]}).set_index("excel_row")
    plans=json.loads((INPUT/(line+"_preop_resampling.json")).read_text())
    part=next(p for p in plans if p["scheme"]=="final_development_inner" and p["fold"]==fold)
    tr=meta.loc[part["train_excel_rows"]];va=meta.loc[part["assessment_excel_rows"]]
    assert set(tr.group_hash).isdisjoint(va.group_hash)
    assert tr[line+"_partition_v2"].eq("development").all() and va[line+"_partition_v2"].eq("development").all()
    tx=features.loc[tr.index,cols];vx=features.loc[va.index,cols]
    keys=lambda m:(m.group_hash.astype(str)+"|"+m.excel_row.astype(str)).tolist()
    imputer=StableMixedImputer(spec,method,seed=42,iterations=5,trees=200,neighbors=5,threads=threads)
    filled_train=imputer.fit_transform(tx,keys(tr));filled_assessment=imputer.transform(vx,keys(va))
    for original,filled in [(tx,filled_train),(vx,filled_assessment)]:
        for c in cols:
            observed=original[c].notna()
            if c in spec["num"]:np.testing.assert_array_equal(original.loc[observed,c].to_numpy(),filled.loc[observed,c].to_numpy())
            else:assert original.loc[observed,c].astype(str).tolist()==filled.loc[observed,c].astype(str).tolist()
    # Real-data independent checks: same-case under single, reordered and batched inference.
    check=vx.loc[vx.isna().any(axis=1)].head(8)
    if len(check):
        cm=meta.loc[check.index];batch=imputer.transform(check,keys(cm))
        rev=imputer.transform(check.iloc[::-1],keys(cm.iloc[::-1])).iloc[::-1]
        solo=pd.concat([imputer.transform(check.iloc[i:i+1],keys(cm.iloc[i:i+1])) for i in range(len(check))])
        for other in [rev,solo]:
            for c in spec["num"]:np.testing.assert_allclose(batch[c],other[c],rtol=1e-12,atol=1e-12)
            for c in spec["cat"]:assert batch[c].tolist()==other[c].tolist()
    outputs={}
    for name,frame in [("train.csv",filled_train),("assessment.csv",filled_assessment),("train_metadata.csv",tr),("assessment_metadata.csv",va)]:
        temp=dest/(name+".tmp")
        frame.to_csv(temp,index=(name in ["train.csv","assessment.csv"]),index_label="excel_row")
        os.replace(temp,dest/name);outputs[name]=sha(dest/name)
    temp=dest/"imputer.joblib.tmp";joblib.dump(imputer,temp,compress=3);os.replace(temp,dest/"imputer.joblib");outputs["imputer.joblib"]=sha(dest/"imputer.joblib")
    receipt={"task":[line,fold,method],"train_n":len(tr),"inner_assessment_n":len(va),"train_AKI":int(tr.y.sum()),"inner_assessment_AKI":int(va.y.sum()),"imputer_received_no_outcome_fields":True,"global_hospital_time_evaluation_not_fitted":True,"observed_predictors_preserved":True,"case_stable_real_data_checks_passed":True,"inference_version":imputer.inference_version,"source_input_acceptance_sha256":sha(INPUT/"INPUT_ACCEPTANCE.json"),"iterations":5,"trees":200,"neighbors":5,"threads":threads,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"seconds":time.time()-start,"outputs_sha256":outputs}
    atomic(dest/"COMPLETE.json",receipt)
    print(json.dumps({"task":receipt["task"],"seconds":receipt["seconds"],"peak_rss_bytes":receipt["peak_rss_bytes"],"complete":True}),flush=True)
if __name__=="__main__":main()
