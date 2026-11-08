"""Qualified frozen inputs for post-audit support only; preserve source artefacts."""
from pathlib import Path
import os,sys,json,hashlib,pickle
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
O=R/"outputs/process_resolution_20261025_v1";W=Path(__file__).resolve().parent
S=R/"outputs/v4_development_inputs_20261018"
sources={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):p=Path(p);sources[str(p)]=sha(p);return p
def read(p):return json.loads(track(p).read_text())
def csv(p):return pd.read_csv(track(p),dtype={"group_hash":str},float_precision="round_trip")
def normalized(df,public=False):
    d=df.copy()
    if public:d=d.rename(columns={"y_scr":"y","record_key":"excel_row"});d["center"]="INSPIRE"
    d["excel_row"]=d.excel_row.astype(str);d["group_hash"]=d.group_hash.astype(str);d["center"]=d.center.astype(str)
    assert d.excel_row.is_unique and d.y.isin([0,1]).all()
    order=sorted(range(len(d)),key=lambda i:(d.center.iloc[i],d.group_hash.iloc[i],d.excel_row.iloc[i]))
    return d.iloc[order].reset_index(drop=True)
def dataset(df):
    m=df[["excel_row","group_hash","center","y"]].copy()
    groups=m.group_hash.to_numpy();cent=m.center.to_numpy();y=m.y.to_numpy(dtype=int)
    strata=[]
    for c in sorted(set(cent)):
        strata.append([np.flatnonzero((cent==c)&(groups==g)) for g in sorted(set(groups[cent==c]))])
    fold=np.full(len(m),-1,dtype=int)
    for k,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261025).split(np.zeros(len(m)),y,groups)):
        assert set(groups[tr]).isdisjoint(groups[va])
        fold[va]=k
    assert (fold>=0).all()
    return {"meta":m,"fold":fold,"strata":strata,"pred":{},"cal_routes":{},"policies":{}}
def put(ds,name,df):
    assert ds["meta"].equals(df[["excel_row","group_hash","center","y"]])
    ds["pred"][name]=df.prob_calibrated.to_numpy(dtype=np.float64)
def modelname(m):return "_".join(m["identity"][x] for x in ["stage","branch","role","family"])
def sensitivity(route,scenario):
    q=read(R/"outputs/fixed_flow_sensitivity_20261024_v1/routes"/route/scenario/"COMPLETE.json")
    fit=Path(q["fit_cache"]);assert sha(fit/"COMPLETE.json")==q["fit_receipt_sha256"]
    rec=read(fit/"COMPLETE.json")
    p=fit/"PRIVATE_HELDOUT_PREDICTIONS.csv";assert sha(p)==rec["files"][p.name]
    return normalized(csv(p))
def main():
    assert read(O/"ANALYSIS_CONTRACT.json")["preserve_original_V1_models_thresholds_data_and_evaluations"]
    data={};locks={};path_name={}
    accepted=read(R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json")
    for ref in accepted["models"]:
        if not ref["primary_route_winner"]:continue
        assert sha(ref["path"])==ref["sha256"];m=read(ref["path"]);i=m["identity"];ctx=i["line"]+"_"+i["cohort"];name=modelname(m)
        path=R/"outputs/formal_evaluation_20261023_v1"/ctx/"private_predictions"/name/"PREDICTIONS.csv"
        q=read(path.with_name("COMPLETE.json"));assert sha(path)==q["sha256"]
        df=normalized(csv(path))
        if ctx not in data:data[ctx]=dataset(df)
        put(data[ctx],name,df);data[ctx]["cal_routes"][name]={"id":i["line"]+"_"+i["cohort"]+"_"+i["stage"]+"_"+i["branch"],"thresholds":m["training_thresholds"],"manifest_path":ref["path"]}
        locks[(ctx,name)]=m;path_name[ref["path"]]=(ctx,name)
    for ref in read(R/"outputs/selective_policy_lock_20261023_v1/STAGE_POLICY_LOCK_ACCEPTANCE.json")["policy_manifests"]:
        assert sha(ref["path"])==ref["sha256"];m=read(ref["path"]);i=m["identity"];ctx=i["line"]+"_"+i["cohort"]
        cal=m["calibrator"];assert sha(cal["path"])==cal["sha256"];track(cal["path"])
        name=f"policy_{i['upgrade_fraction']}_{i['mode']}_{i['kind']}"
        ds=data[ctx];ds["policies"][name]={"lock":m,"calibrator_path":cal["path"],"base":path_name[m["base_model"]["path"]][1],"upgrade":path_name[m["upgrade_model"]["path"]][1]}
    pub=read(R/"outputs/common6_public_evaluation_20261024_v1/PRE_SCORE_LOCK.json")
    for ref in pub["models"]:
        assert sha(ref["path"])==ref["sha256"];m=read(ref["path"]);line=m["identity"]["line"]
        df=normalized(csv(R/"outputs/common6_public_evaluation_20261024_v1"/line/"PRIVATE_PREDICTIONS.csv"),True)
        ctx="INSPIRE_common6"
        if ctx not in data:data[ctx]=dataset(df)
        name="common6_"+line
        put(data[ctx],name,df);data[ctx]["cal_routes"][name]={"id":name,"thresholds":m["training_thresholds"],"manifest_path":ref["path"]}
    for ctx in ["geography_preop","temporal_preop","geography_paired","temporal_paired"]:
        ds=data[ctx];line,cohort=ctx.split("_")
        stages=[("preop","compact"),("preop","full")] if cohort=="preop" else [("preop","full"),("endop","full")]
        alt={}
        for stage,branch in stages:
            route=f"{line}_{cohort}_{stage}_{branch}"
            for sc in ["matched_control","omit_ALT"]:
                df=sensitivity(route,sc)
                assert ds["meta"].equals(df[["excel_row","group_hash","center","y"]])
                alt[(stage,branch,sc)]=df.prob_calibrated.to_numpy(dtype=float)
        ds["alt"]=alt
        if line=="temporal":
            geo=data["geography_"+cohort]["meta"]
            # Remove whole groups, not merely matching operation rows.
            ds["nonoverlap_mask"]=~ds["meta"].group_hash.isin(geo.group_hash).to_numpy()
    # Fold identities and row counts are private source-level artefacts, not new data.
    summary=[]
    for ctx,ds in data.items():
        m=ds["meta"];p=O/(ctx+"_PRIVATE_FOLDS.csv");out=m.copy();out["fold"]=ds["fold"];out.to_csv(p,index=False)
        summary.append({"context":ctx,"n":len(m),"events":int(m.y.sum()),"groups":int(m.group_hash.nunique()),"cal_routes":len(ds["cal_routes"]),
                        "fold_events":[int(m.y.to_numpy()[ds["fold"]==k].sum()) for k in range(3)],
                        "retained_nonoverlap_n":int(ds["nonoverlap_mask"].sum()) if "nonoverlap_mask" in ds else None})
    with (W/"PRIVATE_INPUTS.pkl").open("wb") as f:pickle.dump(data,f,protocol=5)
    current_inputs=read(S/"SOURCE_LOCK.json")
    assert sha(S/"SOURCE_LOCK.json")=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
    baseline=read(R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json")
    protected={f:v["sha256"] for f,v in baseline["files"].items()}
    protected.update(sources)
    expected={"geography_preop":(1159,72),"temporal_preop":(1683,66),"geography_paired":(1158,72),"temporal_paired":(1011,36),"INSPIRE_common6":(6043,253)}
    assert set(data)==set(expected)
    for ctx,ds in data.items():
        assert (len(ds["meta"]),int(ds["meta"].y.sum()))==expected[ctx]
        for k in range(3):
            fold=ds["fold"]
            assert set(ds["meta"].loc[fold==k,"group_hash"]).isdisjoint(ds["meta"].loc[fold!=k,"group_hash"])
    for p,h in protected.items():assert sha(p)==h
    manifest={"status":"PASS_source_identity_and_group_fold_preparation","datasets":summary,"sources_sha256":sources,
              "protected_frozen_files_sha256":protected,"prepared_pickle_sha256":sha(W/"PRIVATE_INPUTS.pkl"),"contract_sha256":sha(O/"ANALYSIS_CONTRACT.json"),
              "no_predictive_model_refitting":True,"target_updates_not_yet_scored":True}
    (O/"INPUT_ACCEPTANCE.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+chr(10))
    print(json.dumps({"datasets":summary,"source_files":len(sources),"protected_files":len(protected)},ensure_ascii=False))
if __name__=="__main__":main()
