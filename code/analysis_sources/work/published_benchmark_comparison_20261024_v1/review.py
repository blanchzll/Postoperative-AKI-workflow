"""Separate scalar/weighted-metric and all-checkpoint review; no model fitting."""
from pathlib import Path
import json,hashlib,os,fcntl,sys,math,datetime
import pandas as pd
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score
from core import ROOT,es
OUT=ROOT/"outputs/published_benchmark_comparison_20261024_v1"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((OUT/"FIT_AND_EVALUATION_ACCEPTANCE.json").read_text())
for fn,h in receipt["files_sha256"].items():assert sha(OUT/fn)==h,fn
contract=json.loads((OUT/"ANALYSIS_CONTRACT.json").read_text())
for fn,h in contract["sources_sha256"].items():assert sha(ROOT/fn)==h,fn
for fn,h in json.loads((OUT/"ORIGINAL_PREDICTION_RECEIPTS.json").read_text()).items():assert sha(ROOT/fn)==h["sha256"],fn
owner=json.loads((OUT/"OWNER.json").read_text());p=Path(f"/proc/{owner['pid']}/stat")
assert not p.exists() or p.read_text().split(") ",1)[1].split()[19]!=owner["start_ticks"],"Owner still running"
with open(OUT/".run.lock","a") as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
flow=pd.read_csv(OUT/"COHORT_FLOW.csv")
assert flow.loc[flow.reference=="SIM_AKI","eligible_all"].to_list()==[2389,2389]
assert flow.loc[flow.reference=="Chen2026","eligible_all"].to_list()==[276,276]
point=pd.read_csv(OUT/"POINT_ESTIMATES.csv")
interval=pd.read_csv(OUT/"INTERVALS.csv");delta=pd.read_csv(OUT/"PAIRED_DIFFERENCES.csv")
boot=json.loads((OUT/"BOOTSTRAP_RECEIPTS.json").read_text())
params=json.loads((ROOT/"outputs/published_benchmark_SIM_AKI_20261024_v1/PUBLISHED_COEFFICIENTS.json").read_text())
cal=json.loads((OUT/"DEVELOPMENT_RECALIBRATORS.json").read_text())
x=pd.read_csv(ROOT/"outputs/v4_development_inputs_20261018/paired_endop_features.csv").set_index("excel_row")
xf=pd.read_csv(ROOT/"outputs/v4_development_inputs_20261018/full_source_endop_features.csv").set_index("excel_row")
def renal(a,s,cr,yr):
    cr=cr/88.4;k=.9 if s==1 else .7
    if yr==2009:
        power=(-.411 if s==1 else -.329) if cr<=k else -1.209
        return 141*(cr/k)**power*.993**a*(1 if s==1 else 1.018)
    power=(-.302 if s==1 else -.241) if cr<=k else -1.2
    return 142*(cr/k)**power*.9938**a*(1 if s==1 else 1.012)
def independent_summary(y,v,g,r,w=None):
    y=np.asarray(y);v=np.asarray(v);w=np.ones(len(y),dtype=int) if w is None else w
    n=int(w.sum());e=int(w@y);q={}
    if 0<e<n:q["AUROC"]=float(roc_auc_score(y,v,sample_weight=w))
    if e:q["AP"]=float(average_precision_score(y,v,sample_weight=w))
    order=sorted(range(len(y)),key=lambda i:(-v[i],str(g[i]),str(r[i])))
    for frac in [.1,.2]:
        left=math.ceil(frac*n);capt=0
        for ix in order:
            k=min(left,int(w[ix]));capt+=k*int(y[ix]);left-=k
            if not left:break
        q[f"capacity{int(frac*100)}_captured_n"]=capt
        q[f"capacity{int(frac*100)}_captured_per1000"]=1000*capt/n
    return q
checks=0;rows_checked=0
for name,br in boot.items():
    line=br["line"];is_sim=name.startswith("SIM")
    file=OUT/(f"private_SIM_{line}_predictions.csv" if is_sim else f"private_Chen_{line}_scores.csv")
    d=pd.read_csv(file);d["record_key"]=d.excel_row.astype(str);d["center_key"]=d.center.astype(str)
    d=d.sort_values(["center_key","group_hash","record_key"]).reset_index(drop=True)
    cols=[v for v in d.columns if v.startswith("our_") or v.startswith("SIM_") or v=="Chen_Table4_ordinal_score"]
    for row in d.itertuples():
        z=(x if is_sim else xf).loc[row.excel_row]
        if is_sim:
            for year in [2009,2021]:
                g=renal(z.Age,z.Sex,z.PreopCr,year)
                cov=[z.Age,z.Sex,int(z.ASA>=3),z.Diabetes,0,1,1,int(z.PreopAlb<35),
                     int(z.PreopHb<(130 if z.Sex==1 else 120)),int(30<=g<45),int(45<=g<60),
                     int(z.IntraopTransfusion>0),z.OperationTime/60]
                odds=[math.exp(params["intercept"][j]+sum(a*b for a,b in zip(cov,params["beta"][j]))) for j in range(2)]
                p=sum(odds)/(1+sum(odds));np.testing.assert_allclose(getattr(row,f"SIM_original_{year}"),p,atol=2e-15)
                cc=cal[f"{line}_{year}"];mapped=expit(cc["intercept"]+cc["slope"]*logit(np.clip(p,1e-6,1-1e-6)))
                np.testing.assert_allclose(getattr(row,f"SIM_development_recalibrated_{year}"),mapped,atol=2e-14)
        else:
            # Source Table4, independently scalar, no intercept or fake probabilities.
            raw=.32*z.Age-.39*z.PreopAlb-.27*z.PreopHb+2.06+1.42*z.Diabetes+2.63*z.Hypertension
            raw+=1.32*int(z.PreopAlb<35)+1.37*int(z.PreopHb<(130 if z.Sex==1 else 120))
            np.testing.assert_allclose(row.Chen_Table4_ordinal_score,raw,atol=2e-14)
        rows_checked+=1
    y=d.y.to_numpy();g=d.group_hash.to_numpy();r=d.record_key.to_numpy()
    yy,gg,rr,cc,strata,_=es._design(y,g,r,d.center_key.to_numpy(),{"design":np.full(len(d),.5)})
    assert np.array_equal(y,yy) and np.array_equal(g,gg)
    for col in cols:
        q=independent_summary(y,d[col].to_numpy(),g,r)
        p=point.loc[(point.context==name)&(point.model==col)].iloc[0]
        for k,v in q.items():np.testing.assert_allclose(p[k],v,atol=2e-14)
    rng=np.random.default_rng(20261017);proof=hashlib.sha256();stream=hashlib.sha256()
    by={};paired={};zero=allpos=0
    comparison_names=delta.loc[delta.context==name,"comparison"].unique()
    for rep in range(2000):
        path=OUT/"checkpoints"/name/f"{rep:04d}.json";q=json.loads(path.read_text())
        assert q["signature"]==receipt["signature"]
        weights=es.group_weights(strata,len(d),rng);proof.update(weights.astype("<i8").tobytes());stream.update(path.read_bytes())
        assert hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest()==q["weight_sha256"]
        zero+=q["zero_events"];allpos+=q["all_events"]
        if rep in [0,1,99,1999] or q["zero_events"]:
            for col in cols:
                qq=independent_summary(y,d[col].to_numpy(),g,r,weights)
                for k,v in qq.items():np.testing.assert_allclose(q["scores"][col][k],v,atol=2e-13)
        for model,metrics in q["scores"].items():
            for k,v in metrics.items():
                if k not in ["n","events"] and v is not None and np.isfinite(v):by.setdefault((model,k),[]).append(v)
        for comparison in comparison_names:
            a,b=comparison.split(" MINUS ")
            for k,v in q["scores"][a].items():
                bv=q["scores"][b].get(k)
                if k not in ["n","events"] and v is not None and bv is not None and np.isfinite(v) and np.isfinite(bv):
                    paired.setdefault((comparison,k),[]).append(v-bv)
        checks+=1
    assert proof.hexdigest()==br["resample_weight_sha256"]
    assert stream.hexdigest()==br["checkpoint_stream_sha256"]
    assert zero==br["zero_event_replicates"] and allpos==br["all_event_replicates"]
    for tab,values,who in [(interval,by,"model"),(delta,paired,"comparison")]:
        for row in tab.loc[tab.context==name].itertuples():
            key=(getattr(row,who),row.metric);v=values[key]
            assert len(v)==row.valid_replicates
            np.testing.assert_allclose([row.lower95,row.upper95],np.quantile(v,[.025,.975]),atol=2e-12)
    if not is_sim:
        assert not interval.loc[interval.context==name,"metric"].str.contains("Brier|calibration|net_benefit|expected").any()
result={"status":"PASS_published_formula_predictions_and_all_bootstrap_intervals_reviewed",
 "reviewed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 "source_prediction_rows_scalar_reverified":rows_checked,"checkpoints_reverified":checks,
 "checks":["source and accepted predictions hashes","source renal/clinical formula scalar reimplementation",
 "development-only recalibration arithmetic","independent sklearn weighted AUC/AP and weighted exact-capacity allocations",
 "all8000 hospital-group resample seeds and checkpoint hashes","every reported interval and paired interval recomputed",
 "rank-only Chen has no invented native probability metrics","original owner exited and lock released"],
 "review_type":"separate verification implementation under Codex review, not independent scientific reviewer",
 "computation_receipt_sha256":sha(OUT/"FIT_AND_EVALUATION_ACCEPTANCE.json"),
 "review_code_sha256":sha(__file__),"main_cohorts_and_original_models_unchanged":True,"whole_project_complete":False}
(OUT/"REVIEW_ACCEPTANCE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result))
