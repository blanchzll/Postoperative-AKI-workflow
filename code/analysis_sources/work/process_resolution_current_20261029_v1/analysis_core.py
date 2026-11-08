"""Targeted support estimands. No re-selection or predictive-component fitting."""
from pathlib import Path
import os,sys,json,pickle,hashlib,math
import numpy as np,pandas as pd
from scipy.special import expit
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
O=R/"outputs/process_resolution_20261025_v1";W=Path(__file__).resolve().parent
sys.path.insert(0,str(R/"work/formal_evaluation_20261023_v1"))
import evaluation_stats as es
from fixed_prediction_metrics import FixedRisk
from calibration_solver import fit_update,apply_update
import joblib
METHODS=["original_V1","offset_MAP","logistic_MAP"]
DCA=[.01,.02,.03,.05,.075,.1,.15,.2]
DATA={};BOOTS={};BOOT_PROOF={};PROTECTED={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+"."+str(os.getpid())+".tmp")
    t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+chr(10));t.replace(p)
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,)):return bool(x)
    return x
def weights_for(meta,reps=2000,seed=20261017):
    centers=meta.center.to_numpy();groups=meta.group_hash.to_numpy();n=len(meta)
    rng=np.random.default_rng(seed);w=np.zeros((reps,n),dtype=np.int64)
    for rep in range(reps):
        for c in sorted(set(centers)):
            idx=np.flatnonzero(centers==c);gg=sorted(set(groups[idx]));gi={g:i for i,g in enumerate(gg)}
            counts=np.bincount(rng.integers(0,len(gg),len(gg)),minlength=len(gg))
            w[rep,idx]=counts[np.array([gi[g] for g in groups[idx]])]
    return w
def initialize():
    global DATA,BOOTS,PROTECTED
    rec=json.loads((O/"INPUT_ACCEPTANCE.json").read_text())
    assert sha(W/"PRIVATE_INPUTS.pkl")==rec["prepared_pickle_sha256"]
    assert sha(O/"ANALYSIS_CONTRACT.json")==rec["contract_sha256"]
    with (W/"PRIVATE_INPUTS.pkl").open("rb") as f:DATA=pickle.load(f)
    PROTECTED=rec["protected_frozen_files_sha256"]
    for ctx,ds in DATA.items():
        for item in ds["policies"].values():
            item["calibrator"]=joblib.load(item["calibrator_path"])
        BOOTS[ctx]=weights_for(ds["meta"])
        proof=hashlib.sha256(BOOTS[ctx].astype("<i8").tobytes()).hexdigest();BOOT_PROOF[ctx]=proof
        if ctx!="INSPIRE_common6":
            original=json.loads((R/"outputs/formal_evaluation_20261023_v1"/ctx/"EVALUATION.json").read_text())["receipt"]["resample_weight_sha256"]
            assert proof==original,(ctx,"weight identities differ")
        if "nonoverlap_mask" in ds:
            m=ds["meta"].loc[ds["nonoverlap_mask"]].reset_index(drop=True)
            BOOTS[ctx+"__nonoverlap"]=weights_for(m);BOOT_PROOF[ctx+"__nonoverlap"]=hashlib.sha256(BOOTS[ctx+"__nonoverlap"].astype("<i8").tobytes()).hexdigest()
    for arr in BOOTS.values():arr.setflags(write=False)
def counts(y,a,w):
    n=float(w.sum());events=float(w@y);review=float(w@a);tp=float(w@(a*y));fp=review-tp
    return {"reviewed_n":review,"captured_n":tp,"missed_n":events-tp,"false_alerts_n":fp,
            "reviewed_per1000":review/n*1000,"captured_per1000":tp/n*1000,"missed_per1000":(events-tp)/n*1000,
            "sensitivity":tp/events if events else np.nan,"PPV":tp/review if review else np.nan}
def metrics(y,p,g,r,w=None,cuts=None,localmask=None):
    if w is None:w=np.ones(len(y),dtype=np.int64)
    risk=FixedRisk(y,p,g,r);q=risk.score(w)
    q["logloss"]=float(np.average(np.logaddexp(0,np.log(np.clip(p,1e-15,1-1e-15)/(1-np.clip(p,1e-15,1-1e-15))))-y*np.log(np.clip(p,1e-15,1-1e-15)/(1-np.clip(p,1e-15,1-1e-15))),weights=w))
    n=float(w.sum());events=float(w@y)
    for cap in [10,20]:
        a=risk.review_counts(cap/100,w);review=float(a.sum());tp=float(a@y)
        vals={"reviewed_n":review,"captured_n":tp,"missed_n":events-tp,"false_alerts_n":review-tp,
              "captured_per1000":1000*tp/n,"sensitivity":tp/events if events else np.nan,"PPV":tp/review if review else np.nan}
        q.update({f"capacity{cap}_{k}":v for k,v in vals.items()})
    rules={"fixed05":.05}
    if cuts:
        for rule in ["training_sensitivity80","training_capacity10","training_capacity20"]:
            if rule in cuts:rules["source_"+rule]=cuts[rule]
    for rule,t in rules.items():
        c=counts(y,(p>=t).astype(float),w)
        q.update({rule+"_"+k:v for k,v in c.items()})
    if localmask is not None:q.update({"adaptation_capacity20_"+k:v for k,v in counts(y,localmask.astype(float),w).items()})
    for t in DCA:q["NB_"+str(t)]=risk.net_benefit(t,w)["net_benefit"]
    return q
def higher_quantile(p,w,q=.8):
    order=np.argsort(p,kind="stable");cum=np.cumsum(w[order]);n=int(cum[-1])
    assert n>0
    ix=math.ceil(q*(n-1));j=np.searchsorted(cum,ix,side="right")
    return float(p[order[j]])
def crossed(ds,p,w,kind):
    y=ds["meta"].y.to_numpy(dtype=int);fold=ds["fold"];q=np.zeros(len(y));mask=np.zeros(len(y),dtype=bool);coefs=[]
    for k in range(3):
        tr=fold!=k;te=~tr
        if kind=="original_V1":c=np.array([0.,1.]);q[te]=p[te];train=p[tr]
        else:
            c=fit_update(y[tr],p[tr],w[tr],"offset" if kind=="offset_MAP" else "logistic")
            q[te]=apply_update(p[te],c);train=apply_update(p[tr],c)
        cut=higher_quantile(train,w[tr],.8);mask[te]=q[te]>=cut
        coefs.append({"fold":k,"a":c[0],"b":c[1],"adaptation_records_weighted":int(w[tr].sum()),"adaptation_events_weighted":int(w[tr]@y[tr]),"capacity20_threshold":cut})
    return q,mask,coefs
def alt_scores(ctx,w):
    ds=DATA[ctx];m=ds["meta"];y=m.y.to_numpy(dtype=int);g=m.group_hash.to_numpy();r=m.excel_row.to_numpy();a=ds["alt"]
    cohort=ctx.split("_")[1]
    base=("preop","compact") if cohort=="preop" else ("preop","full")
    upgrade=("preop","full") if cohort=="preop" else ("endop","full")
    result={}
    for sc in ["matched_control","omit_ALT"]:
        result["base_"+sc]=metrics(y,a[(*base,sc)],g,r,w)
        result["upgrade_"+sc]=metrics(y,a[(*upgrade,sc)],g,r,w)
    fields=["AUROC","AP","Brier","capacity10_captured_per1000","capacity20_captured_per1000"]
    for sc in ["matched_control","omit_ALT"]:
        result["gain_"+sc]={k:result["upgrade_"+sc][k]-result["base_"+sc][k] for k in fields}
    result["change_in_gain_omit_ALT_minus_control"]={k:result["gain_omit_ALT"][k]-result["gain_matched_control"][k] for k in fields}
    return result
def cal_scores(ctx,name,w,with_details=False):
    ds=DATA[ctx];m=ds["meta"];y=m.y.to_numpy(dtype=int);g=m.group_hash.to_numpy();r=m.excel_row.to_numpy();p=ds["pred"][name];cuts=ds["cal_routes"][name]["thresholds"]
    result={};detail={}
    for kind in METHODS:
        q,mask,coef=crossed(ds,p,w,kind)
        result[kind]=metrics(y,q,g,r,w,cuts,mask)
        if with_details:
            result[kind].update(es.calibration(y,q))
            detail[kind]={"prob":q,"adaptation_mask":mask,"coefficients":coef}
    for kind in METHODS[1:]:
        result[kind+"_MINUS_original_V1"]={k:result[kind][k]-result["original_V1"][k] if result[kind][k] is not None and result["original_V1"][k] is not None else np.nan for k in result["original_V1"] if k in result[kind]}
    return (result,detail) if with_details else result
def nonoverlap_scores(ctx,w):
    ds=DATA[ctx];keep=ds["nonoverlap_mask"];m=ds["meta"].loc[keep].reset_index(drop=True)
    y=m.y.to_numpy(dtype=int);g=m.group_hash.to_numpy();r=m.excel_row.to_numpy()
    pred={k:v[keep] for k,v in ds["pred"].items()};result={}
    for name,p in pred.items():result[name]=metrics(y,p,g,r,w)
    idx,gg,rr=es.expanded(w,g,r);yy=y[idx]
    for name,it in ds["policies"].items():
        raw,cal,mask=es.route(it,pred[it["base"]][idx],pred[it["upgrade"]][idx],gg,rr)
        result[name]=metrics(yy,cal,gg,rr);result[name]["upgraded_per1000"]=1000*mask.mean()
    base=next(k for k in pred if k.startswith("preop_"+("compact" if ctx.endswith("preop") else "full")))
    up=next(k for k in pred if k.startswith(("preop" if ctx.endswith("preop") else "endop")+"_full"))
    comparisons=[(up,base)]
    for name,it in ds["policies"].items():
        comparisons.extend([(name,base),(name,up)])
        if it["lock"]["identity"]["mode"]=="capacity" and it["lock"]["identity"]["kind"]=="selective":
            i=it["lock"]["identity"];comparisons.append((name,f"policy_{i['upgrade_fraction']}_capacity_random"))
    for a,b in comparisons:
        result[a+" MINUS "+b]={k:result[a][k]-result[b][k] if result[a][k] is not None and result[b][k] is not None else np.nan for k in result[b] if k in result[a]}
    return result
def random_draws(ctx,start,stop):
    ds=DATA[ctx];m=ds["meta"];y=m.y.to_numpy(dtype=int);g=m.group_hash.to_numpy();r=m.excel_row.to_numpy();n=len(y);rows=[]
    order=np.array(sorted(range(n),key=lambda i:(g[i],r[i])),dtype=int)
    for j in range(start,stop):
        perm=order[np.random.default_rng(20261025+j).permutation(n)]
        for fraction in [.25,.5]:
            it=ds["policies"][f"policy_{fraction}_capacity_random"]
            mask=np.zeros(n,dtype=bool);mask[perm[:math.ceil(n*fraction)]]=True
            mixed=ds["pred"][it["base"]].copy();mixed[mask]=ds["pred"][it["upgrade"]][mask]
            cal=es.apply_calibration(it["calibrator"],mixed)
            q=metrics(y,cal,g,r)
            rows.append({"context":ctx,"draw":j,"upgrade_fraction":fraction,"upgraded_n":int(mask.sum()),**{k:v for k,v in q.items() if k.startswith("capacity")}})
    return rows
def task(job):
    kind,ctx,name,start,stop=job
    if kind=="random":return random_draws(ctx,start,stop)
    result=[];key=ctx+"__nonoverlap" if kind=="nonoverlap" else ctx
    for rep in range(start,stop):
        w=BOOTS[key][rep]
        if kind=="alt":q=alt_scores(ctx,w)
        elif kind=="cal":q=cal_scores(ctx,name,w)
        else:q=nonoverlap_scores(ctx,w)
        result.append(q)
    return result
