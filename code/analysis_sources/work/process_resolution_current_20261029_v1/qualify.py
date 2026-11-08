"""Numerical and boundary qualification for the specified post-audit methods."""
from pathlib import Path
import json,hashlib,time,resource
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
import analysis_core as ac
from calibration_solver import fit_update,apply_update
def independent_profile(y,p,w,kind,sd=2.5):
    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    def ia(b):return brentq(lambda a:np.sum(w*(expit(a+b*z)-y))+a/(sd*sd),-80,80,xtol=1e-11)
    if kind=="offset":return np.array([ia(1.),1.])
    def sb(b):
        a=ia(b);return np.sum(w*(expit(a+b*z)-y)*z)+(b-1)/(sd*sd)
    lo=1e-6
    if sb(lo)>=0:b=lo
    else:
        hi=2.
        while sb(hi)<0:hi*=2
        b=brentq(sb,lo,hi,xtol=1e-10)
    return np.array([ia(b),b])
def one(y,p,w):
    checks=[]
    for kind in ["offset","logistic"]:
        a=fit_update(y,p,w,kind);b=independent_profile(y,p,w,kind)
        err=float(np.max(np.abs(apply_update(p,a)-apply_update(p,b))))
        assert err<=1e-6,(kind,err,a,b)
        rev=fit_update(y[::-1],p[::-1],w[::-1],kind)
        assert np.max(np.abs(apply_update(p,a)-apply_update(p,rev)))<1e-6
        batch=apply_update(p,a);single=np.array([apply_update(p[i:i+1],a)[0] for i in range(len(p))])
        assert np.max(np.abs(batch-single))<=1e-12
        checks.append(err)
    return checks
def main():
    start=time.monotonic();allerrors=[];rng=np.random.default_rng(20261025)
    for i in range(24):
        n=40+i*3;p=expit(rng.normal(-2.5,.8,n));y=rng.binomial(1,expit(-1+np.log(p/(1-p))*.8)).astype(float);w=rng.integers(0,4,n).astype(float)
        if not w.sum():w[0]=1
        allerrors+=one(y,p,w)
    for y in [np.zeros(20),np.ones(20),np.r_[np.ones(10),np.zeros(10)]]:
        p=np.linspace(.01,.4,20);allerrors+=one(y,p,np.ones(20))
    y=np.array([0.,1.,0.,1.]);p=np.array([.05,.3,.2,.7]);w=np.array([2,1,0,3])
    for kind in ["offset","logistic"]:
        a=fit_update(y,p,w,kind);b=fit_update(np.repeat(y,w),np.repeat(p,w),np.ones(w.sum()),kind)
        assert np.max(np.abs(apply_update(p,a)-apply_update(p,b)))<1e-6
    rejected=0
    for args in [(y,p,np.zeros(4)),(y,p,np.array([1,1,-1,1])),(y,np.array([-.1,.2,.3,.4]),np.ones(4)),(np.array([0,1,2,0]),p,np.ones(4))]:
        try:fit_update(*args,"offset")
        except ValueError:rejected+=1
    assert rejected==4
    for i in range(20):
        x=rng.uniform(size=17);w=rng.integers(0,5,17)
        assert ac.higher_quantile(x,w)==np.quantile(np.repeat(x,w),.8,method="higher")
    ac.initialize();realchecks=0;metricchecks=0
    assert ac.BOOT_PROOF["INSPIRE_common6"]==json.loads((ac.R/"outputs/common6_public_evaluation_20261024_v1/EVALUATION.json").read_text())["receipt"]["resample_weight_sha256"]
    for ctx,ds in ac.DATA.items():
        m=ds["meta"];y=m.y.to_numpy(dtype=int);w=np.ones(len(m),dtype=int)
        for name,p in ds["pred"].items():
            q=ac.metrics(y,p,m.group_hash.to_numpy(),m.excel_row.to_numpy())
            assert abs(q["AUROC"]-roc_auc_score(y,p))<1e-12
            assert abs(q["AP"]-average_precision_score(y,p))<1e-12
            assert abs(q["Brier"]-brier_score_loss(y,p))<1e-12
            metricchecks+=3
            # All actual out-fold training boundaries, profile solver and weighted case.
            for k in range(3):
                tr=ds["fold"]!=k;te=~tr
                assert set(m.group_hash.to_numpy()[tr]).isdisjoint(m.group_hash.to_numpy()[te])
                allerrors+=one(y[tr].astype(float),p[tr],ac.BOOTS[ctx][19,tr].astype(float));realchecks+=1
            scores,detail=ac.cal_scores(ctx,name,w,True)
            for method in ac.METHODS:
                assert np.isfinite(detail[method]["prob"]).all()
                assert (detail[method]["prob"]>=0).all() and (detail[method]["prob"]<=1).all()
        if "alt" in ds:
            q=ac.alt_scores(ctx,w)
            base=("preop","compact") if ctx.endswith("preop") else ("preop","full")
            up=("preop","full") if ctx.endswith("preop") else ("endop","full")
            def independent_capture(key):
                order=np.lexsort((m.excel_row.to_numpy(),m.group_hash.to_numpy(),-ds["alt"][key]))
                return int(y[order[:int(np.ceil(len(m)*.2))]].sum())
            independent_DID=1000/len(m)*(independent_capture((*up,"omit_ALT"))-independent_capture((*base,"omit_ALT"))-independent_capture((*up,"matched_control"))+independent_capture((*base,"matched_control")))
            assert abs(q["change_in_gain_omit_ALT_minus_control"]["capacity20_captured_per1000"]-independent_DID)<1e-10
            draws=ac.random_draws(ctx,0,2)
            assert len(draws)==4
            for d in draws:assert d["upgraded_n"]==int(np.ceil(len(m)*d["upgrade_fraction"]))
        if "nonoverlap_mask" in ds:
            mask=ds["nonoverlap_mask"];geo=ac.DATA["geography_"+ctx.split("_")[1]]["meta"]
            assert set(m.loc[mask].group_hash).isdisjoint(geo.group_hash)
            out=ac.nonoverlap_scores(ctx,np.ones(int(mask.sum()),dtype=int))
            assert out and next(iter(out.values()))["n"]==int(mask.sum())
    for p,h in ac.PROTECTED.items():assert ac.sha(p)==h
    rec={"status":"PASS_solver_formula_profile_crosscheck_folds_and_existing_metric_adapters",
         "profile_max_probability_difference":max(allerrors),"real_training_fold_checks":realchecks,
         "independent_point_metric_checks":metricchecks,"invalid_inputs_rejected":rejected,
         "same_objective_numerical_Newton_refinement":True,"no_predictive_components_refitted":True,
         "source_protected_files_unchanged":True,"bootstrap_proofs":ac.BOOT_PROOF,
         "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         "seconds":time.monotonic()-start,
         "code_sha256":{p.name:ac.sha(p) for p in ac.W.glob("*.py")}}
    ac.save(ac.O/"QUALIFICATION.json",rec)
    print(json.dumps({k:v for k,v in rec.items() if k not in ["code_sha256","bootstrap_proofs"]}))
if __name__=="__main__":main()
