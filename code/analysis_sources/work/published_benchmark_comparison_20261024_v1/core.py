"""Source equations and matched benchmark statistics. No training or selection."""
from pathlib import Path
import sys, math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
sys.path.insert(0,str(ROOT/"work/published_benchmark_SIM_AKI_20261024_v1"))
from multinomial_helper import multinomial_probabilities
sys.path.insert(0,str(ROOT/"work/formal_evaluation_20261023_v1"))
import evaluation_stats as es

def egfr(age, male, creatinine_umol, year=2009):
    age=np.asarray(age,dtype=float);male=np.asarray(male,dtype=float)
    cr=np.asarray(creatinine_umol,dtype=float)/88.4
    if age.shape!=male.shape or age.shape!=cr.shape:raise ValueError("shape")
    if not np.isfinite(np.c_[age,male,cr]).all() or (cr<=0).any() or not np.isin(male,[0,1]).all():
        raise ValueError("Invalid observed eGFR inputs")
    female=male==0;k=np.where(female,.7,.9)
    if year==2009:
        return 141*np.minimum(cr/k,1)**np.where(female,-.329,-.411)*np.maximum(cr/k,1)**-1.209*.993**age*np.where(female,1.018,1.)
    if year==2021:
        return 142*np.minimum(cr/k,1)**np.where(female,-.241,-.302)*np.maximum(cr/k,1)**-1.2*.9938**age*np.where(female,1.012,1.)
    raise ValueError("Equation vintage")

def sim_matrix(frame, year=2009):
    a=frame.Age.to_numpy(float);s=frame.Sex.to_numpy(float);g=egfr(a,s,frame.PreopCr.to_numpy(float),year)
    x=np.column_stack([a,s,(frame.ASA.to_numpy(float)>=3),frame.Diabetes.to_numpy(float),
        np.zeros(len(frame)),np.ones(len(frame)),np.ones(len(frame)),
        frame.PreopAlb.to_numpy(float)<35,frame.PreopHb.to_numpy(float)<np.where(s==1,130,120),
        (g>=30)&(g<45),(g>=45)&(g<60),
        frame.IntraopTransfusion.to_numpy(float)>0,frame.OperationTime.to_numpy(float)/60])
    if not np.isfinite(x).all():raise ValueError("Incomplete inputs")
    return x,g

def sim_risk(frame, params,year=2009):
    x,g=sim_matrix(frame,year)
    pp=multinomial_probabilities(x,params["beta"],params["intercept"])
    return pp[:,1:].sum(axis=1),pp,g

def chen_score(frame):
    # aCCI>=4 follows standard age60+ plus active cancer; constant cannot change rank.
    if not (frame.Age.to_numpy(float)>=60).all():raise ValueError("Ineligible age")
    if not frame.SurgicalApproach.astype(str).isin(["1","1.0"]).all():raise ValueError("Pneumoperitoneum unknown")
    s=frame.Sex.to_numpy(float)
    return (.32*frame.Age.to_numpy(float)-.39*frame.PreopAlb.to_numpy(float)
        -.27*frame.PreopHb.to_numpy(float)+2.06
        +1.42*frame.Diabetes.to_numpy(float)+2.63*frame.Hypertension.to_numpy(float)
        +1.32*(frame.PreopAlb.to_numpy(float)<35)
        +1.37*(frame.PreopHb.to_numpy(float)<np.where(s==1,130,120)))

def rank_summary(y,score,g,r):
    y=np.asarray(y,dtype=float);score=np.asarray(score,dtype=float)
    if y.shape!=score.shape or not len(y) or not np.isfinite(score).all():raise ValueError("Unaligned score")
    if not np.isin(y,[0,1]).all():raise ValueError("Nonbinary outcome")
    ids=list(zip(map(str,g),map(str,r)))
    if len(set(ids))!=len(y):raise ValueError("Duplicate identity")
    e=int(y.sum())
    out={"n":len(y),"events":e,"event_rate":e/len(y),
         "AUROC":float(roc_auc_score(y,score)) if 0<e<len(y) else None,
         "AP":float(average_precision_score(y,score)) if e else None}
    order=np.array(sorted(range(len(y)),key=lambda i:(-score[i],ids[i])))
    for f in [.1,.2]:
        mask=np.zeros(len(y),dtype=bool);mask[order[:math.ceil(f*len(y))]]=True
        for k,v in es.counts(y,mask).items():
            if k not in ["n","events"]:out[f"capacity{int(f*100)}_{k}"]=v
    return out

def canonical_context(d,predictions,kind):
    order=np.array(sorted(range(len(d)),key=lambda i:(str(d.center.iloc[i]),str(d.group_hash.iloc[i]),str(d.excel_row.iloc[i]))))
    z=d.iloc[order].reset_index(drop=True)
    y=z.y.to_numpy(int);g=z.group_hash.astype(str).to_numpy();r=z.excel_row.astype(str).to_numpy()
    cc=z.center.astype(str).to_numpy()
    yy,gg,rr,ccc,strata,_=es._design(y,g,r,cc,{"constant_design_only":np.full(len(z),.5)})
    assert np.array_equal(yy,y) and np.array_equal(gg,g) and np.array_equal(rr,r)
    return {"y":y,"g":g,"r":r,"strata":strata,
            "predictions":{k:np.asarray(v)[order] for k,v in predictions.items()},"kind":kind,
            "excel_rows":z.excel_row.to_list()}

def context_scores(ctx,idx=None,g=None,r=None):
    y=ctx["y"] if idx is None else ctx["y"][idx]
    g=ctx["g"] if g is None else g;r=ctx["r"] if r is None else r
    out={}
    for name,values in ctx["predictions"].items():
        v=values if idx is None else values[idx]
        out[name]=rank_summary(y,v,g,r) if ctx["kind"]=="rank" else es.summary(y,v,g,r)
    return out

def finite_number(v):
    return v is not None and np.isfinite(v)
