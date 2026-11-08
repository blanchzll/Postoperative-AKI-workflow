"""Finite same-patient source-model benchmark; all learning confined to source development."""
from pathlib import Path
import sys,json,hashlib,os,time,datetime,fcntl,traceback,resource,multiprocessing as mp,warnings
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
import sklearn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"work/published_benchmark_comparison_20261024_v1"))
from core import *
OUT=ROOT/"outputs/published_benchmark_comparison_20261024_v1"
WORK=ROOT/"work/published_benchmark_comparison_20261024_v1"
INPUT=ROOT/"outputs/v4_development_inputs_20261018"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,p)
def serial(value):
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value
def observed(d,cols):
    return d[cols].apply(pd.to_numeric,errors="coerce").notna().all(axis=1)
def eligible_sim(d):
    cols=["Age","Sex","ASA","Diabetes","PreopHb","PreopAlb","PreopCr","IntraopTransfusion","OperationTime"]
    d=d.copy();valid=observed(d,cols)
    valid &= d.Sex.isin([0,1])&d.Diabetes.isin([0,1])&d.ASA.isin([1,2,3,4,5])
    valid &= (d.Age>=18)&(d.PreopCr>0)&(d.PreopCr<176.8)&(d.PreopHb>0)&(d.PreopAlb>0)
    valid &= (d.IntraopTransfusion>=0)&(d.OperationTime>0)
    # Compute renal eligibility only after observing complete valid inputs; no filling.
    z=d.loc[valid];g09=egfr(z.Age,z.Sex,z.PreopCr,2009);g21=egfr(z.Age,z.Sex,z.PreopCr,2021)
    ix=z.index[(g09>=30)&(g21>=30)]
    eligible=pd.Series(False,index=d.index);eligible.loc[ix]=True
    return eligible
def eligible_chen(d):
    cols=["Age","Sex","PreopHb","PreopAlb","Diabetes","Hypertension"]
    return (observed(d,cols)&d.Age.ge(60)&d.SurgicalApproach.astype(str).isin(["1","1.0"])
        &d.Sex.isin([0,1])&d.Diabetes.isin([0,1])&d.Hypertension.isin([0,1])
        &(d.PreopHb>0)&(d.PreopAlb>0))
def current_winner(line,cohort,stage,branch):
    refs=json.loads((ROOT/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json").read_text())["models"]
    candidates=[]
    for ref in refs:
        assert sha(ref["path"])==ref["sha256"]
        m=json.loads(Path(ref["path"]).read_text());identity=m["identity"]
        if ref["primary_route_winner"] and (identity["line"],identity["cohort"],identity["stage"],identity["branch"])==(line,cohort,stage,branch):
            candidates.append("_".join(identity[k] for k in ["stage","branch","role","family"]))
    assert len(candidates)==1
    return candidates[0]
PRED_RECEIPTS={}
def frozen_prediction(line,cohort,route,rows):
    directory=ROOT/f"outputs/formal_evaluation_20261023_v1/{line}_{cohort}/private_predictions/{route}"
    p=directory/"PREDICTIONS.csv";receipt=json.loads((directory/"COMPLETE.json").read_text())
    assert sha(p)==receipt["sha256"],p
    d=pd.read_csv(p).set_index("excel_row",verify_integrity=True)
    chosen=d.loc[rows.excel_row].reset_index()
    for k in ["excel_row","group_hash","center","y","stage"]:
        assert chosen[k].astype(str).tolist()==rows[k].astype(str).tolist(),(p,k)
    PRED_RECEIPTS[str(p.relative_to(ROOT))]={"sha256":sha(p),"receipt_sha256":sha(directory/"COMPLETE.json")}
    return chosen
def prepare():
    contract=json.loads((OUT/"ANALYSIS_CONTRACT.json").read_text())
    qual=json.loads((OUT/"QUALIFICATION.json").read_text())
    assert qual["core_sha256"]==sha(WORK/"core.py")
    assert qual["contract_sha256"]==sha(OUT/"ANALYSIS_CONTRACT.json")
    for fn,h in contract["sources_sha256"].items():assert sha(ROOT/fn)==h,fn
    params=json.loads((ROOT/"outputs/published_benchmark_SIM_AKI_20261024_v1/PUBLISHED_COEFFICIENTS.json").read_text())
    full=pd.read_csv(INPUT/"full_source_endop_features.csv").merge(pd.read_csv(INPUT/"preop_metadata.csv"),on="excel_row",validate="one_to_one")
    paired=pd.read_csv(INPUT/"paired_endop_features.csv").merge(pd.read_csv(INPUT/"paired_metadata.csv"),on="excel_row",validate="one_to_one")
    assert (len(full),int(full.y.sum()))==(3710,205)
    assert (len(paired),int(paired.y.sum()))==(2428,138)
    simmask=eligible_sim(paired);chenmask=eligible_chen(full)
    for d,mask,name in [(paired,simmask,"SIM_AKI"),(full,chenmask,"Chen2026")]:
        led=d[["excel_row","group_hash","center","y","geography_partition_v2","temporal_partition_v2"]].copy()
        led["eligible"]=mask.to_numpy();led.to_csv(OUT/f"private_{name}_eligibility.csv",index=False)
    contexts={};calibrators={};flows=[]
    for line in ["geography","temporal"]:
        part=f"{line}_partition_v2"
        s=paired.loc[simmask].copy();dev=s.loc[s[part]=="development"];val=s.loc[s[part]=="validation"].copy().reset_index(drop=True)
        assert not set(dev.group_hash)&set(val.group_hash)
        eroute=current_winner(line,"paired","endop","full");proute=current_winner(line,"paired","preop","full")
        end=frozen_prediction(line,"paired",eroute,val);pre=frozen_prediction(line,"paired",proute,val)
        predictions={"our_endop_cal":end.prob_calibrated.to_numpy(),"our_endop_raw":end.prob_raw.to_numpy(),
                     "our_preop_carryforward_cal":pre.prob_calibrated.to_numpy()}
        for year in [2009,2021]:
            pdv,_,_=sim_risk(dev,params,year);pv,classes,g=sim_risk(val,params,year)
            assert dev.y.nunique()==2
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cal=LogisticRegression(C=1e6,solver="lbfgs",tol=1e-10,max_iter=2000).fit(logit(np.clip(pdv,1e-6,1-1e-6)).reshape(-1,1),dev.y.to_numpy())
            if caught:raise RuntimeError("Development recalibration warning: "+str([str(w.message) for w in caught]))
            mapped=cal.predict_proba(logit(np.clip(pv,1e-6,1-1e-6)).reshape(-1,1))[:,1]
            predictions[f"SIM_original_{year}"]=pv;predictions[f"SIM_development_recalibrated_{year}"]=mapped
            calibrators[f"{line}_{year}"]={"intercept":float(cal.intercept_[0]),"slope":float(cal.coef_[0,0]),
                "development_n":len(dev),"development_events":int(dev.y.sum()),
                "development_group_sha256":hashlib.sha256("\n".join(sorted(set(dev.group_hash))).encode()).hexdigest(),
                "source_predictions_clipped":1e-6,"fit_only_on_development":True}
            assert np.allclose(mapped,expit(cal.intercept_[0]+cal.coef_[0,0]*logit(np.clip(pv,1e-6,1-1e-6))),atol=1e-14)
            val[f"eGFR{year}"]=g
        c=canonical_context(val,predictions,"prob")
        c["comparisons"]=[("our_endop_cal",name) for name in predictions if name.startswith("SIM_")]
        c["metadata"]={"reference":"SIM-AKI","line":line,"assessment":"end_of_operation",
                       "n":len(val),"events":int(val.y.sum()),"eligible_development_n":len(dev),"eligible_development_events":int(dev.y.sum())}
        contexts[f"SIM_{line}"]=c
        priv=val[["excel_row","group_hash","center","y","stage","eGFR2009","eGFR2021"]].copy()
        for k,v in predictions.items():priv[k]=v
        priv.to_csv(OUT/f"private_SIM_{line}_predictions.csv",index=False)
        flows.append({"reference":"SIM_AKI","line":line,"source_records":len(paired),"eligible_all":int(simmask.sum()),
                      "excluded_any_input_or_source_renal_eligibility":int((~simmask).sum()),"development_n":len(dev),
                      "development_events":int(dev.y.sum()),"validation_n":len(val),"validation_events":int(val.y.sum())})
        z=full.loc[chenmask].copy();cd=z.loc[z[part]=="development"];cv=z.loc[z[part]=="validation"].copy().reset_index(drop=True)
        assert not set(cd.group_hash)&set(cv.group_hash)
        fr=current_winner(line,"preop","preop","full");cr=current_winner(line,"preop","preop","compact")
        fp=frozen_prediction(line,"preop",fr,cv);cp=frozen_prediction(line,"preop",cr,cv)
        preds={"Chen_Table4_ordinal_score":chen_score(cv),"our_full_preop_carryforward_cal":fp.prob_calibrated.to_numpy(),
               "our_compact_preop_carryforward_cal":cp.prob_calibrated.to_numpy()}
        c=canonical_context(cv,preds,"rank")
        c["comparisons"]=[(k,"Chen_Table4_ordinal_score") for k in preds if k.startswith("our_")]
        c["metadata"]={"reference":"Chen2026_Table4","line":line,"assessment":"known_open_surgery_endop_assessment",
                       "n":len(cv),"events":int(cv.y.sum()),"eligible_development_n":len(cd),"eligible_development_events":int(cd.y.sum()),
                       "probability_metrics_qualified":False,"aCCI_threshold_constant_under_standard_including_index_cancer_definition":True}
        contexts[f"Chen_{line}"]=c
        priv=cv[["excel_row","group_hash","center","y","stage"]].copy()
        for k,v in preds.items():priv[k]=v
        priv.to_csv(OUT/f"private_Chen_{line}_scores.csv",index=False)
        flows.append({"reference":"Chen2026","line":line,"source_records":len(full),"eligible_all":int(chenmask.sum()),
                      "excluded_any_missing_or_population_mismatch":int((~chenmask).sum()),"development_n":len(cd),
                      "development_events":int(cd.y.sum()),"validation_n":len(cv),"validation_events":int(cv.y.sum())})
    write_json(OUT/"DEVELOPMENT_RECALIBRATORS.json",calibrators)
    pd.DataFrame(flows).to_csv(OUT/"COHORT_FLOW.csv",index=False)
    write_json(OUT/"ORIGINAL_PREDICTION_RECEIPTS.json",PRED_RECEIPTS)
    return contexts
CONTEXTS={};WEIGHTS={};SIGNATURE=""
def evaluate_repeat(task):
    name,rep=task;c=CONTEXTS[name];w=WEIGHTS[name][rep]
    p=OUT/"checkpoints"/name/f"{rep:04d}.json"
    if p.exists():
        q=json.loads(p.read_text());assert q["signature"]==SIGNATURE and q["weight_sha256"]==hashlib.sha256(w.tobytes()).hexdigest()
        return name,rep
    idx,g,r=es.expanded(w,c["g"],c["r"])
    scores=context_scores(c,idx,g,r);y=c["y"][idx]
    q=serial({"signature":SIGNATURE,"replicate":rep,"context":name,"weight_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
              "zero_events":int(y.sum())==0,"all_events":int(y.sum())==len(y),"scores":scores})
    write_json(p,q)
    return name,rep
def cgroup_resources():
    avail=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
    limits=[];quota=len(os.sched_getaffinity(0))
    cg=Path("/sys/fs/cgroup")/Path("/proc/self/cgroup").read_text().strip().split("::")[-1].lstrip("/")
    for parent in [cg,*cg.parents]:
        if not str(parent).startswith("/sys/fs/cgroup"):continue
        m=parent/"memory.max";current=parent/"memory.current";cpu=parent/"cpu.max"
        if m.exists() and m.read_text().strip()!="max":
            remaining=int(m.read_text().strip())-int(current.read_text().strip())
            avail=min(avail,remaining);limits.append([str(parent),remaining])
        if cpu.exists():
            q,p=cpu.read_text().split()
            if q!="max":quota=min(quota,max(1,int(q)//int(p)))
    return avail,quota,limits
def aggregate(point):
    rows=[];intervals=[];deltas=[];receipts={}
    for name,c in CONTEXTS.items():
        estimates=point[name];v={k:{} for k in estimates};diffs={a+" MINUS "+b:{} for a,b in c["comparisons"]}
        proof=hashlib.sha256();zero=allone=0;cp_hash=hashlib.sha256()
        for rep in range(2000):
            path=OUT/"checkpoints"/name/f"{rep:04d}.json";q=json.loads(path.read_text())
            assert q["signature"]==SIGNATURE
            w=WEIGHTS[name][rep];proof.update(w.tobytes());cp_hash.update(path.read_bytes())
            assert q["weight_sha256"]==hashlib.sha256(w.tobytes()).hexdigest()
            zero+=q["zero_events"];allone+=q["all_events"]
            for model,metrics in q["scores"].items():
                for key,value in metrics.items():
                    if key not in ["n","events"] and finite_number(value):v[model].setdefault(key,[]).append(value)
            for a,b in c["comparisons"]:
                dd=diffs[a+" MINUS "+b]
                for key in estimates[a]:
                    aa=q["scores"][a].get(key);bb=q["scores"][b].get(key)
                    if key not in ["n","events"] and finite_number(aa) and finite_number(bb):dd.setdefault(key,[]).append(aa-bb)
        for model,metrics in estimates.items():rows.append({"context":name,"model":model,**metrics})
        for model,cols in v.items():
            for metric,samples in cols.items():
                lo,hi=np.quantile(samples,[.025,.975],method="linear")
                intervals.append({"context":name,"model":model,"metric":metric,"estimate":estimates[model].get(metric),
                                  "lower95":float(lo),"upper95":float(hi),"valid_replicates":len(samples),"requested_replicates":2000})
        for comparison,cols in diffs.items():
            a,b=comparison.split(" MINUS ")
            for metric,samples in cols.items():
                av=estimates[a].get(metric);bv=estimates[b].get(metric);lo,hi=np.quantile(samples,[.025,.975],method="linear")
                deltas.append({"context":name,"comparison":comparison,"metric":metric,"estimate":av-bv if finite_number(av) and finite_number(bv) else None,
                              "lower95":float(lo),"upper95":float(hi),"valid_replicates":len(samples),"requested_replicates":2000})
        receipts[name]={**c["metadata"],"replicates":2000,"seed":20261017,"resample_weight_sha256":proof.hexdigest(),
                        "checkpoint_stream_sha256":cp_hash.hexdigest(),"zero_event_replicates":zero,"all_event_replicates":allone}
    pd.DataFrame(rows).to_csv(OUT/"POINT_ESTIMATES.csv",index=False)
    pd.DataFrame(intervals).to_csv(OUT/"INTERVALS.csv",index=False)
    pd.DataFrame(deltas).to_csv(OUT/"PAIRED_DIFFERENCES.csv",index=False)
    write_json(OUT/"BOOTSTRAP_RECEIPTS.json",receipts)
def run():
    global CONTEXTS,WEIGHTS,SIGNATURE
    started=time.monotonic()
    lock=open(OUT/".run.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=Path(f"/proc/{os.getpid()}/stat").read_text().split(") ",1)[1].split()[19]
    write_json(OUT/"OWNER.json",{"pid":os.getpid(),"start_ticks":tick,"launched_at":now(),"code_sha256":sha(__file__)})
    CONTEXTS=prepare()
    SIGNATURE=hashlib.sha256((sha(WORK/"core.py")+sha(__file__)+sha(OUT/"ANALYSIS_CONTRACT.json")+sha(OUT/"DEVELOPMENT_RECALIBRATORS.json")+sha(OUT/"ORIGINAL_PREDICTION_RECEIPTS.json")).encode()).hexdigest()
    point={n:context_scores(c) for n,c in CONTEXTS.items()}
    # Calibrators locked and checkpoint identity fixed before first holdout score output.
    for name,c in CONTEXTS.items():
        rng=np.random.default_rng(20261017)
        WEIGHTS[name]=np.stack([es.group_weights(c["strata"],len(c["y"]),rng) for _ in range(2000)]).astype("<i8")
        c["signature"]=SIGNATURE
    write_json(OUT/"MODEL_MAPPING_LOCK.json",{"signature":SIGNATURE,"contract_sha256":sha(OUT/"ANALYSIS_CONTRACT.json"),
              "recalibration_sha256":sha(OUT/"DEVELOPMENT_RECALIBRATORS.json"),"context_metadata":{n:c["metadata"] for n,c in CONTEXTS.items()},
              "code":{"core":sha(WORK/"core.py"),"run":sha(__file__)},"fixed_original_models_untouched":True})
    # Four real first replicates also serve as measured pilots and are reused.
    for name in CONTEXTS:evaluate_repeat((name,0))
    avail,cpus,limits=cgroup_resources();hwm=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    headroom=max(3*1024**3,int(avail*.12));budget=max(220*1024**2,int(hwm*1.35))
    workers=max(1,min(cpus,int(max(0,avail-headroom)//budget)))
    write_json(OUT/"PILOT_ACCEPTANCE.json",{"status":"PASS_four_real_context_bootstrap_pilots","signature":SIGNATURE,
        "MemAvailable_or_effective_remaining_bytes":avail,"cgroup_limits":limits,"effective_CPU":cpus,
        "parent_highwater_bytes":hwm,"conservative_worker_budget_bytes":budget,"headroom_bytes":headroom,
        "workers":workers,"GPU":"not useful for small fixed-score GLM diagnostics; original frozen numerical backend preserved"})
    tasks=[(n,i) for n in CONTEXTS for i in range(2000)]
    done=0
    write_json(OUT/"STATUS.json",{"state":"running_fixed_published_benchmarks","completed":4,"total":8000,"workers":workers,"time":now()})
    with mp.get_context("fork").Pool(workers) as pool:
        for name,i in pool.imap_unordered(evaluate_repeat,tasks,chunksize=1):
            done+=1
            if done%200==0:
                write_json(OUT/"STATUS.json",{"state":"running_fixed_published_benchmarks","completed":done,"total":8000,"workers":workers,"time":now()})
    aggregate(point)
    files=["ANALYSIS_CONTRACT.json","QUALIFICATION.json","MODEL_MAPPING_LOCK.json","DEVELOPMENT_RECALIBRATORS.json",
           "ORIGINAL_PREDICTION_RECEIPTS.json","COHORT_FLOW.csv","POINT_ESTIMATES.csv","INTERVALS.csv",
           "PAIRED_DIFFERENCES.csv","BOOTSTRAP_RECEIPTS.json","PILOT_ACCEPTANCE.json"]
    write_json(OUT/"FIT_AND_EVALUATION_ACCEPTANCE.json",{"status":"PASS_fixed_published_benchmark_computation_independent_review_pending",
        "signature":SIGNATURE,"contexts":4,"replicates_per_context":2000,"checkpoints":8000,"seconds":time.monotonic()-started,
        "completed_at":now(),"files_sha256":{f:sha(OUT/f) for f in files},"numpy":np.__version__,"sklearn":sklearn.__version__,
        "source_fitting":"published formula fixed; only labelled development recalibration fitted",
        "clinical_interpretation_review_completed":False,"whole_project_complete":False})
    write_json(OUT/"STATUS.json",{"state":"computed_review_required","completed":8000,"total":8000,"workers":0,"time":now()})
    print("DONE",time.monotonic()-started,flush=True)
if __name__=="__main__":
    try:run()
    except BaseException:
        write_json(OUT/"FAILURE.json",{"time":now(),"error":traceback.format_exc()})
        raise
