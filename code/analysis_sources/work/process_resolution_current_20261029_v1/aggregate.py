"""Aggregate support results; do not select gates/models or overwrite V1."""
from pathlib import Path
import json,gzip,pickle,math,time,os
from collections import defaultdict
import numpy as np,pandas as pd
import analysis_core as ac
from runner import key,paths,valid
def finite(x):return x is not None and np.isscalar(x) and np.isfinite(x)
def table(point,draws,context,route=""):
    out=[]
    for name,values in point.items():
        for metric,estimate in values.items():
            if not finite(estimate):continue
            nums=np.array([d[name].get(metric,np.nan) for d in draws],dtype=float)
            nums=nums[np.isfinite(nums)]
            if len(nums)==0:continue
            lo,hi=np.quantile(nums,[.025,.975])
            out.append(dict(context=context,route=route,comparison=name,metric=metric,estimate=float(estimate),lower=float(lo),upper=float(hi),valid_bootstraps=len(nums)))
    return out
def write(name,rows):pd.DataFrame(rows).to_csv(ac.O/name,index=False)
def main():
    started=time.monotonic()
    status=json.loads((ac.O/"RUN_STATUS.json").read_text());assert status["status"]=="completed"
    manifest=json.loads((ac.O/"RUN_MANIFEST.json").read_text());sig=manifest["signature"]
    for path,h in manifest["files"].items():assert ac.sha(ac.R/path)==h,path
    ac.initialize();chunks=defaultdict(list);receipts=[]
    for j in manifest["jobs"]:
        assert valid(j,sig),j
        a,b=paths(j);rec=json.loads(b.read_text());receipts.append(rec)
        with gzip.open(a,"rb") as f:rows=pickle.load(f)
        assert len(rows)==(j[4]-j[3])*(2 if j[0]=="random" else 1),j
        chunks[tuple(j[:3])].append((j[3],rows))
    data={k:[item for _,rows in sorted(blocks) for item in rows] for k,blocks in chunks.items()}
    assert all(len(v)==(4000 if k[0]=="random" else 2000) for k,v in data.items())
    alt=[];cal=[];non=[];mc=[];calpoint=[];coefs=[];predrefs=[]
    chinese=[c for c in ac.DATA if c!="INSPIRE_common6"]
    for ctx in chinese:
        d=ac.DATA[ctx];n=len(d["meta"]);w=np.ones(n,dtype=int)
        alt+=table(ac.alt_scores(ctx,w),data[("alt",ctx,"")],ctx)
    for ctx,ds in ac.DATA.items():
        m=ds["meta"];w=np.ones(len(m),dtype=int)
        for name in ds["cal_routes"]:
            pt,details=ac.cal_scores(ctx,name,w,True)
            cal+=table(pt,data[("cal",ctx,name)],ctx,name)
            out=m.copy();out["target_adaptation_fold"]=ds["fold"]
            for method,item in details.items():
                out[method+"_prob"]=item["prob"];out[method+"_adaptation_capacity20_alert"]=item["adaptation_mask"].astype(int)
                for c in item["coefficients"]:coefs.append(dict(context=ctx,route=name,method=method,**c))
                calpoint.append(dict(context=ctx,route=name,method=method,**pt[method]))
            path=ac.O/"private_updated_predictions"/ctx/(name+".csv")
            path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);out.to_csv(path,index=False,float_format="%.17g")
            predrefs.append({"path":str(path),"sha256":ac.sha(path),"n":len(out),"events":int(m.y.sum())})
    ledgers=[];partitions=[]
    for cohort in ["preop","paired"]:
        src=ac.R/"outputs/v4_development_inputs_20261018"/(cohort+"_metadata.csv")
        m=pd.read_csv(src,dtype={"group_hash":str})
        for (geo,tmp),x in m.groupby(["geography_partition_v2","temporal_partition_v2"],dropna=False):
            partitions.append(dict(cohort=cohort,geography_partition=geo,temporal_partition=tmp,n=len(x),events=int(x.y.sum()),patient_groups=x.group_hash.nunique(),date_min=x.surgery_date.min(),date_max=x.surgery_date.max()))
        for line,col in [("geography","geography_partition_v2"),("temporal","temporal_partition_v2")]:
            tr=m.loc[m[col]=="development"];te=m.loc[m[col]=="validation"]
            overlap=set(tr.group_hash)&set(te.group_hash)
            assert not overlap
            ledgers.append(dict(cohort=cohort,comparison=line+"_own_training_vs_evaluation",train_n=len(tr),train_events=int(tr.y.sum()),evaluation_n=len(te),evaluation_events=int(te.y.sum()),overlapping_records=len(set(tr.excel_row)&set(te.excel_row)),overlapping_patient_groups=len(overlap)))
        geo=m.loc[m.geography_partition_v2=="validation"];tmp=m.loc[m.temporal_partition_v2=="validation"]
        over=set(geo.group_hash)&set(tmp.group_hash);ret=tmp.loc[~tmp.group_hash.isin(over)]
        ledgers.append(dict(cohort=cohort,comparison="geographic_vs_temporal_evaluation",train_n=len(geo),train_events=int(geo.y.sum()),evaluation_n=len(tmp),evaluation_events=int(tmp.y.sum()),overlapping_records=len(set(geo.excel_row)&set(tmp.excel_row)),overlapping_patient_groups=len(over),retained_temporal_n=len(ret),retained_temporal_events=int(ret.y.sum()),retained_temporal_patient_groups=ret.group_hash.nunique()))
        ctx="temporal_"+cohort;ds=ac.DATA[ctx]
        assert len(ret)==int(ds["nonoverlap_mask"].sum())
        non+=table(ac.nonoverlap_scores(ctx,np.ones(len(ret),dtype=int)),data[("nonoverlap",ctx,"")],ctx)
    for ctx in chinese:
        ds=ac.DATA[ctx];m=ds["meta"];y=m.y.to_numpy(dtype=int);g=m.group_hash.to_numpy();r=m.excel_row.to_numpy()
        df=pd.DataFrame(data[("random",ctx,"")])
        p=ac.O/"random_allocations"/(ctx+".csv");p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False)
        for fraction in [.25,.5]:
            name=f"policy_{fraction}_capacity_selective";it=ds["policies"][name]
            raw,pred,mask=ac.es.route(it,ds["pred"][it["base"]],ds["pred"][it["upgrade"]],g,r)
            actual=ac.metrics(y,pred,g,r);d=df.loc[df.upgrade_fraction==fraction]
            assert len(d)==2000 and np.all(d.upgraded_n==math.ceil(len(m)*fraction))
            for cap in [10,20]:
                field=f"capacity{cap}_captured_n";vals=d[field].to_numpy();sel=actual[field]
                lo,med,hi=np.quantile(vals,[.025,.5,.975]);s=float(vals.std(ddof=1))
                mc.append(dict(context=ctx,upgrade_fraction=fraction,review_fraction=cap/100,upgraded_n=int(mask.sum()),n=len(m),events=int(y.sum()),reviewed_n=actual[f"capacity{cap}_reviewed_n"],selective_captured_n=sel,random_mean_captured_n=vals.mean(),random_mean_MCSE=s/np.sqrt(2000),random_allocation_q025=lo,random_allocation_median=med,random_allocation_q975=hi,selective_minus_random_mean_per1000=(sel-vals.mean())/len(m)*1000,share_random_strictly_better=float(np.mean(vals>sel)),share_random_tied=float(np.mean(vals==sel)),selective_PPV=actual[f"capacity{cap}_PPV"],random_mean_PPV=float(d[f"capacity{cap}_PPV"].mean()),selective_missed_n=int(y.sum()-sel),random_mean_missed_n=float(y.sum()-vals.mean()),interpretation="Conditional allocation variability, not population CI or trial P value"))
    write("ALT_SUPPORT.csv",alt);write("CALIBRATION_UPDATE_SUPPORT.csv",cal);write("CALIBRATION_POINT_DIAGNOSTICS.csv",calpoint)
    write("CALIBRATION_FOLD_PARAMETERS.csv",coefs);write("NONOVERLAP_SUPPORT.csv",non);write("PARTITION_CROSS_TAB.csv",partitions)
    write("OVERLAP_LEDGER.csv",ledgers);write("SELECTIVE_VS_RANDOM_MC.csv",mc)
    for p,h in ac.PROTECTED.items():assert ac.sha(ac.R/p)==h,p
    frozen_changed=[p for p,h in manifest["files"].items() if ac.sha(ac.R/p)!=h];assert not frozen_changed
    expected=["ALT_SUPPORT.csv","CALIBRATION_UPDATE_SUPPORT.csv","CALIBRATION_POINT_DIAGNOSTICS.csv","CALIBRATION_FOLD_PARAMETERS.csv","NONOVERLAP_SUPPORT.csv","PARTITION_CROSS_TAB.csv","OVERLAP_LEDGER.csv","SELECTIVE_VS_RANDOM_MC.csv"]
    ac.save(ac.O/"RESULT_ACCEPTANCE.json",{"status":"PASS_targeted_support_calculations_and_aggregation","chunks_verified":len(receipts),"signature":sig,"outputs_sha256":{p:ac.sha(ac.O/p) for p in expected},"private_crossfitted_prediction_files":predrefs,"bootstrap_replicates":2000,"MC_random_allocations_per_context":2000,"all_source_frozen_files_preserved":len(ac.PROTECTED),"no_primary_model_or_policy_reselection":True,"no_manuscript_review":True,"rows":{"ALT":len(alt),"calibration":len(cal),"nonoverlap":len(non),"MC_conditions":len(mc)},"seconds":time.monotonic()-started,"aggregation_code_sha256":ac.sha(Path(__file__))})
    print(json.dumps({"status":"PASS","rows":{"ALT":len(alt),"calibration":len(cal),"nonoverlap":len(non),"MC_conditions":len(mc)},"seconds":time.monotonic()-started}))
if __name__=="__main__":main()
