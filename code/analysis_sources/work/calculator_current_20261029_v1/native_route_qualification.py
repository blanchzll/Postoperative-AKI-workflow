from pathlib import Path
import os,sys,json,time,resource,hashlib,traceback,multiprocessing as mp,fcntl
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];O=R/"outputs/calculator_release_20261025_v1";A=O/"app"
sys.path.insert(0,str(A))
from engine import Calculator
sys.path.insert(0,str(R/"work/locked_risk_curves_20261024_v1"))
import curves as cv
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    q=p.with_name(p.name+"."+str(os.getpid())+".tmp");q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");q.replace(p)
def records_from(x,meta):
    out=[]
    for pos,(_,row) in enumerate(x.iterrows()):
        out.append({"case_key":str(meta.group_hash.iloc[pos])+"|"+str(meta.excel_row.iloc[pos]),**{c:None if pd.isna(row[c]) else float(row[c]) if c in cv.rt.source_spec("endop")["num"] else str(row[c]) for c in x.columns}})
    return out
def payload(calc,rid,records):return {"route":rid,"units":{f["name"]:f["unit"] for f in calc.catalog[rid]["fields"] if f["unit"]},"records":records}
def many(calc,rid,records):
    a=[];b=[]
    for start in range(0,len(records),1000):
        q=calc.predict(payload(calc,rid,records[start:start+1000]))
        a.extend(v["raw_risk"] for v in q["predictions"]);b.extend(v["risk"] for v in q["predictions"])
    return np.array(a),np.array(b)
def execute(rid):
    try:
        with threadpool_limits(limits=1):
            began=time.monotonic();calc=Calculator(R);m,sp,meta,x,ref,contract=cv.inputs(rid);ident=m["identity"]
            name="_".join(ident[k] for k in ["stage","branch","role","family"])
            pp=R/"outputs/formal_evaluation_20261023_v1"/(ident["line"]+"_"+ident["cohort"])/"private_predictions"/name/"PREDICTIONS.csv"
            old=pd.read_csv(pp,dtype={"group_hash":str},float_precision="round_trip").set_index("excel_row",drop=False)
            frame=pd.read_csv(R/"outputs/v4_development_inputs_20261018"/(ident["cohort"]+"_"+ident["stage"]+"_features.csv"),dtype={c:"string" for c in sp["cat"]},float_precision="round_trip").set_index("excel_row").loc[old.index,m["input_columns"]]
            records=records_from(frame,old);units=payload(calc,rid,records[:1])["units"]
            t=time.monotonic();cold=many(calc,rid,records[:1]);coldsecs=time.monotonic()-t
            newraw,newcal=many(calc,rid,records)
            errors={"saved_heldout_raw":float(np.max(np.abs(newraw-old.prob_raw.to_numpy()))),"saved_heldout_cal":float(np.max(np.abs(newcal-old.prob_calibrated.to_numpy())))}
            devel=x.head(128);dm=meta.loc[devel.index];dr=records_from(devel,dm)
            native=cv.Predictor(m);keys=cv.rt.keys(dm)
            for n in [100,128]:
                aa,bb=native(devel.head(n),keys[:n]);pa,pb=many(calc,rid,dr[:n])
                errors[f"development_original{n}_raw"]=float(np.max(np.abs(aa-pa)))
                errors[f"development_original{n}_cal"]=float(np.max(np.abs(bb-pb)))
            sample=records[:128];ba,bc=many(calc,rid,sample)
            ra,rc=many(calc,rid,sample[::-1]);errors["reverse_raw"]=float(np.max(np.abs(ra[::-1]-ba)));errors["reverse_cal"]=float(np.max(np.abs(rc[::-1]-bc)))
            single=np.array([many(calc,rid,[v])[1][0] for v in sample[:16]])
            errors["singleton_cal"]=float(np.max(np.abs(single-bc[:16])))
            fresh=Calculator(R);fa,fc=many(fresh,rid,sample[:16]);errors["reloaded_cal"]=float(np.max(np.abs(fc-bc[:16])))
            # Completely observed synthetic cases must not acquire case-ID predictors.
            synth=json.loads((O/"synthetic_examples"/(rid+".json")).read_text());s1=synth["records"][0]
            s2=dict(s1,case_key="ANOTHER_complete_synthetic_key")
            a=many(calc,rid,[s1])[1][0];b=many(calc,rid,[s2])[1][0];errors["complete_case_key_independence"]=abs(float(a-b))
            allmissing={"case_key":"DEMO_allmissing",**{f["name"]:None for f in calc.catalog[rid]["fields"]}}
            q1=many(calc,rid,[allmissing])[1][0];q2=many(calc,rid,[allmissing])[1][0];errors["all_missing_stability"]=abs(float(q1-q2))
            timings={"cold_first_case_seconds":coldsecs}
            for n in [1,100,1000]:
                t=time.monotonic();many(calc,rid,records[:n]);timings[f"warm_{n}_seconds"]=time.monotonic()-t
            dest=O/"private_replay"/(rid+".csv");dest.parent.mkdir(exist_ok=True)
            pd.DataFrame({"case_key":[v["case_key"] for v in records],"original_raw":old.prob_raw.to_numpy(),"canonical_raw":newraw,"original_calibrated":old.prob_calibrated.to_numpy(),"canonical_calibrated":newcal}).to_csv(dest,index=False,float_format="%.17g")
            passed=all(v<=1e-12 for v in errors.values())
            d={"route":rid,"status":"PASS_full_route_inference" if passed else "HOLD_numeric_difference_not_waived","heldout_reference_n":len(records),"development_reference_n":128,"strict_absolute_tolerance":1e-12,"errors":errors,"timings":timings,"all_missing_risk_finite":bool(np.isfinite(q1)),"private_replay_path":str(dest),"private_replay_sha256":sha(dest),"source_prediction_sha256":sha(pp),"model_or_imputer_refitted":False,"elapsed_seconds":time.monotonic()-began,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            save(O/"route_checks"/(rid+".json"),d);return d
    except Exception:
        d={"route":rid,"status":"ERROR","traceback":traceback.format_exc()};save(O/"route_checks"/(rid+".json"),d);return d
