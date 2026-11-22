
"""Frozen-pipeline heldout scoring and paired patient-group uncertainty."""
from pathlib import Path
import sys
import json
import os
import time
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as e
import finalize_models as f
import statistics_core as s
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

OUT = e.OUT / "evaluation"

def model_identifier(identity):
    return "new_" + identity["method"] + "_" + identity["rule"]

def raw_table(cohort):
    spec = e.legacy.spec_for("preop")
    return pd.read_csv(
        e.legacy.INPUT / f"{cohort}_preop_features.csv",
        dtype={c:"string" for c in spec["cat"]}, float_precision="round_trip"
    ).set_index("excel_row")

def guard_models():
    acceptance = json.loads((f.POST/"MODEL_LOCK_ACCEPTANCE.json").read_text())
    assert acceptance["status"]=="PASS_NEW_TRAINING_PIPELINES_CALIBRATION_AND_LOCKS"
    assert acceptance["protocol_sha256"]==e.protocol_hash()
    assert acceptance["post_code_sha256"]==e.sha(f.__file__)
    for item in acceptance["locks"]:
        assert e.sha(item["path"])==item["sha256"]
    code=json.loads((e.ROOT/"EVALUATION_CODE_LOCK.json").read_text())
    for path,expected in code["files"].items():
        assert e.sha(path)==expected,path
    return acceptance["locks"]

def predict_model(item):
    lock=json.loads(Path(item["path"]).read_text())
    line=lock["identity"]["line"]
    meta=f.metadata()
    selected=meta.loc[meta[line+"_partition_v2"].eq("validation")].copy()
    train_rows,_,_=f.context_rows(line,"final")
    assert set(selected.group_hash).isdisjoint(meta.loc[train_rows].group_hash)
    assert e.sha(lock["raw_bundle"]["path"])==lock["raw_bundle"]["sha256"]
    assert e.sha(lock["calibrator"]["path"])==lock["calibrator"]["sha256"]
    bundle=joblib.load(lock["raw_bundle"]["path"])
    calibrator=joblib.load(lock["calibrator"]["path"])
    raw=raw_table("preop").loc[selected.excel_row]
    with threadpool_limits(limits=1):
        probability=f.predict_raw(bundle,raw,e.legacy.keys(selected))
        calibrated=f.apply_calibration(calibrator,probability)
    name=model_identifier(lock["identity"])
    directory=OUT/"predictions"/line/name
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/"PREDICTIONS.csv"
    selected.assign(prob_raw=probability,prob_calibrated=calibrated).to_csv(target,index=False)
    return {
        "line":line,"name":name,"method":lock["identity"]["method"],"rule":lock["identity"]["rule"],
        "family":lock["identity"]["family"],"raw_count":lock["actual_input_count"],
        "input_columns":lock["input_columns"],"prediction_file":str(target),
        "prediction_sha256":e.sha(target),"lock":item["path"],"lock_sha256":item["sha256"],
        "thresholds":lock["training_thresholds"]
    }

def old_reference(line,cohort,stage,branch,label):
    paths=list((e.SOURCE/f"outputs/model_lock_20261022_v1/locked_routes/{line}/{cohort}/{stage}/{branch}/champion").rglob("MODEL_LOCK.json"))
    assert len(paths)==1
    path=paths[0];lock=json.loads(path.read_text())
    name=f"{stage}_{branch}_champion_{lock['family']}"
    prediction=e.SOURCE/f"outputs/formal_evaluation_20261023_v1/{line}_{cohort}/private_predictions/{name}/PREDICTIONS.csv"
    return {
        "name":label,"prediction_file":str(prediction),"prediction_sha256":e.sha(prediction),
        "probability_column":"prob_calibrated","rank_only":False,
        "family":lock["family"],"raw_count":lock["actual_input_count"],
        "thresholds":lock["training_thresholds"],"new_fit":False
    }

def models_for_context(line,cohort,new_models):
    entries=[]
    for item in new_models:
        if item["line"]!=line:
            continue
        entries.append({**item,"probability_column":"prob_calibrated","rank_only":False,"new_fit":True})
        if item["method"]=="RF_RFE":
            entries.append({**item,"name":item["name"]+"__raw","probability_column":"prob_raw","rank_only":False,"thresholds":{},"new_fit":True})
    if cohort=="preop":
        entries.append(old_reference(line,cohort,"preop","full","reference_full27"))
    else:
        entries.append(old_reference(line,cohort,"preop","full","reference_paired27"))
        entries.append(old_reference(line,cohort,"endop","full","reference_endop38"))
    for entry in list(entries):
        if entry["name"].startswith("reference_"):
            entries.append({**entry,"name":entry["name"]+"__raw","probability_column":"prob_raw","thresholds":{}})
    return entries

def context_frame(line,cohort):
    path=e.legacy.INPUT/f"{cohort}_metadata.csv"
    meta=pd.read_csv(path,dtype={"group_hash":str})
    frame=meta.loc[meta[line+"_partition_v2"].eq("validation")].copy()
    if cohort=="paired":
        main=raw_table("preop")
        paired=raw_table("paired")
        columns=e.legacy.spec_for("preop")
        columns=columns["num"]+columns["cat"]
        pd.testing.assert_frame_equal(
            main.loc[frame.excel_row,columns],paired.loc[frame.excel_row,columns],
            check_dtype=False,check_exact=True
        )
    return frame

def bind_entries(frame,entries):
    data=frame.copy()
    for entry in entries:
        assert e.sha(entry["prediction_file"])==entry["prediction_sha256"]
        pred=pd.read_csv(entry["prediction_file"],dtype={"group_hash":str},float_precision="round_trip").set_index("excel_row")
        assert set(data.excel_row)<=set(pred.index)
        aligned=pred.loc[data.excel_row]
        np.testing.assert_array_equal(data.y.to_numpy(),aligned.y.to_numpy())
        np.testing.assert_array_equal(data.group_hash.to_numpy(dtype=str),aligned.group_hash.to_numpy(dtype=str))
        value=aligned[entry["probability_column"]].to_numpy()
        if entry.get("rank_only"):
            low,high=float(value.min()),float(value.max())
            value=(value-low)/(high-low) if high>low else np.full(len(value),.5)
            entry["ranking_affine_range"]=[low,high]
        data[entry["name"]]=value
    return data

def prepare_contexts(new_models):
    contexts=[]
    for line in ["geography","temporal"]:
        for cohort in ["preop","paired"]:
            name=line+"_"+cohort
            entries=models_for_context(line,cohort,new_models)
            frame=bind_entries(context_frame(line,cohort),entries)
            frame,weights,proof=s.aligned_weights(name,frame)
            contexts.append(save_context(name,frame,weights,entries,proof))
        benchmark=e.SOURCE/"outputs/published_benchmark_comparison_20261024_v1"
        source=benchmark/f"private_Chen_{line}_scores.csv"
        reference=pd.read_csv(source,dtype={"group_hash":str},float_precision="round_trip")
        entries=[
            {"name":"reference_Chen_Table4_ranking","prediction_file":str(source),"prediction_sha256":e.sha(source),
             "probability_column":"Chen_Table4_ordinal_score","rank_only":True,"thresholds":{},"new_fit":False},
            {"name":"reference_Chen_subset_full27","prediction_file":str(source),"prediction_sha256":e.sha(source),
             "probability_column":"our_full_preop_carryforward_cal","rank_only":False,"thresholds":{},"new_fit":False}
        ]
        for item in new_models:
            if item["line"]==line and item["method"]=="RF_RFE":
                entries.append({**item,"probability_column":"prob_calibrated","rank_only":False,"new_fit":True})
        frame=bind_entries(reference,entries)
        expected=json.loads((benchmark/"BOOTSTRAP_RECEIPTS.json").read_text())["Chen_"+line]["resample_weight_sha256"]
        frame,weights,proof=s.eligible_weights(frame,expected)
        contexts.append(save_context(line+"_Chen",frame,weights,entries,proof))
    e.atomic(OUT/"CONTEXTS.json",contexts)
    return contexts

def save_context(name,frame,weights,entries,proof):
    directory=OUT/"contexts"/name
    directory.mkdir(parents=True,exist_ok=True)
    frame.to_csv(directory/"PRIVATE_DATA.csv",index=False)
    np.savez_compressed(directory/"WEIGHTS_PRIVATE.npz",weights=weights)
    receipt={
        "context":name,"n":len(frame),"events":int(frame.y.sum()),"patient_groups":frame.group_hash.nunique(),
        "data":str(directory/"PRIVATE_DATA.csv"),"data_sha256":e.sha(directory/"PRIVATE_DATA.csv"),
        "weights":str(directory/"WEIGHTS_PRIVATE.npz"),"weights_sha256":e.sha(directory/"WEIGHTS_PRIVATE.npz"),
        "models":entries,"weight_lineage":proof
    }
    e.atomic(directory/"INPUTS.json",receipt)
    return receipt

def sample_block(task):
    context,entry,start,stop=task
    directory=OUT/"samples"/context["context"]/entry["name"]/f"block_{start:04d}_{stop:04d}"
    directory.mkdir(parents=True,exist_ok=True)
    signature={
        "context_data":context["data_sha256"],"weights":context["weights_sha256"],"entry":entry,
        "start":start,"stop":stop,"code":e.sha(__file__),"statistics_code":e.sha(s.__file__),
        "downstream_protocol":e.sha(e.ROOT/"DOWNSTREAM_PROTOCOL_LOCK.json")
    }
    if e.qualified(directory,signature):
        return str(directory)
    frame=pd.read_csv(context["data"],dtype={"group_hash":str},float_precision="round_trip")
    weights=np.load(context["weights"],allow_pickle=False)["weights"]
    risk=s.FixedRisk(frame.y,frame[entry["name"]],frame.group_hash,frame.excel_row.astype(str))
    point=s.score(risk,np.ones(len(frame),dtype=np.int64),entry.get("thresholds"),entry.get("rank_only",False))
    fields=list(point)
    values=[]
    for draw in range(start,stop):
        row=s.score(risk,weights[draw],entry.get("thresholds"),entry.get("rank_only",False))
        values.append([row[field] if row[field] is not None else np.nan for field in fields])
    np.savez_compressed(directory/"SAMPLES.npz",fields=np.asarray(fields),samples=np.asarray(values),draws=np.arange(start,stop))
    e.atomic(directory/"POINT.json",point)
    e.atomic(directory/"COMPLETE.json",{
        "signature":signature,"files":{f:e.sha(directory/f) for f in ["SAMPLES.npz","POINT.json"]},
        "patient_replicates":stop-start
    })
    return str(directory)

def aggregate(contexts):
    points=[]
    intervals=[]
    differences=[]
    for context in contexts:
        scores={}
        point_by_model={}
        for entry in context["models"]:
            chunks=[]
            fields=None
            for start in range(0,2000,50):
                root=OUT/"samples"/context["context"]/entry["name"]/f"block_{start:04d}_{min(start+50,2000):04d}"
                assert e.qualified(root)
                payload=np.load(root/"SAMPLES.npz",allow_pickle=False)
                if fields is None:
                    fields=payload["fields"].tolist()
                assert fields==payload["fields"].tolist()
                chunks.append(payload["samples"])
            matrix=np.concatenate(chunks)
            assert matrix.shape==(2000,len(fields))
            values={name:matrix[:,index] for index,name in enumerate(fields)}
            scores[entry["name"]]=values
            point=json.loads((OUT/"samples"/context["context"]/entry["name"]/"block_0000_0050/POINT.json").read_text())
            point_by_model[entry["name"]]=point
            points.append({"context":context["context"],"model":entry["name"],**point})
            for row in s.intervals(entry["name"],point,values):
                intervals.append({"context":context["context"],**row})
            target=OUT/"metric_samples"/context["context"]
            target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/(entry["name"]+".npz"),fields=np.asarray(fields),samples=matrix)
        names=list(scores)
        # Preserve all same-record comparisons. Primary pairs are tagged explicitly.
        for first,second in itertools.combinations(names,2):
            if not first.startswith("new_") and second.startswith("new_"):
                first,second=second,first
            name=first+" MINUS "+second
            common=[key for key in scores[first] if key in scores[second] and key not in ["n","events"]]
            estimates={}
            samples={}
            for key in common:
                a,b=point_by_model[first].get(key),point_by_model[second].get(key)
                estimates[key]=a-b if a is not None and b is not None else None
                samples[key]=scores[first][key]-scores[second][key]
            for row in s.intervals(name,estimates,samples):
                row.update(context=context["context"],
                    primary=(first=="new_RF_RFE_oneSE" and second in ["new_RF_RFE_maxAP","reference_full27","reference_paired27","reference_endop38"]))
                differences.append(row)
    s.rows_to_csv(OUT/"POINT_ESTIMATES.csv",points)
    s.rows_to_csv(OUT/"INTERVALS.csv",intervals)
    s.rows_to_csv(OUT/"PAIRED_DIFFERENCES.csv",differences)
    e.atomic(OUT/"EVALUATION_ACCEPTANCE.json",{
        "status":"PASS_NEW_FIXED_PREDICTIONS_PAIRED_PATIENT_BOOTSTRAP",
        "protocol_sha256":e.protocol_hash(),"downstream_protocol_sha256":e.sha(e.ROOT/"DOWNSTREAM_PROTOCOL_LOCK.json"),
        "contexts":len(contexts),"model_context_rows":len(points),"interval_rows":len(intervals),
        "paired_difference_rows":len(differences),"replicates":2000,
        "source_models_retrained":False,"target_recalibration_or_selection":False,
        "files":{name:e.sha(OUT/name) for name in ["POINT_ESTIMATES.csv","INTERVALS.csv","PAIRED_DIFFERENCES.csv","CONTEXTS.json"]},
        "independent_numerical_and_scientific_review_pending":True
    })

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    assert args.run
    locks=guard_models()
    assert json.loads((e.OUT/"statistics_preflight/QUALIFICATION.json").read_text())["status"].startswith("PASS")
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/"RUN.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        predictions=[predict_model(item) for item in locks]
        e.atomic(OUT/"PREDICTION_BANK_ACCEPTANCE.json",{
            "status":"PASS_ALL12_FROZEN_MODEL_PREDICTIONS","models":predictions,
            "all_locks_precede_target_scoring":True
        })
        contexts=prepare_contexts(predictions)
        snapshot=e.legacy.resource_snapshot()
        reserve=max(4*1024**3,int(.25*snapshot["available_bytes"]))
        workers=min(snapshot["effective_cpu"],max(1,(snapshot["available_bytes"]-reserve-1024**3)//(512*1024**2)))
        tasks=[(context,entry,start,min(2000,start+50)) for context in contexts for entry in context["models"] for start in range(0,2000,50)]
        complete=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(sample_block,task) for task in tasks]
            for future in as_completed(futures):
                complete.append(future.result())
                if len(complete)%10==0 or len(complete)==len(tasks):
                    e.atomic(OUT/"STATUS.json",{"phase":"PAIRED_PATIENT_BOOTSTRAP","accepted_blocks":len(complete),"total_blocks":len(tasks),"workers":workers,"updated":time.time()})
        aggregate(contexts)
        e.atomic(OUT/"STATUS.json",{"phase":"NUMERICAL_EVALUATION_COMPLETE_REVIEW_REQUIRED","updated":time.time()})
        print("NUMERICAL_EVALUATION_COMPLETE_REVIEW_REQUIRED",flush=True)
if __name__=="__main__":
    try:
        main()
    except Exception:
        import traceback
        e.atomic(OUT/"FAILURE.json",{"traceback":traceback.format_exc(),"time":time.time()})
        raise

