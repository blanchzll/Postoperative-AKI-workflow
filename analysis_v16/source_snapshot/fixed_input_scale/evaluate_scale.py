"""Paired fixed-prediction evaluation of every prespecified K on identical patients."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import time
import hashlib
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from fit_scale import ROOT, PROJECT, FROZEN, e, f, read, write, update, PROTOCOL_SHA
sys.path.insert(0, str(e.SOURCE / "work/formal_evaluation_20261023_v1"))
import evaluation_stats as native
from fixed_prediction_metrics import FixedRisk, _design, group_weights

OUT = ROOT / "outputs/evaluation"
GLOBAL_METRICS = ["AUROC", "AP", "Brier", "observed_expected", "calibration_intercept", "calibration_slope", "calibration_joint_intercept"]
CAPACITY_METRICS = ["reviewed_n", "captured_n", "missed_n", "false_alerts_n", "sensitivity", "PPV", "reviewed_per1000", "captured_per1000", "missed_per1000", "false_alerts_per1000"]
METRICS = GLOBAL_METRICS + [f"capacity{q}_{m}" for q in (10, 20) for m in CAPACITY_METRICS]


def predict(item):
    lock = read(item["path"])
    assert e.sha(item["path"]) == item["sha256"]
    assert lock["protocol_sha256"] == PROTOCOL_SHA
    line = lock["identity"]["line"]
    k = lock["actual_input_count"]
    directory = OUT / "predictions" / line / f"K{k:02d}"
    signature = {"lock_sha256": item["sha256"], "code_sha256": e.sha(__file__), "source_sha256": e.sha(e.legacy.INPUT / "preop_preop_features.csv")}
    if e.qualified(directory, signature):
        return read(directory / "COMPLETE.json")["entry"]
    directory.mkdir(parents=True, exist_ok=True)
    meta = f.metadata()
    target = meta.loc[meta[line + "_partition_v2"].eq("validation")].copy()
    rows, _, _ = f.context_rows(line, "final")
    assert set(target.group_hash).isdisjoint(meta.loc[rows].group_hash)
    for component in ("raw_bundle", "calibrator"):
        assert e.sha(lock[component]["path"]) == lock[component]["sha256"]
    bundle = joblib.load(lock["raw_bundle"]["path"])
    calibrator = joblib.load(lock["calibrator"]["path"])
    with threadpool_limits(limits=1):
        raw = f.predict_raw(bundle, f.raw_frame(target.excel_row.tolist(), e.legacy.spec_for("preop")), e.legacy.keys(target))
        calibrated = f.apply_calibration(calibrator, raw)
    path = directory / "PREDICTIONS.csv"
    target.assign(prob_raw=raw, prob_calibrated=calibrated).to_csv(path, index=False)
    entry = {"line": line, "K": k, "name": f"K{k:02d}", "lock": item["path"], "lock_sha256": item["sha256"], "path": str(path), "sha256": e.sha(path), "family": lock["family"], "input_columns": lock["input_columns"], "source_calibration_slope": lock["calibration_slope"], "thresholds": lock["training_thresholds"]}
    write(directory / "COMPLETE.json", {"signature": signature, "files": {"PREDICTIONS.csv": e.sha(path)}, "entry": entry, "training_group_disjointness": True})
    return entry


def reference_files(line, scope):
    evaluation = e.SOURCE / "outputs/formal_evaluation_20261023_v1"
    family = "LR" if line == "geography" else "Voting"
    files = {
        "original_nine": evaluation / f"{line}_preop/private_predictions/preop_compact_champion_{family}/PREDICTIONS.csv",
        "original_full27": evaluation / f"{line}_preop/private_predictions/preop_full_champion_{family}/PREDICTIONS.csv",
        "prior_RF_RFE_oneSE": FROZEN / f"outputs/evaluation/predictions/{line}/new_RF_RFE_oneSE/PREDICTIONS.csv",
        "prior_RF_RFE_maxAP": FROZEN / f"outputs/evaluation/predictions/{line}/new_RF_RFE_maxAP/PREDICTIONS.csv",
    }
    if scope == "common_paired":
        paired_family = "Voting" if line == "geography" else "SGBT"
        files.update({"original_endop38": evaluation / f"{line}_paired/private_predictions/endop_full_champion_{family}/PREDICTIONS.csv", "original_paired_development27": evaluation / f"{line}_paired/private_predictions/preop_full_champion_{paired_family}/PREDICTIONS.csv"})
    return files


def prepare_context(line, scope, entries):
    directory = OUT / f"{line}_{scope}"
    directory.mkdir(parents=True, exist_ok=True)
    files = reference_files(line, scope)
    bank = {name: pd.read_csv(path, dtype={"excel_row": str, "group_hash": str, "center": str}, float_precision="round_trip") for name, path in files.items()}
    anchor = bank["original_nine"] if scope == "preop" else bank["original_endop38"]
    anchor = anchor.sort_values(["center", "group_hash", "excel_row"]).reset_index(drop=True)
    rows = anchor.excel_row.tolist()
    if scope == "common_paired":
        spec = e.legacy.spec_for("preop")
        columns = spec["num"] + spec["cat"]
        main = pd.read_csv(e.legacy.INPUT / "preop_preop_features.csv", dtype={c: "string" for c in spec["cat"]}, float_precision="round_trip").set_index("excel_row")
        paired = pd.read_csv(e.legacy.INPUT / "paired_preop_features.csv", dtype={c: "string" for c in spec["cat"]}, float_precision="round_trip").set_index("excel_row")
        pd.testing.assert_frame_equal(main.loc[list(map(int, rows)), columns], paired.loc[list(map(int, rows)), columns], check_dtype=False, check_exact=True)
    wide = anchor[["excel_row", "group_hash", "center", "y", "surgery_date"]].copy()
    provenance = []
    thresholds = {}
    for entry in entries:
        if entry["line"] == line:
            bank[entry["name"]] = pd.read_csv(entry["path"], dtype={"excel_row": str, "group_hash": str, "center": str}, float_precision="round_trip")
            files[entry["name"]] = Path(entry["path"])
            thresholds[entry["name"]] = entry["thresholds"]
    for name, data in bank.items():
        assert data.excel_row.is_unique
        aligned = data.set_index("excel_row").loc[rows].reset_index()
        for column in ("group_hash", "center", "y", "surgery_date"):
            assert aligned[column].tolist() == anchor[column].tolist(), (line, scope, name, column)
        for mode, column in (("calibrated", "prob_calibrated"), ("raw", "prob_raw")):
            if column in aligned:
                wide[name + "__" + mode] = aligned[column].to_numpy()
        provenance.append({"name": name, "path": str(files[name]), "sha256": e.sha(files[name])})
    # The canonical hospital/group/record order exactly matches the original source bootstrap.
    yy, gg, rr, cc, strata, _ = _design(wide.y, wide.group_hash, wide.excel_row, wide.center, {"anchor": wide.original_nine__calibrated})
    assert np.array_equal(rr, wide.excel_row.to_numpy(dtype=str))
    path = directory / "PREDICTIONS_WIDE_PRIVATE.csv"
    wide.to_csv(path, index=False)
    weights = np.lib.format.open_memmap(directory / "BOOTSTRAP_WEIGHTS_PRIVATE.npy", mode="w+", dtype=np.int64, shape=(2000, len(wide)))
    rng = np.random.default_rng(20261017)
    proof = hashlib.sha256()
    for rep in range(2000):
        weights[rep] = group_weights(strata, len(wide), rng)
        proof.update(weights[rep].astype("<i8").tobytes())
    weights.flush()
    del weights
    writer = PROJECT / f"writing_runs/v16_continuation_20261117/paired_comparison_20261117/{line}_{scope}/ACCEPTANCE.json"
    assert proof.hexdigest() == read(writer)["resample_weight_sha256"]
    receipt = {"status": "PASS_IDENTICAL_CASES_INPUTS_AND_PATIENT_RESAMPLING", "line": line, "scope": scope, "n": len(wide), "events": int(wide.y.sum()), "groups": wide.group_hash.nunique(), "source_entries": provenance, "prediction_sha256": e.sha(path), "bootstrap_stream_sha256": proof.hexdigest(), "weight_file_sha256": e.sha(directory / "BOOTSTRAP_WEIGHTS_PRIVATE.npy"), "seed": 20261017, "replicates": 2000, "original_writer_bootstrap_match": True, "thresholds": thresholds}
    write(directory / "CONTEXT_ACCEPTANCE.json", receipt)
    return {"line": line, "scope": scope, "directory": str(directory), "names": list(bank)}


def score(fixed, weights):
    base = fixed.score(weights)
    index = np.repeat(np.arange(len(fixed.y)), weights)
    result = dict(base)
    result.update(native.calibration(fixed.y[index], fixed.p[index]))
    for q in (10, 20):
        result.update({f"capacity{q}_{key}": value for key, value in fixed.capacity(q / 100, weights, base).items() if key in CAPACITY_METRICS})
    return result


def sample_model(task):
    started = time.time()
    directory = Path(task["directory"])
    name = task["name"]
    dest = directory / "models" / name
    dest.mkdir(parents=True, exist_ok=True)
    context = read(directory / "CONTEXT_ACCEPTANCE.json")
    signature = {"context_sha256": e.sha(directory / "CONTEXT_ACCEPTANCE.json"), "model": name, "code_sha256": e.sha(__file__), "protocol_sha256": PROTOCOL_SHA}
    if e.qualified(dest, signature):
        return {"directory": str(dest), "status": "reused_current_complete"}
    frame = pd.read_csv(directory / "PREDICTIONS_WIDE_PRIVATE.csv", dtype={"excel_row": str, "group_hash": str, "center": str}, float_precision="round_trip")
    fixed = FixedRisk(frame.y, frame[name + "__calibrated"], frame.group_hash, frame.excel_row)
    weights = np.load(directory / "BOOTSTRAP_WEIGHTS_PRIVATE.npy", mmap_mode="r", allow_pickle=False)
    point = score(fixed, np.ones(len(frame), dtype=np.int64))
    samples = np.empty((2000, len(METRICS)), dtype=float)
    for rep in range(2000):
        values = score(fixed, weights[rep])
        samples[rep] = [np.nan if values.get(metric) is None else values[metric] for metric in METRICS]
    np.savez_compressed(dest / "METRIC_SAMPLES.npz", values=samples, metrics=np.asarray(METRICS))
    intervals = []
    for j, metric in enumerate(METRICS):
        valid = samples[np.isfinite(samples[:, j]), j]
        ci = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
        intervals.append({"metric": metric, "estimate": point.get(metric), "lower95": ci[0], "upper95": ci[1], "valid_replicates": len(valid), "requested_replicates": 2000})
    pd.DataFrame(intervals).to_csv(dest / "INTERVALS.csv", index=False)
    point_rows = [{"mode": "calibrated", **point}]
    if name + "__raw" in frame:
        raw_fixed = FixedRisk(frame.y, frame[name + "__raw"], frame.group_hash, frame.excel_row)
        point_rows.append({"mode": "raw", **score(raw_fixed, np.ones(len(frame), dtype=np.int64))})
    pd.DataFrame(point_rows).to_csv(dest / "POINT_ESTIMATES.csv", index=False)
    burden = []
    for key, threshold in context.get("thresholds", {}).get(name, {}).items():
        if key.startswith("training_") and isinstance(threshold, (int, float)):
            flagged = fixed.p >= threshold
            tp = int(np.sum(fixed.y[flagged]))
            burden.append({"source_threshold_name": key, "frozen_threshold": threshold, "n": len(frame), "events": int(frame.y.sum()), "flagged_n": int(flagged.sum()), "captured_n": tp, "reviewed_per1000": 1000 * flagged.mean(), "captured_per1000": 1000 * tp / len(frame)})
    if burden:
        pd.DataFrame(burden).to_csv(dest / "SOURCE_THRESHOLD_BURDEN.csv", index=False)
    files = {p.name: e.sha(p) for p in dest.iterdir() if p.suffix in (".csv", ".npz")}
    write(dest / "COMPLETE.json", {"signature": signature, "files": files, "line": task["line"], "scope": task["scope"], "name": name, "replicates": 2000, "seconds": time.time() - started})
    return {"line": task["line"], "scope": task["scope"], "name": name, "seconds": time.time() - started}


def aggregate(contexts):
    all_points, all_intervals, differences = [], [], []
    for context in contexts:
        directory = Path(context["directory"])
        banks = {}
        points = {}
        for name in context["names"]:
            dest = directory / "models" / name
            assert e.qualified(dest)
            arr = np.load(dest / "METRIC_SAMPLES.npz", allow_pickle=False)
            assert arr["metrics"].tolist() == METRICS
            banks[name] = arr["values"]
            table = pd.read_csv(dest / "POINT_ESTIMATES.csv")
            points[name] = table.loc[table["mode"].eq("calibrated")].iloc[0]
            all_points.extend(table.assign(line=context["line"], scope=context["scope"], model=name).to_dict("records"))
            all_intervals.extend(pd.read_csv(dest / "INTERVALS.csv").assign(line=context["line"], scope=context["scope"], model=name).to_dict("records"))
        refs = [name for name in context["names"] if not name.startswith("K")]
        pairs = [(f"K{k:02d}", ref) for k in range(1, 28) for ref in refs]
        pairs += [("original_full27", "original_nine")]
        if context["scope"] == "common_paired":
            pairs += [("original_endop38", "original_nine"), ("original_endop38", "original_full27"), ("original_endop38", "original_paired_development27")]
        for a, b in pairs:
            delta = banks[a] - banks[b]
            for j, metric in enumerate(METRICS):
                valid = delta[np.isfinite(delta[:, j]), j]
                ci = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
                differences.append({"line": context["line"], "scope": context["scope"], "model": a, "reference": b, "direction": "model MINUS reference", "metric": metric, "estimate": points[a][metric] - points[b][metric], "lower95": ci[0], "upper95": ci[1], "valid_replicates": len(valid), "requested_replicates": 2000})
    pd.DataFrame(all_points).to_csv(OUT / "POINT_ESTIMATES.csv", index=False)
    pd.DataFrame(all_intervals).to_csv(OUT / "INTERVALS.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "PAIRED_DIFFERENCES.csv", index=False)
    # Original fixed9/full27/endop38 paired intervals remain exactly reconciled.
    reconciled = 0
    table = pd.DataFrame(differences)
    for context in contexts:
        old = pd.read_csv(PROJECT / f"writing_runs/v16_continuation_20261117/paired_comparison_20261117/{context['line']}_{context['scope']}/PAIRED_DIFFERENCES.csv")
        name_map = {"nine": "original_nine", "full27": "original_full27", "endop38": "original_endop38", "paired27": "original_paired_development27"}
        for row in old.to_dict("records"):
            a, b = row["difference"].split(" MINUS ")
            if a not in name_map or b not in name_map:
                continue
            metric = row["metric"]
            if metric not in GLOBAL_METRICS:
                metric = f"capacity{int(100 * row['review_fraction'])}_{metric}"
            match = table.loc[(table.line == context["line"]) & (table.scope == context["scope"]) & (table.model == name_map[a]) & (table.reference == name_map[b]) & (table.metric == metric)]
            assert len(match) == 1
            for field in ("estimate", "lower95", "upper95"):
                assert abs(float(match.iloc[0][field]) - row[field]) < 1e-10
            reconciled += 1
    return {"point_rows": len(all_points), "interval_rows": len(all_intervals), "paired_difference_rows": len(differences), "original_direct_comparison_intervals_reconciled": reconciled}


def main():
    start = time.time()
    assert read(ROOT / "outputs/FIT_AND_LOCK_ACCEPTANCE.json")["final_systems"] == 54
    assert read(ROOT / "inputs/EVALUATION_CODE_LOCK.json")["sha256"] == e.sha(__file__)
    models = read(ROOT / "outputs/post_development/MODEL_MANIFEST.json")
    OUT.mkdir(parents=True, exist_ok=True)
    workers = min(48, len(os.sched_getaffinity(0)))
    update("PREDICTING_ALL54_LOCKED_FIXED_K_MODELS")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        entries = list(pool.map(predict, models))
    write(OUT / "PREDICTION_MANIFEST.json", entries)
    contexts = [prepare_context(line, scope, entries) for line in ("geography", "temporal") for scope in ("preop", "common_paired")]
    write(OUT / "CONTEXTS.json", contexts)
    tasks = [{**context, "name": name} for context in contexts for name in context["names"]]
    update("PAIRED2000_PATIENT_BOOTSTRAP_ALL_K", prediction_banks=54, statistics_tasks=len(tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(sample_model, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if i % 10 == 0 or i == len(futures):
                update("PAIRED2000_PATIENT_BOOTSTRAP_ALL_K", tasks_complete=i, tasks_total=len(futures), latest=result)
                print(json.dumps({"complete": i, "total": len(futures), "latest": result}), flush=True)
    counts = aggregate(contexts)
    write(OUT / "NUMERICAL_COMPUTATION_ACCEPTANCE.json", {"status": "PASS_ALL_K_FIXED_PREDICTIONS_PAIRED_PATIENT_BOOTSTRAP_COMPUTATION", "protocol_sha256": PROTOCOL_SHA, "code_sha256": e.sha(__file__), "prediction_banks": len(entries), "contexts": len(contexts), "statistics_tasks": len(tasks), **counts, "replicates_per_context_model": 2000, "all_context_streams_match_original": True, "new_target_side_fitting": False, "independent_and_scientific_review_pending": True, "seconds": time.time() - start})
    update("ALL_K_NUMERIC_COMPLETE_INDEPENDENT_REVIEW_PENDING", **counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write(ROOT / "outputs/EVALUATION_FAILURE.json", {"traceback": traceback.format_exc(), "code_sha256": e.sha(__file__), "protocol_sha256": PROTOCOL_SHA})
        raise
