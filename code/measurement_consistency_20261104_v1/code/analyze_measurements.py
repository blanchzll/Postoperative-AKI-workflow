"""Finite fixed-prediction measurement and consistency evaluation."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import csv
import datetime
import fcntl
import hashlib
import json
import math
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

RUN = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/measurement_consistency_20261104_v1")
REFERENCE = RUN.parent / "full_data_20261025_postslot"
PREVIOUS = REFERENCE / "outputs/novelty_strengthening_20261101_v1"
SOURCE = PREVIOUS / "mimic_calendar_v2"
OUTPUT = RUN / "outputs"
sys.path.insert(0, str(REFERENCE / "work/final_evaluation_engine_20261022_v1"))
from fixed_prediction_metrics import FixedRisk
sys.path.insert(0, str(REFERENCE / "work/formal_evaluation_20261023_v1"))
from evaluation_stats import calibration

QUALITY = ["event_rate", "AUROC", "AP", "Brier", "observed_expected",
           "mean_predicted_probability", "calibration_intercept", "calibration_slope"]
COUNTS = ["reviewed_n", "captured_n", "missed_n", "false_alerts_n", "sensitivity",
          "PPV", "specificity", "reviewed_per1000", "captured_per1000",
          "missed_per1000", "false_alerts_per1000"]
METRICS = QUALITY + [operation + "_" + field
                     for operation in ["global_capacity20", "source_training_capacity20"]
                     for field in COUNTS]

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def inputs():
    protocol = load_json(RUN / "PROTOCOL_LOCK.json")
    assert digest(RUN / "REQUEST.json") == protocol["request_sha256"]
    assert digest(OUTPUT / "PRIVATE_QUALIFIED_RECORDS.csv") == protocol["qualified_record_sha256"]
    cache = np.load(SOURCE / "BOOTSTRAP_INPUT_PRIVATE.npz", allow_pickle=False)
    records = cache["records"].astype(str)
    groups = cache["groups"].astype(str)
    weights = cache["weights"].astype(np.int64)
    table = pd.read_csv(OUTPUT / "PRIVATE_QUALIFIED_RECORDS.csv",
                        dtype={"record_key": str, "group_hash": str, "Sex": str},
                        float_precision="round_trip").set_index("record_key").loc[records]
    np.testing.assert_array_equal(table.group_hash.to_numpy(), groups)
    y7 = table.y_7.to_numpy(dtype=int)
    np.testing.assert_array_equal(y7, cache["y"])
    assert len(records) == 759 and y7.sum() == 59 and len(set(groups)) == 759
    stream = hashlib.sha256()
    for row in weights:
        stream.update(row.astype("<i8").tobytes())
    assert stream.hexdigest() == "b7fbeff201293e306e07a197aa96b5a92a36be0855aa01d14fe60d7ff4a6226d"
    eligible = table.eligible_3.to_numpy(dtype=bool)
    y3 = table.y_3.fillna(-1).to_numpy(dtype=int)
    assert eligible.sum() == 757 and y3[eligible].sum() == 44
    assert np.all(y3[~eligible] == -1)
    profiles = [
        {"id": "full7", "mask": np.ones(759, dtype=bool), "y": y7, "parent": "full"},
        {"id": "common7", "mask": eligible, "y": y7, "parent": "common"},
        {"id": "common3", "mask": eligible, "y": y3, "parent": "common"}
    ]
    for item in protocol["groups"]:
        for level in item["levels"]:
            mask = table[item["dimension"]].astype(str).to_numpy() == level
            profiles.append({"id": item["dimension"] + "__" + level, "mask": mask,
                             "y": y7, "parent": "full"})
    assert len(profiles) == 13
    locks = load_json(REFERENCE / "outputs/common6_model_lock_20261024_v1/STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json")
    thresholds = {}
    for item in locks["models"]:
        assert digest(item["path"]) == item["sha256"]
        model = load_json(item["path"])
        thresholds[model["identity"]["line"] + "__calibrated"] = model["training_thresholds"]["training_capacity20"]
    return protocol, cache, records, groups, weights, table, profiles, thresholds, stream.hexdigest()

def allocation(order, weights):
    budget = math.ceil(int(weights.sum()) * .2)
    before = np.r_[0, np.cumsum(weights[order])[:-1]]
    selected = np.minimum(weights[order], np.maximum(0, budget - before))
    result = np.zeros_like(weights)
    result[order] = selected
    assert int(result.sum()) == budget
    return result

def operation_counts(y, weights, selected):
    population = int(weights.sum())
    events = int(weights @ y)
    reviewed = int(selected.sum())
    captured = int(selected @ y)
    missed = events - captured
    false_alerts = reviewed - captured
    return {
        "reviewed_n": reviewed, "captured_n": captured, "missed_n": missed,
        "false_alerts_n": false_alerts,
        "sensitivity": captured / events if events else None,
        "PPV": captured / reviewed if reviewed else None,
        "specificity": 1 - false_alerts / (population - events) if population > events else None,
        "reviewed_per1000": 1000 * reviewed / population,
        "captured_per1000": 1000 * captured / population,
        "missed_per1000": 1000 * missed / population,
        "false_alerts_per1000": 1000 * false_alerts / population
    }

def prepare_profiles(profiles, probabilities, groups, records):
    result = []
    for profile in profiles:
        indices = np.flatnonzero(profile["mask"])
        y = profile["y"][indices]
        events = int(y.sum())
        size = len(indices)
        result.append({
            **profile, "indices": indices, "n": size, "events": events,
            "risk": FixedRisk(y, probabilities[indices], groups[indices], records[indices]),
            "allow_auc": events >= 15 and size - events >= 20,
            "allow_calibration": events >= 20 and size - events >= 20
        })
    return result

def evaluate_replica(weights, prepared, probabilities, threshold, orders):
    full_selected = allocation(orders["full"], weights)
    common_indices = orders["common_indices"]
    common_selected = np.zeros_like(weights)
    common_selected[common_indices] = allocation(orders["common"], weights[common_indices])
    threshold_selected = weights * (probabilities >= threshold)
    values = np.full((len(prepared), len(METRICS)), np.nan)
    for index, profile in enumerate(prepared):
        indices = profile["indices"]
        local_weights = weights[indices]
        if local_weights.sum() == 0:
            continue
        risk = profile["risk"]
        score = risk.score(local_weights)
        current = {name: score.get(name) for name in QUALITY}
        if not profile["allow_auc"]:
            current["AUROC"] = None
            current["AP"] = None
        current["mean_predicted_probability"] = float(local_weights @ risk.p / local_weights.sum())
        if profile["allow_calibration"]:
            expanded = np.repeat(np.arange(len(indices)), local_weights)
            diagnostic = calibration(risk.y[expanded], risk.p[expanded])
            current["calibration_intercept"] = diagnostic["calibration_intercept"]
            current["calibration_slope"] = diagnostic["calibration_slope"]
        else:
            current["calibration_intercept"] = None
            current["calibration_slope"] = None
        chosen = full_selected if profile["parent"] == "full" else common_selected
        for operation, selected in [
            ("global_capacity20", chosen[indices]),
            ("source_training_capacity20", threshold_selected[indices])
        ]:
            current.update({operation + "_" + key: value
                            for key, value in operation_counts(risk.y, local_weights, selected).items()})
        values[index] = [np.nan if current[name] is None else current[name] for name in METRICS]
    return values

def interval_rows(model, prepared, point, samples):
    rows = []
    for index, profile in enumerate(prepared):
        for column, metric in enumerate(METRICS):
            values = samples[:, index, column]
            defined = values[np.isfinite(values)]
            bounds = np.quantile(defined, [.025, .975]).tolist() if len(defined) else [None, None]
            reason = ""
            if metric in ["AUROC", "AP"] and not profile["allow_auc"]:
                reason = "prespecified_observed_event_or_nonevent_count_guard"
            if metric.startswith("calibration_") and not profile["allow_calibration"]:
                reason = "prespecified_observed_count_calibration_guard"
            estimate = point[index, column]
            rows.append({
                "model": model, "profile": profile["id"], "n": profile["n"],
                "events": profile["events"], "metric": metric,
                "estimate": float(estimate) if np.isfinite(estimate) else None,
                "lower95": bounds[0], "upper95": bounds[1],
                "valid_replicates": len(defined), "requested_replicates": len(samples),
                "not_estimable_reason": reason
            })
    return rows

def run_model(model):
    started = time.time()
    protocol, cache, records, groups, weights, table, profiles, thresholds, stream = inputs()
    directory = OUTPUT / model
    directory.mkdir(mode=0o700, exist_ok=True)
    signature = {
        "protocol_sha256": digest(RUN / "PROTOCOL_LOCK.json"),
        "code_sha256": digest(__file__),
        "source_cache_sha256": digest(SOURCE / "BOOTSTRAP_INPUT_PRIVATE.npz")
    }
    if (directory / "COMPLETE.json").exists():
        existing = load_json(directory / "COMPLETE.json")
        assert existing["signature"] == signature
        return existing
    probabilities = cache[model]
    prepared = prepare_profiles(profiles, probabilities, groups, records)
    common_indices = np.flatnonzero(profiles[1]["mask"])
    orders = {
        "full": np.array(sorted(range(759), key=lambda i: (-probabilities[i], groups[i], records[i]))),
        "common": np.array(sorted(range(len(common_indices)),
                                  key=lambda j: (-probabilities[common_indices[j]],
                                                 groups[common_indices[j]], records[common_indices[j]]))),
        "common_indices": common_indices
    }
    point = evaluate_replica(np.ones(759, dtype=int), prepared, probabilities, thresholds[model], orders)
    # Reuse the accepted overall estimates as a guard for all new paired functions.
    old_point = pd.read_csv(SOURCE / "POINT_ESTIMATES.csv").set_index("model").loc[model]
    for metric in ["AUROC", "AP", "Brier", "observed_expected", "calibration_intercept", "calibration_slope"]:
        assert abs(point[0, METRICS.index(metric)] - float(old_point[metric])) < 1e-12, metric
    assert point[0, METRICS.index("global_capacity20_reviewed_n")] == 152
    samples = np.full((2000, len(prepared), len(METRICS)), np.nan)
    completed = 0
    checkpoint = directory / "CHECKPOINT_PRIVATE.npz"
    if checkpoint.exists():
        previous = np.load(checkpoint, allow_pickle=False)
        assert json.loads(str(previous["signature"])) == signature
        completed = len(previous["samples"])
        samples[:completed] = previous["samples"]
    for replicate in range(completed, 2000):
        samples[replicate] = evaluate_replica(weights[replicate], prepared, probabilities, thresholds[model], orders)
        if replicate == 24:
            save_json(directory / "PILOT25.json", {
                "status": "ACTUAL25REPLICATES_POINT_REFERENCE_AND_ALLOCATION_PASS",
                "signature": signature, "profiles": 13, "replicates": 25,
                "source_weight_sha256": stream, "no_prediction_fit_or_update": True
            })
        if (replicate + 1) % 100 == 0 or replicate == 24:
            temporary = checkpoint.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, samples=samples[:replicate+1],
                                    signature=json.dumps(signature, sort_keys=True))
            os.replace(temporary, checkpoint)
            save_json(directory / "PROGRESS.json", {
                "phase": "fixed_prediction_bootstrap",
                "done": replicate+1, "total": 2000, "pid": os.getpid()
            })
    intervals = interval_rows(model, prepared, point, samples)
    pd.DataFrame(intervals).to_csv(directory / "ALL_ESTIMATES_AND_INTERVALS.csv", index=False)
    comparisons = []
    profile_index = {profile["id"]: index for index, profile in enumerate(prepared)}
    for left, right, purpose in protocol["contrasts"]:
        li, ri = profile_index[left], profile_index[right]
        for column, metric in enumerate(METRICS):
            differences = samples[:, li, column] - samples[:, ri, column]
            defined = differences[np.isfinite(differences)]
            bounds = np.quantile(defined, [.025, .975]).tolist() if len(defined) else [None, None]
            estimate = point[li, column] - point[ri, column]
            comparisons.append({
                "model": model, "left": left, "right": right, "purpose": purpose,
                "metric": metric,
                "estimate": float(estimate) if np.isfinite(estimate) else None,
                "lower95": bounds[0], "upper95": bounds[1],
                "valid_replicates": len(defined), "requested_replicates": 2000
            })
    pd.DataFrame(comparisons).to_csv(directory / "PAIRED_CONTRASTS.csv", index=False)
    np.savez_compressed(directory / "SAMPLES_PRIVATE.npz", samples=samples, point=point,
                        profiles=np.array([p["id"] for p in prepared]), metrics=np.array(METRICS))
    result = {
        "status": "COMPUTED_LOCKED_NEW_MEASUREMENT_AND_CONSISTENCY",
        "model": model, "signature": signature, "profile_count": 13,
        "interval_rows": len(intervals), "contrast_rows": len(comparisons),
        "replicates": 2000, "source_weight_sha256": stream,
        "threshold": thresholds[model], "elapsed_seconds": time.time()-started,
        "files": {name: digest(directory / name) for name in [
            "ALL_ESTIMATES_AND_INTERVALS.csv", "PAIRED_CONTRASTS.csv", "SAMPLES_PRIVATE.npz", "PILOT25.json"
        ]}
    }
    save_json(directory / "COMPLETE.json", result)
    return result

def main():
    with (RUN / "OWNER.lock").open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUTPUT / "NUMERICAL_COMPUTATION_ACCEPTANCE.json").exists()
        protocol = load_json(RUN / "PROTOCOL_LOCK.json")
        assert digest(__file__) == load_json(RUN / "NUMERICAL_CODE_LOCK.json")["analysis_code_sha256"]
        status = load_json(RUN / "STATUS.json")
        status.update(state="LOCKED_NEW_COMPUTATION_RUNNING", new_computation_started=True,
                      main_unit="aki-measurement-compute-20261104-v1.service")
        save_json(RUN / "STATUS.json", status)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(run_model, protocol["population"]["models"]))
        intervals = pd.concat([pd.read_csv(OUTPUT / model / "ALL_ESTIMATES_AND_INTERVALS.csv")
                               for model in protocol["population"]["models"]], ignore_index=True)
        contrasts = pd.concat([pd.read_csv(OUTPUT / model / "PAIRED_CONTRASTS.csv")
                               for model in protocol["population"]["models"]], ignore_index=True)
        intervals.to_csv(OUTPUT / "ALL_ESTIMATES_AND_INTERVALS.csv", index=False)
        contrasts.to_csv(OUTPUT / "PAIRED_CONTRASTS.csv", index=False)
        save_json(OUTPUT / "NUMERICAL_COMPUTATION_ACCEPTANCE.json", {
            "status": "COMPUTED_NEW_REQUEST_BOUND_ANALYSES_REVIEW_PENDING",
            "task_id": protocol["task_id"], "request_sha256": protocol["request_sha256"],
            "protocol_sha256": digest(RUN / "PROTOCOL_LOCK.json"),
            "analysis_code_sha256": digest(__file__), "models": results,
            "new_prediction_fits": 0,
            "files": {name: digest(OUTPUT / name) for name in [
                "ALL_ESTIMATES_AND_INTERVALS.csv", "PAIRED_CONTRASTS.csv"
            ]}
        })
        status.update(state="COMPUTED_REQUIRES_NUMERICAL_AND_SCIENTIFIC_REVIEW", main_unit=None)
        save_json(RUN / "STATUS.json", status)
        print(json.dumps({"status": "NEW_COMPUTATION_FINISHED",
                          "interval_rows": len(intervals), "contrasts": len(contrasts)}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        save_json(OUTPUT / "COMPUTATION_FAILURE.json", {"traceback": traceback.format_exc()})
        raise
