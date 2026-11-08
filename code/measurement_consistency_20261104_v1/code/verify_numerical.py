"""Independent sklearn and statsmodels replay of all new fixed-prediction functionals."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import csv
import hashlib
import json
import math
import multiprocessing
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score

RUN = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/measurement_consistency_20261104_v1")
SOURCE = RUN.parent / "full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1/mimic_calendar_v2"
OUTPUT = RUN / "outputs"

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def percentile_interval(values):
    finite = sorted(float(v) for v in values if np.isfinite(v))
    if not finite:
        return np.nan, np.nan, 0
    results = []
    for probability in [.025, .975]:
        position = probability * (len(finite) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        results.append(finite[lower] + (position - lower) * (finite[upper] - finite[lower]))
    return results[0], results[1], len(finite)

def calibration_irls(y, probabilities):
    clipped = np.clip(probabilities, 1e-6, 1-1e-6)
    z = np.log(clipped / (1-clipped))
    if y.min() == y.max():
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        offset_fit = sm.GLM(y, np.ones((len(y), 1)), offset=z,
                            family=sm.families.Binomial()).fit(tol=1e-11, maxiter=100)
        offset = float(offset_fit.params[0])
        if np.std(z) < 1e-10 or min(z[y == 1]) >= max(z[y == 0]) or min(z[y == 0]) >= max(z[y == 1]):
            return offset, None
        joint = sm.GLM(y, np.column_stack([np.ones(len(y)), z]),
                       family=sm.families.Binomial()).fit(tol=1e-11, maxiter=100)
        assert joint.converged
        x = (z-z.mean()) / z.std()
        q = joint.predict()
        variance = q*(1-q)
        hessian = np.array([[variance.mean(), np.mean(variance*x)],
                            [np.mean(variance*x), np.mean(variance*x*x)]])
        slope = float(joint.params[1]) if np.linalg.cond(hessian) <= 1e10 else None
    return offset, slope

def load_design(model):
    protocol = read_json(RUN / "PROTOCOL_LOCK.json")
    assert read_json(OUTPUT / "INDEPENDENT_INPUT_VERIFICATION.json")["status"].startswith("PASS")
    cache = np.load(SOURCE / "BOOTSTRAP_INPUT_PRIVATE.npz", allow_pickle=False)
    with (OUTPUT / "PRIVATE_QUALIFIED_RECORDS.csv").open(newline="") as handle:
        by_record = {r["record_key"]: r for r in csv.DictReader(handle)}
    records, groups = cache["records"].astype(str), cache["groups"].astype(str)
    data = [by_record[r] for r in records]
    np.testing.assert_array_equal(groups, [r["group_hash"] for r in data])
    y7 = np.array([int(r["y_7"]) for r in data])
    y3 = np.array([int(r["y_3"]) if r["eligible_3"] == "1" else -1 for r in data])
    eligible3 = np.array([r["eligible_3"] == "1" for r in data])
    profiles = [
        ("full7", np.ones(759, dtype=bool), y7, "full"),
        ("common7", eligible3, y7, "common"),
        ("common3", eligible3, y3, "common")
    ]
    for item in protocol["groups"]:
        for level in item["levels"]:
            membership = np.array([r[item["dimension"]] == level for r in data])
            profiles.append((item["dimension"]+"__"+level, membership, y7, "full"))
    weights = cache["weights"].astype(int)
    proof = hashlib.sha256()
    generator = np.random.default_rng(20261017)
    assert list(groups) == sorted(set(groups))
    for row in weights:
        draw = generator.integers(0, 759, 759)
        expected = np.bincount(draw, minlength=759)
        np.testing.assert_array_equal(row, expected)
        proof.update(row.astype("<i8").tobytes())
    assert proof.hexdigest() == "b7fbeff201293e306e07a197aa96b5a92a36be0855aa01d14fe60d7ff4a6226d"
    native = np.load(OUTPUT / model / "SAMPLES_PRIVATE.npz", allow_pickle=False)
    assert list(native["profiles"]) == [p[0] for p in profiles]
    metrics = list(native["metrics"].astype(str))
    assert metrics == protocol["metrics"]["quality"] + [
        name+"_"+field for name in protocol["metrics"]["operations"] for field in protocol["metrics"]["decisions"]]
    threshold = read_json(OUTPUT / model / "COMPLETE.json")["threshold"]
    return protocol, records, groups, weights, cache[model], profiles, metrics, threshold, native

def literal_allocation(expanded, probabilities, groups, records):
    ordered = sorted(expanded, key=lambda i: (-probabilities[i], groups[i], records[i]))
    count = math.ceil(.20 * len(ordered))
    return np.array(ordered[:count], dtype=int)

def independent_scores(expanded, probabilities, profiles, selected_full, selected_common,
                       selected_threshold, metrics):
    result = np.full((len(profiles), len(metrics)), np.nan)
    for i, (name, membership, labels, parent) in enumerate(profiles):
        population = expanded[membership[expanded]]
        n = len(population)
        if not n:
            continue
        y, p = labels[population], probabilities[population]
        e = int(y.sum())
        observed_e = int(labels[membership].sum())
        observed_n = int(membership.sum())
        values = {
            "event_rate": e/n, "AUROC": None, "AP": None,
            "Brier": float(np.mean((y-p)**2)),
            "observed_expected": float(e/p.sum()) if p.sum() > 0 else None,
            "mean_predicted_probability": float(p.mean()),
            "calibration_intercept": None, "calibration_slope": None
        }
        if observed_e >= 15 and observed_n-observed_e >= 20:
            values["AUROC"] = float(roc_auc_score(y, p)) if 0 < e < n else None
            values["AP"] = float(average_precision_score(y, p)) if e else None
        if observed_e >= 20 and observed_n-observed_e >= 20:
            values["calibration_intercept"], values["calibration_slope"] = calibration_irls(y, p)
        for operation, selected in [
            ("global_capacity20", selected_full if parent == "full" else selected_common),
            ("source_training_capacity20", selected_threshold)
        ]:
            selected = selected[membership[selected]]
            reviewed = len(selected)
            captured = int(labels[selected].sum())
            missed, false_alerts = e-captured, reviewed-captured
            counts = {
                "reviewed_n": reviewed, "captured_n": captured, "missed_n": missed,
                "false_alerts_n": false_alerts,
                "sensitivity": captured/e if e else None,
                "PPV": captured/reviewed if reviewed else None,
                "specificity": 1-false_alerts/(n-e) if n > e else None,
                "reviewed_per1000": 1000*reviewed/n, "captured_per1000": 1000*captured/n,
                "missed_per1000": 1000*missed/n, "false_alerts_per1000": 1000*false_alerts/n
            }
            values.update({operation+"_"+k: v for k, v in counts.items()})
        result[i] = [np.nan if values[m] is None else values[m] for m in metrics]
    return result

def verify_model(model):
    start = time.time()
    protocol, records, groups, weights, probabilities, profiles, metrics, threshold, native = load_design(model)
    own_samples = np.full_like(native["samples"], np.nan)
    max_error = {metric: 0.0 for metric in metrics}
    assertions = 0
    def compare(left, right):
        nonlocal assertions
        assert left.shape == right.shape
        np.testing.assert_array_equal(np.isfinite(left), np.isfinite(right))
        for j, metric in enumerate(metrics):
            good = np.isfinite(left[..., j])
            differences = np.abs(left[..., j][good]-right[..., j][good])
            error = float(differences.max()) if len(differences) else 0.0
            max_error[metric] = max(max_error[metric], error)
            tolerance = 1e-4 if metric.startswith("calibration_") else 1e-9
            assert error <= tolerance, (model, metric, error, tolerance)
            assertions += len(left[..., j].reshape(-1))
    for replicate in range(-1, 2000):
        multiplicity = np.ones(759, dtype=int) if replicate < 0 else weights[replicate]
        expanded = np.repeat(np.arange(759), multiplicity)
        eligible = profiles[1][1]
        common_expanded = expanded[eligible[expanded]]
        selected_full = literal_allocation(expanded, probabilities, groups, records)
        selected_common = literal_allocation(common_expanded, probabilities, groups, records)
        selected_threshold = expanded[probabilities[expanded] >= threshold]
        score = independent_scores(expanded, probabilities, profiles, selected_full, selected_common,
                                   selected_threshold, metrics)
        if replicate < 0:
            own_point = score
            compare(score, native["point"])
        else:
            own_samples[replicate] = score
            compare(score, native["samples"][replicate])
        if replicate >= 0 and (replicate+1) % 100 == 0:
            save_json(OUTPUT / model / "INDEPENDENT_PROGRESS.json",
                      {"done": replicate+1, "total": 2000, "pid": os.getpid()})
    table = pd.read_csv(OUTPUT / model / "ALL_ESTIMATES_AND_INTERVALS.csv")
    for row in table.itertuples():
        i = [p[0] for p in profiles].index(row.profile)
        j = metrics.index(row.metric)
        low, high, count = percentile_interval(own_samples[:, i, j])
        tolerance = 1e-4 if row.metric.startswith("calibration_") else 1e-9
        for actual, expected in [(own_point[i, j], row.estimate), (low, row.lower95), (high, row.upper95)]:
            assert np.isfinite(actual) == np.isfinite(expected)
            if np.isfinite(actual):
                error = abs(float(actual)-float(expected))
                max_error[row.metric] = max(max_error[row.metric], error)
                assert error <= tolerance
            assertions += 1
        assert count == row.valid_replicates and row.requested_replicates == 2000
        assertions += 1
    differences = pd.read_csv(OUTPUT / model / "PAIRED_CONTRASTS.csv")
    for row in differences.itertuples():
        left = [p[0] for p in profiles].index(row.left)
        right = [p[0] for p in profiles].index(row.right)
        j = metrics.index(row.metric)
        sample = own_samples[:, left, j]-own_samples[:, right, j]
        low, high, count = percentile_interval(sample)
        point = own_point[left, j]-own_point[right, j]
        tolerance = 2e-4 if row.metric.startswith("calibration_") else 1e-9
        for actual, expected in [(point, row.estimate), (low, row.lower95), (high, row.upper95)]:
            assert np.isfinite(actual) == np.isfinite(expected)
            if np.isfinite(actual):
                error = abs(float(actual)-float(expected))
                max_error[row.metric] = max(max_error[row.metric], error)
                assert error <= tolerance
            assertions += 1
        assert count == row.valid_replicates
        assertions += 1
    result = {
        "status": "PASS_INDEPENDENT_ALL_POINT_REPLICATE_FUNCTIONS_INTERVALS_AND_PAIRED_CONTRASTS",
        "model": model, "replicates": 2000, "profiles": 13, "metrics": len(metrics),
        "interval_rows": len(table), "contrast_rows": len(differences),
        "assertions": assertions, "max_abs_error_by_metric": max_error,
        "elapsed_seconds": time.time()-start, "no_native_analysis_imported": True,
        "implementation": "Literal expanded patient copies, sklearn AUROC/AP, direct count arithmetic and statsmodels BinomialIRLS diagnostics. Native uses fixed weighted ranks and scipy brentq/BFGS.",
        "code_sha256": digest(__file__),
        "native_complete_sha256": digest(OUTPUT / model / "COMPLETE.json")
    }
    save_json(OUTPUT / model / "INDEPENDENT_NUMERICAL_VERIFICATION.json", result)
    return result

def main():
    assert read_json(OUTPUT / "NUMERICAL_COMPUTATION_ACCEPTANCE.json")["status"].startswith("COMPUTED")
    protocol = read_json(RUN / "PROTOCOL_LOCK.json")
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(verify_model, protocol["population"]["models"]))
    save_json(OUTPUT / "INDEPENDENT_NUMERICAL_VERIFICATION.json", {
        "status": "PASS_INDEPENDENT_FULL_NEW_MEASUREMENT_AND_CONSISTENCY_NUMERICAL_REPLAY",
        "request_sha256": digest(RUN / "REQUEST.json"),
        "protocol_sha256": digest(RUN / "PROTOCOL_LOCK.json"),
        "source_weight_stream_verified": True, "replicates": 2000,
        "models": results, "assertions": sum(r["assertions"] for r in results),
        "source_inputs_verified": "INDEPENDENT_INPUT_VERIFICATION.json",
        "code_sha256": digest(__file__)
    })
    print(json.dumps({"status": "INDEPENDENT_NUMERICAL_PASS",
                      "assertions": sum(r["assertions"] for r in results)}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        save_json(OUTPUT / "INDEPENDENT_VERIFICATION_FAILURE.json", {"traceback": traceback.format_exc()})
        raise
