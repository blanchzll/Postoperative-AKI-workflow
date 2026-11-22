"""Independent sklearn/expanded-record verification of all fixed-K paired metrics."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import math
import time
import traceback
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from fit_scale import ROOT, PROJECT, FROZEN, e, f, read, write, PROTOCOL_SHA

OUT = ROOT / "outputs/independent_verification"
EVALUATION = ROOT / "outputs/evaluation"


def verify_model(task):
    directory = Path(task["directory"])
    name = task["name"]
    dest = directory / "models" / name
    data = pd.read_csv(directory / "PREDICTIONS_WIDE_PRIVATE.csv", dtype={"excel_row": str, "group_hash": str, "center": str}, float_precision="round_trip")
    y = data.y.to_numpy(dtype=int)
    p = data[name + "__calibrated"].to_numpy()
    group_record = list(zip(data.group_hash, data.excel_row))
    order = np.array(sorted(range(len(y)), key=lambda i: (-p[i], group_record[i])))
    w_bank = np.load(directory / "BOOTSTRAP_WEIGHTS_PRIVATE.npy", mmap_mode="r", allow_pickle=False)
    bank = np.load(dest / "METRIC_SAMPLES.npz", allow_pickle=False)
    columns = bank["metrics"].tolist()
    original = bank["values"]
    maximum_error = 0.0
    checks = 0
    recalculated = {}
    for rep in range(2000):
        w = w_bank[rep]
        n, events = int(w.sum()), int(w @ y)
        expected = float(w @ p)
        values = {"AUROC": roc_auc_score(y, p, sample_weight=w), "AP": average_precision_score(y, p, sample_weight=w), "Brier": brier_score_loss(y, p, sample_weight=w), "observed_expected": events / expected}
        # Literal repeated records preserve all operations and source group/record tie order.
        expanded_order = np.repeat(order, w[order])
        assert len(expanded_order) == n
        for q in [10, 20]:
            reviewed = math.ceil(n * q / 100)
            captured = int(y[expanded_order[:reviewed]].sum())
            false_alerts = reviewed - captured
            missed = events - captured
            values.update({f"capacity{q}_reviewed_n": reviewed, f"capacity{q}_captured_n": captured, f"capacity{q}_missed_n": missed, f"capacity{q}_false_alerts_n": false_alerts, f"capacity{q}_sensitivity": captured / events, f"capacity{q}_PPV": captured / reviewed, f"capacity{q}_reviewed_per1000": 1000 * reviewed / n, f"capacity{q}_captured_per1000": 1000 * captured / n, f"capacity{q}_missed_per1000": 1000 * missed / n, f"capacity{q}_false_alerts_per1000": 1000 * false_alerts / n})
        for metric, value in values.items():
            error = abs(value - original[rep, columns.index(metric)])
            assert error < 2e-10, (name, rep, metric, error)
            maximum_error = max(maximum_error, error)
            checks += 1
            recalculated.setdefault(metric, []).append(value)
    intervals = pd.read_csv(dest / "INTERVALS.csv").set_index("metric")
    for metric, samples in recalculated.items():
        lower, upper = np.quantile(samples, [.025, .975])
        error = max(abs(lower - intervals.loc[metric, "lower95"]), abs(upper - intervals.loc[metric, "upper95"]))
        assert error < 2e-10
        maximum_error = max(maximum_error, error)
        checks += 2
    # Different fitting implementation verifies the probability-calibration diagnostics.
    calibration_checks = 0
    for rep in [None, 0, 999, 1999]:
        w = np.ones(len(y), dtype=int) if rep is None else w_bank[rep]
        index = np.repeat(np.arange(len(y)), w)
        yy, pp = y[index], p[index]
        z = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
        offset = sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(), offset=z).fit(tol=1e-10, maxiter=200)
        point = pd.read_csv(dest / "POINT_ESTIMATES.csv").query("mode == 'calibrated'").iloc[0] if rep is None else None
        expected_offset = point.calibration_intercept if rep is None else original[rep, columns.index("calibration_intercept")]
        assert abs(offset.params[0] - expected_offset) < 2e-6
        calibration_checks += 1
        expected_slope = point.calibration_slope if rep is None else original[rep, columns.index("calibration_slope")]
        if np.isfinite(expected_slope):
            standardized = (z - z.mean()) / z.std()
            fitted = sm.GLM(yy, sm.add_constant(standardized), family=sm.families.Binomial()).fit(tol=1e-10, maxiter=200)
            slope = fitted.params[1] / z.std()
            assert np.isclose(slope, expected_slope, atol=2e-5, rtol=2e-5), (name, rep, slope, expected_slope)
            calibration_checks += 1
    receipt = {"line": task["line"], "scope": task["scope"], "model": name, "all_replicates": 2000, "independent_metric_checks": checks, "independent_calibration_checks": calibration_checks, "maximum_error": maximum_error}
    write(OUT / "models" / f"{task['line']}_{task['scope']}_{name}.json", receipt)
    return receipt


def verify_training_and_locks():
    selected = {(x["line"], x["context"], x["K"]): x for x in read(ROOT / "inputs/SELECTED_BY_SIZE.json")}
    nodes = read(ROOT / "outputs/post_development/REFITS.json")
    bindings = {(x["line"], x["context"]): x for x in read(ROOT / "inputs/FULL_RANKING_BINDINGS.json")}
    checks = 0
    for node in nodes:
        line, context, k = node["line"], node["context"], node["K"]
        directory = Path(node["directory"])
        assert e.sha(directory / "COMPLETE.json") == node["receipt_sha256"]
        assert e.qualified(directory)
        receipt = read(directory / "COMPLETE.json")
        candidate = selected[line, context, k]["candidate"]
        rows, valid_rows, method = f.context_rows(line, context)
        metadata = f.metadata().loc[rows]
        assert receipt["signature"]["params"] == candidate["task"]["params"]
        assert receipt["signature"]["family"] == candidate["task"]["family"]
        assert receipt["input_columns"] == bindings[line, context]["features_by_count"][str(k)]
        assert receipt["raw_input_count"] == k
        assert receipt["training_n"] == len(rows) and receipt["training_events"] == int(metadata.y.sum())
        assert receipt["training_rows_sha256"] == e.digest(rows)
        assert not receipt["target_evaluation_used"]
        if valid_rows:
            assert set(metadata.group_hash).isdisjoint(f.metadata().loc[valid_rows].group_hash)
            outer = pd.read_csv(directory / "OUTER_PREDICTIONS.csv")
            assert set(outer.excel_row) == set(valid_rows)
            np.testing.assert_array_equal(f.metadata().loc[outer.excel_row].y, outer.y)
        checks += 9
    for item in read(ROOT / "outputs/post_development/MODEL_MANIFEST.json"):
        lock = read(item["path"])
        oof_path = Path(item["path"]).parent / "PRIVATE_DEVELOPMENT_OOF.csv"
        assert e.sha(oof_path) == lock["honest_development_raw_oof"]["sha256"]
        data = pd.read_csv(oof_path, dtype={"group_hash": str}, float_precision="round_trip")
        rows, _, _ = f.context_rows(lock["identity"]["line"], "final")
        assert data.excel_row.is_unique and set(data.excel_row) == set(rows)
        z = np.clip(data.prob_raw.to_numpy(), 1e-6, 1 - 1e-6)
        model = LogisticRegression(C=1e6, penalty="l2", solver="lbfgs", max_iter=5000, random_state=42)
        with threadpool_limits(limits=1):
            model.fit(np.log(z / (1 - z)).reshape(-1, 1), data.y)
        np.testing.assert_allclose(model.coef_[0, 0], lock["calibration_slope"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(model.intercept_[0], lock["calibration_intercept"], rtol=0, atol=1e-10)
        p = model.predict_proba(np.log(z / (1 - z)).reshape(-1, 1))[:, 1]
        positive = sorted(p[data.y.to_numpy() == 1], reverse=True)
        assert abs(positive[math.ceil(.8 * len(positive)) - 1] - lock["training_thresholds"]["training_sensitivity80"]) < 1e-12
        for q in [10, 20]:
            rank = math.ceil((len(p) - 1) * (1 - q / 100))
            independent = sorted(p)[rank]
            assert abs(independent - lock["training_thresholds"][f"training_capacity{q}"]) < 1e-12
        checks += 7
    return checks


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    assert read(EVALUATION / "NUMERICAL_COMPUTATION_ACCEPTANCE.json")["prediction_banks"] == 54
    contexts = read(EVALUATION / "CONTEXTS.json")
    tasks = [{**context, "name": name} for context in contexts for name in context["names"]]
    results = []
    with ProcessPoolExecutor(max_workers=min(48, len(os.sched_getaffinity(0))), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(verify_model, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 16 == 0:
                print(json.dumps({"independently_verified_models": len(results), "total": len(tasks)}), flush=True)
    training_checks = verify_training_and_locks()
    receipt = {"status": "PASS_INDEPENDENT_ALL_K_TRAINING_LOCKS_AND_ALL_PATIENT_REPLICATE_KEY_METRICS", "protocol_sha256": PROTOCOL_SHA, "code_sha256": e.sha(__file__), "context_model_banks": len(results), "independent_scalar_checks": sum(x["independent_metric_checks"] for x in results), "independent_calibration_checks": sum(x["independent_calibration_checks"] for x in results), "training_and_lock_checks": training_checks, "maximum_key_metric_difference": max(x["maximum_error"] for x in results), "all2000_primary_replicates_independently_recomputed": True, "calibration_diagnostics_independently_fitted_at": ["point", "bootstrap0", "bootstrap999", "bootstrap1999"], "scientific_review_pending": True, "seconds": time.time() - started}
    write(OUT / "NUMERICAL_ACCEPTANCE.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write(OUT / "FAILURE.json", {"traceback": traceback.format_exc(), "code_sha256": e.sha(__file__)})
        raise
