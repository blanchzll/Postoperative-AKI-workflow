
"""Development-bound final refits, honest OOF calibration and frozen model locks.

This stage consumes accepted development selection and never computes a target
performance metric. Heldout prediction and scoring have separate entry points.
"""
from pathlib import Path
import sys
import os
import json
import time
import fcntl
import warnings
import resource
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as e
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

POST = e.OUT / "post_development"
METHODS = ["RF_RFE", "LASSO", "EN"]
RULES = ["oneSE", "maxAP"]

@lru_cache(maxsize=1)
def this_code_hash():
    return e.sha(__file__)

@lru_cache(maxsize=1)
def metadata():
    return pd.read_csv(e.legacy.INPUT / "preop_metadata.csv", dtype={"group_hash": str}).set_index("excel_row", drop=False)

def context_rows(line, context):
    parts = json.loads((e.legacy.INPUT / f"{line}_preop_resampling.json").read_text())
    if context == "final":
        inner = [p for p in parts if p["scheme"] == "final_development_inner"]
        train_rows = sorted({row for p in inner for row in p["assessment_excel_rows"]})
        valid_rows = []
    else:
        fold = int(context.split("_")[1])
        outer = next(p for p in parts if p["scheme"] == "outer" and p["fold"] == fold)
        train_rows = outer["train_excel_rows"]
        valid_rows = outer["assessment_excel_rows"]
    meta = metadata()
    train = meta.loc[train_rows]
    valid = meta.loc[valid_rows]
    assert train[line + "_partition_v2"].eq("development").all()
    assert valid[line + "_partition_v2"].eq("development").all()
    assert set(train.group_hash).isdisjoint(valid.group_hash)
    method = next(d["method"] for d in e.descriptors() if d["line"] == line and d["context"] == context)
    return train_rows, valid_rows, method

def raw_frame(rows, spec):
    data = pd.read_csv(
        e.legacy.INPUT / "preop_preop_features.csv",
        dtype={c: "string" for c in spec["cat"]}, float_precision="round_trip"
    ).set_index("excel_row")
    return data.loc[rows, spec["num"] + spec["cat"]]

def checked_train_cache(directory, rows, spec):
    receipt = json.loads((directory / "COMPLETE.json").read_text())
    assert e.qualified(directory)
    frame = pd.read_csv(
        directory / "train.csv", dtype={c: "string" for c in spec["cat"]},
        float_precision="round_trip"
    ).set_index("excel_row")
    assert frame.index.tolist() == rows
    assert list(frame.columns) == spec["num"] + spec["cat"]
    assert not frame.isna().any().any()
    return directory, frame

def training_imputer(line, context, columns):
    rows, valid_rows, method = context_rows(line, context)
    spec = e.legacy.spec_for("preop", columns)
    task = {
        "kind": "impute", "line": line, "cohort": "preop", "stage": "preop",
        "context": context, "fold": "refit", "method": method, "branch": "selected",
        "columns": columns, "train_rows": rows, "assessment_rows": valid_rows
    }
    old_signature = e.legacy.cache_signature(task)
    key = e.digest(old_signature)[:24]
    locations = [
        e.SOURCE / "outputs/formal_development_20261018_v1/cache" / key,
        e.SOURCE / "outputs/ensemble_calibration_20261020_v1/cache" / key,
        e.OUT / "imputation" / key
    ]
    for directory in locations:
        if (directory / "COMPLETE.json").exists():
            assert e.qualified(directory, old_signature)
            return checked_train_cache(directory, rows, spec)
    # The original final-model cache has a train-only signature and no target transform.
    legacy_signature = {
        "input": e.sha(e.legacy.INPUT / "preop_preop_features.csv"),
        "rows": rows, "spec": spec, "method": method, "seed": 42,
        "iterations": 5, "trees": 200, "neighbors": 5,
        "runtime_code": e.sha(e.SOURCE / "work/model_lock_20261022_v1/runtime.py"),
        "imputer_sources": old_signature["imputer_code"]
    }
    directory = e.SOURCE / "outputs/model_lock_20261022_v1/cache" / e.digest(legacy_signature)[:24]
    if (directory / "COMPLETE.json").exists():
        assert e.qualified(directory, legacy_signature)
        return checked_train_cache(directory, rows, spec)
    signature = {
        "source_lock": e.contract()["source_lock_sha256"],
        "input": old_signature["input_sha256"], "rows": rows, "spec": spec,
        "method": method, "seed": 42, "iterations": 5, "trees": 200, "neighbors": 5,
        "imputer_sources": old_signature["imputer_code"]
    }
    directory = POST / "imputation" / e.digest(signature)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if e.qualified(directory, signature):
            return checked_train_cache(directory, rows, spec)
        started = time.time()
        train = raw_frame(rows, spec)
        meta = metadata().loc[rows]
        imputer = e.StableMixedImputer(spec, method, seed=42, iterations=5, trees=200, neighbors=5, threads=1)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            filled = imputer.fit_transform(train, e.legacy.keys(meta))
        assert not filled.isna().any().any()
        for column in columns:
            observed = train[column].notna()
            if column in spec["num"]:
                np.testing.assert_array_equal(train.loc[observed, column], filled.loc[observed, column])
            else:
                assert train.loc[observed, column].astype(str).tolist() == filled.loc[observed, column].astype(str).tolist()
        joblib.dump(imputer, directory / "imputer.joblib", compress=3)
        filled.to_csv(directory / "train.csv", index_label="excel_row")
        e.atomic(directory / "COMPLETE.json", {
            "signature": signature,
            "files": {f: e.sha(directory / f) for f in ["train.csv", "imputer.joblib"]},
            "input_columns": list(filled.columns),
            "training_only": True, "no_outcome_in_imputer": True,
            "seconds": time.time() - started,
            "warnings": sorted(set(str(w.message)[:600] for w in caught)),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        })
    return directory, filled

def selected_rfe_columns(line, context, count):
    full_spec = e.legacy.spec_for("preop")
    columns = full_spec["num"] + full_spec["cat"]
    directory, train = training_imputer(line, context, columns)
    rows, _, _ = context_rows(line, context)
    meta = metadata().loc[rows]
    source = directory / "RFE"
    inherited = {}
    original_receipt = None
    if (source / "COMPLETE.json").exists():
        assert e.qualified(source)
        original_receipt = json.loads((source / "COMPLETE.json").read_text())
        signature = original_receipt["signature"]
        assert signature["imputation_receipt"] == e.sha(directory / "COMPLETE.json")
        assert signature["training_y"] == e.digest(meta.y.tolist())
        assert signature["training_groups"] == e.digest(meta.group_hash.tolist())
        assert signature["rf"] == {"trees": 200, "leaf": 5, "max_features": "sqrt", "seed": 42}
        algorithm = signature.get("source_frozen_algorithm", signature.get("selector_code"))
        assert algorithm in {
            e.sha(e.SOURCE / "work/formal_development_20261018_v1/preprocess.py"),
            e.sha(e.SOURCE / "work/ensemble_calibration_20261020_v1/preprocess.py")
        }
        inherited = json.loads((source / "path.json").read_text())["features_by_count"]
        if str(count) in inherited:
            return inherited[str(count)], {
                "ranking_source": str(source / "path.json"),
                "ranking_sha256": e.sha(source / "path.json"),
                "original_receipt_sha256": e.sha(source / "COMPLETE.json"),
                "training_only": True
            }
    signature = {
        "code": this_code_hash(), "protocol": e.protocol_hash(),
        "imputation_receipt": e.sha(directory / "COMPLETE.json"),
        "training_y": e.digest(meta.y.tolist()), "training_groups": e.digest(meta.group_hash.tolist()),
        "inherited_sha256": e.sha(source / "path.json") if inherited else None
    }
    target = POST / "ranking" / e.digest(signature)[:24]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if e.qualified(target, signature):
            paths = json.loads((target / "path.json").read_text())["features_by_count"]
        else:
            paths = dict(inherited)
            retained = list(inherited[str(min(map(int, inherited)))]) if inherited else list(columns)
            while True:
                paths[str(len(retained))] = list(retained)
                if len(retained) == 1:
                    break
                spec = e.legacy.spec_for("preop", retained)
                encoder, _ = e.models.encoder(spec, train, {})
                encoded = encoder.fit_transform(train)
                keep = VarianceThreshold().fit(encoded)
                owners = np.asarray(spec["num"] + [c for c in spec["cat"] for _ in spec["cat_levels"][c]])[keep.get_support()]
                model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=1)
                with threadpool_limits(limits=1):
                    model.fit(keep.transform(encoded), meta.y)
                importance = {c: float(model.feature_importances_[owners == c].sum()) for c in retained}
                retained.remove(min(retained, key=lambda c: (importance[c], c)))
            e.atomic(target / "path.json", {"features_by_count": paths, "training_only": True})
            e.atomic(target / "COMPLETE.json", {
                "signature": signature, "files": {"path.json": e.sha(target / "path.json")}
            })
    return paths[str(count)], {"ranking_source": str(target / "path.json"), "ranking_sha256": e.sha(target / "path.json"), "training_only": True}

def selected_sparse_columns(line, context, params):
    spec = e.legacy.spec_for("preop")
    directory, train = training_imputer(line, context, spec["num"] + spec["cat"])
    rows, _, _ = context_rows(line, context)
    meta = metadata().loc[rows]
    encoder, _ = e.models.encoder(spec, train, {})
    encoded = encoder.fit_transform(train)
    keep = VarianceThreshold().fit(encoded)
    model = e.sparse_estimator(params)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(keep.transform(encoded), meta.y)
    assert int(model.n_iter_.max()) < 20000, "Sparse full-training screening did not converge at locked budget"
    owners = np.asarray(spec["num"] + [c for c in spec["cat"] for _ in spec["cat_levels"][c]])[keep.get_support()]
    active = set(owners[np.abs(model.coef_[0]) > 1e-8])
    columns = [c for c in spec["num"] + spec["cat"] if c in active]
    return columns, {
        "training_only": True, "raw_count": len(columns),
        "full_imputation_receipt_sha256": e.sha(directory / "COMPLETE.json"),
        "iterations": int(model.n_iter_.max()),
        "warnings": sorted(set(str(w.message)[:600] for w in caught))
    }

def predict_raw(bundle, frame, case_keys):
    columns = bundle["input_columns"]
    if not columns:
        return np.full(len(frame), bundle["intercept_probability"])
    filled = bundle["imputer"].transform(frame.loc[:, columns], case_keys)
    probability = bundle["pipeline"].predict_proba(filled)[:, 1]
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    return probability

def fit_pipeline(line, context, method, candidate):
    task = candidate["task"]
    signature = {
        "line": line, "context": context, "method": method, "family": task["family"], "params": task["params"],
        "count": task.get("count"), "protocol": e.protocol_hash(),
        "code": this_code_hash(), "engine": e.code_hash()
    }
    directory = POST / "refits" / e.digest(signature)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if e.qualified(directory, signature):
            return directory
        started = time.time()
        rows, valid_rows, imputation_method = context_rows(line, context)
        train_meta = metadata().loc[rows]
        if method == "RF_RFE":
            columns, selection = selected_rfe_columns(line, context, task["count"])
        else:
            columns, selection = selected_sparse_columns(line, context, task["params"])
        spec = e.legacy.spec_for("preop", columns)
        notes = []
        if columns:
            cache, train = training_imputer(line, context, columns)
            imputer = joblib.load(cache / "imputer.joblib")
            assert set(imputer.cols) == set(columns)
            if method == "RF_RFE":
                fit = e.xgb_fit if task["family"] == "XGB" else None
                if fit:
                    pipe, _, notes, fallback = fit(train, train_meta, train.iloc[:1], spec, task["params"])
                else:
                    pipe, _, notes, fallback = e.models.one_fit(train, train_meta, train.iloc[:1], spec, task["family"], task["params"], 1)
            else:
                pipe, _, notes, converged = e.sparse_refit(train, train_meta, train.iloc[:1], spec, task["params"])
                assert converged, "Sparse subset full refit did not converge at locked budget"
                fallback = []
            cache_path = str(cache / "COMPLETE.json")
            cache_sha = e.sha(cache / "COMPLETE.json")
            prevalence = None
        else:
            imputer = None
            pipe = None
            fallback = []
            prevalence = float(train_meta.y.mean())
            cache_path = None
            cache_sha = None
        bundle = {
            "input_columns": columns, "imputer": imputer, "pipeline": pipe,
            "intercept_probability": prevalence, "spec": spec,
            "method": method, "family": task["family"], "params": task["params"],
            "training_rows_sha256": e.digest(rows),
            "training_groups_sha256": e.digest(train_meta.group_hash.tolist()),
            "training_labels_sha256": e.digest(train_meta.y.tolist()),
            "protocol_sha256": e.protocol_hash(), "contains_training_information_keep_private": True
        }
        bundle_path = directory / "raw_bundle.joblib"
        joblib.dump(bundle, bundle_path, compress=3)
        check_rows = rows[:5]
        check_spec = e.legacy.spec_for("preop")
        check_frame = raw_frame(check_rows, check_spec)
        check_meta = metadata().loc[check_rows]
        keys = e.legacy.keys(check_meta)
        with threadpool_limits(limits=1):
            batch = predict_raw(bundle, check_frame, keys)
            single = np.asarray([predict_raw(bundle, check_frame.iloc[i:i+1], keys[i:i+1])[0] for i in range(len(keys))])
            loaded = predict_raw(joblib.load(bundle_path), check_frame, keys)
            unused = [c for c in check_frame.columns if c not in columns]
            perturbed = check_frame.copy()
            perturbed[unused] = np.nan
            excluded = predict_raw(bundle, perturbed, keys)
        np.testing.assert_allclose(batch, single, rtol=0, atol=1e-12)
        np.testing.assert_allclose(batch, loaded, rtol=0, atol=1e-12)
        np.testing.assert_allclose(batch, excluded, rtol=0, atol=1e-12)
        files = {"raw_bundle.joblib": e.sha(bundle_path)}
        if valid_rows:
            valid_meta = metadata().loc[valid_rows]
            frame = raw_frame(valid_rows, e.legacy.spec_for("preop"))
            with threadpool_limits(limits=1):
                raw_probability = predict_raw(bundle, frame, e.legacy.keys(valid_meta))
            valid_meta.assign(prob_raw=raw_probability).to_csv(directory / "OUTER_PREDICTIONS.csv", index=False)
            files["OUTER_PREDICTIONS.csv"] = e.sha(directory / "OUTER_PREDICTIONS.csv")
        e.atomic(directory / "COMPLETE.json", {
            "signature": signature, "files": files,
            "input_columns": columns, "raw_input_count": len(columns),
            "selection_evidence": selection,
            "training_n": len(rows), "training_events": int(train_meta.y.sum()),
            "training_group_count": train_meta.group_hash.nunique(),
            "training_rows_sha256": e.digest(rows), "imputation_method": imputation_method,
            "imputation_receipt": cache_path, "imputation_receipt_sha256": cache_sha,
            "warnings": notes, "spline_fallbacks": fallback,
            "batch_single_reload_max_abs_error": float(max(np.max(np.abs(batch-single)), np.max(np.abs(batch-loaded)))),
            "seconds": time.time() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "target_evaluation_used": False
        })
    return directory

def fit_calibrator(y, probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1-1e-6)
    model = LogisticRegression(C=1e6, penalty="l2", solver="lbfgs", max_iter=5000, random_state=42)
    with threadpool_limits(limits=1):
        model.fit(np.log(probability / (1-probability)).reshape(-1, 1), y)
    return model

def apply_calibration(model, probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1-1e-6)
    return model.predict_proba(np.log(probability / (1-probability)).reshape(-1, 1))[:, 1]

def thresholds(y, probability):
    y = np.asarray(y)
    positive = np.sort(probability[y == 1])[::-1]
    index = int(np.ceil(.8 * len(positive))) - 1
    return {
        "training_sensitivity80": float(positive[index]),
        "training_capacity10": float(np.quantile(probability, .9, method="higher")),
        "training_capacity20": float(np.quantile(probability, .8, method="higher")),
        "alert_comparison": ">=", "ties_included": True,
        "source": "Honest algorithm-selection raw OOF transformed by final development-fitted calibrator"
    }

def selection(line, context, method):
    path = e.OUT / f"selection/{line}/{context}/{method}/SELECTION.json"
    value = json.loads(path.read_text())
    return value, path

def one_refit(item):
    line, context, method, rule = item
    chosen, _ = selection(line, context, method)
    directory = fit_pipeline(line, context, method, chosen[rule])
    return {"line": line, "context": context, "method": method, "rule": rule, "directory": str(directory), "receipt_sha256": e.sha(directory/"COMPLETE.json")}

def assemble_system(line, method, rule, refits):
    pieces = []
    for fold in range(5):
        context = "outer_" + str(fold)
        rec = next(r for r in refits if (r["line"], r["context"], r["method"], r["rule"]) == (line, context, method, rule))
        directory = Path(rec["directory"])
        assert e.sha(directory / "COMPLETE.json") == rec["receipt_sha256"]
        outer = pd.read_csv(directory / "OUTER_PREDICTIONS.csv", dtype={"group_hash": str}, float_precision="round_trip")
        chosen, _ = selection(line, context, method)
        source = chosen[rule]
        assert e.sha(source["prediction_file"]) == source["prediction_sha256"]
        inner = np.load(source["prediction_file"], allow_pickle=False)
        assert set(inner["group_hash"]).isdisjoint(outer.group_hash)
        training_rows, _, _ = context_rows(line, context)
        assert set(inner["excel_row"]) == set(training_rows)
        source_meta = metadata().loc[inner["excel_row"]]
        np.testing.assert_array_equal(source_meta.y.to_numpy(), inner["y"])
        np.testing.assert_array_equal(source_meta.group_hash.to_numpy(dtype=str), inner["group_hash"])
        fold_calibrator = fit_calibrator(inner["y"], inner["prob_raw"])
        outer["prob_calibrated_outer_only"] = apply_calibration(fold_calibrator, outer.prob_raw)
        outer["outer_fold"] = fold
        outer["selected_raw_count"] = json.loads((directory / "COMPLETE.json").read_text())["raw_input_count"]
        pieces.append(outer)
    oof = pd.concat(pieces, ignore_index=True)
    assert oof.excel_row.is_unique
    final_rows, _, _ = context_rows(line, "final")
    assert set(oof.excel_row) == set(final_rows)
    final_calibrator = fit_calibrator(oof.y.to_numpy(), oof.prob_raw.to_numpy())
    oof["prob_calibrated_final_map"] = apply_calibration(final_calibrator, oof.prob_raw)
    source_thresholds = thresholds(oof.y.to_numpy(), oof.prob_calibrated_final_map.to_numpy())
    rec = next(r for r in refits if (r["line"],r["context"],r["method"],r["rule"]) == (line,"final",method,rule))
    raw_directory = Path(rec["directory"])
    receipt = json.loads((raw_directory / "COMPLETE.json").read_text())
    chosen, chosen_path = selection(line, "final", method)
    system = POST / "locked_models" / line / method / rule
    system.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_calibrator, system / "calibrator.joblib", compress=3)
    oof.to_csv(system / "PRIVATE_DEVELOPMENT_OOF.csv", index=False)
    lock = {
        "identity": {"line": line, "cohort":"preop","stage":"preop","branch":method.lower()+"_"+rule,"role":"champion","method": method, "rule": rule, "family": chosen[rule]["task"]["family"]},
        "family":chosen[rule]["task"]["family"],
        "protocol_sha256": e.protocol_hash(), "post_code_sha256": this_code_hash(),
        "input_source_sha256": e.contract()["source_lock_sha256"],
        "raw_bundle": {"path": str(raw_directory / "raw_bundle.joblib"), "sha256": e.sha(raw_directory / "raw_bundle.joblib")},
        "calibrator": {"path": str(system / "calibrator.joblib"), "sha256": e.sha(system / "calibrator.joblib")},
        "input_columns": receipt["input_columns"], "actual_input_count": receipt["raw_input_count"],
        "training_n": receipt["training_n"], "training_events": receipt["training_events"],
        "training_rows_sha256": receipt["training_rows_sha256"],
        "training_thresholds": source_thresholds,
        "development_selection": {"path": str(chosen_path), "sha256": e.sha(chosen_path)},
        "honest_development_raw_oof": {"path": str(system / "PRIVATE_DEVELOPMENT_OOF.csv"), "sha256": e.sha(system / "PRIVATE_DEVELOPMENT_OOF.csv")},
        "calibration_slope": float(final_calibrator.coef_[0,0]),
        "calibration_intercept": float(final_calibrator.intercept_[0]),
        "previous_target_evaluation_exposure": True,
        "new_target_metrics_computed_before_lock": False,
        "contains_training_information_keep_private": True
    }
    if method == "RF_RFE":
        lock.update(base_components={lock["family"]:lock["raw_bundle"]},members=[lock["family"]],meta=None)
    e.atomic(system / "MODEL_LOCK.json", lock)
    return {"path": str(system / "MODEL_LOCK.json"), "sha256": e.sha(system / "MODEL_LOCK.json"), "identity": lock["identity"], "input_count": lock["actual_input_count"]}

def guard_development():
    acceptance = json.loads((e.OUT / "DEVELOPMENT_ACCEPTANCE.json").read_text())
    assert acceptance["status"] == "PASS_TRAINING_CONTEXT_SEARCH_AND_SELECTION_ONLY"
    assert acceptance["protocol_sha256"] == e.protocol_hash()
    assert acceptance["engine_sha256"] == e.code_hash()
    assert e.sha(e.OUT / "DEVELOPMENT_SELECTIONS.json") == acceptance["selection_sha256"]
    import subprocess
    state = subprocess.run(
        ["systemctl", "--user", "show", "aki-formal-rfe-one-se-20261116-v2.service", "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID", "-p", "Result"],
        capture_output=True, text=True
    ).stdout
    assert "MainPID=0" in state and "ActiveState=inactive" in state, state
    assert "Result=success" in state or "LoadState=not-found" in state, state
    with (e.ROOT / "RUN.lock").open("a") as completed_lock:
        fcntl.flock(completed_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code_lock = json.loads((e.ROOT / "POST_DEVELOPMENT_CODE_LOCK.json").read_text())
    for path, expected in code_lock["files"].items():
        assert e.sha(path) == expected, path

def report(phase, **values):
    status = {"phase":phase,"protocol_sha256":e.protocol_hash(),"updated":time.time(),**values}
    e.atomic(POST / "STATUS.json", status)
    e.atomic(e.ROOT.parent / "STATUS.json", {
        "task_id":e.contract()["task_id"],"state":phase,
        "formal_rf_rfe":{"state":phase,"status_path":str(POST/"STATUS.json")},
        "sparse_diagnostic":{"state":phase,"exact9_gate":False},
        "main_unit":os.environ.get("AKI_UNIT_NAME"),"complete":False,"updated":time.time()
    })

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    assert args.run, "Use --run after accepted development and resource admission"
    guard_development()
    POST.mkdir(parents=True, exist_ok=True)
    with (POST / "RUN.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        e.atomic(e.ROOT.parent / "OWNER.json", {
            "owner_thread":"01a170c0-3584-74e2-b44c-4d0711ee1860","unit":os.environ.get("AKI_UNIT_NAME"),
            "pid":os.getpid(),"start_ticks":int(Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19]),
            "phase":"POST_DEVELOPMENT_REFITS","protocol_sha256":e.protocol_hash()
        })
        report("POST_DEVELOPMENT_REFITS_STARTING")
        tasks = [(line, context, method, rule) for line in ["geography","temporal"] for context in ["outer_"+str(i) for i in range(5)]+["final"] for method in METHODS for rule in RULES]
        resource_state = e.legacy.resource_snapshot()
        reserve = max(4 * 1024**3, int(.25 * resource_state["available_bytes"]))
        worker_reserve = 1024**3
        safe_workers = min(resource_state["effective_cpu"], max(1, (resource_state["available_bytes"] - reserve - 1024**3) // worker_reserve))
        workers = min(args.workers, safe_workers) if args.workers else safe_workers
        assert workers > 0
        e.atomic(POST / "RESOURCE_ADMISSION.json", {"resources": resource_state, "workers": workers, "worker_reserve_bytes":worker_reserve,"OS_reserve_bytes":reserve})
        unique = {}
        for task in tasks:
            line, context, method, rule = task
            chosen, _ = selection(line, context, method)
            value = chosen[rule]["task"]
            key = e.digest({"line":line,"context":context,"method":method,"family":value["family"],"params":value["params"],"count":value.get("count")})
            unique.setdefault(key, []).append(task)
        complete = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(one_refit, group[0]): group for group in unique.values()}
            for future in as_completed(futures):
                result = future.result()
                for line, context, method, rule in futures[future]:
                    complete.append({**result,"line":line,"context":context,"method":method,"rule":rule})
                e.atomic(POST / "REFITS.json", complete)
                report("TRAINING_REFITS",accepted=len(complete),total=len(tasks),workers=workers)
        locks = [assemble_system(line, method, rule, complete) for line in ["geography","temporal"] for method in METHODS for rule in RULES]
        e.atomic(POST / "MODEL_LOCK_ACCEPTANCE.json", {
            "status":"PASS_NEW_TRAINING_PIPELINES_CALIBRATION_AND_LOCKS",
            "protocol_sha256":e.protocol_hash(),"post_code_sha256":this_code_hash(),
            "refit_routes":len(complete),"locks":locks,"new_target_metrics_computed":False
        })
        report("ALL12_MODELS_LOCKED_HELDOUT_EVALUATION_REQUIRED",locks=len(locks))
        print("ALL12_MODELS_LOCKED_HELDOUT_EVALUATION_REQUIRED",flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        e.atomic(POST / "FAILURE.json", {"traceback":traceback.format_exc(),"time":time.time()})
        raise

