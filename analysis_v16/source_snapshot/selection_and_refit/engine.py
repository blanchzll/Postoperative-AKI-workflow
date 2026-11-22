
"""Source-bound, raw-group nested RF-RFE and sparse diagnostic engine."""
from pathlib import Path
import os
import sys
import json
import hashlib
import time
import fcntl
import gc
import resource
import warnings
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ["AKI_WORKSPACE"]) / "correction_runs/full_data_20261025_postslot"
OUT = ROOT / "outputs"
PROTOCOL = ROOT / "PROTOCOL_LOCK.json"
sys.path.insert(0, str(SOURCE / "work/formal_development_20261018_v1"))
import core as legacy
import model_engine as models
from stable_imputer import StableMixedImputer

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return legacy.digest(value)

def atomic(path, value):
    legacy.atomic(path, value)

def qualified(directory, signature=None):
    return legacy.ready(directory, signature)

@lru_cache(maxsize=1)
def contract():
    return json.loads(PROTOCOL.read_text())

@lru_cache(maxsize=1)
def code_hash():
    return sha(__file__)

@lru_cache(maxsize=1)
def protocol_hash():
    return sha(PROTOCOL)

@lru_cache(maxsize=1)
def descriptors():
    source = ROOT.parent / "sparse_v1/FOLD_INPUTS_PRIVATE.json"
    return json.loads(source.read_text())["descriptors"]

def descriptor(line, context, fold):
    return next(d for d in descriptors() if (d["line"], d["context"], d["fold"]) == (line, context, fold))

def task_for(desc, columns=None):
    task = {
        "kind": "impute", "line": desc["line"], "cohort": "preop",
        "stage": "preop", "context": desc["context"], "fold": desc["fold"],
        "method": desc["method"], "branch": "full",
        "train_rows": desc["train_rows"], "assessment_rows": desc["validation_rows"]
    }
    if columns is not None:
        task.update(branch="selected", columns=list(columns))
    return task

def load_frames(task, directory):
    spec = legacy.cache_signature(task)["spec"]
    dtype = {c: "string" for c in spec["cat"]}
    train = pd.read_csv(directory / "train.csv", dtype=dtype, float_precision="round_trip").set_index("excel_row")
    valid = pd.read_csv(directory / "assessment.csv", dtype=dtype, float_precision="round_trip").set_index("excel_row")
    meta = pd.read_csv(legacy.INPUT / "preop_metadata.csv", dtype={"group_hash": str}).set_index("excel_row", drop=False)
    train_meta = meta.loc[task["train_rows"]]
    valid_meta = meta.loc[task["assessment_rows"]]
    assert train.index.tolist() == train_meta.index.tolist()
    assert valid.index.tolist() == valid_meta.index.tolist()
    assert set(train_meta.group_hash).isdisjoint(valid_meta.group_hash)
    assert train_meta[task["line"] + "_partition_v2"].eq("development").all()
    assert valid_meta[task["line"] + "_partition_v2"].eq("development").all()
    assert list(train.columns) == spec["num"] + spec["cat"]
    return train, train_meta, valid, valid_meta, spec

def imputed(task):
    """Read a qualified legacy cache or create a task-owned subset-only cache."""
    signature = legacy.cache_signature(task)
    source = legacy.cache_dir(task)
    if (source / "COMPLETE.json").exists():
        if not qualified(source, signature):
            raise ValueError("Incomplete source cache " + str(source))
        return source
    target = OUT / "imputation" / digest(signature)[:24]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if qualified(target, signature):
            return target
        started = time.time()
        spec = signature["spec"]
        train_raw, train_meta, valid_raw, valid_meta = legacy.frame_and_meta(task, spec)
        imputer = StableMixedImputer(spec, task["method"], seed=42, iterations=5, trees=200, neighbors=5, threads=1)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            train = imputer.fit_transform(train_raw, legacy.keys(train_meta))
            valid = imputer.transform(valid_raw, legacy.keys(valid_meta)) if len(valid_raw) else valid_raw.copy()
        assert not train.isna().any().any() and not valid.isna().any().any()
        for raw, filled in [(train_raw, train), (valid_raw, valid)]:
            for column in spec["num"] + spec["cat"]:
                mask = raw[column].notna()
                if column in spec["num"]:
                    np.testing.assert_array_equal(raw.loc[mask, column], filled.loc[mask, column])
                else:
                    assert raw.loc[mask, column].astype(str).tolist() == filled.loc[mask, column].astype(str).tolist()
        joblib.dump(imputer, target / "imputer.joblib", compress=3)
        train.to_csv(target / "train.csv", index_label="excel_row")
        valid.to_csv(target / "assessment.csv", index_label="excel_row")
        atomic(target / "COMPLETE.json", {
            "signature": signature,
            "files": {f: sha(target / f) for f in ["train.csv", "assessment.csv", "imputer.joblib"]},
            "train_n": len(train), "assessment_n": len(valid),
            "input_columns": list(train.columns), "owner_root": str(ROOT),
            "warnings": sorted(set(str(w.message)[:600] for w in caught)),
            "seconds": time.time() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "no_outcome_in_imputer": True, "no_final_evaluation_used": True
        })
    return target

def ranking(desc):
    return cached_ranking(desc["line"], desc["context"], desc["fold"])

@lru_cache(maxsize=36)
def cached_ranking(line, context, fold):
    desc = descriptor(line, context, fold)
    full_task = task_for(desc)
    full_dir = imputed(full_task)
    train, meta, _, _, spec = load_frames(full_task, full_dir)
    source = full_dir / "RFE"
    inherited = None
    if (source / "COMPLETE.json").exists():
        original_signature = {
            "imputation_receipt": sha(full_dir / "COMPLETE.json"),
            "training_y": digest(meta.y.tolist()),
            "training_groups": digest(meta.group_hash.tolist()),
            "selector_code": sha(SOURCE / "work/formal_development_20261018_v1/preprocess.py"),
            "rf": {"trees": 200, "leaf": 5, "max_features": "sqrt", "seed": 42}
        }
        assert qualified(source, original_signature)
        inherited = json.loads((source / "path.json").read_text())["features_by_count"]
        assert set(inherited) == set(map(str, range(3, 28)))
    signature = {
        "protocol": protocol_hash(), "code": code_hash(),
        "full_receipt": sha(full_dir / "COMPLETE.json"),
        "y": digest(meta.y.tolist()), "groups": digest(meta.group_hash.tolist()),
        "inherited_receipt": sha(source / "COMPLETE.json") if inherited else None
    }
    target = OUT / "ranking" / digest(signature)[:24]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if qualified(target, signature):
            return target
        paths = dict(inherited) if inherited else {}
        columns = list(paths["3"]) if inherited else list(train.columns)
        new_importances = []
        started = time.time()
        while True:
            paths[str(len(columns))] = list(columns)
            if len(columns) == 1:
                break
            nums = [c for c in spec["num"] if c in columns]
            cats = [c for c in spec["cat"] if c in columns]
            subset_spec = legacy.spec_for("preop", columns)
            encoder, _ = models.encoder(subset_spec, train[subset_spec["num"] + subset_spec["cat"]], {})
            encoded = encoder.fit_transform(train)
            keep = VarianceThreshold().fit(encoded)
            owner = np.asarray(nums + [c for c in cats for _ in spec["cat_levels"][c]])[keep.get_support()]
            forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=1)
            with threadpool_limits(limits=1):
                forest.fit(keep.transform(encoded), meta.y)
            importance = {c: float(forest.feature_importances_[owner == c].sum()) for c in columns}
            removed = min(columns, key=lambda c: (importance[c], c))
            new_importances.append({"raw_count": len(columns), "importance": importance, "removed": removed})
            columns.remove(removed)
        assert set(paths) == set(map(str, range(1, 28)))
        for size, columns in paths.items():
            assert len(columns) == int(size) and len(set(columns)) == int(size)
            assert set(columns) <= set(spec["num"] + spec["cat"])
        atomic(target / "path.json", {
            "features_by_count": paths, "new_importances": new_importances,
            "inherited_counts": sorted(map(int, inherited)) if inherited else [],
            "training_only": True, "descriptor": {k: v for k, v in desc.items() if not k.endswith("_rows")}
        })
        atomic(target / "COMPLETE.json", {
            "signature": signature, "files": {"path.json": sha(target / "path.json")},
            "seconds": time.time() - started
        })
    return target

def subset(desc, count):
    path = ranking(desc) / "path.json"
    columns = json.loads(path.read_text())["features_by_count"][str(count)]
    task = task_for(desc, columns)
    directory = imputed(task)
    return task, directory, sha(path)

def sparse_estimator(params):
    args = {
        "solver": "saga", "C": params["C"], "max_iter": 20000,
        "tol": 1e-5, "random_state": 42, "class_weight": None,
        "warm_start": False, "fit_intercept": True
    }
    if params["l1_ratio"] == 1:
        return LogisticRegression(penalty="l1", **args)
    return LogisticRegression(penalty="elasticnet", l1_ratio=params["l1_ratio"], **args)

def sparse_screen(desc, params):
    task = task_for(desc)
    directory = imputed(task)
    signature = {
        "code": code_hash(), "protocol": protocol_hash(),
        "full_receipt": sha(directory / "COMPLETE.json"), "params": params
    }
    target = OUT / "sparse_screen" / digest(signature)[:24]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if qualified(target, signature):
            return json.loads((target / "screen.json").read_text()), target
        train, meta, _, _, spec = load_frames(task, directory)
        encoder, _ = models.encoder(spec, train, {})
        encoded = encoder.fit_transform(train)
        keep = VarianceThreshold().fit(encoded)
        model = sparse_estimator(params)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            model.fit(keep.transform(encoded), meta.y)
        owners = np.asarray(spec["num"] + [c for c in spec["cat"] for _ in spec["cat_levels"][c]])[keep.get_support()]
        active = set(owners[np.abs(model.coef_[0]) > 1e-8])
        columns = [c for c in spec["num"] + spec["cat"] if c in active]
        result = {
            "input_columns": columns, "raw_count": len(columns),
            "converged": int(model.n_iter_.max()) < 20000,
            "iterations": int(model.n_iter_.max()),
            "warnings": sorted(set(str(w.message)[:600] for w in caught))
        }
        atomic(target / "screen.json", result)
        atomic(target / "COMPLETE.json", {
            "signature": signature, "files": {"screen.json": sha(target / "screen.json")}
        })
    return result, target

def sparse_refit(train, meta, valid, spec, params):
    if len(train.columns) == 0:
        return None, np.full(len(valid), float(meta.y.mean())), [], True
    encoder, _ = models.encoder(spec, train, {})
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("encode", encoder), ("constant", VarianceThreshold()), ("model", sparse_estimator(params))])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        pipe.fit(train, meta.y)
        pred = pipe.predict_proba(valid)[:, 1]
    converged = int(pipe.named_steps["model"].n_iter_.max()) < 20000
    return pipe, pred, sorted(set(str(w.message)[:600] for w in caught)), converged

def xgb_fit(train, meta, valid, spec, params):
    backend = json.loads((OUT / "engineering/BACKEND_ACCEPTANCE.json").read_text())["selected_backend"]
    if backend == "cpu":
        return models.one_fit(train, meta, valid, spec, "XGB", params, 1)
    from sklearn.pipeline import Pipeline
    gpu_lock = OUT / "engineering/GPU.lock"
    with gpu_lock.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        encoder, fallback = models.encoder(spec, train, params)
        model = models.estimator("XGB", params, 1)
        model.set_params(device="cuda")
        pipe = Pipeline([("encode", encoder), ("constant", VarianceThreshold()), ("model", model)])
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            pipe.fit(train, meta.y)
            pred = pipe.predict_proba(valid)[:, 1]
        device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        assert device.startswith("cuda"), device
        return pipe, pred, sorted(set(str(w.message)[:600] for w in caught)), fallback

def score_candidate(task):
    """Each candidate evaluates three original development folds only."""
    started = time.time()
    signature = {
        "code": code_hash(), "protocol": protocol_hash(), "task": task
    }
    target = OUT / "candidates" / task["line"] / task["context"] / task["kind"] / digest(task)[:24]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if qualified(target, signature):
            return json.loads((target / "RESULT.json").read_text())
        metrics = []
        frames = []
        predictions = []
        inputs = []
        all_warnings = []
        converged = True
        for fold in range(3):
            desc = descriptor(task["line"], task["context"], fold)
            if task["kind"] == "rfe":
                input_task, directory, selection_sha = subset(desc, task["count"])
                train, meta, valid, valid_meta, spec = load_frames(input_task, directory)
                model, pred, notes, fallback = (xgb_fit(train, meta, valid, spec, task["params"]) if task["family"] == "XGB" else models.one_fit(train, meta, valid, spec, task["family"], task["params"], 1))
                screen_converged = True
            else:
                screening, screening_directory = sparse_screen(desc, task["params"])
                columns = screening["input_columns"]
                screen_converged = screening["converged"]
                selection_sha = sha(screening_directory / "screen.json")
                if columns:
                    input_task = task_for(desc, columns)
                    directory = imputed(input_task)
                    train, meta, valid, valid_meta, spec = load_frames(input_task, directory)
                else:
                    full_task = task_for(desc)
                    train, meta, valid, valid_meta, _ = load_frames(full_task, imputed(full_task))
                    train = train.iloc[:, :0]
                    valid = valid.iloc[:, :0]
                    spec = {"num": [], "cat": [], "cat_levels": {}}
                    directory = None
                model, pred, notes, fit_converged = sparse_refit(train, meta, valid, spec, task["params"])
                screen_converged = screen_converged and fit_converged
                notes += screening["warnings"]
                fallback = []
            assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
            converged = converged and screen_converged
            metrics.append({
                "fold": fold, "AP": float(average_precision_score(valid_meta.y, pred)),
                "AUROC": float(roc_auc_score(valid_meta.y, pred)), "Brier": float(brier_score_loss(valid_meta.y, pred)),
                "raw_count": len(train.columns), "input_columns": list(train.columns),
                "imputation_receipt": str(directory / "COMPLETE.json") if directory else None,
                "imputation_receipt_sha256": sha(directory / "COMPLETE.json") if directory else None,
                "selection_sha256": selection_sha, "spline_fallbacks": fallback,
                "converged": bool(screen_converged)
            })
            frames.append(valid_meta.assign(inner_fold=fold))
            predictions.append(pred)
            all_warnings.extend(notes)
            del model
            gc.collect()
        combined = pd.concat(frames)
        assert combined.excel_row.is_unique
        prediction_file = target / "PREDICTIONS.npz"
        np.savez_compressed(
            prediction_file, excel_row=combined.excel_row.to_numpy(),
            group_hash=combined.group_hash.to_numpy(dtype=str), y=combined.y.to_numpy(),
            fold=combined.inner_fold.to_numpy(), prob_raw=np.concatenate(predictions)
        )
        result = {
            "task": task, "candidate_id": digest(task), "signature": signature,
            "fold_metrics": metrics, "mean_AP": float(np.mean([x["AP"] for x in metrics])),
            "mean_Brier": float(np.mean([x["Brier"] for x in metrics])),
            "mean_raw_count": float(np.mean([x["raw_count"] for x in metrics])),
            "max_raw_count": max(x["raw_count"] for x in metrics),
            "eligible": bool(converged), "warnings": sorted(set(all_warnings)),
            "prediction_file": str(prediction_file), "prediction_sha256": sha(prediction_file),
            "seconds": time.time() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        }
        atomic(target / "RESULT.json", result)
        atomic(target / "COMPLETE.json", {
            "signature": signature, "files": {f: sha(target / f) for f in ["RESULT.json", "PREDICTIONS.npz"]}
        })
    return result

def preprocess_task(item):
    desc, count = item
    started = time.time()
    task, directory, ranking_sha = subset(desc, count)
    return {
        "line": desc["line"], "context": desc["context"], "fold": desc["fold"], "count": count,
        "cache": str(directory), "receipt_sha256": sha(directory / "COMPLETE.json"),
        "ranking_sha256": ranking_sha, "seconds": time.time() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    }

def candidate_tasks(kind):
    pools = json.loads((ROOT / "inputs/HYPERPARAMETER_POOLS.json").read_text())["pools"]
    if kind == "rfe":
        return [
            {"kind": "rfe", "line": p["line"], "context": p["context"], "family": p["family"],
             "count": count, "pool_index": index, "params": entry["params"]}
            for p in pools for count in range(1, 28) for index, entry in enumerate(p["entries"])
        ]
    return [
        {"kind": "sparse", "line": line, "context": context,
         "family": "LASSO" if ratio == 1 else "EN", "params": {"C": value, "l1_ratio": ratio}}
        for line in ["geography", "temporal"] for context in ["final"] + ["outer_" + str(i) for i in range(5)]
        for ratio in [1., .25, .5, .75] for value in [10 ** (-4 + .5 * i) for i in range(13)]
    ]

def choose(results, sparse=False):
    candidates = [r for r in results if r["eligible"]]
    assert candidates, "No qualified converged candidates"
    best_ap = max(r["mean_AP"] for r in candidates)
    best_pool = [r for r in candidates if r["mean_AP"] >= best_ap - 1e-12]
    best_brier = min(r["mean_Brier"] for r in best_pool)
    best = min([r for r in best_pool if r["mean_Brier"] <= best_brier + 1e-12], key=lambda r: r["candidate_id"])
    se = float(np.std([x["AP"] for x in best["fold_metrics"]], ddof=1) / np.sqrt(3))
    eligible = [r for r in candidates if r["mean_AP"] >= best["mean_AP"] - se - 1e-12]
    if sparse:
        parsimonious = min(eligible, key=lambda r: (
            r["mean_raw_count"], r["max_raw_count"], r["task"]["params"]["C"],
            -r["task"]["params"]["l1_ratio"], -r["mean_AP"], r["mean_Brier"], r["candidate_id"]
        ))
    else:
        min_count = min(r["task"]["count"] for r in eligible)
        fewest = [r for r in eligible if r["task"]["count"] == min_count]
        compact_ap = max(r["mean_AP"] for r in fewest)
        fewest = [r for r in fewest if r["mean_AP"] >= compact_ap - 1e-12]
        compact_brier = min(r["mean_Brier"] for r in fewest)
        parsimonious = min([r for r in fewest if r["mean_Brier"] <= compact_brier + 1e-12], key=lambda r: r["candidate_id"])
    return {
        "maxAP": best, "oneSE": parsimonious, "standard_error_best_fold_AP": se,
        "cutoff_mean_AP": best["mean_AP"] - se,
        "eligible_count": len(eligible), "total_candidates": len(results)
    }

