"""Recover only three failed KNN prediction checks from their saved fitted states."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
import fcntl
import time
import traceback
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import fit_scale as runtime
from fit_scale import ROOT, e, f, POST, read, write, update, PROTOCOL_SHA
from knn_pipeline_wrapper_v2 import ExactEuclideanKNNPipeline
import knn_exact_query


def recover(record):
    line, context, k = record["line"], record["context"], record["K"]
    task = record["candidate"]["task"]
    original_signature = {"line": line, "context": context, "method": "RF_RFE", "family": task["family"], "params": task["params"], "count": k, "protocol": PROTOCOL_SHA, "code": f.this_code_hash(), "engine": e.code_hash()}
    old_directory = POST / "refits" / e.digest(original_signature)[:24]
    original_path = old_directory / "raw_bundle.joblib"
    original_sha = e.sha(original_path)
    original_bundle = joblib.load(original_path)
    assert original_bundle["family"] == "KNN" and original_bundle["params"] == task["params"]
    rows, valid_rows, imputation_method = f.context_rows(line, context)
    meta = f.metadata().loc[rows]
    assert original_bundle["training_rows_sha256"] == e.digest(rows)
    assert original_bundle["training_labels_sha256"] == e.digest(meta.y.tolist())
    assert original_bundle["training_groups_sha256"] == e.digest(meta.group_hash.tolist())
    fitted_state_hash = joblib.hash(original_bundle["pipeline"])
    bundle = dict(original_bundle)
    bundle["pipeline"] = ExactEuclideanKNNPipeline(original_bundle["pipeline"])
    assert joblib.hash(bundle["pipeline"].fitted_pipeline) == fitted_state_hash
    bundle["prediction_numerical_adapter"] = {"file": str(Path(knn_exact_query.__file__)), "sha256": e.sha(knn_exact_query.__file__), "reason": "Near-self Euclidean squared-norm cancellation produced batch-dependent inverse-distance probabilities. Fitted state and neighbor-count/weighting metric retained; singleton neighbor identities and direct-subtraction distances give a deterministic query."}
    directory = POST / "query_adapted" / line / context / f"K{k:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / "raw_bundle.joblib", compress=3)
    raw = f.raw_frame(rows[:5], e.legacy.spec_for("preop"))
    keys = e.legacy.keys(meta.iloc[:5])
    with threadpool_limits(limits=1):
        batch = f.predict_raw(bundle, raw, keys)
        single = np.array([f.predict_raw(bundle, raw.iloc[i:i+1], keys[i:i+1])[0] for i in range(5)])
        reloaded = f.predict_raw(joblib.load(directory / "raw_bundle.joblib"), raw, keys)
        perturbed = raw.copy()
        perturbed[[c for c in raw.columns if c not in bundle["input_columns"]]] = np.nan
        excluded = f.predict_raw(bundle, perturbed, keys)
        for other in [single, reloaded, excluded]:
            np.testing.assert_allclose(batch, other, atol=1e-12, rtol=0)
        valid_meta = f.metadata().loc[valid_rows]
        raw_valid = f.raw_frame(valid_rows, e.legacy.spec_for("preop"))
        probability = f.predict_raw(bundle, raw_valid, e.legacy.keys(valid_meta))
        # All held-outer predictions also satisfy singleton/reload consistency.
        singleton = np.array([f.predict_raw(bundle, raw_valid.iloc[i:i+1], e.legacy.keys(valid_meta.iloc[i:i+1]))[0] for i in range(len(valid_meta))])
        np.testing.assert_allclose(probability, singleton, atol=1e-12, rtol=0)
        original_probability = f.predict_raw(original_bundle, raw_valid, e.legacy.keys(valid_meta))
    valid_meta.assign(prob_raw=probability).to_csv(directory / "OUTER_PREDICTIONS.csv", index=False)
    spec = e.legacy.spec_for("preop", bundle["input_columns"])
    cache, _ = f.training_imputer(line, context, bundle["input_columns"])
    signature = {**original_signature, "query_adapter_sha256": e.sha(knn_exact_query.__file__), "wrapper_sha256": e.sha(Path(__file__).parent / "knn_pipeline_wrapper_v2.py")}
    receipt = {"signature": signature, "files": {n: e.sha(directory / n) for n in ["raw_bundle.joblib", "OUTER_PREDICTIONS.csv"]}, "input_columns": bundle["input_columns"], "raw_input_count": k, "selection_evidence": runtime.selected_columns(line, context, k)[1], "training_n": len(rows), "training_events": int(meta.y.sum()), "training_group_count": meta.group_hash.nunique(), "training_rows_sha256": e.digest(rows), "imputation_method": imputation_method, "imputation_receipt": str(cache / "COMPLETE.json"), "imputation_receipt_sha256": e.sha(cache / "COMPLETE.json"), "warnings": ["Original batch-dependent KNN query rejected; exact Euclidean query adapter qualified under version2."], "spline_fallbacks": [], "batch_single_reload_max_abs_error": float(max(np.max(abs(batch-single)), np.max(abs(batch-reloaded)))), "outer_batch_single_max_abs_error": float(np.max(abs(probability-singleton))), "original_outer_vs_adapted_max_abs_difference": float(np.max(abs(original_probability-probability))), "original_fitted_bundle": str(original_path), "original_fitted_bundle_sha256": original_sha, "fitted_pipeline_state_hash": fitted_state_hash, "new_predictive_refits": 0, "target_evaluation_used": False}
    assert e.sha(original_path) == original_sha
    write(directory / "COMPLETE.json", receipt)
    node = {"line": line, "context": context, "K": k, "method": "RF_RFE", "rule": f"K{k:02d}", "directory": str(directory), "receipt_sha256": e.sha(directory / "COMPLETE.json"), "raw_bundle_sha256": e.sha(directory / "raw_bundle.joblib"), "source_candidate_id": record["candidate"]["candidate_id"], "reused_existing_raw_fit": False, "reused_current_partial_fitted_state": True, "raw_input_count": k, "protocol_sha256": PROTOCOL_SHA, "query_adapter_version": 2}
    write(ROOT / "outputs/nodes" / line / context / f"K{k:02d}.json", node)
    return receipt


def main():
    start = time.time()
    with (ROOT / "work/.active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / "OWNER.json", {"unit": "aki-input-scale-fits-recovery-20261117-v2.service", "PID": os.getpid(), "start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]), "protocol_sha256": PROTOCOL_SHA, "recovery_code_sha256": e.sha(__file__)})
        qualification = read(ROOT / "outputs/knn_query_repair_v2/INNER_QUERY_QUALIFICATION.json")
        assert qualification["all_original_inner_scores_unchanged_within1e12"]
        assert qualification["code_sha256"] == e.sha(knn_exact_query.__file__)
        records = read(ROOT / "inputs/SELECTED_BY_SIZE.json")
        current = list((ROOT / "outputs/nodes").glob("*/*/K*.json"))
        assert len(current) == 321
        update("RECOVERING_THREE_KNN_QUERY_CHECKS", valid_completed_nodes=321)
        recovered = []
        for record in records:
            path = ROOT / "outputs/nodes" / record["line"] / record["context"] / f"K{record['K']:02d}.json"
            if not path.exists():
                assert (record["line"], record["context"]) == ("temporal", "outer_3") and record["K"] in [20, 21, 23]
                recovered.append(recover(record))
        nodes = [read(p) for p in (ROOT / "outputs/nodes").glob("*/*/K*.json")]
        assert len(nodes) == 324
        nodes.sort(key=lambda n: (n["line"], n["context"], n["K"]))
        write(POST / "REFITS.json", nodes)
        selected = {(x["line"], x["context"], x["K"]): x for x in records}
        update("ASSEMBLING_FIXED_K_SOURCE_CALIBRATION_AND_LOCKS", logical_nodes_complete=324)
        systems = [runtime.assemble(line, k, nodes, selected) for line in ["geography", "temporal"] for k in range(1, 28)]
        write(POST / "MODEL_MANIFEST.json", systems)
        write(ROOT / "outputs/knn_query_repair_v2/RECOVERY_ACCEPTANCE.json", {"status": "PASS_THREE_KNN_QUERY_ADAPTERS_WITHOUT_REFITTING_SAVED_MODELS", "recovered_nodes": len(recovered), "valid_initial_nodes_retained": 321, "source_inner_qualification": qualification, "repairs": recovered, "code_sha256": e.sha(__file__), "frozen_source_files_edited": False, "target_results_used": False, "seconds": time.time() - start})
        write(ROOT / "outputs/FIT_AND_LOCK_ACCEPTANCE.json", {"status": "PASS_COMPLETE_FIXED_K_SOURCE_FITS_OOF_CALIBRATION_AND_MODEL_LOCKS", "protocol_sha256": PROTOCOL_SHA, "fit_code_sha256": e.sha(runtime.__file__), "recovery_code_sha256": e.sha(__file__), "logical_context_nodes": 324, "new_raw_refits": 302, "reused_raw_refits": 22, "reused_new_partial_fitted_states": 3, "engineering_validation_fits": 9, "final_systems": 54, "model_manifest_sha256": e.sha(POST / "MODEL_MANIFEST.json"), "refit_index_sha256": e.sha(POST / "REFITS.json"), "new_target_metrics_computed": 0})
        update("ALL54_MODELS_LOCKED_TARGET_PREDICTION_READY", logical_nodes_complete=324, locked_systems=54)
        print("PASS recovered3, retained321, locked54", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write(ROOT / "outputs/knn_query_repair_v2/RECOVERY_FAILURE.json", {"traceback": traceback.format_exc(), "code_sha256": e.sha(__file__)})
        raise
