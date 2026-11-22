"""Fit and freeze the complete fixed-K path using the unchanged source algorithms."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import time
import fcntl
import traceback
import resource
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
FROZEN = PROJECT / "correction_runs/rfe_mainline_revision_20261116_v1/formal_v2"
sys.path.insert(0, str(FROZEN / "work"))
import engine as e
import finalize_models as f
POST = ROOT / "outputs/post_development"
PROTOCOL_SHA = e.sha(ROOT / "PROTOCOL_LOCK.json")
f.POST = POST
# This process-local adapter leaves every frozen source file and source contract intact.
e.protocol_hash = lambda: PROTOCOL_SHA
_original_selected_rfe_columns = f.selected_rfe_columns
_BINDINGS = None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e.atomic(path, value)


def update(phase, **values):
    write(ROOT / "STATUS.json", {"phase": phase, "complete": False, "updated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA, **values})


def prepare_rankings():
    bindings = read(ROOT / "inputs/RANKING_BINDINGS.json")
    extensions = []
    for binding in bindings:
        path = Path(binding["path"])
        assert e.sha(path) == binding["sha256"]
        if "1" not in binding["features_by_count"] or "2" not in binding["features_by_count"]:
            _, evidence = _original_selected_rfe_columns(binding["line"], binding["context"], 1)
            path = Path(evidence["ranking_source"])
            assert path.is_relative_to(ROOT)
            paths = read(path)["features_by_count"]
            for k, columns in binding["features_by_count"].items():
                assert paths[k] == columns
            extensions.append({"line": binding["line"], "context": binding["context"], "path": str(path), "sha256": e.sha(path), "inherited_minimum_K": min(map(int, binding["features_by_count"])), "RF_selector_fits": 2})
            binding.update(path=str(path), sha256=e.sha(path), features_by_count=paths)
        assert set(binding["features_by_count"]) == set(map(str, range(1, 28)))
    write(ROOT / "inputs/FULL_RANKING_BINDINGS.json", bindings)
    write(ROOT / "outputs/RANKING_EXTENSION_ACCEPTANCE.json", {"status": "PASS_SOURCE_ONLY_RFE_PATH_EXTENSION", "protocol_sha256": PROTOCOL_SHA, "extended_contexts": extensions, "new_RF_selector_fits": sum(x["RF_selector_fits"] for x in extensions), "all_inherited_features_retained": True, "target_rows_or_outcomes_used": False})


def selected_columns(line, context, count):
    global _BINDINGS
    if _BINDINGS is None:
        _BINDINGS = {(b["line"], b["context"]): b for b in read(ROOT / "inputs/FULL_RANKING_BINDINGS.json")}
    binding = _BINDINGS[line, context]
    return binding["features_by_count"][str(count)], {"ranking_source": binding["path"], "ranking_sha256": binding["sha256"], "training_only": True, "training_rows_sha256": binding["training_rows_sha256"], "training_groups_sha256": binding["training_groups_sha256"], "training_y_sha256": binding["training_y_sha256"]}


f.selected_rfe_columns = selected_columns


def fit_node(record):
    started = time.time()
    line, context, k = record["line"], record["context"], record["K"]
    reused = record["reuse_raw_refit"]
    if reused:
        directory = Path(reused["directory"])
        assert e.sha(directory / "COMPLETE.json") == reused["receipt_sha256"]
        assert e.qualified(directory)
        receipt = read(directory / "COMPLETE.json")
        expected, evidence = selected_columns(line, context, k)
        assert receipt["input_columns"] == expected
        bundle = joblib.load(directory / "raw_bundle.joblib")
        assert bundle["training_rows_sha256"] == evidence["training_rows_sha256"]
        assert bundle["training_groups_sha256"] == evidence["training_groups_sha256"]
        assert bundle["training_labels_sha256"] == evidence["training_y_sha256"]
        assert bundle["family"] == record["candidate"]["task"]["family"]
        assert bundle["params"] == record["candidate"]["task"]["params"]
    else:
        directory = f.fit_pipeline(line, context, "RF_RFE", record["candidate"])
    receipt = read(directory / "COMPLETE.json")
    node = {"line": line, "context": context, "K": k, "method": "RF_RFE", "rule": f"K{k:02d}", "directory": str(directory), "receipt_sha256": e.sha(directory / "COMPLETE.json"), "raw_bundle_sha256": e.sha(directory / "raw_bundle.joblib"), "source_candidate_id": record["candidate"]["candidate_id"], "reused_existing_raw_fit": bool(reused), "raw_input_count": receipt["raw_input_count"], "seconds_this_call": time.time() - started, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "protocol_sha256": PROTOCOL_SHA}
    assert node["raw_input_count"] == k
    destination = ROOT / "outputs/nodes" / line / context / f"K{k:02d}.json"
    if destination.exists():
        old = read(destination)
        assert old["receipt_sha256"] == node["receipt_sha256"]
        return old
    write(destination, node)
    return node


def assemble(line, k, nodes, selected):
    key = f"K{k:02d}"
    selections = {c: selected[line, c, k]["candidate"] for c in ["final"] + [f"outer_{i}" for i in range(5)]}
    f.selection = lambda query_line, context, method: ({key: selections[context]}, ROOT / "inputs/SELECTED_BY_SIZE.json")
    existing = POST / "locked_models" / line / "RF_RFE" / key / "MODEL_LOCK.json"
    if existing.exists():
        lock = read(existing)
        assert lock["protocol_sha256"] == PROTOCOL_SHA
        assert e.sha(lock["raw_bundle"]["path"]) == lock["raw_bundle"]["sha256"]
        assert e.sha(lock["calibrator"]["path"]) == lock["calibrator"]["sha256"]
        return {"path": str(existing), "sha256": e.sha(existing), "identity": lock["identity"], "input_count": k}
    record = f.assemble_system(line, "RF_RFE", key, nodes)
    lock = read(record["path"])
    lock["identity"]["role"] = "exploratory_fixed_K"
    lock["identity"]["K"] = k
    lock["selection_scope"] = "Fixed K, independently source-selected family and parameters within each outer-training context. Final calibrator uses fixed-K raw source OOF, distinct from adaptive oneSE/maxAP OOF."
    lock["adapter_code_sha256"] = e.sha(__file__)
    lock["frozen_source_protocol_sha256"] = e.sha(FROZEN / "PROTOCOL_LOCK.json")
    write(record["path"], lock)
    record["sha256"] = e.sha(record["path"])
    record["identity"] = lock["identity"]
    return record


def available_memory():
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
    return mem["MemAvailable"]


def main():
    started = time.time()
    lock_file = (ROOT / "work/.active.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(ROOT / "OWNER.json", {"unit": "aki-input-scale-fits-20261117-v1.service", "PID": os.getpid(), "start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]), "protocol_sha256": PROTOCOL_SHA, "code_sha256": e.sha(__file__)})
    lock = read(ROOT / "inputs/FIT_CODE_LOCK.json")
    assert lock["sha256"] == e.sha(__file__)
    records = read(ROOT / "inputs/SELECTED_BY_SIZE.json")
    update("PREPARING_SOURCE_ONLY_RANKING_EXTENSIONS")
    prepare_rankings()
    selected = {(x["line"], x["context"], x["K"]): x for x in records}
    pilot = next(x for x in records if (x["line"], x["context"], x["K"]) == ("geography", "final", 9))
    update("NECESSARY_FIT_PILOT")
    first = fit_node(pilot)
    write(ROOT / "outputs/FIT_PILOT.json", first)
    memory = available_memory()
    reserve = max(4 * 1024**3, memory // 4)
    per_worker = max(512 * 1024**2, 2 * first["peak_rss_bytes"])
    workers = min(len(os.sched_getaffinity(0)), 48, (memory - reserve - 2 * 1024**3) // per_worker, (30 * 1024**3) // per_worker)
    assert workers >= 1
    admission = {"MemAvailable_bytes": memory, "OS_reserve_bytes": reserve, "per_worker_budget_bytes": per_worker, "pilot_peak_rss_bytes": first["peak_rss_bytes"], "workers": workers, "task_memory_ceiling_bytes": 32 * 1024**3, "swap_bytes": 0, "BLAS_threads_per_worker": 1, "other_jobs_signalled": False}
    write(ROOT / "outputs/RESOURCE_ADMISSION.json", admission)
    nodes = [first]
    remaining = [r for r in records if r is not pilot]
    update("FITTING_COMPLETE_K_PATH", logical_nodes_complete=1, logical_nodes_total=324, **admission)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(fit_node, record): record for record in remaining}
        for future in as_completed(futures):
            node = future.result()
            nodes.append(node)
            if len(nodes) % 10 == 0 or len(nodes) == 324:
                update("FITTING_COMPLETE_K_PATH", logical_nodes_complete=len(nodes), logical_nodes_total=324, latest={key: node[key] for key in ("line", "context", "K", "reused_existing_raw_fit")}, admitted_workers=workers)
                print(json.dumps({"nodes_complete": len(nodes), "last": [node["line"], node["context"], node["K"]]}), flush=True)
    nodes.sort(key=lambda r: (r["line"], r["context"], r["K"]))
    write(POST / "REFITS.json", nodes)
    update("ASSEMBLING_FIXED_K_SOURCE_CALIBRATION_AND_LOCKS", logical_nodes_complete=len(nodes))
    systems = []
    for line in ("geography", "temporal"):
        for k in range(1, 28):
            systems.append(assemble(line, k, nodes, selected))
    write(POST / "MODEL_MANIFEST.json", systems)
    receipt = {"status": "PASS_COMPLETE_FIXED_K_SOURCE_FITS_OOF_CALIBRATION_AND_MODEL_LOCKS", "protocol_sha256": PROTOCOL_SHA, "fit_code_sha256": e.sha(__file__), "logical_context_nodes": len(nodes), "new_raw_refits": sum(not n["reused_existing_raw_fit"] for n in nodes), "reused_raw_refits": sum(n["reused_existing_raw_fit"] for n in nodes), "final_systems": len(systems), "model_manifest_sha256": e.sha(POST / "MODEL_MANIFEST.json"), "refit_index_sha256": e.sha(POST / "REFITS.json"), "new_target_metrics_computed": 0, "seconds": time.time() - started}
    write(ROOT / "outputs/FIT_AND_LOCK_ACCEPTANCE.json", receipt)
    update("ALL54_MODELS_LOCKED_TARGET_PREDICTION_READY", logical_nodes_complete=324, locked_systems=54)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        write(ROOT / "outputs/FIT_FAILURE.json", {"type": sys.exc_info()[0].__name__, "traceback": traceback.format_exc(), "code_sha256": e.sha(__file__), "protocol_sha256": PROTOCOL_SHA})
        raise
