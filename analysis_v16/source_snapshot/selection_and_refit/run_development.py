
"""Finite, resumable development-only execution for the locked formal revision."""
from pathlib import Path
import os
import sys
import json
import time
import math
import fcntl
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as e

def process_start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19])

def snapshot():
    resource = e.legacy.resource_snapshot()
    resource["host_memavailable_bytes"] = int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )) * 1024
    return resource

def update_status(phase, **details):
    state = {
        "task_id": e.contract()["task_id"], "phase": phase,
        "owner_pid": os.getpid(), "owner_start_ticks": process_start_ticks(os.getpid()),
        "protocol_sha256": e.protocol_hash(), "engine_sha256": e.code_hash(),
        "updated_unix": time.time(), **details
    }
    e.atomic(e.OUT / "STATUS.json", state)
    e.atomic(e.ROOT.parent / "STATUS.json", {
        "task_id": e.contract()["task_id"], "state": phase,
        "formal_rf_rfe": {"state": phase, "status_path": str(e.OUT / "STATUS.json")},
        "sparse_diagnostic": {"state": details.get("sparse_state", "PENDING"), "exact9_gate": False},
        "main_unit": os.environ.get("AKI_UNIT_NAME"), "complete": False,
        "updated_unix": time.time()
    })

def execute_phase(name, function, tasks, worker_count, per_worker_bytes):
    target = e.OUT / "execution" / name
    target.mkdir(parents=True, exist_ok=True)
    accepted = []
    pending = {}
    iterator = iter(tasks)
    exhausted = False
    last_save = 0
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as pool:
        while pending or not exhausted:
            free = snapshot()["available_bytes"]
            reserve = max(4 * 1024**3, int(.25 * free))
            pressure = free - reserve < per_worker_bytes * max(1, worker_count - len(pending))
            while not exhausted and len(pending) < worker_count and not pressure:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(function, task)
                pending[future] = task
            if not pending:
                if exhausted:
                    break
                update_status(name, accepted=len(accepted), total=len(tasks), waiting="AVAILABLE_MEMORY", resources=snapshot())
                time.sleep(15)
                continue
            finished, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    e.atomic(target / "FAILURE.json", {"task": task, "traceback": traceback.format_exc(), "time": time.time()})
                    raise
                if name == "PREPROCESS":
                    accepted.append(result)
                else:
                    accepted.append({
                        "candidate_id": result["candidate_id"], "task": result["task"],
                        "prediction_file": result["prediction_file"],
                        "prediction_sha256": result["prediction_sha256"],
                        "eligible": result["eligible"], "seconds": result["seconds"],
                        "peak_rss_bytes": result["peak_rss_bytes"]
                    })
            if time.time() - last_save >= 20 or len(accepted) == len(tasks):
                e.atomic(target / "ACCEPTED.json", accepted)
                update_status(
                    name, accepted=len(accepted), total=len(tasks), workers=worker_count,
                    resources=snapshot(), sparse_state="RUNNING" if name == "SPARSE_DEVELOPMENT" else "PENDING"
                )
                print(json.dumps({"phase": name, "accepted": len(accepted), "total": len(tasks), "time": time.time()}), flush=True)
                last_save = time.time()
    e.atomic(target / "ACCEPTED.json", accepted)
    e.atomic(target / "COMPLETE.json", {
        "protocol_sha256": e.protocol_hash(), "engine_sha256": e.code_hash(),
        "task_count": len(tasks), "accepted_count": len(accepted),
        "accepted_sha256": e.sha(target / "ACCEPTED.json")
    })
    assert len(accepted) == len(tasks)
    return accepted

def consolidate():
    selections = []
    for kind in ["rfe", "sparse"]:
        tasks = e.candidate_tasks(kind)
        grouped = {}
        for task in tasks:
            result = e.score_candidate(task)
            key = (task["line"], task["context"], task["family"] if kind == "sparse" else "RF_RFE")
            grouped.setdefault(key, []).append(result)
        for (line, context, method), results in grouped.items():
            selection = e.choose(results, sparse=(kind == "sparse"))
            record = {"line": line, "context": context, "method": method, **selection}
            destination = e.OUT / "selection" / line / context / method / "SELECTION.json"
            e.atomic(destination, record)
            selections.append({
                "line": line, "context": context, "method": method,
                "file": str(destination), "sha256": e.sha(destination),
                "oneSE_count_mean_inner": selection["oneSE"]["mean_raw_count"],
                "oneSE_candidate_id": selection["oneSE"]["candidate_id"],
                "maxAP_count_mean_inner": selection["maxAP"]["mean_raw_count"],
                "maxAP_candidate_id": selection["maxAP"]["candidate_id"],
                "oneSE_AP": selection["oneSE"]["mean_AP"], "maxAP_AP": selection["maxAP"]["mean_AP"],
                "cutoff": selection["cutoff_mean_AP"], "SE_best": selection["standard_error_best_fold_AP"]
            })
    e.atomic(e.OUT / "DEVELOPMENT_SELECTIONS.json", selections)
    return selections

def main():
    e.OUT.mkdir(parents=True, exist_ok=True)
    with (e.ROOT / "RUN.lock").open("w") as owner_lock:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engineering = json.loads((e.OUT / "engineering/ENGINEERING_ACCEPTANCE.json").read_text())
        assert engineering["engine_sha256"] == e.code_hash()
        assert engineering["protocol_sha256"] == e.protocol_hash()
        lock = json.loads((e.ROOT / "CODE_LOCK.json").read_text())
        for path, expected in lock["files"].items():
            assert e.sha(path) == expected, path
        before = snapshot()
        per_worker = max(768 * 1024**2, math.ceil(engineering["max_pilot_peak_rss_bytes"] * 1.5))
        reserve = max(4 * 1024**3, math.ceil(before["available_bytes"] * .25))
        budget = before["available_bytes"] - reserve - 1024**3
        workers = min(before["effective_cpu"], max(1, budget // per_worker))
        assert workers >= 1 and budget >= per_worker
        e.atomic(e.OUT / "RESOURCE_ADMISSION.json", {
            "before": before, "workers": workers,
            "per_worker_reserved_bytes": per_worker, "OS_reserve_bytes": reserve,
            "controller_reserved_bytes": 1024**3,
            "pilot_peak_rss_bytes": engineering["max_pilot_peak_rss_bytes"],
            "scientific_parameters_changed": False
        })
        e.atomic(e.ROOT.parent / "OWNER.json", {
            "owner_thread": "01a170c0-3584-74e2-b44c-4d0711ee1860",
            "unit": os.environ.get("AKI_UNIT_NAME"), "pid": os.getpid(),
            "start_ticks": process_start_ticks(os.getpid()),
            "root": str(e.ROOT.parent), "protocol_sha256": e.protocol_hash()
        })
        update_status("PREPROCESS", workers=workers, resources=before)
        items = [(desc, count) for desc in e.descriptors() for count in range(1, 28)]
        execute_phase("PREPROCESS", e.preprocess_task, items, workers, per_worker)
        execute_phase("RFE_DEVELOPMENT", e.score_candidate, e.candidate_tasks("rfe"), workers, per_worker)
        execute_phase("SPARSE_DEVELOPMENT", e.score_candidate, e.candidate_tasks("sparse"), workers, per_worker)
        selections = consolidate()
        e.atomic(e.OUT / "DEVELOPMENT_ACCEPTANCE.json", {
            "status": "PASS_TRAINING_CONTEXT_SEARCH_AND_SELECTION_ONLY",
            "protocol_sha256": e.protocol_hash(), "engine_sha256": e.code_hash(),
            "selections": len(selections), "selection_sha256": e.sha(e.OUT / "DEVELOPMENT_SELECTIONS.json"),
            "new_heldout_evaluation_completed": False,
            "final_pipeline_refits_and_downstream_pending": True
        })
        update_status("DEVELOPMENT_COMPLETE_FINAL_REFITS_REQUIRED", selections=len(selections), sparse_state="DEVELOPMENT_COMPLETE")
        print("DEVELOPMENT_COMPLETE_FINAL_REFITS_REQUIRED", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = {"traceback": traceback.format_exc(), "time": time.time(), "owner_pid": os.getpid()}
        e.atomic(e.OUT / "FAILURE.json", error)
        update_status("FAILED_DIAGNOSIS_REQUIRED", failure_path=str(e.OUT / "FAILURE.json"))
        raise

