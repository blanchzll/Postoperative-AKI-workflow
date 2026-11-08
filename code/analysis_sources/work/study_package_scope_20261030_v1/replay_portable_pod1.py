"""Finite POD1 relocation replay, around the unchanged native inference adapter.

Default invocation is read-only admission checking. Actual inference requires
--run, completed formal training, joint resource headroom and the exact bounded
owned unit. No model fitting, parameter changes, or bootstrap is performed.
"""
from pathlib import Path
import argparse, csv, fcntl, gc, hashlib, importlib.util, json, os, resource
import shutil, subprocess, sys, tempfile, threading, time, traceback
from datetime import datetime
from zoneinfo import ZoneInfo

R = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W = R/"work/study_package_scope_20261030_v1"
B = R/"outputs/study_package_scope_20261030_v1"
PREP = B/"portable_pod1_runtime_preparation_r1"
O = B/"portable_pod1_numeric_replay_v1"
LOCK = W/"POD1_PORTABLE_REPLAY_CODE_LOCK.json"
UNIT = "aki-postslot-portable-pod1-replay-20261030-v1.service"
BUDGET = 4*1024**3
GIB = 1024**3
TOL = 1e-12

def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, data):
    # New outcomes only. Unknown or failed outcomes must be inspected explicitly.
    with Path(path).open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def code_guard():
    lock = read(LOCK)
    for relative, expected in lock["files"].items():
        if sha(R/relative) != expected:
            raise RuntimeError("Pinned replay input changed: " + relative)
    return lock

def gpu_snapshot():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = []
    for row in csv.reader(text.splitlines()):
        index, name, total, free = [item.strip() for item in row]
        rows.append(dict(index=int(index), name=name,
                         total_bytes=int(total)*1024**2, free_bytes=int(free)*1024**2))
    if len(rows) != 1 or rows[0]["name"] != "Tesla P40":
        raise RuntimeError("Original one-device P40 qualification must be rechecked")
    return rows

def preflight():
    code_guard()
    g = load("_aki_portable_pod1_admission",
             R/"work/postformal_priority_admission_20261030_v1/admission.py")
    g.code_guard()
    state = g.unit_state(g.FORMAL_UNIT)
    ap = g.F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json"
    acceptance = g.read(ap) if ap.exists() else None
    reason = g.completion_reason(state, acceptance, g.read(g.F/"STATUS.json"))
    result = dict(time=now(), unit=UNIT, code_lock_sha256=sha(LOCK),
                  formal_unit_state=state, actual_numeric_replay_started=False,
                  memory_budget_bytes=BUDGET)
    if reason:
        return dict(result, status=reason, eligible=False)
    # Includes this unit if already launched; does not omit other owners.
    plan = g.resource_plan(False)
    remaining = min(plan["available_for_new_growth"],
                    plan["AKI_family_remaining_cap"])-2*GIB
    devices = gpu_snapshot()
    eligible = bool(plan["accepted"] and remaining >= BUDGET
                    and devices[0]["free_bytes"] >= 20*GIB)
    result.update(formal_acceptance_sha256=sha(ap), resource_plan=plan,
                  separately_retained_monitoring_bytes=2*GIB,
                  remaining_after_monitoring_bytes=remaining,
                  GPU=devices, required_GPU_free_bytes=20*GIB,
                  preserve_original_model_device=True)
    return dict(result, eligible=eligible,
                status="READY_FOR_FINITE_POD1_RELOCATION_REPLAY" if eligible
                       else "WAIT_JOINT_RAM_OR_GPU_HEADROOM")

def execute():
    admission = preflight()
    if not admission["eligible"]:
        print(json.dumps(admission, ensure_ascii=False))
        return
    g = load("_aki_portable_pod1_owner",
             R/"work/postformal_priority_admission_20261030_v1/admission.py")
    unit = g.unit_state(UNIT)
    if unit["ActiveState"] != "active" or int(unit["MainPID"]) != os.getpid():
        raise RuntimeError("Exact finite unit process ownership required")
    if not 2*GIB <= int(unit["MemoryMax"]) <= BUDGET:
        raise RuntimeError("Owned 2-4 GiB memory cgroup required")
    cg = Path("/sys/fs/cgroup")/unit["ControlGroup"].lstrip("/")
    if (cg/"memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("Swap-free cgroup required")
    runtime_max = subprocess.check_output(
        ["systemctl","--user","show",UNIT,"--property=RuntimeMaxUSec","--value"],
        text=True).strip()
    if runtime_max != "15min":
        raise RuntimeError("Exact 15-minute wall-time ceiling required")
    os.sched_setaffinity(0, {admission["resource_plan"]["CPU_affinity"][-1]})
    for name in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (850,850))
    # Preserve the original XGBoost device/precision; do not hide/retarget CUDA.
    if O.exists():
        raise RuntimeError("Existing replay outcome requires inspection; no overwrite")
    O.mkdir(mode=0o700, parents=True)
    owner = (O/"OWNER.lock").open("x")
    fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = time.monotonic()
    save(O/"ADMISSION.json", admission)
    scratch = None
    sampler_stop = threading.Event()
    sampler = None
    audit = {"enforce": False, "forbidden_original_reads": [], "allowed_package_reads": 0}
    gpu = {"samples": 0, "sampled_owned_peak_bytes": 0, "errors": [],
           "sampling_interval_seconds": 1.0, "not_allocator_exact_peak": True}
    try:
        scalar = load("_aki_private_protection", W/"audit_fixed_policy_records.py")
        protection = scalar.protection()
        manifest = read(PREP/"PRIVATE_PATH_MAP_WITH_POD1_SOURCE_INTERNAL.json")
        routes = read(PREP/"POD1_22_ROUTE_REPLAY_PLAN_INTERNAL.json")
        if len(manifest["files"]) != 613 or len(routes) != 22:
            raise RuntimeError("Frozen package/route set changed")
        identities = [tuple(sorted(row["identity"].items())) for row in routes]
        if len(set(identities)) != 22 or any(row["identity"]["role"] != "final" for row in routes):
            raise RuntimeError("Expected 22 distinct original final routes")
        source_snapshot = {}
        scratch = Path(tempfile.mkdtemp(prefix="owned-pod1-relocation-", dir=O)).resolve()
        copied_bytes = 0
        for item in manifest["files"]:
            relative = item["path"]
            src = R/relative
            if src.stat().st_size != item["bytes"] or sha(src) != item["sha256"]:
                raise RuntimeError("Original private payload changed: " + relative)
            dest = scratch/relative
            if not dest.resolve().is_relative_to(scratch):
                raise RuntimeError("Manifest path escapes owned relocation directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as source, dest.open("xb") as target:
                shutil.copyfileobj(source,target,length=1024*1024)
                target.flush(); os.fsync(target.fileno())
            if dest.stat().st_size != item["bytes"] or sha(dest) != item["sha256"]:
                raise RuntimeError("Relocated payload mismatch: " + relative)
            source_snapshot[relative] = item["sha256"]
            copied_bytes += item["bytes"]
        save(O/"COPY_VERIFICATION.json", dict(files=len(source_snapshot),
             bytes=copied_bytes, original_SHA256_preserved=True, private_only=True))
        # Run both support modules from the relocated copy as well. The map is
        # copied before the read audit; numerical inference never opens PREP/LOCK.
        helper_hashes = {}
        for name in ["portable_pod1_runtime.py", "locked_package_paths.py"]:
            src = W/name
            dest = scratch/"work/study_package_scope_20261030_v1"/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as source, dest.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush(); os.fsync(target.fileno())
            if sha(dest) != sha(src):
                raise RuntimeError("Copied replay helper mismatch")
            helper_hashes[name] = sha(dest)
        manifest_copy = scratch/"PRIVATE_REPLAY_PATH_MAP.json"
        with (PREP/"PRIVATE_PATH_MAP_WITH_POD1_SOURCE_INTERNAL.json").open("rb") as source, manifest_copy.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush(); os.fsync(target.fileno())
        manifest_hash = sha(manifest_copy)
        expected_manifest_hash = read(LOCK)["files"][str((PREP/"PRIVATE_PATH_MAP_WITH_POD1_SOURCE_INTERNAL.json").relative_to(R))]
        if manifest_hash != expected_manifest_hash:
            raise RuntimeError("Copied exact path map mismatch")
        adapter = load("_aki_portable_pod1_inference",
                       scratch/"work/study_package_scope_20261030_v1/portable_pod1_runtime.py")
        original_roots = [R.resolve(), (R/"outputs").resolve()]
        own_output = O.resolve()

        def audit_open(event, args):
            if event != "open" or not audit["enforce"] or not args:
                return
            path = args[0]
            if not isinstance(path, (str, bytes, os.PathLike)):
                return
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved.is_relative_to(scratch):
                audit["allowed_package_reads"] += 1
                return
            if resolved.is_relative_to(own_output):
                return
            if any(resolved.is_relative_to(root) for root in original_roots):
                audit["forbidden_original_reads"].append(str(resolved))
                raise RuntimeError("Original project fallback read rejected")

        sys.addaudithook(audit_open)

        def sample_owned_gpu():
            while not sampler_stop.is_set():
                try:
                    data = subprocess.check_output([
                        "nvidia-smi","--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"], text=True, timeout=3)
                    amount = 0
                    for row in csv.reader(data.splitlines()):
                        if row and row[0].strip() == str(os.getpid()):
                            amount += int(row[1].strip())*1024**2
                    gpu["samples"] += 1
                    gpu["sampled_owned_peak_bytes"] = max(gpu["sampled_owned_peak_bytes"], amount)
                except Exception as error:
                    gpu["errors"].append(type(error).__name__ + ": " + str(error))
                sampler_stop.wait(1.0)

        audit["enforce"] = True
        sampler = threading.Thread(target=sample_owned_gpu, daemon=False)
        sampler.start()
        runtime = adapter.create_native_runtime(scratch, manifest_copy, expected_manifest_hash)
        import numpy as np
        from threadpoolctl import threadpool_limits
        results = []
        for route in routes:
            for key, hash_key in [
                ("task_receipt","task_receipt_sha256"),("bundle","bundle_sha256"),
                ("calibrator","calibrator_sha256"),("expected_predictions","expected_predictions_sha256")]:
                if sha(runtime.resolver.resolve(route[key])) != route[hash_key]:
                    raise RuntimeError("Route payload mismatch: " + key)
            expected = runtime.read_csv(route["expected_predictions"],
                                        dtype={"group_hash":str}, float_precision="round_trip")
            rows = expected.excel_row.astype(int).tolist()
            if len(rows) != route["records"] or len(set(rows)) != len(rows):
                raise RuntimeError("Route count or record identity duplication")
            if int(expected.y.sum()) != route["events"]:
                raise RuntimeError("Native endpoint count mismatch")
            observed = runtime.data.loc[rows]
            for name in ["group_hash","center","surgery_date"]:
                if observed[name].astype(str).tolist() != expected[name].astype(str).tolist():
                    raise RuntimeError("Native metadata identity/order mismatch: " + name)
            for name in ["y","clinical_y"]:
                if not np.array_equal(observed[name].to_numpy(),expected[name].to_numpy()):
                    raise RuntimeError("Native endpoint alignment mismatch: " + name)
            raw = np.asarray(runtime.predict_bundle(route["bundle"], rows), dtype=float)
            cal = runtime.load_state(route["calibrator"])
            if cal.classes_.tolist() != [0,1]:
                raise RuntimeError("Calibrator class direction mismatch")
            with threadpool_limits(limits=1):
                calibrated = cal.predict_proba(runtime.logit(raw))[:,1]
            errors = {}
            for name, values in [("prob_raw",raw),("prob_calibrated",calibrated)]:
                if values.shape != (len(rows),) or not np.isfinite(values).all():
                    raise RuntimeError("Non-finite or wrong-size prediction")
                if ((values<0)|(values>1)).any():
                    raise RuntimeError("Probability outside [0,1]")
                difference = float(np.max(np.abs(values-expected[name].to_numpy())))
                if difference > TOL:
                    raise RuntimeError("Replay exceeded absolute tolerance: " + name + " " + str(difference))
                errors[name+"_maximum_absolute_error"] = difference
            results.append(dict(identity=route["identity"], records=len(rows),
                                events=route["events"], **errors))
            del cal, raw, calibrated, expected, observed
            gc.collect()
        sampler_stop.set(); sampler.join(timeout=5)
        if sampler.is_alive() or gpu["errors"] or gpu["samples"] == 0:
            raise RuntimeError("Resource sampler did not finish with usable evidence")
        if audit["forbidden_original_reads"]:
            raise RuntimeError("Original-project fallback was attempted")
        trace = runtime.resolver.trace
        versions = runtime.numerical_versions
        del runtime
        gc.collect()
        audit["enforce"] = False
        for relative, expected_hash in source_snapshot.items():
            if sha(R/relative) != expected_hash:
                raise RuntimeError("Original payload changed during replay")
        if scalar.protection() != protection:
            raise RuntimeError("Protected manuscript changed during replay")
        save(O/"NUMERIC_ROUTE_COMPARISONS.json", results)
        save(O/"PRIVATE_PACKAGE_READ_TRACE.json", trace)
        save(O/"ORIGINAL_READ_REJECTION_AUDIT.json", audit)
        save(O/"RESOURCE_MEASUREMENT.json", dict(
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            GPU=gpu, CPU_affinity=sorted(os.sched_getaffinity(0)),
            cgroup_memory_events=(cg/"memory.events").read_text(),
            cgroup_memory_peak_bytes=int((cg/"memory.peak").read_text()),
            seconds=time.monotonic()-started))
        save(O/"POD1_RELOCATION_NUMERIC_ACCEPTANCE.json", dict(
            status="PASS_PRIVATE_POD1_22_ROUTE_NUMERIC_RELOCATION_ONLY", time=now(),
            code_lock_sha256=sha(LOCK), routes=22, original_payload_files_copied=613,
            replay_support_code_copies=helper_hashes, copied_path_map_sha256=manifest_hash,
            records_checked=sum(x["records"] for x in results),
            raw_max_abs=max(x["prob_raw_maximum_absolute_error"] for x in results),
            calibrated_max_abs=max(x["prob_calibrated_maximum_absolute_error"] for x in results),
            atol=TOL, rtol=0, numerical_versions=versions,
            original_inputs_models_and_92_manuscripts_unchanged=True,
            original_project_fallback_reads=0, new_fits=0, new_bootstraps=0,
            not_full_study_package_or_scientific_acceptance=True,
            evidence_hashes={p.name:sha(p) for p in O.glob("*.json")}))
        # Remove only this owned temporary copy, after durable acceptance evidence.
        shutil.rmtree(scratch)
        save(O/"OWNED_SCRATCH_CLEANUP.json",
             dict(time=now(),removed=str(scratch),exists=scratch.exists(),
                  original_files_removed=0, owned_process_exits_after_completion=True))
        print(json.dumps(dict(status="COMPLETE_PRIVATE_POD1_RELOCATION_REPLAY",
                              receipt=str(O/"POD1_RELOCATION_NUMERIC_ACCEPTANCE.json"))))
    except BaseException as error:
        audit["enforce"] = False
        sampler_stop.set()
        if sampler is not None:
            sampler.join(timeout=5)
        path = O/"FAILURE.json"
        if not path.exists():
            save(path, dict(time=now(),error=repr(error),traceback=traceback.format_exc(),
                            owned_scratch=str(scratch),scratch_preserved=True,
                            new_fits=0,original_outputs_overwritten=False))
        raise
    finally:
        sampler_stop.set()
        owner.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        execute()
    else:
        print(json.dumps(preflight(),ensure_ascii=False))
