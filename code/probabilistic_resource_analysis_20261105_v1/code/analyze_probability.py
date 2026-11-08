"""Conditional parameter-uncertainty evaluation of frozen resource rankings."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import betainc, betaincinv, ndtr

RUN = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/probabilistic_resource_analysis_20261105_v1")
REFERENCE = RUN.parent / "full_data_20261025_postslot"
PREVIOUS = REFERENCE / "outputs/novelty_strengthening_20261101_v1"
SOURCE = PREVIOUS / "resource_comparison"
OUTPUT = RUN / "outputs"
METRICS = [
    "p_win", "p_tie", "p_loss", "p_infeasible", "p_win_given_feasible",
    "mean_delta_per1000_given_feasible", "mean_delta_per1000_fallback",
    "expected_regret_base_per1000", "expected_regret_extra_fallback_per1000",
    "mean_base_captured_per1000", "mean_extra_captured_per1000_given_feasible",
    "mean_extra_reviews_per1000_given_feasible",
]

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()

def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def write_table(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def frozen_module():
    path = PREVIOUS / "run_resource_comparison.py"
    specification = importlib.util.spec_from_file_location("original_resource_functions", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

def load_job(comparison, mode):
    protocol = read(RUN / "PROTOCOL_LOCK.json")
    old = read(PREVIOUS / "ANALYSIS_PROTOCOL_LOCK.json")
    item = next(row for row in old["matrix"] if row["comparison_id"] == comparison)
    context = item["context"]
    receipt = read(SOURCE / "CONTEXT_INPUT_AND_RESAMPLING_ACCEPTANCE.json")["contexts"][context]
    weights_path = SOURCE / (context + "_BOOTSTRAP_WEIGHTS_PRIVATE.npz")
    cache_path = SOURCE / (context + "_PREDICTION_CACHE_PRIVATE.npz")
    assert digest(weights_path) == receipt["weight_file_sha256"]
    assert digest(cache_path) == receipt["prediction_cache_sha256"]
    weights_file = np.load(weights_path, allow_pickle=False)
    cache = np.load(cache_path, allow_pickle=False)
    original = frozen_module()
    risks = []
    for side in ["base", "extra_information"]:
        source = item[side]
        assert digest(source["prediction"]) == source["prediction_sha256"]
        key = source["prediction"] + "|" + mode
        code = hashlib.sha256(key.encode()).hexdigest()[:20]
        risk = original.stats.FixedRisk(
            weights_file["y"], cache[code + "_p"],
            weights_file["groups"], weights_file["records"]
        )
        np.testing.assert_array_equal(risk.order, cache[code + "_order"])
        risks.append(risk)
    return protocol, item, weights_file, risks, original

def prepare_parameter_draws(protocol):
    path = OUTPUT / "PARAMETER_DRAWS_PRIVATE.npz"
    if path.exists():
        assert read(OUTPUT / "PARAMETER_DRAWS.json")["sha256"] == digest(path)
        return
    count = protocol["precision"]["maximum_IID_draws"]
    rng = np.random.default_rng(20261105)
    uniform = rng.random(count)
    normal = rng.standard_normal((count, 2))
    arrays = {}
    for family in protocol["normalized_coordinate"]["families"]:
        arrays[family["id"]] = family["scale"] * betaincinv(
            family["alpha"], family["beta"], uniform
        )
    arrays["acquisition_minutes"] = 3.0 * ndtr(normal[:, 0])
    diagnostics = []
    for rho, name in [(-.7, "negative"), (0.0, "independent"), (.7, "positive")]:
        probability = ndtr(rho * normal[:, 0] + math.sqrt(1-rho*rho) * normal[:, 1])
        review = np.where(
            probability <= .5,
            5.0 + np.sqrt(50.0 * probability),
            15.0 - np.sqrt(50.0 * (1-probability))
        )
        arrays["review_" + name] = review
        diagnostics.append(dict(
            name=name, latent_gaussian_rho=rho,
            transformed_Pearson_correlation=float(np.corrcoef(arrays["acquisition_minutes"], review)[0, 1]),
            mean_review_minutes=float(review.mean())
        ))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    save(OUTPUT / "PARAMETER_DRAWS.json", dict(
        seed=20261105, maximum_draws=count, prefix_reuse=True,
        protocol_sha256=digest(RUN / "PROTOCOL_LOCK.json"),
        sha256=digest(path), acquisition_mean_minutes=float(arrays["acquisition_minutes"].mean()),
        dependence_diagnostics=diagnostics
    ))

def review_count_mass(n, budget, family):
    count = (budget * n + 99) // 100
    reviews = np.arange(count + 1)
    upper = np.clip((count - reviews) / (n * family["scale"]), 0, 1)
    lower = np.clip((count - reviews - 1) / (n * family["scale"]), 0, 1)
    mass = betainc(family["alpha"], family["beta"], upper) - betainc(
        family["alpha"], family["beta"], lower
    )
    infeasible = 1 - float(betainc(
        family["alpha"], family["beta"],
        np.clip(count / (n * family["scale"]), 0, 1)
    ))
    assert np.all(mass >= -1e-14)
    mass = np.maximum(mass, 0)
    assert abs(float(mass.sum()) + infeasible - 1) < 1e-12
    return count, reviews, mass, infeasible

def integrated_values(base_curve, extra_curve, n, budget, family, cached_masses):
    key = (n, budget, family["id"])
    if key not in cached_masses:
        cached_masses[key] = review_count_mass(n, budget, family)
    count, reviews, mass, infeasible = cached_masses[key]
    baseline = int(base_curve[count])
    extra = extra_curve[reviews].astype(float)
    difference = extra - baseline
    feasible = float(mass.sum())
    scale = 1000.0 / n
    win = float(mass[difference > 0].sum())
    tie = float(mass[difference == 0].sum())
    loss = float(mass[difference < 0].sum())
    assert abs(win + tie + loss + infeasible - 1) < 1e-12
    delta = float(mass @ difference) * scale
    return np.array([
        win, tie, loss, infeasible, win / feasible,
        delta / feasible, delta,
        float(mass @ np.maximum(difference, 0)) * scale,
        float(mass @ np.maximum(-difference, 0)) * scale,
        baseline * scale,
        float(mass @ extra) / feasible * scale,
        float(mass @ reviews) / feasible * scale,
    ]), count, baseline

def draw_values(base_curve, extra_curve, n, acquisition_ratio=None, acquisition=None,
                review=None, minute_budget=None, reference_count=None):
    if acquisition_ratio is not None:
        remaining = reference_count - n * acquisition_ratio
        feasible = remaining >= 0
        base_counts = np.full(len(remaining), reference_count, dtype=np.int64)
        extra_counts = np.floor(np.maximum(remaining, 0)).astype(np.int64)
    else:
        remaining = minute_budget - n * acquisition
        feasible = remaining >= 0
        base_counts = np.minimum(n, np.floor(minute_budget / review).astype(np.int64))
        extra_counts = np.minimum(n, np.floor(np.maximum(remaining, 0) / review).astype(np.int64))
    baseline = base_curve[base_counts]
    extra = extra_curve[extra_counts]
    difference = extra - baseline
    fallback = np.where(feasible, difference, 0).astype(float) * 1000.0 / n
    win = feasible & (difference > 0)
    tie = feasible & (difference == 0)
    loss = feasible & (difference < 0)
    arrays = [
        win.astype(float), tie.astype(float), loss.astype(float), (~feasible).astype(float),
        win[feasible].astype(float),
        difference[feasible].astype(float) * 1000.0 / n, fallback,
        np.maximum(fallback, 0), np.maximum(-fallback, 0),
        baseline.astype(float) * 1000.0 / n,
        extra[feasible].astype(float) * 1000.0 / n,
        extra_counts[feasible].astype(float) * 1000.0 / n,
    ]
    means = np.array([float(value.mean()) for value in arrays])
    errors = np.array([
        float(value.std(ddof=1) / math.sqrt(len(value))) if len(value) > 1 else 0.0
        for value in arrays
    ])
    return means, errors, int(feasible.sum()), base_counts, extra_counts, feasible

def wilson_halfwidth(probabilities, count):
    z = 1.959963984540054
    p = np.asarray(probabilities)
    return z * np.sqrt(p * (1-p) / count + z*z/(4*count*count)) / (1+z*z/count)

def simulate_precision(base, extra, n, protocol, parameters, *, budget=None, family=None, dependence=None):
    size = protocol["precision"]["initial_IID_parameter_draws"]
    maximum = protocol["precision"]["maximum_IID_draws"]
    while True:
        if family is not None:
            kwargs = dict(acquisition_ratio=parameters[family["id"]][:size],
                          reference_count=(budget*n+99)//100)
        else:
            kwargs = dict(acquisition=parameters["acquisition_minutes"][:size],
                          review=parameters["review_"+dependence][:size],
                          minute_budget=10*((budget*n+99)//100))
        values, errors, valid, _, _, _ = draw_values(base, extra, n, **kwargs)
        probability_error = float(wilson_halfwidth(values[:4], size).max())
        capture_error = 1.959963984540054 * errors[METRICS.index("mean_delta_per1000_fallback")]
        precision_ok = probability_error <= .005 and capture_error <= .25
        if precision_ok or size == maximum:
            return values, errors, size, valid, probability_error, float(capture_error), precision_ok
        size = min(2*size, maximum)

def run_job(job):
    comparison, mode = job
    name = comparison + "__" + mode
    directory = OUTPUT / name
    directory.mkdir(mode=0o700, exist_ok=True)
    signature = dict(
        protocol_sha256=digest(RUN / "PROTOCOL_LOCK.json"),
        code_sha256=digest(__file__),
        parameters_sha256=read(OUTPUT / "PARAMETER_DRAWS.json")["sha256"]
    )
    if (directory / "COMPLETE.json").exists():
        complete = read(directory / "COMPLETE.json")
        assert complete["signature"] == signature
        return complete
    protocol, item, source, risks, original = load_job(comparison, mode)
    weights = source["weights"].astype(np.int64)
    n_observed = len(source["y"])
    all_weights = np.vstack([np.ones(n_observed, dtype=np.int64), weights])
    counts = all_weights.sum(axis=1)
    event_counts = all_weights @ source["y"]
    maximum = int(counts.max())
    base_curves = np.full((len(all_weights), maximum+1), -1, dtype=np.int16)
    extra_curves = base_curves.copy()
    specifications = [
        (budget, family) for budget in protocol["normalized_budgets"]
        for family in protocol["normalized_coordinate"]["families"]
    ]
    results = np.empty((2001, len(specifications), len(METRICS)))
    zero_capture = np.zeros((2001, len(specifications)), dtype=bool)
    mass_cache = {}
    started = time.time()
    for index, weight in enumerate(all_weights):
        n = int(counts[index])
        integers = np.arange(n+1)
        base = original.tp_at(original.prefix(risks[0], weight), integers)
        extra = original.tp_at(original.prefix(risks[1], weight), integers)
        base_curves[index, :n+1] = base
        extra_curves[index, :n+1] = extra
        for position, (budget, family) in enumerate(specifications):
            results[index, position], _, captured = integrated_values(
                base, extra, n, budget, family, mass_cache
            )
            zero_capture[index, position] = captured == 0
        if index == 25:
            save(directory / "PILOT25.json", dict(
                status="ACTUAL_FIRST25_BOOTSTRAP_FUNCTIONS_COMPLETED_AND_REUSED",
                point_completed=True, bootstrap_completed=25, signature=signature
            ))
        if index % 100 == 0 or index == 2000:
            save(directory / "PROGRESS.json", dict(
                state="NORMALIZED_INTEGRATION", done_bootstrap=index, total_bootstrap=2000
            ))
    curve_path = directory / "CURVES_PRIVATE.npz"
    np.savez_compressed(curve_path, base=base_curves, extra=extra_curves,
                        n=counts, events=event_counts)
    sample_path = directory / "INTEGRATED_SAMPLES_PRIVATE.npz"
    np.savez_compressed(sample_path, values=results, metrics=np.array(METRICS),
                        zero_capture=zero_capture)
    summary = []
    for position, (budget, family) in enumerate(specifications):
        lower, upper = np.quantile(results[1:, position], [.025, .975], axis=0, method="linear")
        count = (budget*n_observed+99)//100
        for metric_index, metric in enumerate(METRICS):
            summary.append(dict(
                comparison=comparison, context=item["context"], mode=mode,
                budget_percent=budget, family=family["id"], metric=metric,
                n=n_observed, events=int(event_counts[0]), base_reviews=count,
                estimate=float(results[0, position, metric_index]),
                lower95=float(lower[metric_index]), upper95=float(upper[metric_index]),
                valid_replicates=2000, requested_replicates=2000,
                zero_reference_capture_bootstrap=int(zero_capture[1:, position].sum()),
                zero_event_bootstrap=int((event_counts[1:] == 0).sum()),
                source_grade="D_CONDITIONAL_SCENARIO",
            ))
    write_table(directory / "NORMALIZED_ESTIMATES.csv", summary)
    parameters = np.load(OUTPUT / "PARAMETER_DRAWS_PRIVATE.npz", allow_pickle=False)
    validation = []
    absolute = []
    base = base_curves[0, :n_observed+1]
    extra = extra_curves[0, :n_observed+1]
    for position, (budget, family) in enumerate(specifications):
        values, errors, draws, valid, halfwidth, capture_halfwidth, ok = simulate_precision(
            base, extra, n_observed, protocol, parameters, budget=budget, family=family
        )
        error = np.abs(values-results[0, position])
        tolerance = 5*errors + np.array([2e-5]*5+[1e-8]*7)
        assert np.all(error <= tolerance), (name, budget, family["id"], error, tolerance)
        for i, metric in enumerate(METRICS):
            validation.append(dict(
                comparison=comparison, mode=mode, budget_percent=budget, family=family["id"],
                metric=metric, exact=float(results[0, position, i]), MC_estimate=float(values[i]),
                MCSE=float(errors[i]), absolute_error=float(error[i]), allowed_error=float(tolerance[i]),
                draws=draws, feasible_draws=valid, probability_max95_halfwidth=halfwidth,
                fallback_capture95_halfwidth_per1000=capture_halfwidth, precision_pass=ok,
            ))
    for budget in protocol["absolute_time_sensitivity"]["budgets"]:
        for dependence in ["negative", "independent", "positive"]:
            values, errors, draws, valid, halfwidth, capture_halfwidth, ok = simulate_precision(
                base, extra, n_observed, protocol, parameters, budget=budget, dependence=dependence
            )
            for i, metric in enumerate(METRICS):
                absolute.append(dict(
                    comparison=comparison, mode=mode, reference_budget_percent=budget,
                    absolute_total_minutes=10*((budget*n_observed+99)//100),
                    dependence=dependence, metric=metric, n=n_observed,
                    events=int(event_counts[0]), estimate=float(values[i]), MCSE=float(errors[i]),
                    MC95_halfwidth=float(1.959963984540054*errors[i]),
                    draws=draws, feasible_draws=valid, precision_pass=ok,
                    probability_max95_halfwidth=halfwidth,
                    fallback_capture95_halfwidth_per1000=capture_halfwidth,
                    patient_sampling_interval="NOT_COMPUTED_IN_SUPPORTIVE_COORDINATE_CHECK",
                    source_grade="D_CONDITIONAL_JOINT_SCENARIO",
                ))
    write_table(directory / "NORMALIZED_MC_VALIDATION.csv", validation)
    write_table(directory / "ABSOLUTE_TIME_ESTIMATES.csv", absolute)
    paths = [curve_path, sample_path, directory/"NORMALIZED_ESTIMATES.csv",
             directory/"NORMALIZED_MC_VALIDATION.csv", directory/"ABSOLUTE_TIME_ESTIMATES.csv"]
    complete = dict(
        status="COMPUTED_REQUIRES_INDEPENDENT_AND_SCIENTIFIC_REVIEW",
        comparison=comparison, mode=mode, signature=signature, n=n_observed,
        normalized_profiles=len(specifications), normalized_rows=len(summary),
        absolute_profiles=6, absolute_rows=len(absolute), MC_validation_rows=len(validation),
        bootstrap_replicates=2000, actual_variable_bootstrap_n_range=[int(counts[1:].min()), int(counts[1:].max())],
        elapsed_seconds=time.time()-started, files={p.name:digest(p) for p in paths}
    )
    save(directory / "COMPLETE.json", complete)
    save(directory / "PROGRESS.json", dict(state="COMPUTED", done_bootstrap=2000, total_bootstrap=2000))
    return complete

def main():
    with (RUN / "RUN.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(RUN / "PROTOCOL_LOCK.json")
        assert digest(RUN / "REQUEST.json") == protocol["request_sha256"]
        code_lock = read(RUN / "NUMERICAL_CODE_LOCK.json")
        assert digest(__file__) == code_lock["code_sha256"]
        assert digest(RUN / "PROTOCOL_LOCK.json") == code_lock["protocol_sha256"]
        prepare_parameter_draws(protocol)
        jobs = [(comparison, mode) for comparison in protocol["comparisons"]
                for mode in protocol["probability_modes"]]
        receipts = []
        with ProcessPoolExecutor(
                max_workers=protocol["resources"]["workers"],
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_job, job):job for job in jobs}
            for future in as_completed(futures):
                receipt = future.result()
                receipts.append(receipt)
                save(OUTPUT / "ACCEPTED_TASKS.json", dict(completed=len(receipts), total=len(jobs), receipts=receipts))
        for filename in ["NORMALIZED_ESTIMATES.csv", "NORMALIZED_MC_VALIDATION.csv", "ABSOLUTE_TIME_ESTIMATES.csv"]:
            rows = []
            for comparison, mode in jobs:
                with (OUTPUT / (comparison+"__"+mode) / filename).open() as handle:
                    rows.extend(csv.DictReader(handle))
            write_table(OUTPUT / filename, rows)
        save(OUTPUT / "NUMERICAL_COMPUTATION_ACCEPTANCE.json", dict(
            status="COMPUTATION_COMPLETE_REVIEW_PENDING",
            task_id=protocol["task_id"], request_sha256=protocol["request_sha256"],
            protocol_sha256=digest(RUN / "PROTOCOL_LOCK.json"), code_sha256=digest(__file__),
            jobs=receipts, normalized_profiles=240, normalized_interval_rows=2880,
            absolute_time_profiles=48, absolute_time_rows=576,
            original_predictive_fits=0, original_bootstrap_weights_reused=True,
            numerical_method="ExactCDF on integer budgets; separate finiteIIDabsolute-cost integration",
            files={name:digest(OUTPUT/name) for name in [
                "NORMALIZED_ESTIMATES.csv","NORMALIZED_MC_VALIDATION.csv","ABSOLUTE_TIME_ESTIMATES.csv"
            ]}
        ))
        print(json.dumps(dict(status="COMPUTED", completed=len(receipts))), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        save(OUTPUT / "COMPUTATION_FAILURE.json", dict(error=traceback.format_exc(), pid=os.getpid()))
        raise
