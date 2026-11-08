"""Wait for required technical stages, then execute already frozen decision workloads once."""
from pathlib import Path
import json, sys, os, time, fcntl, subprocess, traceback
ROOT = Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/decision_continuation_20261024_v1'
DEC = ROOT / 'outputs/decision_workload_20261024_v1'
sys.path.insert(0, str(ROOT / 'work/formal_evaluation_20261023_v1'))
import common as base
REQUIRED = {'public': (ROOT / 'outputs/common6_public_evaluation_20261024_v1/REVIEW_ACCEPTANCE.json', 'PASS_common6_development_locks_and_public_results_reverified_and_interpreted'), 'main_strata': (ROOT / 'outputs/stratified_evaluation_20261023_v1/STAGE_STRATIFIED_ACCEPTANCE.json', 'PASS_24_prespecified_stratified_evaluations'), 'fixed_flow_sensitivity': (ROOT / 'outputs/fixed_flow_sensitivity_evaluation_20261024_v1/REVIEW_ACCEPTANCE.json', 'PASS_fixed_flow_sensitivity_artifacts_verified_and_interpreted'), 'label_sensitivity': (ROOT / 'outputs/outcome_sensitivity_20261024_v1/REVIEW_ACCEPTANCE.json', 'PASS_eight_scenario_receipts_outputs_and_report'), 'conditional_stability': (ROOT / 'outputs/conditional_stability_20261024_v1/REVIEW_ACCEPTANCE.json', 'PASS_2000_saved_receipts_and_outputs_reverified'), 'nonlinear_LR': (ROOT / 'outputs/nonlinear_LR_coverage_20261024_v1/ACCEPTANCE.json', 'PASS_60_main_LR_studies_include_linear_and_spline_variants'), 'SHAP': (ROOT / 'outputs/locked_pipeline_shap_20261024_v1/STAGE_SHAP_ACCEPTANCE.json', 'PASS_ten_locked_model_permutation_SHAP_routes'), 'reconstruction': (ROOT / 'outputs/paired_extension_recovery_20261024_v1/REVIEW_ACCEPTANCE.json', 'PASS_48_corrected_input_reconstruction_tasks_reverified'), 'cohort_extension': (ROOT / 'outputs/paired_extension_evaluation_20261024_v1/STAGE_EXTENSION_EVALUATION_ACCEPTANCE.json', 'PASS_ten_fixed_cohort_and_common_reference_evaluations')}
ACTIVE_UPSTREAM = ['locked_pipeline_shap_20261024_v1', 'paired_extension_clinical_20261024_v1', 'paired_extension_evaluation_20261024_v1']

def check():
    for f, h in base.read(OUT / 'CODE_LOCK.json').items():
        assert base.sha(CODE / f) == h
    for f, h in base.read(OUT / 'DEPENDENCIES.json').items():
        assert base.sha(f) == h
    for f, h in base.read(DEC / 'CODE_LOCK.json').items():
        assert base.sha(ROOT / 'work/decision_workload_20261024_v1' / f) == h
    for f, h in base.read(DEC / 'DEPENDENCIES.json').items():
        assert base.sha(f) == h

def readiness():
    missing = []
    receipts = {}
    for name, (p, status) in REQUIRED.items():
        if not p.exists():
            missing.append(name)
            continue
        r = base.read(p)
        assert status is None or r.get('status') == status, (name, r.get('status'))
        receipts[name] = {'receipt': str(p), 'sha256': base.sha(p), 'status': r.get('status', r.get('state'))}
    for name in ACTIVE_UPSTREAM:
        p = ROOT / 'outputs' / name
        if (p / 'FAILURE.json').exists():
            raise RuntimeError('Upstream failure requires diagnosis: ' + name)
        if (p / 'LAUNCH.json').exists():
            o = base.read(p / 'LAUNCH.json')
            proc = Path('/proc', str(o['pid']), 'stat')
            if proc.exists() and proc.read_text().split()[21] == str(o['start_ticks']):
                missing.append(name + '_owned_process_not_yet_exited')
    return (missing, receipts)

def main():
    guard = (OUT / 'SUPERVISOR.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check()
    base.atomic(OUT / 'LAUNCH.json', {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21], 'time_UTC': base.stamp(), 'code_lock_sha256': base.sha(OUT / 'CODE_LOCK.json')})
    assert base.read(DEC / 'PREPARATION_ACCEPTANCE.json')['status'] == 'PASS_math_and_source_adapter_preparation_only'
    while True:
        check()
        missing, receipts = readiness()
        if not missing:
            break
        base.atomic(OUT / 'status.json', {'state': 'waiting_required_supporting_stages', 'waiting_for': missing, 'pid': os.getpid(), 'time_UTC': base.stamp(), 'clinical_scenarios_started': False})
        time.sleep(30)
    decguard = (DEC / 'SUPERVISOR.lock').open('a')
    fcntl.flock(decguard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release = {'status': 'PASS_required_preceding_stages_accounted_for', 'code_lock_sha256': base.sha(DEC / 'CODE_LOCK.json'), 'nonlinear_LR_status_explicit': True, 'public_and_necessary_supporting_stage_statuses': receipts, 'scope': 'Original V1 fixed clinical rules and prespecified effect/workload assumptions; no thresholds selected from new results', 'unrelated_optional_or_posterior_stages': {'POD1': 'planned_after_decision', '': 'planned_after_decision', 'published_model_benchmarks': 'eligibility_review_only_no_exact_qualified_model_yet', 'calculator_package_manuscript': 'pending'}, 'technical_stage_acceptance_is_not_full_scientific_report_delivery': True}
    rp = DEC / 'RELEASE_FOR_CLINICAL_SCENARIOS.json'
    if rp.exists():
        assert base.read(rp) == release, 'Changed release; inspect existing decisions before resuming'
    else:
        base.atomic(rp, release)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    receipts2 = {}
    pending = [(line, cohort) for line in ['geography', 'temporal'] for cohort in ['preop', 'paired']]
    active = {}
    peak_mem_available = base.mem()
    headroom = max(4 * 1024 ** 3, int(peak_mem_available * 0.25))
    reserve_each = 2 * 1024 ** 3
    cpu_slots = len(os.sched_getaffinity(0))

    def collect(task):
        p = DEC / task / 'COMPLETE.json'
        r = base.read(p)
        assert r['status'] == 'PASS_frozen_workload_scenarios_not_clinical_efficacy'
        assert r['signature']['release_sha256'] == base.sha(rp)
        for f, h in r['files'].items():
            assert base.sha(p.parent / f) == h
        receipts2[str(p)] = base.sha(p)

    def rss(pid):
        try:
            return int(Path('/proc', str(pid), 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError:
            return 0
    try:
        while pending or active:
            check()
            for task, (proc, log) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del active[task]
                assert rc == 0, 'Frozen clinical runner failed; diagnose before any retry: ' + task
                collect(task)
            while pending and len(active) < min(4, cpu_slots):
                reserved = sum((max(0, reserve_each - rss(proc.pid)) for proc, _ in active.values()))
                if base.mem() < headroom + reserved + reserve_each:
                    break
                line, cohort = pending.pop(0)
                task = line + '_' + cohort
                if (DEC / task / 'COMPLETE.json').exists():
                    collect(task)
                    continue
                log = (logs / (task + '.log')).open('ab')
                proc = subprocess.Popen([str(base.PYTHON), '-B', str(ROOT / 'work/decision_workload_20261024_v1/clinical.py'), line, cohort], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[task] = (proc, log)
            base.atomic(OUT / 'status.json', {'state': 'running_frozen_decision_workloads', 'completed': len(receipts2), 'total': 4, 'owned_workers': {k: v[0].pid for k, v in active.items()}, 'pending': pending, 'memory_available_bytes': base.mem(), 'memory_headroom_bytes': headroom, 'cpu_slots': cpu_slots, 'task_threads': 1, 'pid': os.getpid(), 'time_UTC': base.stamp()})
            if pending or active:
                time.sleep(5)
    finally:
        for proc, log in active.values():
            proc.wait()
            log.close()
    base.atomic(OUT / 'STAGE_DECISION_ACCEPTANCE.json', {'status': 'PASS_four_frozen_clinical_decision_workload_contexts', 'receipts': receipts2, 'source_release_sha256': base.sha(rp), 'no_threshold_optimisation_or_effect_assumption_selection': True, 'no_claim_of_observed_AKI_prevention': True, 'interpretation_pending': True, 'whole_project_complete': False})
    base.atomic(OUT / 'status.json', {'state': 'complete_decision_workloads_interpretation_pending', 'completed': 4, 'total': 4, 'owned_workers': 0, 'time_UTC': base.stamp()})
    decguard.close()
    guard.close()
if __name__ == '__main__':
    try:
        main()
    except Exception:
        base.atomic(OUT / 'FAILURE.json', {'time_UTC': base.stamp(), 'traceback': traceback.format_exc()})
        raise
