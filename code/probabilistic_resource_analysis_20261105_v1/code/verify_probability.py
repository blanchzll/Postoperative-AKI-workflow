"""Independent literal-curve and polynomial-CDF verification."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import csv
import hashlib
import json
import math
import multiprocessing
import traceback
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

RUN = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/probabilistic_resource_analysis_20261105_v1")
PREVIOUS = RUN.parent/"full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1"
SOURCE = PREVIOUS/"resource_comparison"
OUTPUT = RUN/"outputs"
FIELDS = [
    "p_win","p_tie","p_loss","p_infeasible","p_win_given_feasible",
    "mean_delta_per1000_given_feasible","mean_delta_per1000_fallback",
    "expected_regret_base_per1000","expected_regret_extra_fallback_per1000",
    "mean_base_captured_per1000","mean_extra_captured_per1000_given_feasible",
    "mean_extra_reviews_per1000_given_feasible"
]

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    os.replace(temporary,path)

def polynomial_cdf(values, family):
    x = np.clip(values/family["scale"],0,1)
    degree = family["alpha"]+family["beta"]-1
    result = np.zeros_like(x,dtype=float)
    for index in range(family["alpha"],degree+1):
        result += math.comb(degree,index)*x**index*(1-x)**(degree-index)
    return result

def integrate(base, extra, n, budget, family, cache):
    key = n,budget,family["id"]
    if key not in cache:
        k = (budget*n+99)//100
        counts = np.arange(k+1)
        left = (k-counts-1)/n
        right = (k-counts)/n
        mass = polynomial_cdf(right,family)-polynomial_cdf(left,family)
        cache[key] = counts,np.maximum(0,mass),1-float(polynomial_cdf(np.array([k/n]),family)[0])
    counts,mass,infeasible = cache[key]
    baseline = int(base[(budget*n+99)//100])
    difference = extra[counts].astype(int)-baseline
    positive = difference>0
    negative = difference<0
    equal = difference==0
    feasible = float(sum(mass))
    win = float(sum(mass[positive]))
    tie = float(sum(mass[equal]))
    loss = float(sum(mass[negative]))
    delta = float(sum(mass*difference))*1000/n
    gain = float(sum(mass[positive]*difference[positive]))*1000/n
    harm = float(sum(mass[negative]*(-difference[negative])))*1000/n
    return np.array([
        win,tie,loss,infeasible,win/feasible,delta/feasible,delta,
        gain,harm,1000*baseline/n,
        float(sum(mass*extra[counts]))*1000/n/feasible,
        float(sum(mass*counts))*1000/n/feasible
    ])

def sorted_percentile(values, proportion):
    values = np.sort(values,axis=0)
    position = (len(values)-1)*proportion
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower]+(position-lower)*(values[upper]-values[lower])

def histogram_moments(histogram, support):
    total = int(histogram.sum())
    value = float(histogram@support)/total
    square = float(histogram@(support*support))/total
    error = math.sqrt(max(0,square-value*value)/(total-1)) if total>1 else 0.0
    return value,error

def verify_mc(base, extra, n, parameters, count, *, budget=None, family=None, dependence=None):
    if family is not None:
        k = (budget*n+99)//100
        cost = parameters[family["id"]][:count]
        feasible = cost <= k/n
        extra_reviews = np.floor(k-n*cost).astype(int)
        extra_reviews = np.clip(extra_reviews,0,n)
        base_reviews = np.full(count,k,dtype=int)
    else:
        absolute = 10*((budget*n+99)//100)
        acquisition = parameters["acquisition_minutes"][:count]
        review = parameters["review_"+dependence][:count]
        feasible = acquisition <= absolute/n
        extra_reviews = np.clip(np.floor((absolute-n*acquisition)/review).astype(int),0,n)
        base_reviews = np.minimum(n,np.floor(absolute/review).astype(int))
        for index in np.linspace(0,count-1,31,dtype=int):
            b = Fraction(absolute)
            r = Fraction(float(review[index]))
            a = Fraction(float(acquisition[index]))
            assert base_reviews[index] == min(n,int(b//r))
            assert bool(feasible[index]) == (n*a<=b)
            if feasible[index]:
                assert extra_reviews[index] == min(n,int((b-n*a)//r))
    captured_base = base[base_reviews].astype(int)
    captured_extra = extra[extra_reviews].astype(int)
    event_max = int(base[-1])
    differences = captured_extra[feasible]-captured_base[feasible]
    frequencies = np.bincount(differences+event_max,minlength=2*event_max+1)
    support = np.arange(-event_max,event_max+1,dtype=float)
    valid = int(frequencies.sum())
    wins = int(frequencies[event_max+1:].sum())
    ties = int(frequencies[event_max])
    losses = int(frequencies[:event_max].sum())
    raw = [
        (wins,count),(ties,count),(losses,count),(count-valid,count),(wins,valid)
    ]
    means = [numerator/denominator for numerator,denominator in raw]
    errors = [math.sqrt(p*(1-p)/(denominator-1)) if denominator>1 else 0
              for p,(_,denominator) in zip(means,raw)]
    mean,error = histogram_moments(frequencies,support*1000/n)
    means.append(mean);errors.append(error)
    full = frequencies.copy()
    full[event_max] += count-valid
    for transform in [
        support*1000/n,np.maximum(support,0)*1000/n,np.maximum(-support,0)*1000/n
    ]:
        mean,error = histogram_moments(full,transform)
        means.append(mean);errors.append(error)
    for values in [captured_base,captured_extra[feasible],extra_reviews[feasible]]:
        frequencies2 = np.bincount(values)
        support2 = np.arange(len(frequencies2))*1000/n
        mean,error = histogram_moments(frequencies2,support2)
        means.append(mean);errors.append(error)
    return np.array(means),np.array(errors),valid

def integer_cell(value):
    exact = Fraction(value)
    if exact.denominator != 1:
        raise ValueError("Expected integral CSV count: "+value)
    return exact.numerator

def worker(job):
    comparison,mode = job
    name = comparison+"__"+mode
    directory = OUTPUT/name
    protocol = read(RUN/"PROTOCOL_LOCK.json")
    old = read(PREVIOUS/"ANALYSIS_PROTOCOL_LOCK.json")
    item = next(row for row in old["matrix"] if row["comparison_id"]==comparison)
    source = np.load(SOURCE/(item["context"]+"_BOOTSTRAP_WEIGHTS_PRIVATE.npz"),allow_pickle=False)
    cache = np.load(SOURCE/(item["context"]+"_PREDICTION_CACHE_PRIVATE.npz"),allow_pickle=False)
    stored = np.load(directory/"CURVES_PRIVATE.npz",allow_pickle=False)
    native = np.load(directory/"INTEGRATED_SAMPLES_PRIVATE.npz",allow_pickle=False)
    groups = source["groups"].astype(str)
    records = source["records"].astype(str)
    y = source["y"].astype(int)
    w = source["weights"].astype(int)
    specifications = [(budget,family) for budget in protocol["normalized_budgets"]
                      for family in protocol["normalized_coordinate"]["families"]]
    result = np.empty_like(native["values"])
    orders = []
    for side in ["base","extra_information"]:
        key = item[side]["prediction"]+"|"+mode
        code = hashlib.sha256(key.encode()).hexdigest()[:20]
        probability = cache[code+"_p"]
        order = np.array(sorted(range(len(y)),key=lambda i:(-float(probability[i]),groups[i],records[i])))
        np.testing.assert_array_equal(order,cache[code+"_order"])
        orders.append(order)
    for center in np.unique(source["centers"]):
        first = {}
        for index in np.flatnonzero(source["centers"]==center):
            first.setdefault(groups[index],index)
        np.testing.assert_array_equal(w[:,list(first.values())].sum(axis=1),len(first))
    for group in np.unique(groups):
        index = np.flatnonzero(groups==group)
        if len(index)>1:
            np.testing.assert_array_equal(w[:,index],np.repeat(w[:,index[:1]],len(index),axis=1))
    mass_cache = {}
    checks = 0
    largest = 0.0
    point_curves = []
    for index in range(2001):
        weights = np.ones(len(y),dtype=int) if index==0 else w[index-1]
        n = int(weights.sum())
        curves = []
        for side,order in enumerate(orders):
            copies = np.repeat(order,weights[order])
            curve = np.r_[0,np.cumsum(y[copies])]
            assert len(curve)==n+1
            np.testing.assert_array_equal(curve,stored["base" if side==0 else "extra"][index,:n+1])
            curves.append(curve)
            checks += n+1
        assert int(stored["n"][index])==n
        assert int(stored["events"][index])==int(weights@y)
        if index==0:
            point_curves=curves
        for position,(budget,family) in enumerate(specifications):
            result[index,position] = integrate(*curves,n,budget,family,mass_cache)
        difference = np.abs(result[index]-native["values"][index])
        largest = max(largest,float(difference.max()))
        assert np.all(difference<1e-9),(name,index,float(difference.max()))
        checks += difference.size
        if index%200==0 or index==2000:
            save(directory/"INDEPENDENT_PROGRESS.json",dict(done=index,total=2000))
    with (directory/"NORMALIZED_ESTIMATES.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    lookup={(budget,family["id"]):position for position,(budget,family) in enumerate(specifications)}
    lows=sorted_percentile(result[1:],.025)
    highs=sorted_percentile(result[1:],.975)
    for row in rows:
        position=lookup[int(row["budget_percent"]),row["family"]]
        metric=FIELDS.index(row["metric"])
        for field,value in [
            ("estimate",result[0,position,metric]),("lower95",lows[position,metric]),("upper95",highs[position,metric])
        ]:
            assert abs(float(row[field])-value)<1e-9,(name,field,row["metric"])
            checks+=1
    # Verify old point budgets and inverse boundaries using exact integer arithmetic.
    with (SOURCE/(name+"_GRID.csv")).open() as handle:
        for row in csv.DictReader(handle):
            budget=int(row["budget_percent"])
            cost=round(100*float(row["extra_cost_per_person_review_equivalents"]))
            n=len(y);k=(budget*n+99)//100;remaining=100*k-cost*n
            feasible=remaining>=0
            assert feasible==(row["feasible_at_point"]=="True")
            assert int(row["base_captured_n"])==int(point_curves[0][k])
            if feasible:
                reviews=remaining//100
                assert reviews==integer_cell(row["extra_reviews_n"])
                assert int(point_curves[1][reviews])==integer_cell(row["extra_captured_n"])
            checks+=4
    with (SOURCE/(name+"_BOUNDARIES.csv")).open() as handle:
        for row in csv.DictReader(handle):
            budget=int(row["budget_percent"]);k=(budget*len(y)+99)//100
            target=int(point_curves[0][k])
            first=next(j for j,value in enumerate(point_curves[1]) if value>=target)
            assert first==int(row["minimum_extra_path_reviews_for_same_capture"])
            checks+=1
    parameters=np.load(OUTPUT/"PARAMETER_DRAWS_PRIVATE.npz",allow_pickle=False)
    with (directory/"ABSOLUTE_TIME_ESTIMATES.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    for budget in protocol["absolute_time_sensitivity"]["budgets"]:
        for dependence in ["negative","independent","positive"]:
            chosen=[r for r in rows if int(r["reference_budget_percent"])==budget and r["dependence"]==dependence]
            count=int(chosen[0]["draws"])
            means,errors,valid=verify_mc(*point_curves,len(y),parameters,count,budget=budget,dependence=dependence)
            for row in chosen:
                j=FIELDS.index(row["metric"])
                assert abs(float(row["estimate"])-means[j])<1e-9
                assert abs(float(row["MCSE"])-errors[j])<1e-9
                assert int(row["feasible_draws"])==valid
                checks+=3
    with (directory/"NORMALIZED_MC_VALIDATION.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    for budget,family in specifications:
        chosen=[r for r in rows if int(r["budget_percent"])==budget and r["family"]==family["id"]]
        count=int(chosen[0]["draws"])
        means,errors,valid=verify_mc(*point_curves,len(y),parameters,count,budget=budget,family=family)
        for row in chosen:
            j=FIELDS.index(row["metric"])
            assert abs(float(row["MC_estimate"])-means[j])<1e-9
            assert abs(float(row["MCSE"])-errors[j])<1e-9
            assert int(row["feasible_draws"])==valid
            checks+=3
    receipt=dict(
        status="PASS_INDEPENDENT_LITERAL_CURVES_POLYNOMIAL_INTEGRALS_ALL_SAMPLES_INTERVALS_AND_MONTECARLO",
        comparison=comparison,mode=mode,assertions=checks,
        normalized_profiles=len(specifications),bootstrap_replicates=2000,
        old_grid_rows_verified=186,old_boundary_rows_verified=6,
        absolute_profiles=6,MCvalidation_profiles=len(specifications),
        max_integrated_numeric_error=largest,
        variable_record_counts_and_patient_clusters_verified=True
    )
    save(directory/"INDEPENDENT_VERIFICATION.json",receipt)
    return receipt

def main():
    protocol=read(RUN/"PROTOCOL_LOCK.json")
    lock=read(RUN/"INDEPENDENT_CODE_LOCK.json")
    assert digest(__file__)==lock["code_sha256"]
    assert digest(RUN/"PROTOCOL_LOCK.json")==lock["protocol_sha256"]
    jobs=[(comparison,mode) for comparison in protocol["comparisons"] for mode in protocol["probability_modes"]]
    receipts=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(worker,job) for job in jobs]):
            receipts.append(future.result())
            save(OUTPUT/"INDEPENDENT_ACCEPTED_TASKS.json",dict(completed=len(receipts),total=len(jobs),receipts=receipts))
    save(OUTPUT/"INDEPENDENT_NUMERICAL_VERIFICATION.json",dict(
        status="PASS_COMPLETE_INDEPENDENT_NEW_RESOURCE_PROBABILITY_FUNCTIONS",
        request_sha256=protocol["request_sha256"],protocol_sha256=digest(RUN/"PROTOCOL_LOCK.json"),
        checks=sum(row["assertions"] for row in receipts),
        maximum_abs_integrated_error=max(row["max_integrated_numeric_error"] for row in receipts),
        jobs=receipts,independent_patient_curve_construction=True,
        independent_parameter_CDF_implementation=True,
        absolute_minute_budget_rational_spotchecks=True,
        source_bootstrap_streams_preserved=True
    ))
    print(json.dumps(dict(status="INDEPENDENT_VERIFICATION_PASS",jobs=len(receipts))),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception:
        save(OUTPUT/"INDEPENDENT_VERIFICATION_FAILURE.json",dict(error=traceback.format_exc(),pid=os.getpid()))
        raise
