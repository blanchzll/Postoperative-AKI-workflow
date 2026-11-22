
"""Fixed-prediction statistics and source-bound bootstrap input utilities."""
from pathlib import Path
import sys
import json
import math
import importlib.util
import hashlib
import numpy as np
import pandas as pd

import engine as e
sys.path.insert(0, str(e.SOURCE / "work/formal_evaluation_20261023_v1"))
import evaluation_stats as native
from fixed_prediction_metrics import FixedRisk, _design, group_weights

RESOURCE = e.SOURCE / "outputs/novelty_strengthening_20261101_v1/resource_comparison"
EVALUATION = e.OUT / "evaluation"

def rows_to_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def aligned_weights(context, frame):
    acceptance = json.loads((RESOURCE / "CONTEXT_INPUT_AND_RESAMPLING_ACCEPTANCE.json").read_text())
    receipt = acceptance["contexts"][context]
    path = RESOURCE / (context + "_BOOTSTRAP_WEIGHTS_PRIVATE.npz")
    assert e.sha(path) == receipt["weight_file_sha256"]
    source = np.load(path, allow_pickle=False)
    records = source["records"].astype(str)
    data = frame.copy()
    data["_record_string"] = data.excel_row.astype(str)
    data = data.set_index("_record_string", drop=False)
    assert data.index.is_unique and set(records) == set(data.index)
    data = data.loc[records].reset_index(drop=True)
    np.testing.assert_array_equal(source["y"].astype(int), data.y.to_numpy(dtype=int))
    np.testing.assert_array_equal(source["groups"].astype(str), data.group_hash.to_numpy(dtype=str))
    weights = source["weights"]
    assert weights.shape == (2000, len(data))
    assert np.all(weights >= 0)
    return data, weights, {
        "weight_file": str(path), "weight_file_sha256": e.sha(path),
        "original_stream_resampling_sha256": receipt["resampling_sha256"],
        "source": "Original current-source hospital/patient weight stream, exact archive payload reused"
    }

def eligible_weights(frame, expected_hash):
    raw = frame.copy()
    y, groups, records, centers, strata, models = _design(
        raw.y.to_numpy(), raw.group_hash.astype(str).to_numpy(),
        raw.excel_row.astype(str).to_numpy(), raw.center.to_numpy(),
        {"order_only": np.linspace(0, 1, len(raw))}
    )
    # Native _design canonically orders center, patient group and record.
    raw["_record_string"] = raw.excel_row.astype(str)
    raw = raw.set_index("_record_string", drop=False).loc[records.astype(str)].reset_index(drop=True)
    np.testing.assert_array_equal(y, raw.y.to_numpy())
    np.testing.assert_array_equal(groups.astype(str), raw.group_hash.to_numpy(dtype=str))
    rng = np.random.default_rng(20261017)
    weights = np.asarray([group_weights(strata, len(raw), rng) for _ in range(2000)], dtype=np.int64)
    proof = hashlib.sha256()
    for weight in weights:
        proof.update(weight.astype("<i8").tobytes())
    assert proof.hexdigest() == expected_hash, (proof.hexdigest(), expected_hash)
    return raw, weights, {
        "source": "Recreated original eligible-subset stream with exact native recorded weight SHA",
        "original_stream_resampling_sha256": proof.hexdigest(), "seed": 20261017
    }

def score(fixed, weights, thresholds=None, rank_only=False, calibration=True):
    weights = np.asarray(weights, dtype=np.int64)
    y = fixed.y.astype(float)
    n = int(weights.sum())
    events = float(weights @ y)
    output = fixed.score(weights)
    if rank_only:
        output = {key: value for key, value in output.items() if key in ["n", "events", "event_rate", "AUROC", "AP"]}
    if calibration and not rank_only:
        index = np.repeat(np.arange(len(y)), weights)
        output.update(native.calibration(y[index], fixed.p[index]))
    for percent in [10, 20]:
        counts = fixed.review_counts(percent / 100, weights)
        reviewed = float(counts.sum())
        captured = float(counts @ y)
        values = {
            "reviewed_n": reviewed, "captured_n": captured, "missed_n": events-captured,
            "false_alerts_n": reviewed-captured,
            "sensitivity": captured/events if events else None,
            "PPV": captured/reviewed if reviewed else None,
            "specificity": 1-(reviewed-captured)/(n-events) if n>events else None,
            "captured_per1000":1000*captured/n, "missed_per1000":1000*(events-captured)/n,
            "reviewed_per1000":1000*reviewed/n, "false_alerts_per1000":1000*(reviewed-captured)/n
        }
        output.update({f"capacity{percent}_{key}":value for key,value in values.items()})
    if not rank_only:
        for threshold in native.DCA:
            output[f"net_benefit_{threshold}"] = fixed.net_benefit(threshold, weights)["net_benefit"]
        for name, threshold in (thresholds or {}).items():
            if not name.startswith("training_") or not isinstance(threshold, (float, int)):
                continue
            mask = fixed.p >= threshold
            reviewed = float(weights @ mask)
            captured = float(weights @ (mask * y))
            output[f"{name}_reviewed_per1000"] = 1000*reviewed/n
            output[f"{name}_captured_per1000"] = 1000*captured/n
            output[f"{name}_missed_per1000"] = 1000*(events-captured)/n
            output[f"{name}_PPV"] = captured/reviewed if reviewed else None
    return {key: (float(value) if value is not None and np.isfinite(value) else None) for key,value in output.items()}

def capture_prefix(fixed, weights):
    y = fixed.y[fixed.order].astype(np.int64)
    weight = np.asarray(weights, dtype=np.int64)[fixed.order]
    return np.cumsum(weight), np.cumsum(weight*y), y

def captured_at(prefix, count):
    cumulative, events, y = prefix
    count = np.asarray(count, dtype=np.int64)
    position = np.searchsorted(cumulative, count, side="right")
    bounded = np.minimum(position, len(cumulative)-1)
    previous = np.maximum(bounded-1, 0)
    old_count = np.where(bounded>0, cumulative[previous], 0)
    old_events = np.where(bounded>0, events[previous], 0)
    return np.where(position==len(cumulative), events[-1], old_events+(count-old_count)*y[bounded])

def minimum_reviews(prefix, target):
    cumulative, events, y = prefix
    target = np.asarray(target, dtype=np.int64)
    position = np.minimum(np.searchsorted(events, target, side="left"), len(events)-1)
    previous = np.maximum(position-1, 0)
    old_count = np.where(position>0, cumulative[previous], 0)
    old_events = np.where(position>0, events[previous], 0)
    return np.where(target==0, 0, old_count+target-old_events)

def intervals(model, estimates, samples):
    result = []
    for metric, point in estimates.items():
        if metric in ["n", "events"]:
            continue
        array = np.asarray(samples[metric], dtype=float)
        valid = array[np.isfinite(array)]
        lower, upper = np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None]
        result.append({
            "model":model,"metric":metric,"estimate":point,"lower95":lower,"upper95":upper,
            "valid_replicates":len(valid),"requested_replicates":len(array)
        })
    return result

