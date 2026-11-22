"""Describe the existing nested selection variation and auditable input dependencies."""
from pathlib import Path
import json
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from fit_scale import ROOT, PROJECT, FROZEN, e, f, read, write, PROTOCOL_SHA

OUT = ROOT / "outputs/selection_and_burden"


def primitive_inputs(columns):
    result = set(columns)
    if "BMI" in result:
        result.remove("BMI")
        result.update(["Height", "Weight"])
    if "PreopHospitalDays" in result:
        result.remove("PreopHospitalDays")
        result.update(["admission_calendar_date", "operation_calendar_date"])
    return sorted(result)


def workflow(column):
    if column in ["PreopWBC", "PreopHb"]:
        return "routine_blood_count"
    if column in ["PreopAlb", "PreopALT", "PreopBUN", "PreopCr", "PreopGlucose"]:
        return "biochemistry"
    if column in ["Height", "Weight", "BMI", "WeightLoss"]:
        return "anthropometry_or_weight_history"
    if column == "PreopHospitalDays":
        return "calendar_dates"
    if column in ["Age", "Sex"]:
        return "demographics"
    if column in ["TumorSite", "TumorGroup", "NeoadjuvantChemo"]:
        return "tumour_history"
    if column == "ASA":
        return "preanaesthetic_assessment"
    return "medical_history"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = read(ROOT / "inputs/SELECTED_BY_SIZE.json")
    bindings = {(x["line"], x["context"]): x for x in read(ROOT / "inputs/FULL_RANKING_BINDINGS.json")}
    features = e.legacy.spec_for("preop")
    features = features["num"] + features["cat"]
    source_features, fold_features, summary, frequencies, families, burden, rules = [], [], [], [], [], [], []
    for record in records:
        line, context, k = record["line"], record["context"], record["K"]
        columns = bindings[line, context]["features_by_count"][str(k)]
        source_features.append({"line": line, "context": context, "K": k, "family": record["candidate"]["task"]["family"], "params": json.dumps(record["candidate"]["task"]["params"], sort_keys=True), "input_columns": "|".join(columns), "candidate_id": record["candidate"]["candidate_id"]})
        for fold in record["candidate"]["fold_metrics"]:
            fold_features.append({"line": line, "context": context, "inner_fold": fold["fold"], "K": k, "input_columns": "|".join(fold["input_columns"]), "AP": fold["AP"], "Brier": fold["Brier"]})
    for line in ["geography", "temporal"]:
        for context in ["final"] + [f"outer_{i}" for i in range(5)]:
            path = FROZEN / "outputs/selection" / line / context / "RF_RFE/SELECTION.json"
            selection = read(path)
            for rule in ["oneSE", "maxAP"]:
                candidate = selection[rule]
                k = candidate["task"]["count"]
                match = next(r for r in records if (r["line"], r["context"], r["K"]) == (line, context, k))
                assert candidate["candidate_id"] == match["candidate"]["candidate_id"]
                rules.append({"line": line, "context": context, "rule": rule, "K": k, "family": candidate["task"]["family"], "mean_AP": candidate["mean_AP"], "mean_Brier": candidate["mean_Brier"], "standard_error_best_fold_AP": selection["standard_error_best_fold_AP"], "oneSE_cutoff": selection["cutoff_mean_AP"], "source_sha256": e.sha(path), "candidate_id": candidate["candidate_id"]})
        for k in range(1, 28):
            outer = [r for r in source_features if r["line"] == line and r["context"] != "final" and r["K"] == k]
            sets = [set(r["input_columns"].split("|")) for r in outer]
            final = set(bindings[line, "final"]["features_by_count"][str(k)])
            jaccards = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
            final_overlap = [len(final & columns) / k for columns in sets]
            summary.append({"line": line, "K": k, "outer_contexts": 5, "pairwise_comparisons": 10, "mean_pairwise_Jaccard": np.mean(jaccards), "min_pairwise_Jaccard": min(jaccards), "max_pairwise_Jaccard": max(jaccards), "mean_final_overlap_fraction": np.mean(final_overlap), "final_input_columns": "|".join(sorted(final)), "scope": "descriptive variation across five overlapping original outer-training sets"})
            counts = Counter(c for columns in sets for c in columns)
            for column in features:
                frequencies.append({"line": line, "K": k, "feature": column, "selected_outer_training_contexts": counts[column], "outer_contexts": 5, "fraction": counts[column] / 5, "selected_in_final": column in final})
            for (family, params), count in Counter((row["family"], row["params"]) for row in outer).items():
                families.append({"line": line, "K": k, "family": family, "params": params, "outer_contexts_selected": count, "outer_contexts_total": 5})
            primitives = primitive_inputs(final)
            packages = Counter(workflow(c) for c in final)
            burden.append({"line": line, "K": k, "primitive_dependency_count": len(primitives), "workflow_categories_present": len(packages), "primitive_dependencies": "|".join(primitives), "final_input_columns": "|".join(sorted(final)), **{w: packages.get(w, 0) for w in sorted(set(map(workflow, features)))}, "costs_and_joint_order_or_blood_draw_timing": "not measured; field count and categories are proxies"})
    for name, rows in [("CONTEXT_SELECTED_FEATURES.csv", source_features), ("INNER_TRAINING_SELECTED_FEATURES.csv", fold_features), ("FEATURE_STABILITY_BY_K.csv", summary), ("FEATURE_SELECTION_FREQUENCY.csv", frequencies), ("FAMILY_PARAMETER_SELECTION_FREQUENCY.csv", families), ("INPUT_DEPENDENCIES_AND_WORKFLOWS.csv", burden), ("ORIGINAL_SIZE_RULES.csv", rules)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
    dictionary = []
    for column in features:
        note = "Observed source field. Local acquisition time and unit price unknown."
        if column == "BMI":
            note = "Computed BMI requires Height and Weight; count their union if those fields are also selected."
        elif column == "PreopHospitalDays":
            note = "Calendar-derived preoperative stay requires admission and operation dates. Real-time future operation-date availability depends on intended scoring time."
        elif column in ["PreopWBC", "PreopHb"]:
            note = "Potential shared blood-count workflow. Actual joint order, sample and result availability not documented."
        elif column.startswith("Preop"):
            note = "Potential shared biochemistry sampling or panel. Ordering, return times and charges need local records; analyte count is not blood-draw count."
        elif column == "HeartDisease":
            note = "Source-author definition excludes hypertension, which is separately recorded."
        elif column in ["Smoking", "Alcohol"]:
            note = "Retain literal source code2; its clinical semantics remain unconfirmed."
        dictionary.append({"field": column, "workflow_category": workflow(column), "primitive_dependencies": "|".join(primitive_inputs([column])), "note": note})
    pd.DataFrame(dictionary).to_csv(OUT / "FIELD_DICTIONARY.csv", index=False)
    # Honest fixed-K pipeline internal evaluation uses each outer fold's own inner-only calibration.
    internal = []
    import sys
    sys.path.insert(0, str(e.SOURCE / "work/final_evaluation_engine_20261022_v1"))
    from fixed_prediction_metrics import FixedRisk
    for item in read(ROOT / "outputs/post_development/MODEL_MANIFEST.json"):
        lock = read(item["path"])
        data = pd.read_csv(Path(item["path"]).parent / "PRIVATE_DEVELOPMENT_OOF.csv", dtype={"group_hash": str})
        for mode, column in [("raw", "prob_raw"), ("outer_training_only_calibrated", "prob_calibrated_outer_only")]:
            fixed = FixedRisk(data.y, data[column], data.group_hash, data.excel_row)
            for q in [10, 20]:
                internal.append({"line": lock["identity"]["line"], "K": lock["actual_input_count"], "mode": mode, "review_percent": q, **fixed.capacity(q / 100)})
    pd.DataFrame(internal).to_csv(OUT / "HONEST_FIXED_K_SOURCE_OOF_METRICS.csv", index=False)
    write(OUT / "DESCRIPTIVE_ACCEPTANCE.json", {"status": "PASS_COMPLETE_EXISTING_NESTED_SELECTION_AND_DEPENDENCY_DESCRIPTION", "protocol_sha256": PROTOCOL_SHA, "code_sha256": e.sha(__file__), "context_feature_sets": len(source_features), "inner_selected_feature_sets": len(fold_features), "K_line_summaries": len(summary), "original_size_rule_records": len(rules), "new_selection_bootstrap_fits": 0, "selection_frequency_is_population_probability": False, "clinical_costs_measured": False, "files": {x.name: e.sha(x) for x in OUT.glob("*.csv")}})


if __name__ == "__main__":
    main()
