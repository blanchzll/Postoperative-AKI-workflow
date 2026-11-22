"""Derive the complete prespecified loss-margin path and concise analysis tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fit_scale import ROOT, read, write, e, PROTOCOL_SHA

OUT = ROOT / "outputs/summary"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation = ROOT / "outputs/evaluation"
    points = pd.read_csv(evaluation / "POINT_ESTIMATES.csv")
    differences = pd.read_csv(evaluation / "PAIRED_DIFFERENCES.csv")
    rows = []
    for line in ["geography", "temporal"]:
        for scope in ["preop", "common_paired"]:
            for k in range(1, 28):
                for q in [10, 20]:
                    match = differences.loc[(differences.line == line) & (differences.scope == scope) & (differences.model == f"K{k:02d}") & (differences.reference == "original_full27") & (differences.metric == f"capacity{q}_captured_per1000")]
                    assert len(match) == 1
                    value = match.iloc[0]
                    for margin in np.arange(0, 10.25, .5):
                        rows.append({"line": line, "scope": scope, "K": k, "review_percent": q, "loss_direction": "original_full27 MINUS fixed_K", "loss_per1000": -value.estimate, "loss_lower95": -value.upper95, "loss_upper95": -value.lower95, "delta_per1000": margin, "criterion": "unrounded_upper95_loss_strictly_less_than_delta", "criterion_met": bool(-value.lower95 < margin), "clinical_acceptability_predefined": False})
    loss = pd.DataFrame(rows)
    loss.to_csv(OUT / "LOSS_MARGIN_ALL_K.csv", index=False)
    joint = []
    for scope in ["preop", "common_paired"]:
        for k in range(1, 28):
            for q in [10, 20]:
                subset = loss.loc[(loss.scope == scope) & (loss.K == k) & (loss.review_percent == q)]
                upper = subset.loss_upper95.max()
                for margin in np.arange(0, 10.25, .5):
                    joint.append({"scope": scope, "K": k, "review_percent": q, "delta_per1000": margin, "maximum_two_line_upper95_loss": upper, "both_lines_criterion_met": bool(upper < margin)})
    pd.DataFrame(joint).to_csv(OUT / "JOINT_LOSS_MARGIN_ALL_K.csv", index=False)
    core = points.loc[points["mode"].eq("calibrated")].copy()
    core.to_csv(OUT / "ALL_K_AND_REFERENCES_CLINICAL_POINTS.csv", index=False)
    nine = differences.loc[differences.model.eq("K09") & differences.reference.isin(["original_nine", "original_full27", "original_endop38"])]
    nine.to_csv(OUT / "NEW_NINE_DIRECT_PAIRED_COMPARISONS.csv", index=False)
    training = pd.read_csv(ROOT / "outputs/TRAINING_SIZE_PATH.csv")
    frontier = []
    for line in ["geography", "temporal"]:
        path = training.loc[(training.line == line) & training.context.eq("final")].sort_values("K")
        for row in path.to_dict("records"):
            simpler = path.loc[path.K < row["K"]]
            row["nondominated_by_fewer_inputs_in_source_AP"] = bool(simpler.empty or row["mean_AP"] > simpler.mean_AP.max() + 1e-12)
            frontier.append(row)
    pd.DataFrame(frontier).to_csv(OUT / "SOURCE_AP_SIZE_FRONTIER.csv", index=False)
    rules = pd.read_csv(ROOT / "outputs/selection_and_burden/ORIGINAL_SIZE_RULES.csv")
    final_rules = rules.loc[rules.context.eq("final")]
    final_rules.to_csv(OUT / "SOURCE_SELECTED_SIZE_RECOMMENDATIONS.csv", index=False)
    stability = pd.read_csv(ROOT / "outputs/selection_and_burden/FEATURE_STABILITY_BY_K.csv")
    burden = pd.read_csv(ROOT / "outputs/selection_and_burden/INPUT_DEPENDENCIES_AND_WORKFLOWS.csv")
    one_row = core.loc[core.model.str.match(r"K\d\d$")].copy()
    one_row["K"] = one_row.model.str[1:].astype(int)
    joined = one_row.merge(stability.drop(columns=["final_input_columns"]).rename(columns={"scope": "selection_variation_scope"}), on=["line", "K"]).merge(burden, on=["line", "K"])
    joined.to_csv(OUT / "PERFORMANCE_STABILITY_BURDEN_JOINED.csv", index=False)
    write(OUT / "SUMMARY_ACCEPTANCE.json", {"status": "PASS_FULL_PRESPECIFIED_SIZE_AND_MARGIN_PATH", "protocol_sha256": PROTOCOL_SHA, "code_sha256": e.sha(__file__), "loss_sensitivity_rows": len(loss), "joint_sensitivity_rows": len(joint), "clinical_acceptability_threshold": None, "source_recommendation_rules_retained": ["oneSE", "maxAP"], "target_driven_size_reselection": False, "files": {x.name: e.sha(x) for x in OUT.glob("*.csv")}})
    print(final_rules[["line", "rule", "K", "family"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
