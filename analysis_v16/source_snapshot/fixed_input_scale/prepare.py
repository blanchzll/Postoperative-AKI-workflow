"""Verify the frozen per-size candidates and bind existing raw fits without target scoring."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import hashlib
import time
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
FROZEN = PROJECT / "correction_runs/rfe_mainline_revision_20261116_v1/formal_v2"
sys.path.insert(0, str(FROZEN / "work"))
import engine as e
import finalize_models as f


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e.atomic(path, value)


def choose(candidates):
    best_ap = max(c["mean_AP"] for c in candidates)
    tied = [c for c in candidates if c["mean_AP"] >= best_ap - 1e-12]
    best_brier = min(c["mean_Brier"] for c in tied)
    return min((c for c in tied if c["mean_Brier"] <= best_brier + 1e-12), key=lambda c: c["candidate_id"])


def main():
    started = time.time()
    protocol_sha = e.sha(ROOT / "PROTOCOL_LOCK.json")
    protocol = read(ROOT / "PROTOCOL_LOCK.json")
    assert e.sha(FROZEN / "work/engine.py") == protocol["frozen_sources"]["engine_sha256"]
    assert e.sha(FROZEN / "work/finalize_models.py") == protocol["frozen_sources"]["finalizer_sha256"]
    writer_path = PROJECT / "writing_runs/v16_continuation_20261117/nine_candidate_review_20261117/ALL_CONTEXT_SIZE_CHAMPIONS.json"
    writer = read(writer_path)
    writer_map = {(x["line"], x["context"], x["raw_inputs"]): x for x in writer["rows"]}
    nine_path = PROJECT / "outputs/nine_candidate_screen_20261117_v1/NINE_CANDIDATE_SELECTION.json"
    nine = read(nine_path)
    nine_map = {(x["line"], x["context"]): x["candidate"]["candidate_id"] for x in nine["selected_training_candidates"]}
    records = []
    checked = 0
    score_rows = []
    for line in ("geography", "temporal"):
        for context in ("final", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"):
            groups = defaultdict(list)
            for path in sorted((FROZEN / "outputs/candidates" / line / context / "rfe").glob("*/RESULT.json")):
                candidate = read(path)
                task = candidate["task"]
                assert task["line"] == line and task["context"] == context
                assert candidate["signature"]["code"] == protocol["frozen_sources"]["engine_sha256"]
                assert candidate["signature"]["protocol"] == protocol["frozen_sources"]["protocol_sha256"]
                if not candidate["eligible"]:
                    continue
                assert len(candidate["fold_metrics"]) == 3
                for metric, fold_metric in (("mean_AP", "AP"), ("mean_Brier", "Brier")):
                    assert abs(candidate[metric] - np.mean([x[fold_metric] for x in candidate["fold_metrics"]])) < 1e-14
                assert all(x["raw_count"] == task["count"] == len(x["input_columns"]) for x in candidate["fold_metrics"])
                candidate["source_result_path"] = str(path)
                candidate["source_result_sha256"] = e.sha(path)
                groups[task["count"]].append(candidate)
                checked += 1
                score_rows.append({"line": line, "context": context, "K": task["count"], "family": task["family"], "candidate_id": candidate["candidate_id"], "mean_AP": candidate["mean_AP"], "mean_Brier": candidate["mean_Brier"], "params": json.dumps(task["params"], sort_keys=True)})
            assert set(groups) == set(range(1, 28))
            for k in range(1, 28):
                assert len(groups[k]) == 36
                candidate = choose(groups[k])
                independent = writer_map[line, context, k]
                assert candidate["candidate_id"] == independent["candidate_id"]
                assert candidate["source_result_sha256"] == independent["source_sha256"]
                if k == 9:
                    assert candidate["candidate_id"] == nine_map[line, context]
                assert e.sha(candidate["prediction_file"]) == candidate["prediction_sha256"]
                predictions = np.load(candidate["prediction_file"], allow_pickle=False)
                rows, valid, _ = f.context_rows(line, context)
                assert set(predictions["excel_row"]) == set(rows)
                assert len(predictions["excel_row"]) == len(set(predictions["excel_row"]))
                meta = f.metadata().loc[predictions["excel_row"]]
                np.testing.assert_array_equal(meta.y.to_numpy(), predictions["y"])
                np.testing.assert_array_equal(meta.group_hash.to_numpy(dtype=str), predictions["group_hash"])
                assert set(predictions["group_hash"]).isdisjoint(f.metadata().loc[valid].group_hash)
                records.append({"line": line, "context": context, "K": k, "candidate": candidate})
    assert checked == 11664 and len(records) == 324
    old_refits = read(FROZEN / "outputs/post_development/REFITS.json")
    old_rf = []
    ranking_sources = defaultdict(dict)
    for rec in old_refits:
        if rec["method"] != "RF_RFE":
            continue
        directory = Path(rec["directory"])
        receipt = read(directory / "COMPLETE.json")
        assert e.sha(directory / "COMPLETE.json") == rec["receipt_sha256"]
        assert e.qualified(directory)
        path = Path(receipt["selection_evidence"]["ranking_source"])
        assert e.sha(path) == receipt["selection_evidence"]["ranking_sha256"]
        ranking_sources[rec["line"], rec["context"]][str(path)] = read(path)["features_by_count"]
        old_rf.append((rec, receipt))
    # Qualified 1-2-input extensions from the prior study can cover other K bindings.
    extended_paths = list((FROZEN / "outputs/post_development/ranking").glob("*/path.json"))
    ranking_bindings = []
    extensions_needed = []
    for line in ("geography", "temporal"):
        for context in ("final", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"):
            rows, _, _ = f.context_rows(line, context)
            meta = f.metadata().loc[rows]
            candidates = dict(ranking_sources[line, context])
            for path in extended_paths:
                receipt = read(path.parent / "COMPLETE.json")
                signature = receipt["signature"]
                if signature.get("training_y") == e.digest(meta.y.tolist()) and signature.get("training_groups") == e.digest(meta.group_hash.tolist()):
                    assert e.qualified(path.parent)
                    candidates[str(path)] = read(path)["features_by_count"]
            longest = max(candidates, key=lambda path: len(candidates[path]))
            for other, paths in candidates.items():
                for k in set(paths).intersection(candidates[longest]):
                    assert paths[k] == candidates[longest][k]
            existing = candidates[longest]
            if "1" not in existing or "2" not in existing:
                extensions_needed.append({"line": line, "context": context, "inherited_path": longest, "inherited_minimum_K": min(map(int, existing)), "additional_RF_fits_upper_bound": 2})
            ranking_bindings.append({"line": line, "context": context, "path": longest, "sha256": e.sha(longest), "features_by_count": existing, "training_rows_sha256": e.digest(rows), "training_y_sha256": e.digest(meta.y.tolist()), "training_groups_sha256": e.digest(meta.group_hash.tolist())})
    reused = 0
    for record in records:
        task = record["candidate"]["task"]
        binding = next(x for x in ranking_bindings if (x["line"], x["context"]) == (record["line"], record["context"]))
        columns = binding["features_by_count"].get(str(record["K"]))
        matches = []
        for old, receipt in old_rf:
            signature = receipt["signature"]
            if (old["line"], old["context"], signature["count"], signature["family"], signature["params"]) != (record["line"], record["context"], record["K"], task["family"], task["params"]):
                continue
            assert columns == receipt["input_columns"]
            assert receipt["training_rows_sha256"] == binding["training_rows_sha256"]
            matches.append(old)
        record["reuse_raw_refit"] = matches[0] if matches else None
        reused += bool(matches)
    write(ROOT / "inputs/SELECTED_BY_SIZE.json", records)
    write(ROOT / "inputs/RANKING_BINDINGS.json", ranking_bindings)
    write(ROOT / "inputs/RANKING_EXTENSIONS_REQUIRED.json", extensions_needed)
    pd.DataFrame(score_rows).to_csv(ROOT / "outputs/TRAINING_CANDIDATE_SCORES.csv", index=False)
    pd.DataFrame([{"line": r["line"], "context": r["context"], "K": r["K"], "family": r["candidate"]["task"]["family"], "mean_AP": r["candidate"]["mean_AP"], "mean_Brier": r["candidate"]["mean_Brier"], "candidate_id": r["candidate"]["candidate_id"], "params": json.dumps(r["candidate"]["task"]["params"], sort_keys=True), "reused_raw_refit": bool(r["reuse_raw_refit"])} for r in records]).to_csv(ROOT / "outputs/TRAINING_SIZE_PATH.csv", index=False)
    receipt = {"status": "PASS_ALL_K_TRAINING_SELECTION_AND_SOURCE_ALIGNMENT", "protocol_sha256": protocol_sha, "eligible_candidates_verified": checked, "within_K_winners": len(records), "writer_independent_winners_matched": len(records), "nine_node_winners_matched": len(nine_map), "selected_inner_OOF_banks_verified": len(records), "logical_refits": len(records), "reusable_raw_refits": reused, "new_raw_refits_required": len(records) - reused, "full_training_ranking_extensions": extensions_needed, "upper_bound_new_RFE_operator_fits": 2 * len(extensions_needed), "new_target_metrics": 0, "new_predictive_fits": 0, "source_writer_path": str(writer_path), "source_writer_sha256": e.sha(writer_path), "nine_selection_sha256": e.sha(nine_path), "files": {str(path.relative_to(ROOT)): e.sha(path) for path in [ROOT / "inputs/SELECTED_BY_SIZE.json", ROOT / "inputs/RANKING_BINDINGS.json", ROOT / "inputs/RANKING_EXTENSIONS_REQUIRED.json", ROOT / "outputs/TRAINING_CANDIDATE_SCORES.csv", ROOT / "outputs/TRAINING_SIZE_PATH.csv"]}, "seconds": time.time() - started}
    write(ROOT / "outputs/PREPARATION_ACCEPTANCE.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("files", "full_training_ranking_extensions")}), flush=True)


if __name__ == "__main__":
    main()
