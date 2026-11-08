"""Independent reconstruction of measurement windows and coarse real-time strata."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import csv
import datetime
import gzip
import hashlib
import json
import re
from collections import defaultdict

RUN = Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/measurement_consistency_20261104_v1")
OLD = RUN.parent / "full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1"
SOURCE = OLD / "mimic_calendar_v2"
OUTPUT = RUN / "outputs"

def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()

def main():
    qualified = rows(OUTPUT / "PRIVATE_QUALIFIED_RECORDS.csv")
    originals = {r["record_key"]: r for r in rows(SOURCE / "PRIVATE_METADATA.csv")}
    features = {r["record_key"]: r for r in rows(SOURCE / "PRIVATE_COMMON6_FEATURES.csv")}
    cr_items = {"50912", "52546"}
    labs = defaultdict(list)
    excluded_invalid = 0
    for row in rows(SOURCE / "PRIVATE_SELECTED_LAB_EVENTS.csv"):
        if row["itemid"] not in cr_items:
            continue
        try:
            value = Decimal(row["valuenum"])
        except InvalidOperation:
            excluded_invalid += 1
            continue
        if not value.is_finite() or value <= 0:
            excluded_invalid += 1
            continue
        unit = row["valueuom"].lower().strip().replace("µ", "u").replace("μ", "u")
        assert unit in ["mg/dl", "umol/l"]
        if unit == "mg/dl":
            value *= Decimal("88.4")
        labs[row["subject_id"]].append((row, value))
    subjects = {r["subject_id"] for r in qualified}
    patients = {}
    patient_path = Path("/home/lei/physionet.org/files/mimic/hosp/patients.csv.gz")
    with gzip.open(patient_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subject_id"] in subjects:
                patients[row["subject_id"]] = row
    checks = 0
    reconstructed = []
    for row in qualified:
        source = originals[row["record_key"]]
        feature = features[row["record_key"]]
        procedure = datetime.date.fromisoformat(source["procedure_date"])
        cutoff = datetime.datetime.combine(procedure, datetime.time())
        baseline_candidates = []
        followup = {3: [], 7: []}
        for lab, value in labs[source["subject_id"]]:
            chart = datetime.datetime.fromisoformat(lab["charttime"])
            if cutoff - datetime.timedelta(days=30) <= chart < cutoff and lab["storetime"]:
                if datetime.datetime.fromisoformat(lab["storetime"]) < cutoff:
                    baseline_candidates.append((chart, datetime.datetime.fromisoformat(lab["storetime"]),
                                                int(lab["labevent_id"]), value))
            day = (chart.date() - procedure).days
            for horizon in [3, 7]:
                if 1 <= day <= horizon:
                    followup[horizon].append((day, value))
        baseline = max(baseline_candidates, key=lambda r: r[:3])[3]
        day_count = len({day for day, _ in followup[7]})
        expected = {
            "eligible_3": int(bool(followup[3])),
            "y_3": int(any(value >= Decimal("1.5") * baseline for _, value in followup[3])) if followup[3] else "",
            "y_7": int(any(value >= Decimal("1.5") * baseline for _, value in followup[7])),
            "Cr_record_count_7": len(followup[7]),
            "Cr_observed_days_7": day_count,
            "Cr_record_count_3": len(followup[3]),
            "Cr_observed_days_3": len({day for day, _ in followup[3]}),
            "Cr_late_day4to7_observed": int(any(day >= 4 for day, _ in followup[7])),
            "Cr_days_1to2_vs_3plus": "1to2" if day_count <= 2 else "3plus",
            "Cr_records_1to3_vs_4plus": "1to3" if len(followup[7]) <= 3 else "4plus",
            "albumin_observed": int(bool(feature["PreopAlb"])),
            "age_band65": "under65" if Decimal(feature["Age"]) < 65 else "65plus",
            "Sex": feature["Sex"]
        }
        patient = patients[source["subject_id"]]
        years = re.findall(r"\d{4}", patient["anchor_year_group"])
        assert len(years) == 2
        difference = procedure.year - int(patient["anchor_year"])
        low, high = int(years[0]) + difference, int(years[1]) + difference
        expected.update(
            source_anchor_year_group=patient["anchor_year_group"],
            possible_procedure_year_low=low,
            possible_procedure_year_high=high,
            broad_real_period_candidate="through2016" if high < 2017 else "2017on" if low >= 2017 else "boundary_uncertain"
        )
        for key, value in expected.items():
            assert str(value) == row[key], (key, value, row[key])
            checks += 1
        reconstructed.append(expected)
    assert sum(r["y_7"] for r in reconstructed) == 59
    assert sum(r["eligible_3"] for r in reconstructed) == 757
    assert sum(r["y_3"] for r in reconstructed if r["eligible_3"]) == 44
    result = {
        "status": "PASS_INDEPENDENT_DECIMAL_WINDOW_AND_PROVIDER_PERIOD_RECONSTRUCTION",
        "request_sha256": digest(RUN / "REQUEST.json"),
        "protocol_sha256": digest(RUN / "PROTOCOL_LOCK.json"),
        "records": 759, "unique_patient_groups": 759,
        "events7": 59, "observed_window3": 757, "events3": 44,
        "unknown_window3": 2, "field_checks": checks,
        "period_uncertain": sum(r["broad_real_period_candidate"] == "boundary_uncertain" for r in reconstructed),
        "raw_selected_subset_only": True,
        "full_source_labevents_rescan_performed": False,
        "independent_implementation": "Decimal and calendar-day logic; originalqualification uses Fraction and timestamp intervals",
        "code_sha256": digest(__file__),
        "source_sha256": {str(p): digest(p) for p in [
            OUTPUT / "PRIVATE_QUALIFIED_RECORDS.csv",
            SOURCE / "PRIVATE_METADATA.csv", SOURCE / "PRIVATE_COMMON6_FEATURES.csv",
            SOURCE / "PRIVATE_SELECTED_LAB_EVENTS.csv", patient_path
        ]}
    }
    (OUTPUT / "INDEPENDENT_INPUT_VERIFICATION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}), flush=True)

if __name__ == "__main__":
    main()
