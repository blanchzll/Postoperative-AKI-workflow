"""Bind all27 inputs to audited source columns, coding, units and time evidence."""
from pathlib import Path
import json
import csv
import pandas as pd
from fit_scale import ROOT, PROJECT, e, read, write

OUT = ROOT / "outputs/selection_and_burden"


def main():
    audit = PROJECT / "outputs/full_data_audit_20261025_v1/RAW_110_COLUMN_PROFILES.csv"
    dictionary = PROJECT / "raw/legacy_analysis/总体分析/data_dictionary.csv"
    book_path = e.SOURCE / "outputs/llm_supplement_20261028_v1/data_preparation_v1/SOURCE_CODEBOOK.json"
    book = read(book_path)
    for path, expected in book["source_reference_sha256"].items():
        assert e.sha(path) == expected
    profiles = {r["column"]: r for r in csv.DictReader(audit.open()) if r["sheet"].lower() == "sheet1"}
    original_dictionary = {r["Column"]: r for r in csv.DictReader(dictionary.open())}
    data_path = e.legacy.INPUT / "preop_preop_features.csv"
    data = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    spec = e.legacy.spec_for("preop")
    mapping = {"Age": "N", "PreopHospitalDays": "Q", "Height": "Y", "Weight": "Z", "BMI": "AA", "WeightLoss": "AB", "PreopWBC": "CM", "PreopHb": "CN", "PreopAlb": "CO", "PreopALT": "CP", "PreopBUN": "CQ", "PreopCr": "CR", "PreopGlucose": "CS", "Sex": "M", "TumorSite": "B", "TumorGroup": "C", "PreviousAbdominalSurgery": "AC", "OtherMalignancy": "AD", "Comorbidities": "AE", "Diabetes": "AF", "Hypertension": "AG", "CerebrovascularDisease": "AH", "HeartDisease": "AI", "Smoking": "AJ", "Alcohol": "AK", "NeoadjuvantChemo": "AL", "ASA": "AM"}
    aliases = {"Sex": "Gender", "TumorSite": "GastricColorectal", "TumorGroup": "Gastrocolorectal", "ASA": "ASAGrade"}
    clarifications = {"source": "Direct author statements in this analysis chat, recorded without changing source cells", "HeartDisease": "HeartDisease 实际记录的是高血压以外的心脏病", "history_absence": "没有诊断编码就是无病史", "scope": "Interpretation only; no additional patients or recoding/retraining"}
    write(ROOT / "inputs/AUTHOR_HISTORY_CLARIFICATIONS.json", clarifications)
    sources = [audit, dictionary, book_path, e.legacy.INPUT / "FEATURE_SPECS.json", e.legacy.INPUT / "SOURCE_LOCK.json", data_path, PROJECT / "work/full_data_audit_20261025_v1/apply_authorized_rules.py", PROJECT / "work/full_data_audit_20261025_v1/verify_and_review.py", ROOT / "inputs/AUTHOR_HISTORY_CLARIFICATIONS.json"]
    source_hashes = {str(p): e.sha(p) for p in sources}
    basic = pd.read_csv(OUT / "FIELD_DICTIONARY.csv").set_index("field")
    rows = []
    for field in spec["num"] + spec["cat"]:
        original = original_dictionary.get(aliases.get(field, field), {})
        category_levels = spec["cat_levels"].get(field, [])
        meaning = "continuous numeric field"
        status = "documented_numeric_unit"
        if field in spec["cat"]:
            status = "frozen_categorical_levels"
            if field in book["documented_maps"]:
                meaning = json.dumps(book["documented_maps"][field], ensure_ascii=False, sort_keys=True)
            elif field in book["binary_fields_with_documented0_no1_yes"]:
                meaning = "0=无/否；1=有/是"
            elif field == "Comorbidities":
                meaning = "0/1 literal source levels; source header records medical-history presence. Author confirms absence of diagnosis coding as no relevant history. Specific aggregate disease coverage remains incompletely documented."
                status = "author_negative_history_rule_and_literal_source_levels"
            elif field == "ASA":
                meaning = "1=I,2=II,3=III,4=IV documented in original dictionary; accepted frozen spec also retains5. Definition of level5 is absent from that source dictionary."
                status = "level5_semantics_retained_as_source_unknown"
        if field in ["Smoking", "Alcohol"]:
            meaning += "; source2 retained with unknown clinical semantics"
            status = "0_1_documented_2_unknown"
        if field == "HeartDisease":
            meaning += "; author definition excludes hypertension"
        if field.startswith("Preop") and field != "PreopHospitalDays":
            window = "Source header specifies the most recent preoperative measurement. Exact lookback interval, draw time, and result-release clock are unknown."
            timing_basis = "raw source column header and accepted preoperative FEATURE_SPECS"
            dependence = "Existing preoperative laboratory result for this analyte; local order/panel/shared sampling and marginal collection cost unknown."
        elif field == "PreopHospitalDays":
            window = "Recorded source preoperative stayQ, with authorized I-H date/formula reconciliations for identified cases. Scoring anchor is the existing preoperative task; earlier planned versus actual operation-date availability is unknown."
            timing_basis = "rawQ and current source correction/verification scripts"
            dependence = "Use accepted recorded/correctedQ. Reconstruction from dates requires AdmissionDateH and SurgeryDateI. Preserved source durations and repaired date-derived records coexist."
        elif field == "BMI":
            window = "Recorded source BMI and authorized recalculation when qualified Height/Weight exist. Exact anthropometric measurement clock unknown."
            timing_basis = "rawAA and current authorized BMI repair/release"
            dependence = "Use accepted recorded/correctedBMI. Recalculation requires HeightY and WeightZ; these can be existing chart values."
        elif field == "WeightLoss":
            window = "Source body-weight-loss history in kg. Historical lookback length and reference-weight date unknown."
            timing_basis = "sourceAB and dictionary unit"
            dependence = "Recorded weight-loss history. A fresh computation would need historical/current weights and a time window whose source definition is unknown."
        elif field in ["Height", "Weight"]:
            window = "Included as preoperative anthropometric input by frozen spec. Exact measurement/recording clock unknown."
            timing_basis = "source header and accepted preoperative FEATURE_SPECS"
            dependence = "Existing anthropometric record or local measurement; actual marginal staff time unknown."
        elif field in ["TumorSite", "TumorGroup"]:
            window = "Recorded tumour/surgical category assigned to preoperative input spec. Exact original documentation and pathology-confirmation clocks unknown."
            timing_basis = "sourceB/C mappings and accepted preoperative FEATURE_SPECS"
            dependence = "Recorded tumour category. TumorSite missing remains missing; site is not reconstructed from TumorGroup."
        elif field == "ASA":
            window = "Included as preoperative assessment in frozen input spec. Exact assessment/sign-off clock unknown."
            timing_basis = "sourceAM, dictionaryASAGrade and accepted preoperative FEATURE_SPECS"
            dependence = "Recorded anaesthetic assessment, with local availability and marginal workload unknown."
        elif field in ["Age", "Sex"]:
            window = "Recorded demographics in preoperative input spec. Exact source record-entry clock unknown."
            timing_basis = "sourceN/M, documented coding/unit and accepted preoperative FEATURE_SPECS"
            dependence = "Existing demographic record; no new demographic derivation assumed."
        else:
            window = "Relevant history or preoperative treatment history in frozen preoperative spec. Exact source ascertainment and record-entry clocks unknown."
            timing_basis = "raw source history header, source dictionary and accepted preoperative FEATURE_SPECS"
            dependence = "Existing medical-history/treatment record under author absence-of-diagnosis rule; local ascertainment effort unknown."
        observed = sorted(set(data[field]) - {"", "nan", "NA", "None"}) if field in spec["cat"] else []
        if field in spec["cat"]:
            assert set(observed).issubset(set(category_levels)), (field, observed, category_levels)
        rows.append({"field": field, "type": "numeric" if field in spec["num"] else "categorical", "normalized_source_file": str(data_path), "normalized_source_sha256": source_hashes[str(data_path)], "source_excel_sheet": "Sheet1", "source_excel_column": mapping[field], "source_excel_header": profiles[mapping[field]]["header"], "source_description": original.get("Description", ""), "unit": book["units"].get(field, "category"), "frozen_category_levels": json.dumps(category_levels), "observed_accepted_category_levels": json.dumps(observed), "encoding_semantics": meaning, "encoding_evidence_status": status, "preoperative_window_or_anchor": window, "availability_evidence": timing_basis, "individual_record_or_result_clock": "unknown", "actual_input_dependency": dependence, "workflow_category": basic.loc[field, "workflow_category"], "conditional_reconstruction_dependency_union": basic.loc[field, "primitive_dependencies"], "source_evidence_hashes": json.dumps(source_hashes, ensure_ascii=False, sort_keys=True)})
    pd.DataFrame(rows).to_csv(OUT / "FIELD_SOURCE_ENCODING_AVAILABILITY.csv", index=False)
    burden = pd.read_csv(OUT / "INPUT_DEPENDENCIES_AND_WORKFLOWS.csv")
    interpretation = burden[["line", "K", "primitive_dependency_count", "primitive_dependencies"]].rename(columns={"primitive_dependency_count": "conditional_BMI_and_stay_reconstruction_dependency_count", "primitive_dependencies": "conditional_BMI_and_stay_reconstruction_dependency_union"})
    interpretation["accepted_recorded_model_fields"] = interpretation.K
    interpretation["dependency_scenario"] = "Reuse accepted recorded fields; extra Height/Weight or admission/operation date dependencies apply when reconstructing BMI or stay. Existing recorded values can avoid new collection."
    interpretation["marginal_collection_cost"] = "unknown"
    interpretation.to_csv(OUT / "INPUT_DEPENDENCY_INTERPRETATION.csv", index=False)
    write(OUT / "ENCODING_AND_AVAILABILITY_ACCEPTANCE.json", {"status": "PASS_ALL27_FIELD_SOURCE_ENCODING_UNITS_AND_EXPLICIT_TIMING_EVIDENCE", "fields": len(rows), "numeric_fields": len(spec["num"]), "categorical_fields": len(spec["cat"]), "accepted_categories_within_frozen_spec": True, "new_patient_data": 0, "retraining": 0, "historical_lookback_and_result_release_clocks_retained_unknown": True, "recorded_vs_reconstructed_BMI_and_stay_distinguished": True, "sources": source_hashes, "code_sha256": e.sha(__file__), "files": {name: e.sha(OUT / name) for name in ["FIELD_SOURCE_ENCODING_AVAILABILITY.csv", "INPUT_DEPENDENCY_INTERPRETATION.csv"]}})
    print("PASS all27 field-source/encoding/time/conditional-dependency records")


if __name__ == "__main__":
    main()
