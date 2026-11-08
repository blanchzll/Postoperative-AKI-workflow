"""Prepare source-bound paired clinical text and original folds only; no models or performance evaluation."""
from pathlib import Path
from decimal import Decimal
from collections import Counter
import csv,datetime,hashlib,json
R=Path(__file__).resolve().parents[2]
P=R.parents[1]
B=R/"outputs/v4_development_inputs_20261018"
O=R/"outputs/llm_supplement_20261028_v1/data_preparation_v1"
UNITS={"Age":"years","PreopHospitalDays":"days","Height":"cm","Weight":"kg","BMI":"kg/m2","WeightLoss":"kg","PreopWBC":"10^9/L","PreopHb":"g/L","PreopAlb":"g/L","PreopALT":"U/L","PreopBUN":"mmol/L","PreopCr":"umol/L","PreopGlucose":"mmol/L","OperationTime":"minutes","IntraopBloodLoss":"mL","IntraopTransfusion":"mL","IntraopFluid":"mL; excludes blood products","IntraopHES":"mL","IntraopDextran":"mL","IntraopGelatin":"mL","IntraopPlasma":"mL"}
MAPS={"Sex":{"0":"female","1":"male"},"TumorSite":{"0":"gastric","1":"colon","2":"rectum"},"TumorGroup":{"1":"gastric","2":"colorectal"},"SurgicalApproach":{"1":"open","2":"laparoscopic","3":"laparoscopic converted to open","4":"robotic","robotic_text":"robotic documented as source text","mixed_1_2_text":"mixed open/laparoscopic source text; sequence unspecified"}}
BINARY={"PreviousAbdominalSurgery","OtherMalignancy","Diabetes","Hypertension","CerebrovascularDisease","HeartDisease","Smoking","Alcohol","NeoadjuvantChemo","IntraopDiuretics","IntraopVasoactive"}
BANNED={"excel_row","record_key","group_hash","center","surgery_date","y","stage","rrt","added_rrt"}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
def table(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def key(v):
    d=Decimal(str(v));assert d==d.to_integral_value()
    return str(int(d))
def main():
    assert B.resolve()==(P/"outputs/full_data_audit_20261025_v1/candidate_release/inputs").resolve()
    O.mkdir(parents=True,exist_ok=True,mode=0o700)
    assert sha(B/"SOURCE_LOCK.json")=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
    lock=json.loads((B/"SOURCE_LOCK.json").read_text())
    required=["FEATURE_SPECS.json","paired_metadata.csv","paired_preop_features.csv","paired_endop_features.csv","geography_paired_resampling.json","temporal_paired_resampling.json"]
    for name in required:assert sha(B/name)==lock["input_sha256"][name],name
    meta=table(B/"paired_metadata.csv")
    lookup={key(r["excel_row"]):r for r in meta}
    assert len(lookup)==len(meta)==2428
    assert sum(int(r["y"]) for r in meta)==138
    assert len({r["group_hash"] for r in meta})==2411
    assert all(r["y"] in ["0","1"] for r in meta)
    specs=json.loads((B/"FEATURE_SPECS.json").read_text())
    source_refs=[P/"outputs/full_data_audit_20261025_v1/RAW_110_COLUMN_PROFILES.csv",P/"raw/legacy_analysis/总体分析/data_dictionary.csv",B/"FEATURE_SPECS.json",B/"SOURCE_LOCK.json"]
    # Only documented category meanings are translated; all original codes remain visible.
    undocumented=Counter();missing={};input_files={}
    for stage,n in [("preop",27),("endop",38)]:
        spec=specs[stage];fields=spec["num"]+spec["cat"]
        assert len(fields)==len(set(fields))==n and not BANNED.intersection(f.lower() for f in fields)
        if stage=="preop":assert set(fields)==set(specs["preop"]["num"]+specs["preop"]["cat"])
        rows=table(B/("paired_"+stage+"_features.csv"))
        by_id={key(row["excel_row"]):row for row in rows}
        assert set(by_id)==set(lookup) and len(rows)==2428
        missing[stage]=Counter()
        out=O/(stage+"_clinical_text.jsonl")
        with out.open("w") as stream:
            for rid in sorted(by_id,key=int):
                row=by_id[rid];lines=[]
                for field in fields:
                    value=row[field].strip()
                    label=field
                    if field in UNITS:label+=" ["+UNITS[field]+"]"
                    if field=="PreviousAbdominalSurgery":label+=" [excludes hernia, gallbladder, appendix and Caesarean procedures]"
                    if value in ["","NA","NaN","nan"]:
                        text="not recorded";missing[stage][field]+=1
                    elif field in spec["num"]:
                        assert Decimal(value).is_finite(),field
                        text=value # Preserve the accepted CSV literal and numerical precision.
                    else:
                        assert value in spec["cat_levels"][field],(field,value)
                        if field in MAPS:text=MAPS[field][value]+" [source code "+value+"]"
                        elif field in BINARY and value in ["0","1"]:text=("yes" if value=="1" else "no")+" [source code "+value+"]"
                        elif field=="ASA":text="source ASA category "+value
                        else:
                            text="source category "+value+"; clinical meaning not specified"
                            undocumented[(field,value)]+=1
                    lines.append(label+": "+text)
                assert len(lines)==n
                timepoint="before surgery" if stage=="preop" else "by the end of surgery"
                instruction="Predict the study-defined 7-day postoperative AKI outcome using only the following information available "+timepoint+". Not recorded denotes missing information. Do not guess the clinical meaning of an unspecified source category. Answer 0 for no AKI or 1 for AKI."
                body=instruction+"\n"+"\n".join(lines)
                assert all("\n"+k+":" not in body for k in BANNED)
                stream.write(json.dumps({"record_key":rid,"clinical_text":body},ensure_ascii=False)+"\n")
        input_files[str(out)]=sha(out)
    shared_fields=specs["preop"]["num"]+specs["preop"]["cat"]
    pre={key(r["excel_row"]):r for r in table(B/"paired_preop_features.csv")}
    end={key(r["excel_row"]):r for r in table(B/"paired_endop_features.csv")}
    assert all(pre[rid][field]==end[rid][field] for rid in lookup for field in shared_fields),"Paired shared predictor literals differ"
    folds_out=[];partitions={};engineering=[]
    for line in ["geography","temporal"]:
        field=line+"_partition_v2"
        development={rid for rid,row in lookup.items() if row[field]=="development"}
        evaluation={rid for rid,row in lookup.items() if row[field]=="validation"}
        dg={lookup[r]["group_hash"] for r in development};eg={lookup[r]["group_hash"] for r in evaluation}
        assert not development.intersection(evaluation) and not dg.intersection(eg)
        folds=[f for f in json.loads((B/(line+"_paired_resampling.json")).read_text()) if f["scheme"]=="final_development_inner"]
        assert len(folds)==3
        coverage=Counter()
        for f in folds:
            train={key(x) for x in f["train_excel_rows"]};assessment={key(x) for x in f["assessment_excel_rows"]}
            assert train|assessment==development and not train.intersection(assessment)
            assert not {lookup[r]["group_hash"] for r in train}.intersection(lookup[r]["group_hash"] for r in assessment)
            coverage.update(assessment)
            folds_out.append({"line":line,"original_scheme":f["scheme"],"fold":f["fold"],"train_record_keys":sorted(train,key=int),"assessment_record_keys":sorted(assessment,key=int),"n_train":len(train),"train_events":sum(int(lookup[r]["y"]) for r in train),"n_assessment":len(assessment),"assessment_events":sum(int(lookup[r]["y"]) for r in assessment)})
        assert set(coverage)==development and set(coverage.values())=={1}
        partitions[line]={v:sum(row[field]==v for row in meta) for v in sorted({row[field] for row in meta})}
        if line=="geography":
            train=set(folds_out[0]["train_record_keys"]);used=set()
            for outcome in ["0","1"]:
                chosen=[]
                for rid in sorted(train,key=int):
                    if lookup[rid]["y"]==outcome and lookup[rid]["group_hash"] not in used:
                        chosen.append(rid);used.add(lookup[rid]["group_hash"])
                    if len(chosen)==2:break
                assert len(chosen)==2;engineering+=chosen
    with (O/"labels_and_partitions.csv").open("w",newline="") as f:
        columns=["record_key","group_hash","center","y","geography_partition_v2","temporal_partition_v2"]
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for rid in sorted(lookup,key=int):
            row=lookup[rid];writer.writerow({"record_key":rid,**{k:row[k] for k in columns if k!="record_key"}})
    save(O/"ORIGINAL_OOF_FOLDS.json",folds_out)
    save(O/"ENGINEERING_RECORDS.json",{"record_keys":engineering,"source":"geography final_development_inner fold0 train only","balanced4_records_for_computational_checks_only":True,"not_formal_training_sample":True,"not_a_clinical_performance_estimate":True})
    save(O/"SOURCE_CODEBOOK.json",{"units":UNITS,"documented_maps":MAPS,"binary_fields_with_documented0_no1_yes":sorted(BINARY),"source_reference_sha256":{str(p):sha(p) for p in source_refs},"unknown_codes_retained":{field+"="+value:count//2 for (field,value),count in undocumented.items()},"source_semantics_notes":["Comorbidities retains its literal source code because a0/1 clinical interpretation was not established by the inspected dictionary.","One observed Smoking2 and one Alcohol2 are preserved without guessing former/current status; stage-duplicated counts are divided by2.","TumorSite missing values remain missing even when TumorGroup indicates colorectal disease; no colon/rectum subtype is invented."]})
    files={str(p):sha(p) for p in O.iterdir() if p.is_file() and p.name!="DATA_PREPARATION_CHECK.json"}
    receipt={"status":"PASS_CURRENT_SOURCE_FIELDS_ORIGINAL_PATIENT_FOLDS_AND_LOSSLESS_TEXT_PREPARATION_ONLY","time":now(),"declared_parent_source_lock_sha256":"afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3","current_input_lock_sha256":sha(B/"SOURCE_LOCK.json"),"input_sha256":{str(B/n):sha(B/n) for n in required},"paired_records":2428,"events":138,"patient_groups":2411,"partitions":partitions,"field_counts":{"preop":27,"endop":38},"missing_counts":{k:dict(v) for k,v in missing.items()},"original_final_development_folds":6,"patient_train_assessment_overlap":0,"out_of_fold_assessment_coverage_once_per_development_record":True,"identity_center_date_outcome_RRT_fields_in_clinical_text":False,"outcomes_kept_separate_from_inputs":True,"numeric_source_literals_preserved":True,"paired_shared_preop_predictor_literals_identical":True,"prompt_body_shared_across_three_models":True,"chat_or_completion_wrappers_still_require_model_specific_qualification":True,"model_specific_tokenization_context_and_digit_scoring_qualified":False,"formal_training_started":False,"performance_evaluation_performed":False,"script_sha256":sha(Path(__file__)),"files":files}
    save(O/"DATA_PREPARATION_CHECK.json",receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["files","input_sha256","missing_counts"]},ensure_ascii=False))
if __name__=="__main__":main()
