from pathlib import Path
import json,hashlib,gzip,datetime,os,concurrent.futures,re
import numpy as np
import pandas as pd
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
O=R/"outputs/public_patient_qualification_20261024_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
m=json.loads((O/"MIMIC_PATIENT_AUDIT.json").read_text());e=json.loads((O/"EICU_PATIENT_AUDIT.json").read_text())
sources={**m["source_files"],**e["source_files"]}
def check(it):
 p,q=it;assert Path(p).stat().st_size==q["bytes"];h=sha(p);assert h==q["sha256"],p
 return p,h
with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(os.sched_getaffinity(0)),len(sources))) as pool:
 verified=dict(pool.map(check,sources.items()))
for fn in ["MIMIC_LAUNCH.json","eICU_LAUNCH.json"]:
 own=json.loads((O/fn).read_text());p=Path(f"/proc/{own['pid']}/stat")
 assert not p.exists() or p.read_text().split(") ",1)[1].split()[19]!=own["start_ticks"],fn
p=pd.read_csv(O/"PRIVATE_MIMIC_FIRST_TARGET_OPERATIONS.csv")
c=pd.read_csv(O/"PRIVATE_MIMIC_MEASUREMENT_COVERAGE.csv")
z=pd.read_csv(O/"PRIVATE_MIMIC_WINDOWED_LABS.csv.gz",parse_dates=["time","operation_date"])
assert len(p)==p.subject_id.nunique()==1293
assert set(c.subject_id)==set(p.subject_id)
stats={k:0 for k in c.columns if k.startswith("Preop") or k in ["Sex_available","postoperative_calendar_day1_7_SCr","day0_SCr_ambiguous"]}
rows=[]
for person in p.itertuples():
 q=z.loc[z.subject_id==person.subject_id].copy()
 q=q.loc[pd.to_numeric(q.valuenum,errors="coerce").gt(0)]
 date=pd.Timestamp(person.operation_date)
 before=q.time.lt(date);after=q.time.ge(date+pd.Timedelta(days=1))&q.time.lt(date+pd.Timedelta(days=8))
 rec={"subject_id":person.subject_id,"Sex_available":person.gender in ["M","F"]}
 for name in ["PreopCr","PreopBUN","PreopAlb","PreopHb"]:rec[name+"_before_day0"]=bool((before&(q.feature==name)).any())
 rec["postoperative_calendar_day1_7_SCr"]=bool((after&(q.feature=="PreopCr")).any())
 rec["day0_SCr_ambiguous"]=bool((q.time.ge(date)&q.time.lt(date+pd.Timedelta(days=1))&(q.feature=="PreopCr")).any())
 rows.append(rec)
cc=pd.DataFrame(rows).set_index("subject_id").sort_index()
old=c.set_index("subject_id").sort_index()
for col in cc.columns:assert cc[col].tolist()==old[col].tolist(),col
for col in stats:stats[col]=int(cc[col].sum())
stats["baseline_and_later_SCr"]=int((cc.PreopCr_before_day0&cc.postoperative_calendar_day1_7_SCr).sum())
stats["all_six_observed_and_later_SCr"]=int(cc[["Sex_available","PreopCr_before_day0","PreopBUN_before_day0","PreopAlb_before_day0","PreopHb_before_day0","postoperative_calendar_day1_7_SCr"]].all(axis=1).sum())
assert stats==m["coverage"]
# Distinguish actual documented postoperative GI cancer from no treatment-name match.
E=Path("/home/lei/physionet.org/files/eicu/eicu-collaborative-research-database-2.0")
persons=pd.read_csv(E/"patient.csv.gz",usecols=["patientunitstayid","uniquepid","hospitalid","age"])
age=pd.to_numeric(persons.age.astype(str).str.replace(">","",regex=False),errors="coerce")
persons=persons.loc[age.ge(18)]
matched=[];pattern=r"^gastrointestinal\|post-GI surgery\|s/p surgery for cancer\|(?:colon|gastric|rectal)(?:\||$)"
for ch in pd.read_csv(E/"diagnosis.csv.gz",usecols=["patientunitstayid","diagnosisstring","diagnosisoffset"],chunksize=250000):
 q=ch.loc[ch.diagnosisstring.fillna("").str.contains(pattern,case=False,regex=True)]
 q=q.loc[q.patientunitstayid.isin(persons.patientunitstayid)]
 if len(q):matched.append(q)
post=pd.concat(matched,ignore_index=True) if matched else pd.DataFrame()
post.to_csv(O/"PRIVATE_EICU_DOCUMENTED_POST_GI_CANCER_DIAGNOSES.csv",index=False)
selected=persons.loc[persons.patientunitstayid.isin(post.patientunitstayid)]
counts={"documented_post_GI_cancer_diagnosis_rows":len(post),"adult_ICU_stays":len(selected),
 "unique_people":int(selected.uniquepid.nunique()),"hospitals":int(selected.hospitalid.nunique()),
 "exact_operation_clock_present_in_examined_diagnosis_field":False,
 "not_verified_resection_or_elective_histology_or_original_clock":True}
(O/"EICU_POSTOPERATIVE_DOCUMENTATION_COUNTS.json").write_text(json.dumps(counts,indent=2)+"\n")
receipt={"status":"PASS_patient_level_qualification_review_source_hashes_and_coverage",
 "time_beijing":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 "source_hashes_reverified":verified,"MIMIC_1293_individual_coverage_rederived":True,"MIMIC_counts":stats,
 "EICU_documented_postoperative_status":counts,
 "old_zero_treatment_match_does_not_mean_zero_GI_operations":True,
 "earlier1323_is_broad_candidate_screen_not_confirmed_target_cohort":True,
 "current_original9_27_38_and_selective_evaluation_eligible":False,
 "MIMIC_calendar_window_common6_support_may_be_feasible_but_requires_separate_endpoint_lock":True,
 "public_clinical_AKI_event_counts_or_performance_fabricated":False,"owners_exited":True,
 "code_sha256":sha(__file__),"source_audit_sha256":sha(O/"MIMIC_PATIENT_AUDIT.json"),
 "eicu_audit_sha256":sha(O/"EICU_PATIENT_AUDIT.json"),
 "whole_project_complete":False}
(O/"REVIEW_ACCEPTANCE.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="source_hashes_reverified"},ensure_ascii=False))
