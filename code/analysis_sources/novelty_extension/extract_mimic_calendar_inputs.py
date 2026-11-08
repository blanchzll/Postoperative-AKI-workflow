"""Frozen calendar-anchor sensitivity inputs; no model training or scoring."""
from pathlib import Path
import os,json,csv,hashlib,datetime,collections,resource,fcntl,time
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
import pandas as pd,numpy as np
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");Q=A/"qualification";D=A/"mimic_calendar";M=Path("/home/lei/physionet.org/files/mimic")
def read(p):return json.loads(Path(p).read_text())
def save(p,j):
 p=Path(p);z=p.with_name(p.name+".tmp");z.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(z,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def main():
 D.mkdir(mode=0o700,exist_ok=True)
 with (D/"INPUT_RUNNER.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (D/"INPUT_QUALIFICATION.json").exists()
  pre=read(Q/"MIMIC_LAB_AND_ENDPOINT_PRELOCK.json");assert sha(Q/"MIMIC_LAB_AND_ENDPOINT_PRELOCK.json")=="5fda7a45acc64165f0ecb6b93b9dc2736f228ef34a4f788aef45dc10ac7d20ce"
  p=Q/"PRIVATE_MIMIC_FIRST_RESECTION_CANDIDATES.csv";assert sha(p)==pre["candidate_csv_sha256"]
  c=pd.read_csv(p,dtype={"subject_id":str,"hadm_id":str})
  c=c.set_index("subject_id",drop=False);assert c.index.is_unique
  subjects=set(c.subject_id.astype(int));items={int(r["itemid"]):r["label"] for r in pre["selected_dictionary_entries"]}
  day={str(s):str(v) for s,v in c.procedure_date.items()}
  lo={s:(datetime.date.fromisoformat(v)-datetime.timedelta(days=30)).isoformat() for s,v in day.items()}
  hi={s:(datetime.date.fromisoformat(v)+datetime.timedelta(days=8)).isoformat() for s,v in day.items()}
  cols=["labevent_id","subject_id","hadm_id","itemid","charttime","storetime","valuenum","valueuom"];kept=[];scanned=0;start=time.time()
  raw=M/"hosp/labevents.csv.gz";initial={"bytes":raw.stat().st_size,"mtime_ns":raw.stat().st_mtime_ns}
  for ch in pd.read_csv(raw,usecols=cols,dtype={"labevent_id":str,"subject_id":np.int64,"hadm_id":str,"itemid":np.int32,"charttime":str,"storetime":str,"valuenum":str,"valueuom":str},chunksize=500000,low_memory=False):
   scanned+=len(ch);z=ch.loc[ch.subject_id.isin(subjects)&ch.itemid.isin(items)].copy()
   if len(z):
    z["subject_id"]=z.subject_id.astype(str);d=z.charttime.fillna("").str[:10]
    z=z.loc[(d>=z.subject_id.map(lo))&(d<z.subject_id.map(hi))]
    if len(z):kept.append(z)
   if scanned%2500000==0:save(D/"INPUT_PROGRESS.json",{"phase":"stream_existing_MIMIC_labs","raw_rows_scanned":scanned,"kept_rows":sum(len(z) for z in kept),"elapsed_seconds":time.time()-start,"pid":os.getpid()})
  labs=pd.concat(kept,ignore_index=True) if kept else pd.DataFrame(columns=cols)
  assert labs.labevent_id.is_unique
  labs.to_csv(D/"PRIVATE_SELECTED_LAB_EVENTS.csv",index=False)
  counters=collections.Counter();unitcounts=collections.Counter();mapped=[]
  featuremap={"Creatinine":"PreopCr","Urea Nitrogen":"PreopBUN","Albumin":"PreopAlb","Hemoglobin":"PreopHb"}
  for r in labs.itertuples(index=False):
   label=items[r.itemid];unit=str(r.valueuom).strip().lower().replace("µ","u").replace("μ","u")
   unitcounts[label+"|"+unit]+=1
   try:v=float(r.valuenum)
   except(ValueError,TypeError):counters["nonnumeric"]+=1;continue
   if not np.isfinite(v) or v<=0:counters["not_positive_finite"]+=1;continue
   if label=="Creatinine":
    factor=88.4 if unit=="mg/dl" else 1 if unit=="umol/l" else None
   elif label=="Urea Nitrogen":factor=.357 if unit=="mg/dl" else 1 if unit=="mmol/l" else None
   else:factor=10 if unit=="g/dl" else 1 if unit=="g/l" else None
   if factor is None:counters["unknown_unit_"+label]+=1;continue
   mapped.append({"subject_id":r.subject_id,"hadm_id":r.hadm_id,"itemid":int(r.itemid),"labevent_id":int(r.labevent_id),"field":featuremap[label],"value_SI":v*factor,"charttime":str(r.charttime),"storetime":str(r.storetime)})
  by=collections.defaultdict(list)
  for r in mapped:by[r["subject_id"]].append(r)
  features=[];metadata=[];flow=collections.Counter()
  for s,case in c.iterrows():
   flow["phenotype_candidates"]+=1;op=day[s];cut=op+" 00:00:00";lower=lo[s]+" 00:00:00"
   after=(datetime.date.fromisoformat(op)+datetime.timedelta(days=1)).isoformat()+" 00:00:00";end=hi[s]+" 00:00:00"
   recs=by.get(s,[]);baseline={}
   for field in ["PreopCr","PreopBUN","PreopAlb","PreopHb"]:
    eligible=[v for v in recs if v["field"]==field and lower<=v["charttime"]<cut and len(v["storetime"])>=10 and v["storetime"]<cut]
    if eligible:baseline[field]=max(eligible,key=lambda r:(r["charttime"],r["storetime"],r["labevent_id"]))
   post=[v for v in recs if v["field"]=="PreopCr" and after<=v["charttime"]<end]
   if "PreopCr" not in baseline:flow["no_qualifying_pre_day0_baseline_Cr"]+=1;continue
   if not post:flow["no_observed_day1_through7_Cr"]+=1;continue
   if case.Sex not in ["F","M"]:flow["sex_not_F_M"]+=1;continue
   base=baseline["PreopCr"]["value_SI"];positive=[v for v in post if v["value_SI"]>=1.5*base];y=int(bool(positive))
   grp=hashlib.sha256(("MIMIC3.1|subject|"+s).encode()).hexdigest()
   record="MIMIC3.1|calendar|"+str(case.hadm_id)+"|"+op
   x={"record_key":record,"group_hash":grp,"Age":float(case.Age),"Sex":"1" if case.Sex=="M" else "0",**{f:baseline[f]["value_SI"] if f in baseline else None for f in ["PreopCr","PreopBUN","PreopAlb","PreopHb"]}}
   features.append(x);metadata.append({"record_key":record,"group_hash":grp,"subject_id":s,"hadm_id":str(case.hadm_id),"procedure_date":op,"site":case.site,"y_scr_calendar":y,"post_Cr_n":len(post),"first_recorded_positive_charttime":min(v["charttime"] for v in positive) if positive else None,"last_post_Cr_charttime":max(v["charttime"] for v in post),"baseline_Cr_charttime":baseline["PreopCr"]["charttime"],"baseline_Cr_storetime":baseline["PreopCr"]["storetime"],"anonymized_age_proxy":bool(case.anchor_age_over89_proxy),"center":"MIMIC_BIDMC"})
   flow["qualified_calendar_measured_records"]+=1;flow["events_calendar_ratio"]+=y
  x=pd.DataFrame(features);meta=pd.DataFrame(metadata)
  x.to_csv(D/"PRIVATE_COMMON6_FEATURES.csv",index=False);meta.to_csv(D/"PRIVATE_METADATA.csv",index=False)
  SIX=["Age","Sex","PreopCr","PreopBUN","PreopAlb","PreopHb"]
  assert len(x) and set(x.columns)==set(SIX)|{"record_key","group_hash"} and x.record_key.is_unique and x.group_hash.is_unique
  assert x[SIX].notna().any().all(),"Source entire feature column unobserved: noautomatic reduced-field scoring"
  assert len(x)==len(meta) and x.record_key.tolist()==meta.record_key.tolist()
  assert meta.y_scr_calendar.isin([0,1]).all()
  for r in metadata:assert r["baseline_Cr_charttime"]<r["procedure_date"]+" 00:00:00" and r["baseline_Cr_storetime"]<r["procedure_date"]+" 00:00:00"
  assert raw.stat().st_size==initial["bytes"] and raw.stat().st_mtime_ns==initial["mtime_ns"]
  for v in SIX:
   if v!="Sex":assert np.isfinite(x[v].dropna()).all()
  result={"status":"PASS_ONLY_PREDECLARED_CALENDAR_ANCHORED_SIX_INPUT_SENSITIVITY","protocol_sha256":sha(A/"ANALYSIS_PROTOCOL_LOCK.json"),"lab_prelock_sha256":sha(Q/"MIMIC_LAB_AND_ENDPOINT_PRELOCK.json"),"raw_labevents_local_sha256":sha(raw),"raw_labevents_bytes":initial["bytes"],"raw_rows_scanned":scanned,"selected_rows":len(labs),"valid_mapped_lab_rows":len(mapped),"units_and_exclusion_counts":dict(counters),"source_unit_counts":dict(unitcounts),"flow":dict(flow),"n":len(x),"events":int(meta.y_scr_calendar.sum()),"patient_groups":meta.group_hash.nunique(),"ordinary_missing_n":{v:int(x[v].isna().sum()) for v in SIX},"site_counts":meta.site.value_counts().to_dict(),"observed_post_Cr_counts":{"min":int(meta.post_Cr_n.min()),"median":float(meta.post_Cr_n.median()),"max":int(meta.post_Cr_n.max())},"literal_original_exact_clock_validation":False,"calendar_day0_not_actual_incision_or_end":True,"target":"Observed SCr>=1.5*last known baseline in calendar days1..7 after billed procedure date; excludes day0, does not assert complete KDIGO7days.","baseline_availability":"Last prior30day measurement; both chart and storetime before billed procedure day starts. Deidentified age proxy flagged.","source_only_imputer_required":True,"predictive_fit_or_target_calibration":False,"nonsource_labels_or_dates_not_in_features":True,"new_case_source_separate_from_original_Chinese_or_INSPIRE":True,"elapsed_seconds":time.time()-start,"peak_RSS_MiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"files":{n:sha(D/n) for n in ["PRIVATE_COMMON6_FEATURES.csv","PRIVATE_METADATA.csv","PRIVATE_SELECTED_LAB_EVENTS.csv"]}}
  save(D/"INPUT_QUALIFICATION.json",result);print(json.dumps({"status":result["status"],"n":len(x),"events":result["events"],"missing":result["ordinary_missing_n"],"raw_rows_scanned":scanned,"elapsed_seconds":result["elapsed_seconds"]}),flush=True)
if __name__=="__main__":main()
