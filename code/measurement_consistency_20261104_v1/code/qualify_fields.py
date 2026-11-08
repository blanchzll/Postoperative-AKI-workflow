"""Qualify existing frozen MIMIC inputs and counts before any new comparative performance."""
from pathlib import Path
import csv,gzip,json,hashlib,datetime,collections,fcntl,os,re,shutil
from fractions import Fraction
N=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/measurement_consistency_20261104_v1")
R=N.parent/"full_data_20261025_postslot";A=R/"outputs/novelty_strengthening_20261101_v1";D=A/"mimic_calendar_v2";O=N/"outputs"
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def numeric(v):
 try:return Fraction(v)
 except (ValueError,ZeroDivisionError):return None
def main():
 with (N/"OWNER.lock").open("a+") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert sha(N/"REQUEST.json")=="13038849a674c5e380dab1249f52a86bfc3e169278ec2372ce38981362d37caf"
  assert not (N/"PROTOCOL_LOCK.json").exists()
  features={r["record_key"]:r for r in rows(D/"PRIVATE_COMMON6_FEATURES.csv")}
  cases=rows(D/"PRIVATE_METADATA.csv");assert len(cases)==759 and len({r["group_hash"] for r in cases})==759
  contract=json.loads((A/"qualification/MIMIC_LAB_AND_ENDPOINT_PRELOCK.json").read_text())
  crids={r["itemid"] for r in contract["selected_dictionary_entries"] if r["label"]=="Creatinine"}
  raw=rows(D/"PRIVATE_SELECTED_LAB_EVENTS.csv");by=collections.defaultdict(list);unknown_units=collections.Counter()
  for r in raw:
   if r["itemid"] not in crids:continue
   value=numeric(r["valuenum"])
   if value is None or value<=0:continue
   u=r["valueuom"].strip().lower().replace("µ","u").replace("μ","u")
   if u not in {"mg/dl","umol/l"}:unknown_units[u]+=1;continue
   value*=Fraction("88.4") if u=="mg/dl" else 1
   by[r["subject_id"]].append((datetime.datetime.fromisoformat(r["charttime"]),r["storetime"],int(r["labevent_id"]),value))
  ids={r["subject_id"] for r in cases};pt={}
  p=Path("/home/lei/physionet.org/files/mimic/hosp/patients.csv.gz")
  with gzip.open(p,"rt",newline="") as f:
   for r in csv.DictReader(f):
    if r["subject_id"] in ids:pt[r["subject_id"]]=r
  assert set(pt)==ids
  prepared=[];by_day=collections.Counter()
  for r in cases:
   cut=datetime.datetime.combine(datetime.date.fromisoformat(r["procedure_date"]),datetime.time())
   vals=by[r["subject_id"]];bl=[v for v in vals if cut-datetime.timedelta(days=30)<=v[0]<cut and v[1] and datetime.datetime.fromisoformat(v[1])<cut]
   assert bl
   baseline=sorted(bl,key=lambda v:(v[0],datetime.datetime.fromisoformat(v[1]),v[2]))[-1]
   assert baseline[0].isoformat(sep=" ")==r["baseline_Cr_charttime"] and abs(float(baseline[3])-float(features[r["record_key"]]["PreopCr"]))<1e-9
   post=[v for v in vals if cut+datetime.timedelta(days=1)<=v[0]<cut+datetime.timedelta(days=8)]
   assert len(post)==int(r["post_Cr_n"]) and post
   positive=[v for v in post if 2*v[3]>=3*baseline[3]]
   assert int(bool(positive))==int(r["y_scr_calendar"])
   assert min((v[0].isoformat(sep=" ") for v in positive),default="")==r["first_recorded_positive_charttime"]
   early=[v for v in post if v[0]<cut+datetime.timedelta(days=4)]
   earlypos=[v for v in early if 2*v[3]>=3*baseline[3]]
   days=sorted({(v[0].date()-cut.date()).days for v in post})
   for day in days:by_day[day]+=1
   pp=pt[r["subject_id"]];match=re.fullmatch(r"(\d{4})\s*-\s*(\d{4})",pp["anchor_year_group"]);assert match
   delta=cut.year-int(pp["anchor_year"]);lo=int(match[1])+delta;hi=int(match[2])+delta
   # A qualified broad source-period candidate uses whole-provider-bound interval.
   era="through2016" if hi<=2016 else "2017on" if lo>=2017 else "boundary_uncertain"
   out={**r,"Age":features[r["record_key"]]["Age"],"Sex":features[r["record_key"]]["Sex"],"albumin_observed":int(features[r["record_key"]]["PreopAlb"]!=""),"Cr_record_count_7":len(post),"Cr_observed_days_7":len(days),"Cr_record_count_3":len(early),"Cr_observed_days_3":len({v[0].date() for v in early}),"eligible_3":int(bool(early)),"y_3":int(bool(earlypos)) if early else "","y_7":int(bool(positive)),"Cr_late_day4to7_observed":int(any(day>=4 for day in days)),"Cr_days_1to2_vs_3plus":"1to2" if len(days)<=2 else "3plus","Cr_records_1to3_vs_4plus":"1to3" if len(post)<=3 else "4plus","age_band65":"under65" if float(features[r["record_key"]]["Age"])<65 else "65plus","source_anchor_year_group":pp["anchor_year_group"],"possible_procedure_year_low":lo,"possible_procedure_year_high":hi,"broad_real_period_candidate":era}
   prepared.append(out)
  with (O/"PRIVATE_QUALIFIED_RECORDS.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(prepared[0]));w.writeheader();w.writerows(prepared)
  countrows=[]
  for key in ["albumin_observed","Cr_days_1to2_vs_3plus","Cr_records_1to3_vs_4plus","age_band65","Sex","site","broad_real_period_candidate"]:
   for level in sorted(set(str(r[key]) for r in prepared)):
    subset=[r for r in prepared if str(r[key])==level]
    countrows.append({"dimension":key,"level":level,"n":len(subset),"events7":sum(r["y_7"] for r in subset),"nonevents7":sum(1-r["y_7"] for r in subset),"eligible3":sum(r["eligible_3"] for r in subset),"events3":sum(r["y_3"] for r in subset if r["eligible_3"])})
  with (O/"FIELD_AND_COUNT_QUALIFICATION.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(countrows[0]));w.writeheader();w.writerows(countrows)
  accepted=json.loads((R/"outputs/extended_subgroups_20261028_v1/STAGE_EXTENDED_SUBGROUP_ACCEPTANCE.json").read_text())
  for path,h in accepted["receipts"].items():assert sha(path)==h,path
  baseline=json.loads((N/"REQUEST.json").read_text())["current_writing_delivery"]
  delivery=json.loads(Path(baseline["path"]).read_text());assert sha(delivery["archive"])==baseline["archive_sha256"]
  protected={str(p):sha(p) for p in Path(delivery["package"]).rglob("*") if p.is_file()}
  for p in [D/"PRIVATE_METADATA.csv",D/"PRIVATE_COMMON6_FEATURES.csv",D/"PRIVATE_SELECTED_LAB_EVENTS.csv",D/"INPUT_QUALIFICATION.json",D/"BOOTSTRAP_INPUT_PRIVATE.npz",A/"ACCEPTANCE.json"]:
   protected[str(p)]=sha(p)
  for line in ["geography","temporal"]:
   for p in (D/line).glob("*"): 
    if p.is_file():protected[str(p)]=sha(p)
  # Include original2RF source locks and referenced model components in the preservation list.
  stage=R/"outputs/common6_model_lock_20261024_v1/STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json"
  def visit(obj):
   if isinstance(obj,dict):
    if "path" in obj and "sha256" in obj:
     path=Path(obj["path"]);assert sha(path)==obj["sha256"];protected[str(path)]=obj["sha256"]
    for v in obj.values():visit(v)
   elif isinstance(obj,list):
    for v in obj:visit(v)
  for row in json.loads(stage.read_text())["models"]:
   visit(row);visit(json.loads(Path(row["path"]).read_text()))
  protected[delivery["archive"]]=sha(delivery["archive"])
  save(N/"BASELINE_PROTECTION.json",{"files":protected,"writing_package_files":len([p for p in Path(delivery["package"]).rglob("*") if p.is_file()]),"writing_archive_sha256":baseline["archive_sha256"],"content_reviewed":False,"source_models_and_predictions_frozen":True})
  save(O/"FIELD_QUALIFICATION.json",{"status":"PASS_EXISTING_COHORT_FIELDS_AND_COUNTS_REPLAY_BEFORE_COMPARATIVE_METRICS","n":759,"events7":59,"eligible3":sum(r["eligible_3"] for r in prepared),"events3":sum(r["y_3"] for r in prepared if r["eligible_3"]),"no_early_Cr_outcome_missing":sum(not r["eligible_3"] for r in prepared),"late_day4to7_observed":sum(r["Cr_late_day4to7_observed"] for r in prepared),"Cr_patients_by_observed_post_calendar_day":dict(sorted(by_day.items())),"counts":countrows,"private_record_file_sha256":sha(O/"PRIVATE_QUALIFIED_RECORDS.csv"),"original44_subgroups_reused_receipts":accepted["receipts"],"time_qualification":"Provider anchor-year ranges propagated by within-patient year difference; no shifted date interpreted directly as actual year. Broad-period cutoff remains a candidate at qualification stage.","provider_time_source":"https://mimic.mit.edu/docs/iv/modules/hosp/patients.html","unknown_creatinine_units":dict(unknown_units),"performance_metrics_computed":False,"files_sha256":{str(p):sha(p) for p in [N/"REQUEST.json",A/"qualification/MIMIC_LAB_AND_ENDPOINT_PRELOCK.json",D/"PRIVATE_METADATA.csv",D/"PRIVATE_COMMON6_FEATURES.csv",D/"PRIVATE_SELECTED_LAB_EVENTS.csv",Path("/home/lei/physionet.org/files/mimic/hosp/patients.csv.gz"),Path(__file__)]}})
  print(json.dumps({"status":"FIELD_COUNT_QUALIFICATION_PASS","counts":countrows,"eligible3":sum(r["eligible_3"] for r in prepared),"events3":sum(r["y_3"] for r in prepared if r["eligible_3"]),"late4to7":sum(r["Cr_late_day4to7_observed"] for r in prepared)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
