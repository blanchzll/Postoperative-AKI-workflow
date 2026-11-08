"""Independent reconstruction from extracted original lab text, exact-decimal endpoint."""
from pathlib import Path
import csv,json,datetime,hashlib,collections,math
from fractions import Fraction
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");Q=A/"qualification";D=A/"mimic_calendar_v2"
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads((Q/"MIMIC_LAB_AND_ENDPOINT_PRELOCK.json").read_text());items={r["itemid"]:r["label"] for r in contract["selected_dictionary_entries"]}
cases=rows(Q/"PRIVATE_MIMIC_FIRST_RESECTION_CANDIDATES.csv");raw=rows(D/"PRIVATE_SELECTED_LAB_EVENTS.csv")
fields={"Creatinine":"PreopCr","Urea Nitrogen":"PreopBUN","Albumin":"PreopAlb","Hemoglobin":"PreopHb"}
by=collections.defaultdict(list);nonpositive=0;invalid=0
for r in raw:
 try:v=Fraction(r["valuenum"])
 except (ValueError,ZeroDivisionError):invalid+=1;continue
 if v<=0:nonpositive+=1;continue
 lab=items[r["itemid"]];u=r["valueuom"].strip().lower().replace("µ","u").replace("μ","u")
 factors={"Creatinine":{"mg/dl":"88.4","umol/l":"1"},"Urea Nitrogen":{"mg/dl":"0.357","mmol/l":"1"},"Albumin":{"g/dl":"10","g/l":"1"},"Hemoglobin":{"g/dl":"10","g/l":"1"}}
 if u not in factors[lab]:continue
 by[r["subject_id"]].append({**r,"field":fields[lab],"exact_SI":v*Fraction(factors[lab][u])})
expected={};flow=collections.Counter()
for case in cases:
 s=case["subject_id"];day=datetime.date.fromisoformat(case["procedure_date"]);cut=datetime.datetime.combine(day,datetime.time());lower=cut-datetime.timedelta(days=30);after=cut+datetime.timedelta(days=1);end=cut+datetime.timedelta(days=8)
 records=by[s];baseline={}
 for fld in ["PreopCr","PreopBUN","PreopAlb","PreopHb"]:
  a=[r for r in records if r["field"]==fld and lower<=datetime.datetime.fromisoformat(r["charttime"])<cut and r["storetime"] and datetime.datetime.fromisoformat(r["storetime"])<cut]
  if a:baseline[fld]=sorted(a,key=lambda r:(datetime.datetime.fromisoformat(r["charttime"]),datetime.datetime.fromisoformat(r["storetime"]),int(r["labevent_id"])))[-1]
 post=[r for r in records if r["field"]=="PreopCr" and after<=datetime.datetime.fromisoformat(r["charttime"])<end]
 if "PreopCr" not in baseline:flow["no_preop_Cr"]+=1;continue
 if not post:flow["no_post_Cr"]+=1;continue
 if case["Sex"] not in ["F","M"]:flow["unknown_sex"]+=1;continue
 b=baseline["PreopCr"]["exact_SI"];positive=[r for r in post if r["exact_SI"]>=Fraction("1.5")*b];record="MIMIC3.1|calendar|"+case["hadm_id"]+"|"+case["procedure_date"]
 expected[record]={"record_key":record,"y_exact":int(bool(positive)),"first_positive_exact":min((r["charttime"] for r in positive),default=""),"post_Cr_n":len(post),"Age":float(case["Age"]),"Sex":"1" if case["Sex"]=="M" else "0",**{fld:float(r["exact_SI"]) for fld,r in baseline.items()},"baseline_chart":baseline["PreopCr"]["charttime"],"baseline_store":baseline["PreopCr"]["storetime"]}
x=rows(D/"PRIVATE_COMMON6_FEATURES.csv");m={r["record_key"]:r for r in rows(D/"PRIVATE_METADATA.csv")};checks=0;maxerr=0.;differences=[]
assert set(r["record_key"] for r in x)==set(m)==set(expected)
for r in x:
 e=expected[r["record_key"]];old=m[r["record_key"]];checks+=1
 for fld in ["Age","PreopCr","PreopBUN","PreopAlb","PreopHb"]:
  if fld not in e:assert r[fld]==""
  else:
   err=abs(float(r[fld])-e[fld]);maxerr=max(maxerr,err);assert err<=1e-9,(fld,err)
  checks+=1
 assert r["Sex"]==e["Sex"]
 assert old["baseline_Cr_charttime"]==e["baseline_chart"] and old["baseline_Cr_storetime"]==e["baseline_store"]
 assert int(old["post_Cr_n"])==e["post_Cr_n"]
 if int(old["y_scr_calendar"])!=e["y_exact"] or (old["first_recorded_positive_charttime"] or "")!=e["first_positive_exact"]:
  differences.append({"record_key":r["record_key"],"old_y":int(old["y_scr_calendar"]),"exact_decimal_y":e["y_exact"],"old_first_positive":old["first_recorded_positive_charttime"],"exact_first_positive":e["first_positive_exact"]})
with (D/"PRIVATE_EXACT_RATIONAL_TARGET_REPLAY.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["record_key","y_exact","first_positive_exact"]);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in expected.values())
if differences:
 with (D/"PRIVATE_DECIMAL_DIFFERENCES.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(differences[0]));w.writeheader();w.writerows(differences)
result={"status":"EXACT_RATIONAL_REPLAY_REQUIRES_TIMING_OR_LABEL_CORRECTION" if differences else "PASS_INDEPENDENT_EXACT_RATIONAL_INPUT_AND_ENDPOINT_REPLAY","cases":len(x),"original_events":sum(int(r["y_scr_calendar"]) for r in m.values()),"exact_rational_events":sum(r["y_exact"] for r in expected.values()),"disagreement_records":len(differences),"feature_and_identity_checks":checks,"max_abs_feature_SI_difference":maxerr,"selected_raw_lab_records_reprocessed":len(raw),"nonpositive_finite_lab_exclusions_replayed":nonpositive,"nonnumeric_or_missing_lab_exclusions_replayed":invalid,"scope":"Complete extracted lab subset and frozen source/code/protocol; not a second full scan of158moriginal rows or clinical phenotype adjudication.","source_sha256":{str(p):sha(p) for p in [D/"PRIVATE_SELECTED_LAB_EVENTS.csv",Q/"MIMIC_LAB_AND_ENDPOINT_PRELOCK.json",Path(__file__)]}}
(D/"INDEPENDENT_INPUT_ENDPOINT_REVIEW.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");print(json.dumps(result),flush=True)
