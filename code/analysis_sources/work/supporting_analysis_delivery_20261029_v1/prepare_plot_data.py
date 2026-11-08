"""Read only accepted current aggregate tables; prepare plotting inputs without refits."""
from pathlib import Path
import json,csv,hashlib,datetime,math,resource,time,fcntl
R=Path(__file__).absolute().parents[2]
O=R/"outputs"; W=Path(__file__).absolute().parent
OUT=O/"supporting_analysis_delivery_20261029_v1"
SOURCE_SHA="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
refs={}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def record(p,expected=None):
 p=Path(p);p=p if p.is_absolute() else O/p
 assert str(p).startswith(str(R)+"/"),str(p)
 h=sha(p)
 if expected is not None:assert h==expected,("digest",str(p),h,expected)
 refs[str(p)]=h
 return p
def j(p,expected=None):return json.loads(record(p,expected).read_text())
def rows(p,expected=None):
 with record(p,expected).open(newline="") as f:return list(csv.DictReader(f))
def save(p,v):
 t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def receipt(rel):
 p=O/rel;v=j(p);assert v["status"].startswith(("PASS","COMPLETE")),rel
 return p,v
def table(p,v,name,key="files"):return rows(p.parent/name,v[key][name])
def number(x):
 if x is None or x=="":return None
 a=float(x);return a if math.isfinite(a) else None
def main():
 begin=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/"PREPARE.lock").open("a") as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
  source=record(O/"v4_development_inputs_20261018/SOURCE_LOCK.json",SOURCE_SHA)
  old=OUT/"PLOT_DATA_PREPARATION.json"
  if old.exists():
   v=json.loads(old.read_text());assert v["code_sha256"]==sha(__file__)
   for p,h in v["inputs_sha256"].items():record(p,h)
   for p,h in v["outputs_sha256"].items():assert sha(p)==h
   print("REUSED_VALIDATED_SUPPORTING_PLOT_DATA");return
  p,a=receipt("model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json")
  routes=[]
  for x in a["models"]:
   if x["primary_route_winner"]:
    m=j(x["path"],x["sha256"]);t=m["identity"]
    routes.append(dict(context=t["line"]+"_"+t["cohort"],stage=t["stage"],branch=t["branch"],
      model="_".join(t[z] for z in ["stage","branch","role","family"]),
      route="_".join(t[z] for z in ["line","cohort","stage","branch"]),family=t["family"]))
  assert len(routes)==10
  contexts=["geography_preop","geography_paired","temporal_preop","temporal_paired"]
  dataset={"source_sha256":SOURCE_SHA,"routes":routes,"main":{}}
  def full_models(ctx):return [z["model"] for z in routes if z["context"]==ctx and z["branch"]=="full"]
  for ctx in contexts:
   p,v=receipt("formal_evaluation_20261023_v1/"+ctx+"/COMPLETE.json")
   models={x["model"] for x in routes if x["context"]==ctx}
   d={}
   for name,key in [("POINT_ESTIMATES.csv","point"),("INTERVALS.csv","intervals"),("CALIBRATION_CURVES.csv","calibration")]:
    d[key]=[z for z in table(p,v,name) if z["model"] in models]
   assert {z["model"] for z in d["point"]}==models
   dataset["main"][ctx]=d
  p,v=receipt("albumin_continuous_20261024_v2/REVIEW_ACCEPTANCE.json")
  acurves=table(p,v,"CONTINUOUS_DOSE_CURVES.csv")
  interactions=table(p,v,"CONTINUOUS_INTERACTIONS.csv")
  contrasts=table(p,v,"CONTINUOUS_DOSE_CONTRASTS.csv")
  counts=table(p,v,"SUBGROUP_COUNTS.csv")
  acode=record(R/"work/albumin_continuous_20261024_v2/analysis.py")
  assert "alb = np.nan_to_num((d.PreopAlb.to_numpy() - 35) / 10, nan=0)" in acode.read_text()
  contract=j(p.parent/"CONTRACT.json")
  assert contract["positive_reference_g"]==80 and contract["positive_dose_knots_g"]==[20,80,160]
  corrected=0
  for row in acurves:
   row["display_label"]=row["label"]
   if row["modifier"]=="baseline_alb" and float(row["level"])==-1:
    assert row["label"]=="";row["display_label"]="Baseline serum albumin 25 g/L";corrected+=1
   elif row["modifier"]=="baseline_alb":
    row["display_label"]="Baseline serum albumin %g g/L"%(35+10*float(row["level"]))
  assert corrected==106 and len(acurves)==742
  dataset["albumin"]={"curves":acurves,"interactions":interactions,"contrasts":contrasts,"subgroup_counts":counts,"cohort":v["cohort"]}
  correction={"issue_id":"DISPLAY-ALB-BASELINE-LABEL-20261029","classification":"confirmed presentation label omission",
   "evidence":[str(acode)+":31",str(acode)+":133",str(p.parent/"CONTRACT.json")],
   "native_empty_label":"baseline_alb level -1","derivation":"35 + 10*(-1) = 25 g/L baseline serum albumin",
   "corrected_display_rows":corrected,"native_table_changed":False,"numeric_values_changed":False,
   "not_infused_dose_group_or_threshold":True,"remedy":"New delivery plot-data/display label only; retain original label column and native table hash."}
  dataset["presentation_corrections"]=[correction]
  p,v=receipt("common6_public_evaluation_20261024_v1/REVIEW_ACCEPTANCE.json")
  sm=j(p.parent/"REPORT_SOURCES.json",v["files"]["REPORT_SOURCES.json"])["verified_files"]
  epath=p.parent/"EVALUATION.json";ev=j(epath,sm[str(epath)])
  ea=j(p.parent/"EVALUATION_ACCEPTANCE.json",sm[str(p.parent/"EVALUATION_ACCEPTANCE.json")])
  assert sha(epath)==ea["sha256"]
  crows=rows(p.parent/"CALIBRATION_CURVES.csv",sm[str(p.parent/"CALIBRATION_CURVES.csv")])
  dataset["common6"]={"point":[dict(model=k,**x) for k,x in ev["point"].items()],"intervals":ev["intervals"],"calibration":crows}
  assert all(x["n"]==6043 and x["events"]==253 for x in dataset["common6"]["point"])
  p,v=receipt("pod1_corrected_evaluation_20261024_v1/REVIEW_ACCEPTANCE.json")
  pp=table(p,v,"ALL_POINT_ESTIMATES.csv","report_files")
  pd={}
  for ctx in ["geography_all","geography_matched","temporal_all","temporal_matched"]:
   p0,v0=receipt("pod1_corrected_evaluation_20261024_v1/"+ctx+"/COMPLETE.json")
   cp=table(p0,v0,"POINT_ESTIMATES.csv")
   selected=[x for x in pp if x["context"]==ctx]
   assert len(cp)==len(selected) and {x["model"] for x in cp}=={x["model"] for x in selected}
   for x in cp:
    y=next(z for z in selected if z["model"]==x["model"])
    for k,a0 in x.items():
     if k!="model":
      u,b=number(a0),number(y.get(k))
      assert (u is None and b is None) or (u is not None and b is not None and abs(u-b)<1e-12),(ctx,x["model"],k)
   pd[ctx]={"point":cp,"intervals":table(p0,v0,"INTERVALS.csv")}
  assert len(pp)==22;dataset["pod1"]=pd
  p,v=receipt("published_benchmark_comparison_20261024_v1/REVIEW_ACCEPTANCE.json")
  f=j(p.parent/"FIT_AND_EVALUATION_ACCEPTANCE.json",v["computation_receipt_sha256"])
  ds={k:table(p,f,n,"files_sha256") for n,k in [("POINT_ESTIMATES.csv","point"),("INTERVALS.csv","intervals"),("PAIRED_DIFFERENCES.csv","paired")]}
  assert len(ds["point"])==20
  dataset["published"]=ds
  p,v=receipt("hypothesis_aligned_extensions_20261028_v1/LEADTIME_ACCEPTANCE.json")
  lp=O/"benchmark_and_leadtime_20261024_v1"
  ld={k:rows(lp/n,v["result_files"][str(lp/n)]) for n,k in [("LEADTIME_POINT_ESTIMATES.csv","point"),("LEADTIME_INTERVALS.csv","intervals")]}
  assert len(ld["point"])==6 and all(float(x["n"])==6043 and float(x["events"])==253 for x in ld["point"])
  dataset["recorded_window"]=ld
  p,v=receipt("hypothesis_aligned_extensions_20261028_v1/CLINICAL_SCENARIO_ACCEPTANCE.json")
  wp=O/"decision_workload_20261024_v1"
  wd={k:rows(wp/n,v["generated_summary_files"][str(wp/n)]) for n,k in [
   ("ALL_OPERATING_BURDEN.csv","burden"),("ALL_WORKLOAD_SCENARIOS.csv","workload"),("ALL_HYPOTHETICAL_PREOP_EFFECT_SCENARIOS.csv","assumed_effect")]}
  dataset["workflow"]=wd
  p,v=receipt("extended_subgroups_20261028_v1/STAGE_EXTENDED_SUBGROUP_ACCEPTANCE.json")
  br=O/"stratified_evaluation_20261023_v1/STAGE_STRATIFIED_ACCEPTANCE.json"
  bv=j(br,v["receipts"][str(br)])
  all_tasks={**bv["receipts"],**{k:h for k,h in v["receipts"].items() if k!=str(br)}}
  assert len(all_tasks)==44
  subgroup_data={}
  for f,h in sorted(all_tasks.items()):
   pp0=Path(f);vr=j(pp0,h)
   ctx=pp0.parent.parent.name;kind=pp0.parent.name
   pr=table(pp0,vr,"POINT.csv");ir=table(pp0,vr,"INTERVALS.csv")
   # Full original model tables remain native, hashed, and indexed; panels show locked full-route winners only.
   # Selection depends on frozen model role, never observed subgroup performance.
   models=set(full_models(ctx));pr0=[x for x in pr if x["model"] in models];ir0=[x for x in ir if x["model"] in models]
   assert {x["model"] for x in pr0}==models
   subgroup_data[ctx+"/"+kind]={"point":pr0,"intervals":ir0,"all_model_source_point":str(pp0.parent/"POINT.csv"),"all_model_source_intervals":str(pp0.parent/"INTERVALS.csv"),"native_model_count":len({x["model"] for x in pr})}
  dataset["subgroups"]=subgroup_data
  assert not any("llm" in str(p).lower() for p in refs)
  save(OUT/"PLOT_DATA.json",dataset);save(OUT/"PRESENTATION_LABEL_REPAIR.json",correction)
  manifest={"status":"PASS_ACCEPTED_CURRENT_AGGREGATE_DATA_PREPARATION_ONLY","time":now(),"code_sha256":sha(__file__),
   "source_sha256":SOURCE_SHA,"inputs_sha256":refs,"outputs_sha256":{str(OUT/n):sha(OUT/n) for n in ["PLOT_DATA.json","PRESENTATION_LABEL_REPAIR.json"]},
   "source_files_verified":len(refs),"routes":10,"subgroup_tasks":44,"new_fits":0,"new_bootstrap_runs":0,
   "numeric_source_modified":False,"LLM_results_accessed":False,"figure_rendered":False,"actual_visual_review":False,
   "final_scientific_review":False,"seconds":time.monotonic()-begin,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
  save(old,manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ["inputs_sha256","outputs_sha256"]}))
if __name__=="__main__":main()
