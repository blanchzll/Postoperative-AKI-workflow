"""Collect all current sensitivity results from the corrected eight-context receipt."""
from pathlib import Path
import csv,datetime,hashlib,importlib.util,json,math,os,resource,signal,time
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
W=R/"work/fixed_flow_sensitivity_evaluation_current_20261030_r1"
O=R/"outputs/fixed_flow_sensitivity_evaluation_current_20261030_r1"
DEST=O/"derived_tables_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 text=d if isinstance(d,str) else json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
 with Path(p).open("x") as f:f.write(text);f.flush();os.fsync(f.fileno())
def table(p,data):
 assert data
 fields=list(dict.fromkeys(k for row in data for k in row))
 with p.open("x",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
  f.flush();os.fsync(f.fileno())
def collect():
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CPU,(60,)*2)
 signal.alarm(120);os.sched_setaffinity(0,{max(os.sched_getaffinity(0))});start=time.monotonic()
 for rel,h in read(W/"TABLE_COLLECTION_CODE_LOCK.json")["files"].items():assert sha(R/rel)==h
 stage=O/"STAGE_EVALUATION_ACCEPTANCE.json"
 if not stage.exists():
  print("WAIT_CORRECTED_EIGHT_CONTEXT_ACCEPTANCE");return
 a=read(stage);assert a["status"]=="PASS_eight_current_source_matched_refit_sensitivity_evaluations"
 assert a["recomputed_contexts"]==2 and a["retained_unchanged_contexts"]==6 and len(a["receipts"])==8
 proof=O/"INDEPENDENT_CORRECTION_VERIFICATION.json"
 assert sha(proof)==a["independent_verification_sha256"]
 protected=read(R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json")["files"]
 assert len(protected)==92 and all(sha(p)==d["sha256"] for p,d in protected.items())
 if DEST.exists():raise RuntimeError("Existing collection requires inspection; never overwrite")
 DEST.mkdir(mode=0o700)
 sources={str(stage):sha(stage),str(proof):sha(proof)}
 points=[];intervals=[];differences=[];raw_points=[];thresholds=[];contexts=[]
 expected={(line,cohort,scenario) for line in ["geography","temporal"] for cohort in ["preop","paired"]
           for scenario in ["omit_ALT","exclude_high_baseline_Cr"]}
 identities=set()
 for ref,h in sorted(a["receipts"].items()):
  p=Path(ref);assert sha(p)==h;sources[str(p)]=h;c=read(p);sig=c["signature"]
  identity=(sig["line"],sig["cohort"],sig["scenario"]);assert identity not in identities;identities.add(identity)
  for name,digest in c["files"].items():assert sha(p.parent/name)==digest;sources[str(p.parent/name)]=digest
  result=read(p.parent/"EVALUATION.json");receipt=result["receipt"]
  assert receipt["replicates"]==2000 and receipt["seed"]==20261017 and receipt["group_unit_within_hospital"]
  partition=sig["line"]+"_"+sig["cohort"];scenario=sig["scenario"]
  prefix={"partition":partition,"scenario":scenario}
  counts=set()
  for model,v in result["point"].items():
   counts.add((v["n"],v["events"]))
   for cap in ["10","20"]:
    assert v["capacity"+cap+"_captured_n"]+v["capacity"+cap+"_missed_n"]==v["events"]
    assert v["capacity"+cap+"_captured_n"]+v["capacity"+cap+"_false_alerts_n"]==v["capacity"+cap+"_reviewed_n"]
    assert abs(v["capacity"+cap+"_captured_per1000"]-1000*v["capacity"+cap+"_captured_n"]/v["n"])<=1e-12
   points.append(dict(prefix,model=model,**v))
  assert counts=={(c["n"],c["events"])}
  intervals.extend(dict(prefix,**x) for x in result["intervals"])
  for item in result["paired_differences"]:
   left,right=item["model"].split(" MINUS ")
   if item["estimate"] is not None:
    assert abs(item["estimate"]-(result["point"][left][item["metric"]]-result["point"][right][item["metric"]]))<=1e-12
   differences.append(dict(prefix,**item))
  raw=read(p.parent/"RAW_POINT_ESTIMATES.json")
  assert set(raw)==set(result["point"])
  raw_points.extend(dict(prefix,model=model,**v) for model,v in raw.items())
  with (p.parent/"TRAINING_THRESHOLD_COUNTS.csv").open(newline="") as f:
   thresholds.extend(dict(prefix,**x) for x in csv.DictReader(f))
  contexts.append(dict(prefix,records=c["n"],events=c["events"],receipt=str(p),receipt_sha256=h,
      result_path=str(p.parent),result_files_sha256=c["files"],
      current_source_exclusion_count=0 if scenario=="exclude_high_baseline_Cr" else None,
      extra_independent_exclusion_robustness_evidence=False if scenario=="exclude_high_baseline_Cr" else None))
 assert identities==expected
 tables={
  "ALL_SCENARIO_POINT_ESTIMATES.csv":points,
  "ALL_SCENARIO_INTERVALS.csv":intervals,
  "ALL_PAIRED_COMPARISONS.csv":differences,
  "KEY_PAIRED_COMPARISONS.csv":[x for x in differences if x["metric"] in ["AUROC","AP","Brier","capacity10_captured_per1000","capacity20_captured_per1000"]],
  "ALL_RAW_POINT_ESTIMATES.csv":raw_points,
  "ALL_TRAINING_THRESHOLD_COUNTS.csv":thresholds}
 for name,data in tables.items():table(DEST/name,data)
 write(DEST/"CURRENT_CONTEXT_SOURCE_MAP.json",{"source_sha256":a["source_sha256"],"stage_sha256":sha(stage),"contexts":contexts})
 write(DEST/"ANALYSIS_SCOPE_ZH.md",
  "# 当前固定流程敏感性分析汇总（计算资料）\n\n"
  "本目录仅从本轮被接受的8个评价回执读取全部模型/全部对比，不是稿件正文或最终科学审查。2个时间验证排除情景已按纠正源重算并核验，其余6个输入和数值未变。全部表保留阴性、不利和缺定义的值；CSV空单元格对应原JSON null，不填0。\n\n"
  "当前无法核实的高基线肌酐名单为空。因此exclude_high_baseline_Cr情景与对应matched_control没有病例差异，重复列用于追溯，不构成额外独立稳健性证据。去ALT结果仍是同患者、同固定选择流程的敏感性比较，不重新选择主赢家；两种情景各自显著性差异不能代替正式交互检验。\n\n"
  "当前源3710/205、配对2428/138。表中按各实际地理/时间验证分区给分母，模型行不是独立患者队列。所有区间仍为原2000次医院内患者组bootstrap和seed20261017。此汇总不修改模型、概率、阈值或稿件；临床/科学主张由后续正式审查逐条裁定。\n")
 assert all(sha(p)==h for p,h in sources.items())
 assert all(sha(p)==d["sha256"] for p,d in protected.items())
 write(DEST/"TABLE_COLLECTION_ACCEPTANCE.json",{
  "status":"PASS_CURRENT_EIGHT_CONTEXT_TABLE_COLLECTION_ONLY",
  "time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_sha256":sha(__file__),
  "stage_sha256":sha(stage),"sources_sha256":sources,
  "outputs_sha256":{p.name:sha(p) for p in DEST.iterdir() if p.is_file()},
  "row_counts":{name:len(data) for name,data in tables.items()},
  "all_contexts":8,"all_models_and_comparisons_retained":True,
  "old_wrong_context_paths_not_used_as_current":True,"new_fits":0,"new_bootstraps":0,
  "original92_unchanged":True,"final_scientific_review":False,
  "seconds":time.monotonic()-start,"max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
 print("COLLECTED_ALL_CORRECTED_SENSITIVITY_TABLES_REVIEW_REQUIRED")
if __name__=="__main__":collect()
