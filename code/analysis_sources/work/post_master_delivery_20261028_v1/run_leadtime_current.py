"""Current common6 record-time windows: actual qualification/run and independent aggregate QC."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,traceback,fcntl
R=Path(__file__).resolve().parents[2];O=R/"outputs/benchmark_and_leadtime_20261024_v1";H=R/"outputs/hypothesis_aligned_extensions_20261028_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def main():
 with (O/"CURRENT_CONTROLLER.lock").open("a") as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  script=Path(__file__).parent/"leadtime_current_v1.py"
  contract=read(O/"LEADTIME_CONTRACT.json");assert contract["code_sha256"]==sha(script)
  env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
  with (O/"CURRENT_QUALIFICATION.log").open("ab") as log:
   q=subprocess.run([sys.executable,"-B",str(script),"qualify"],env=env,stdout=log,stderr=log,timeout=180)
   assert q.returncode==0
  if not (O/"LEADTIME_FITTING_ACCEPTANCE.json").exists():
   with (O/"CURRENT_EXECUTION.log").open("ab") as log:
    child=subprocess.Popen([sys.executable,"-B",str(script)],env=env,stdout=log,stderr=log)
    save(O/"LEADTIME_LAUNCH.json",{"pid":child.pid,"start_ticks":Path("/proc",str(child.pid),"stat").read_text().split()[21],"time":now(),"code_sha256":sha(script)})
    try:rc=child.wait(timeout=2400)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned record-window worker exceeded finite budget")
   assert rc==0,"Inspect actual record-time execution failure before retry"
  import numpy as np,pandas as pd
  fit=read(O/"LEADTIME_FITTING_ACCEPTANCE.json")
  assert fit["code_sha256"]==sha(script) and fit["contract_sha256"]==sha(O/"LEADTIME_CONTRACT.json")
  for f,h in fit["files"].items():assert sha(O/f)==h
  owner=read(O/"LEADTIME_LAUNCH.json");proc=Path("/proc",str(owner["pid"]),"stat")
  assert not (proc.exists() and proc.read_text().split()[21]==owner["start_ticks"])
  points=pd.read_csv(O/"LEADTIME_POINT_ESTIMATES.csv",float_precision="round_trip").set_index("rule")
  ci=pd.read_csv(O/"LEADTIME_INTERVALS.csv",float_precision="round_trip")
  boots=pd.read_csv(O/"LEADTIME_BOOTSTRAPS.csv",float_precision="round_trip")
  joined=pd.read_csv(O/"PRIVATE_LEADTIME_JOIN.csv",dtype={"record_key":str})
  assert len(joined)==6043 and int(joined.y_scr.sum())==253
  event=joined.y_scr.eq(1)
  interval=(joined.first_observed_criterion_time-joined.opstart_time)/60
  assert interval[event].notna().all() and (interval[event]>0).all()
  assert len(boots)==12000 and (boots.groupby("rule").size()==2000).all()
  assert set(boots.replicate.unique())==set(range(2000))
  assert (boots.all_AKI_detected_atleast48h_n<=boots.all_AKI_detected_atleast24h_n).all()
  assert (boots.all_AKI_detected_atleast24h_n<=boots.all_AKI_detected_atleast12h_n).all()
  assert (boots.all_AKI_detected_atleast12h_n<=boots.captured_n).all()
  assert (boots.captured_n+boots.missed_n==boots.events).all()
  assert (boots.captured_n+boots.false_alerts_n==boots.reviewed_n).all()
  checks=[]
  for line,threshold in fit["source_thresholds"].items():
   pred=R/"outputs/common6_public_evaluation_20261024_v1"/line/"PRIVATE_PREDICTIONS.csv"
   v=pd.read_csv(pred,dtype={"record_key":str},float_precision="round_trip")
   z=joined.merge(v[["record_key","prob_calibrated"]],on="record_key",validate="one_to_one")
   ll=(z.first_observed_criterion_time-z.opstart_time)/60
   flag=z.prob_calibrated.ge(threshold);ev=z.y_scr.eq(1);row=points.loc[line+"__source_sensitivity80"]
   expected={"reviewed_n":int(flag.sum()),"captured_n":int((flag&ev).sum()),"missed_n":int((~flag&ev).sum()),"false_alerts_n":int((flag&~ev).sum())}
   for hours in [12,24,48]:
    expected[f"all_AKI_detected_atleast{hours}h_n"]=int((flag&ev&ll.ge(hours)).sum())
    expected[f"all_AKI_detected_atleast{hours}h_fraction"]=expected[f"all_AKI_detected_atleast{hours}h_n"]/253
   for key,value in expected.items():np.testing.assert_allclose(row[key],value,rtol=1e-12,atol=1e-12)
   checks.append({"line":line,"fixed_source_threshold":threshold,"independent_join_and_event_counts_match":True,"prediction_sha256":sha(pred)})
  for v in ci.itertuples():
   vals=boots.loc[boots.rule==v.rule,v.metric].dropna().to_numpy()
   assert len(vals)==v.valid_replicates
   if len(vals):np.testing.assert_allclose(np.quantile(vals,[.025,.975]),[v.lower95,v.upper95],rtol=1e-10,atol=1e-10)
  all_interval={"n_events":253,"record_time_eligible_events":int(interval[event].notna().sum()),"median_hours":float(interval[event].median()),"Q1_hours":float(interval[event].quantile(.25)),"Q3_hours":float(interval[event].quantile(.75)),"event_time_exclusions":0}
  save(O/"ALL_EVENT_TIMING_ELIGIBILITY.json",all_interval)
  lines=["# 当前六变量模型：固定手术时点至首次记录肌酐达标的间隔","",
  "仅复用已核哈希的INSPIRE6043条手术、253个记录肌酐事件；使用本轮中国纠正源重新训练、锁定并评价的两条common6模型及其原训练阈值。没有复制旧模型、旧预测或旧提前量。全部253个事件均具备记录时点，未触发风险规则的事件仍留在事件分母。","",
  f"所有事件的记录间隔中位数为{all_interval['median_hours']:.2f}小时，Q1–Q3为{all_interval['Q1_hours']:.2f}–{all_interval['Q3_hours']:.2f}小时。以下各规则检出者的条件间隔另列，不替代全部事件比例。","",
  "|模型来源／规则|复核人数|检出／全部253|漏诊|误报|PPV|检出者间隔中位数（小时）|≥12／24／48小时检出数，分母各253|",
  "|---|---:|---:|---:|---:|---:|---:|---|"]
  for rule,v in points.iterrows():
   lines.append(f"|{rule}|{int(v.reviewed_n)}|{int(v.captured_n)}／253|{int(v.missed_n)}|{int(v.false_alerts_n)}|{v.PPV:.4f}|{v.detected_record_interval_hours_median:.2f}|{int(v.all_AKI_detected_atleast12h_n)}／{int(v.all_AKI_detected_atleast24h_n)}／{int(v.all_AKI_detected_atleast48h_n)}|")
  lines+=["","固定阈值来自中国开发集；名称training_sensitivity80不代表公共队列灵敏度达到80%。10%与20%规则是回顾性批次固定容量，不能冒充逐例在线策略。各比例、每千例复核／误报／漏诊及2000次患者组区间保存在完整点估计和区间文件；抽样权重哈希与本轮公共评价一致。","",
  "间隔从记录手术开始时点计算，终点为首次记录的术后肌酐达标时间。检验结果发布、医生可见或实际触发临床告警的时刻未测得，生物学损伤起点未知。采样安排影响终点记录；这是固定七天风险模型的回顾性记录窗口，不是动态多时窗预测、中国27／38变量模型的小时级验证或干预因果效应。","",
  "本次数值和流程复核不等于最终统计／科学严审；完整37项后另行审读。"]
  (O/"LEADTIME_RESULTS_ZH.md").write_text("\n".join(lines)+"\n")
  review={"status":"PASS_current_common6_record_time_windows_independently_reaggregated","time":now(),"cohort":6043,"events":253,"independent_threshold_point_checks":checks,"intervals_reaggregated":len(ci),"original_public_resampling_weight_sha256":fit["public_resampling_proof_matches"],"all_event_denominator_preserved":True,"measured_clinical_alert_or_biological_onset_claimed":False,"formal_scientific_review_complete":False,"source_acceptance_sha256":sha(O/"LEADTIME_FITTING_ACCEPTANCE.json"),"review_code_sha256":sha(__file__),"report_sha256":sha(O/"LEADTIME_RESULTS_ZH.md")}
  save(O/"LEADTIME_REVIEW_ACCEPTANCE.json",review)
  save(H/"LEADTIME_ACCEPTANCE.json",{"status":"PASS_current_common6_all_event_record_time_identification_windows","time":now(),"current_input_lock_sha256":contract["current_input_source_sha256"],"cohort":6043,"events":253,"windows_hours":[12,24,48],"receipts":{str(O/n):sha(O/n) for n in ["LEADTIME_CONTRACT.json","LEADTIME_FUNCTIONAL_QUALIFICATION.json","LEADTIME_FITTING_ACCEPTANCE.json","LEADTIME_REVIEW_ACCEPTANCE.json","ALL_EVENT_TIMING_ELIGIBILITY.json"]},"result_files":{str(O/f):h for f,h in fit["files"].items()},"actual_clinical_alert_lead_time_validated":False,"dynamic_window_or_intervention_effect_claimed":False,"final_scientific_review_completed":False})
if __name__=="__main__":
 try:main()
 except BaseException as e:
  save(O/"CURRENT_CONTROLLER_FAILURE.json",{"time":now(),"error_class":type(e).__name__,"error":str(e)[:400],"stack":[{"file":str(Path(v.filename)),"line":v.lineno,"function":v.name} for v in traceback.extract_tb(e.__traceback__)[-8:]]});raise
