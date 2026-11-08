"""Assemble current accepted summaries. This is neither manuscript prose nor final review."""
from pathlib import Path
import csv,datetime,hashlib,importlib.util,json,resource,time
R=Path(__file__).absolute().parents[2];W=Path(__file__).absolute().parent
O=R/"outputs";OUT=O/"final_analysis_delivery_20261025_v1/analysis_compendium_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def escape(s):return str(s).replace("|",r"\|").replace("\n"," ")
def table(headers,rows):
 return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(escape(s) for s in r)+" |" for r in rows])+"\n"
def link(p,label=None):return "["+(label or Path(p).name)+"]("+str(p)+")"
def main():
 started=time.monotonic()
 resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
 lock=json.loads((W/"COMPENDIUM_INPUT_LOCK.json").read_text())
 assert sha(__file__)==lock["code_sha256"]
 for p,h in lock["inputs_sha256"].items():assert sha(p)==h,p
 spec=importlib.util.spec_from_file_location("support",W/"render_supporting.py")
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 d,new,_,prep,add=m.checked_inputs()
 gate=json.loads((O/"post_reanalysis_process_review_v1/REVIEW_GATE_STATUS.json").read_text())
 formal=json.loads((O/"llm_supplement_20261028_v1/formal_v1/STATUS.json").read_text())
 OUT.mkdir(mode=0o700,exist_ok=True)
 assert not (OUT/"COMPENDIUM_PREPARATION.json").exists(),"Version an existing compendium, never silently overwrite"
 ledger=[];counts={};lines=["# 当前纠正数据分析结果汇编：全研究审查前材料\n",
 "这是分析过程汇编，不是稿件正文、最终科学审查或完整分析验收。仅汇总已完成计算，保留原开发锁、方法、种子和不利结果；无新增拟合、患者或bootstrap。\n",
 f"门槛快照：{gate['checked_Asia_Shanghai']}，{gate['complete_components']}/{gate['required_components']}。源={gate['source']['status']}；92稿件保护={gate['manuscript_protection']['status']}。后续仍需实际刷新，不能以本快照替代最终验收。\n",
 f"LLM状态回执快照：{formal['time']}，fit {formal['accepted_formal_fits']}/48、before银行 {formal['accepted_before_banks']}/6。这里不生成LLM临床指标。\n",
 "## 完整组件与权威来源\n"]
 lines.append(table(["组件","状态","回执"],[[r["component"],r["status"],link(r["receipt"])] for r in gate["components"]]))
 aggregate=O/"supporting_analysis_delivery_20261029_v1/PLOT_DATA.json";aggregate_hash=sha(aggregate)
 def cell(r,ir,metric,identity,section,pointer):
  v=m.num(r.get(metric));est,lo,hi,n=m.interval(ir,metric,identity)
  if v is not None:assert est is not None and abs(v-est)<1e-10,(section,identity,metric,v,est)
  ledger.append(dict(section=section,identity=identity,metric=metric,value=v,lower95=lo,upper95=hi,valid_replicates=n,pointer=pointer))
  if v is None:return "不可估计/不适用"
  return f"{v:.4g}"+(f" [{lo:.4g}, {hi:.4g}]" if lo is not None and hi is not None else " [区间不可估计]")
 def metrics_section(title,section,contexts,metrics,keys=("model",),extra_note=""):
  lines.append("## "+title+"\n");rr=[]
  for ctx,z in contexts.items():
   for i,r in enumerate(z["point"]):
    identity={k:r[k] for k in keys}
    if section=="common6":pointer=f"/common6/point/{i}"
    elif section=="published":pointer="/published/point/"+str(d["published"]["point"].index(r))
    else:pointer=f"/{section}/{ctx}/point/{i}"
    vals=[cell(r,z["intervals"],met,identity,section,pointer) for met in metrics]
    rr.append([ctx,r["model"],int(float(r["n"])),int(float(r["events"])),*vals])
  counts[section]=len(rr)
  lines.append(table(["评价集合","模型/条件","记录","AKI"]+[met+" [95%区间]" for met in metrics],rr));lines.append(extra_note+"\n")
 metrics_section("主评价：十条锁定路线","main",d["main"],("AUROC","AP","Brier","calibration_slope","capacity20_captured_per1000"),
  extra_note="地理与时间集合有重叠，不作两次独立复制；20%容量由整个人群名单分配。并列性能指标不代替原配对差值检验。")
 metrics_section("common6公共迁移：两RF的原始及开发侧校准条件","common6",{"INSPIRE":d["common6"]},("AUROC","AP","Brier","calibration_slope","capacity20_captured_per1000"),
  extra_note="公共六变量结果不等于中国27/38变量模型的外部验证；全部raw/calibrated条件保留。")
 metrics_section("POD1条件性终点：全部22条路线","pod1",d["pod1"],("AUROC","AP","Brier","capacity20_sensitivity"),
  extra_note="POD1人群与剩余结局是有条件定义，不描述为原无条件7天终点或精确24小时动态预测；稀疏事件限制精度。")
 pub={}
 for ctx in dict.fromkeys(r["context"] for r in d["published"]["point"]):
  pub[ctx]={k:[r for r in d["published"][k] if r["context"]==ctx] for k in ("point","intervals")}
 metrics_section("原文模型基准：全部20条件","published",pub,("AUROC","AP"),("context","model"),
  "原式、开发侧再校准和本研究模型分开。Chen缺截距，只提供序数排序比较，不补造风险概率。")
 lines.append("## 固定时点至首次记录SCr达标：全部六条规则\n")
 z=d["recorded_window"];rr=[];mets=("all_AKI_detected_atleast12h_fraction","all_AKI_detected_atleast24h_fraction","all_AKI_detected_atleast48h_fraction","reviewed_per1000","PPV")
 for i,r in enumerate(z["point"]):
  rr.append([r["rule"]]+[cell(r,z["intervals"],met,{"rule":r["rule"]},"recorded_window",f"/recorded_window/point/{i}") for met in mets])
 counts["recorded_window"]=len(rr)
 lines.append(table(["规则","全部AKI中≥12h捕获比例","≥24h","≥48h","复核/千例","PPV"],rr))
 lines.append("分母包括全部253事件及未触发者。间隔是手术开始至首次记录SCr达标，不是生物发病、结果发布或实际告警提前量。\n")
 lines.append("## 白蛋白：连续关联、非线性与交互\n")
 tests=m.rows(O/"albumin_continuous_20261024_v2/CONTINUOUS_DOSE_ASSOCIATION_TESTS.csv")
 lines.append(table(["原生模型","正剂量整体关联P","非线性P"],[[r["model"],r["positive_dose_p"],r["nonlinear_p"]] for r in tests]))
 lines.append(table(["修饰变量","正剂量交互P","三项BH q"],[[r["modifier"],r["recipient_dose_p"],r["BH_q_three_dose_interactions"]] for r in d["albumin"]["interactions"]]))
 lines.append("保留是否输注+正剂量RCS、20/80/160 g结点及80 g中位数参照。当前资料未确定转折、安全剂量或临床推荐剂量。未知剂量不补0。\n")
 lines.append("基线血清白蛋白主展示改为同一3669条完整病例协变量P10/P50/P90：33.4/40.0/45.12 g/L；不是临床分界或三个独立亚组，不改模型或P/q。旧25/35/45保留追溯。"+link(add["report"],"展示依据与核验")+"\n")
 lines.append("## 44项分层与完整工作量/假设网格\n")
 idx=[]
 for task,z in d["subgroups"].items():
  idx.append({"task":task,"full_route_point_rows":len(z["point"]),"native_model_count":z["native_model_count"],
   "point":z["all_model_source_point"],"point_sha256":prep["inputs_sha256"][z["all_model_source_point"]],
   "intervals":z["all_model_source_intervals"],"intervals_sha256":prep["inputs_sha256"][z["all_model_source_intervals"]]})
 save(OUT/"SUBGROUP_44_SOURCE_INDEX.json",idx);lines.append(link(OUT/"SUBGROUP_44_SOURCE_INDEX.json","全部原生模型分层表及哈希")+"\n")
 lines.append("图中full-route由锁定身份决定，不按亚组表现选模型。分期捕获先取全人群any-AKI名单，不在全阳性分期组内算AUROC。ALT观察/缺失不等于正常/升高；单次低eGFR不诊断CKD。\n")
 counts["subgroup_tasks"]=len(idx)
 for name,key in (("WORKFLOW_BURDEN_ALL.csv","burden"),("WORKFLOW_TIME_TEST_GRID_ALL.csv","workload"),("WORKFLOW_ASSUMPTION_GRID_ALL.csv","assumed_effect")):
  m.write_csv(OUT/name,d["workflow"][key]);counts[key]=len(d["workflow"][key])
  lines.append(f"- {link(OUT/name)}：全部{len(d['workflow'][key])}行。\n")
 lines.append("\n保留全部五规则、复核分钟数/额外检验网格、负/零/正假设效果及执行率。条件情景不升级为虚拟RCT或已降低AKI；抽样区间不包含假设效果参数的不确定性。\n")
 lines.append("## 解释、工具及待审读材料\n")
 lines.append("SHAP/风险曲线及计算器的当前回执见组件表；SHAP不作因果机制。计算器数值/真实浏览器完成不等于完整研究复现包。既有28图与新增支持图必须分清已查看/待渲染状态。\n")
 for n in ("RESOLUTION_REPORT_ZH.md","TARGETED_ISSUES_FOR_FINAL_REVIEW.json","POLICY_CALIBRATION_DIRECTION_DIAGNOSTIC.json"):
  lines.append(link(O/"process_resolution_20261025_v1"/n)+"\n")
 lines.append("最终实际审读须覆盖ALT、去重叠、目标侧标签更新边界、随机比较器负校准斜率、白蛋白时间/适应证混杂、LLM暴露与多重比较。此处只索引资料，不预先宣告其科学结论通过。\n")
 lines.append("## 尚未完成\n")
 lines.extend("- "+r["component"]+"："+r["status"]+"\n" for r in gate["components"] if r["status"]!="ACCEPTED")
 lines.append("\n全部37项真就绪后才进行统计→科学推断→整改复核、验收和新写作对话交接。本汇编不会写出最终验收状态。\n")
 report=OUT/"ANALYSIS_COMPENDIUM_CURRENT_RESULTS_ZH.md";report.write_text("\n".join(lines))
 save(OUT/"NUMERIC_SUMMARY_LEDGER.json",{"aggregate_source":str(aggregate),"aggregate_source_sha256":aggregate_hash,
  "source_preparation_receipt":str(O/"supporting_analysis_delivery_20261029_v1/PLOT_DATA_PREPARATION.json"),"cells":ledger})
 assert counts["main"]==10 and counts["common6"]==4 and counts["pod1"]==22 and counts["published"]==20 and counts["recorded_window"]==6 and counts["subgroup_tasks"]==44
 for p,h in lock["inputs_sha256"].items():assert sha(p)==h,p
 files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}
 receipt={"status":"PREPARED_CURRENT_RESULTS_COMPENDIUM_NOT_FINAL_ACCEPTANCE","time":now(),"code_sha256":sha(__file__),
  "input_lock_sha256":sha(W/"COMPENDIUM_INPUT_LOCK.json"),"source_sha256":d["source_sha256"],"source_gate_snapshot":gate,"formal_status_snapshot":formal,
  "counts":counts,"numeric_summary_cells_checked":len(ledger),"output_sha256":files,"new_fits":0,"new_bootstrap":0,
  "manuscript_read_or_modified":False,"LLM_scientific_metrics_generated":False,"final_scientific_review":False,
  "seconds":time.monotonic()-started,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 save(OUT/"COMPENDIUM_PREPARATION.json",receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k not in ("source_gate_snapshot","formal_status_snapshot","output_sha256")}))
if __name__=="__main__":main()
