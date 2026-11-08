"""Render the complete locked LLM comparison as analysis supplements, never manuscript prose."""
import argparse,csv,math,os
from pathlib import Path
import common as c
SHORT={"biomedlm_2_7b":"BioMedLM","medgemma_4b_it":"MedGemma 1.5","llama3_8b_instruct_ngc_v1":"Llama 3 (NGC)"}
NAMES=[f"llm_{m}_{cond}_{stage}_cal" for stage in c.STAGES for m in c.MODELS for cond in ("before","after")]
NAMES += [f"{m}_{stage}"+("_cal" if m=="primary" else "") for stage in c.STAGES for m in ("primary","tabpfn")]
def label(name):
 for m in c.MODELS:
  if name.startswith("llm_"+m+"_"):
   tail=name[len("llm_"+m+"_"):].split("_");return f"{SHORT[m]} | {tail[1]} | {tail[0]}"
 return name.replace("primary_","Primary | ").replace("tabpfn_","TabPFN | ").replace("_cal","")
def style(name):
 if name.startswith("primary_"):return "#000000","s"
 if name.startswith("tabpfn_"):return "#D55E00","D"
 return ("#0072B2","o") if "_after_" in name else ("#666666","v")
def finite(v):return v is not None and math.isfinite(v)
def fmt(v,digits=3):return "NA" if not finite(v) else f"{v:.{digits}f}"
def run():
 c.check_code();c.numeric_guard()
 refs={}
 for name in ("INDEPENDENT_RESULT_VERIFICATION.json","RESOURCE_REPORT_ACCEPTANCE.json"):
  p=c.O/name;a=c.read(p);assert a["code_lock_sha256"]==c.check_code()
  c.verify_files(c.O,a["files"]);refs[str(p)]=c.sha(p)
  c.verify_files(c.O,a["inputs_sha256"])
 for line in c.LINES:refs[str(c.S/line/"RESULTS.json")]=c.sha(c.S/line/"RESULTS.json")
 with c.owned_lock("render"):
  if c.existing_receipt("RENDER_AND_ANALYSIS_REPORT.json",refs):return
  os.environ.setdefault("MPLCONFIGDIR",str(c.O/"matplotlib_cache"))
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  data={line:c.read(c.S/line/"RESULTS.json") for line in c.LINES}
  for line in c.LINES:refs[str(c.S/line/"RESULTS.json")]=c.sha(c.S/line/"RESULTS.json")
  plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.15,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
  dest=c.R/"outputs/final_analysis_delivery_20261025_v1/figures/LLM"
  dest.mkdir(mode=0o700,parents=True,exist_ok=True);files={};plot_rows=[];figures=[]
  def ci(line,name,metric,paired=False):
   rows=data[line]["paired_differences" if paired else "intervals"];key="comparison" if paired else "model"
   found=[x for x in rows if x[key]==name and x["metric"]==metric];assert len(found)==1
   return found[0]
  def dot(ax,row,y,name):
   color,marker=style(name);lo,hi,value=(row[k] for k in ("lower95","upper95","estimate"))
   if finite(lo) and finite(hi):
    ax.plot([lo,hi],[y,y],color=color,lw=1.4)
    ax.plot([lo,lo],[y-.06,y+.06],color=color,lw=1)
    ax.plot([hi,hi],[y-.06,y+.06],color=color,lw=1)
   if finite(value):ax.scatter([value],[y],s=25,c=color,marker=marker,zorder=3)
   else:ax.text(.98,y,"NA",ha="right",va="center",transform=ax.get_yaxis_transform(),fontsize=7)
  def finish(fig,name,foot):
   fig.text(.5,.008,foot,ha="center",va="bottom",fontsize=7)
   for ext in ("png","pdf","svg"):
    p=dest/(name+"."+ext);assert not p.exists(),"Do not overwrite a rendered result without preserved repair evidence"
    fig.savefig(p,dpi=300,bbox_inches="tight");files[str(p)]=c.sha(p)
   plt.close(fig);figures.append(name)
  foot="Exploratory pointwise 95% patient-group bootstrap intervals; fixed fitted predictions; no multiplicity adjustment."
  for line in c.LINES:
   fig,axes=plt.subplots(1,3,figsize=(13,8),sharey=True)
   for ax,metric,title in zip(axes,("AUROC","AP","Brier"),("AUROC (higher better)","Average precision (higher better)","Brier score (lower better)")):
    for i,name in enumerate(NAMES):
     row=ci(line,name,metric);dot(ax,row,i,name)
     plot_rows.append({"figure":"LLM_"+line+"_all_routes","context":line,"series":name,"metric":metric,**row})
    ax.set_xlabel(title);ax.set_ylim(len(NAMES)-.4,-.6)
   axes[0].set_yticks(range(len(NAMES)),[label(x) for x in NAMES])
   fig.suptitle(line.capitalize()+": all before/after calibrated LLM routes and fixed comparators")
   fig.tight_layout(rect=[0,.04,1,.96])
   finish(fig,"LLM_"+line+"_all_routes",foot+"\nBefore and after both use source-development calibration; raw scores remain in the complete tables.")
  # Fixed order, all six within-checkpoint information contrasts, both capacities and contexts.
  entries=[(m,cond) for m in c.MODELS for cond in ("before","after")]
  fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
  for row,line in enumerate(c.LINES):
   for col,capacity in enumerate((10,20)):
    ax=axes[row,col];metric=f"capacity{capacity}_captured_per1000"
    for i,(m,cond) in enumerate(entries):
     end=f"llm_{m}_{cond}_endop_cal";pre=f"llm_{m}_{cond}_preop_cal";name=end+" MINUS "+pre
     val=ci(line,name,metric,True);dot(ax,val,i,end)
     plot_rows.append({"figure":"LLM_paired_information_gain","context":line,"series":name,"metric":metric,**val})
    ax.axvline(0,color="#000000",lw=.8,ls="--");ax.set_ylim(len(entries)-.4,-.6)
    ax.set_title(f"{line.capitalize()} | {capacity}% review capacity")
    ax.set_xlabel("Endop minus preop AKI captured per 1000")
   axes[row,0].set_yticks(range(len(entries)),[SHORT[m]+" | "+cond for m,cond in entries])
  fig.tight_layout(rect=[0,.05,1,1]);finish(fig,"LLM_paired_information_gain",foot+"\nNegative and null increments retained; the two contexts contain overlapping patients.")
  # Full same-information reference comparisons; no selection of the best LLM.
  entries=[(m,stage,ref) for stage in c.STAGES for m in c.MODELS for ref in ("primary","tabpfn")]
  for line in c.LINES:
   fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True)
   for ax,metric,title in zip(axes,("AUROC","capacity10_captured_per1000","capacity20_captured_per1000"),
                             ("AUROC difference","10% capture difference per 1000","20% capture difference per 1000")):
    for i,(m,stage,ref) in enumerate(entries):
     after=f"llm_{m}_after_{stage}_cal";base=ref+"_"+stage+("_cal" if ref=="primary" else "")
     name=after+" MINUS "+base;val=ci(line,name,metric,True);dot(ax,val,i,after)
     plot_rows.append({"figure":"LLM_"+line+"_reference_comparisons","context":line,"series":name,"metric":metric,**val})
    ax.axvline(0,color="#000000",lw=.8,ls="--");ax.set_ylim(len(entries)-.4,-.6);ax.set_xlabel(title)
   axes[0].set_yticks(range(len(entries)),[SHORT[m]+" | "+stage+" vs "+ref for m,stage,ref in entries])
   fig.suptitle(line.capitalize()+": after-task PEFT versus fixed same-information comparators")
   fig.tight_layout(rect=[0,.04,1,.96]);finish(fig,"LLM_"+line+"_reference_comparisons",foot+"\nDifferences do not replace the locked primary winner or isolate architecture, size or pretraining domain.")
  resource_rows=list(csv.DictReader((c.O/"MODEL_RESOURCE_SUMMARY.csv").open()))
  fig,axes=plt.subplots(1,2,figsize=(11,4))
  for i,m in enumerate(c.MODELS):
   rr=[x for x in resource_rows if x["model"]==m and x["kind"]!="pilot"]
   for k,kind in enumerate(("before","fit")):
    z=next(x for x in rr if x["kind"]==kind)
    axes[0].barh(i+(k-.5)*.3,float(z["summed_completed_worker_seconds"])/3600,height=.28,
      color=("#666666","#0072B2")[k],label=("Before banks","Formal OOF/final fits")[k] if i==0 else None)
   axes[1].scatter(max(float(x["maximum_GPU_allocated_bytes"]) for x in rr)/1024**3,i,c="#0072B2",label="CUDA allocated" if i==0 else None)
   axes[1].scatter(max(float(x["maximum_worker_host_RSS_bytes"]) for x in rr)/1024**3,i,c="#D55E00",marker="s",label="Host RSS" if i==0 else None)
  for ax in axes:ax.set_yticks(range(3),[SHORT[m] for m in c.MODELS]);ax.invert_yaxis();ax.legend(fontsize=8)
  axes[0].set_xlabel("Sum of completed worker elapsed time (hours)");axes[1].set_xlabel("Maximum observed per-worker peak (GiB)")
  fig.tight_layout(rect=[0,.13,1,1])
  finish(fig,"LLM_observed_resource_cost","Pilots reported separately; elapsed time includes model loading and prediction. Retries/waits/downloads are not reconstructed.\nThese are the qualified fixed settings on this P40, not isolated architectural effects or total study wall time.")
  c.write_csv(c.O/"ALL_LLM_PLOT_DATA.csv",plot_rows)
  cal=list(csv.DictReader((c.O/"SOURCE_CALIBRATION_INDEPENDENT.csv").open()))
  report=["# 三个固定LLM的探索性比较：分析附件",
    "本文件是当前源分析结果汇总，不是论文正文，也不是全研究科学审查验收。",
    "主源3710/205；三模型比较使用配对2428/138（2411患者组）。地理评价1158/72，时间评价1011/36；两个评价集合存在患者重叠。",
    "三模型、同一检查点任务微调前后、术前27/术毕38字段与地理/时间四组合全部保留。微调前条件仍使用开发结局校准，不称无标签zero-shot。不得通过本补充结果重新选择主模型。",
    "概率来自固定0/1答案token的二元归一化分数；after的源开发预测由原3患者隔离折生成，最终开发模型只预测相应留出集。原始与校准两种结果均在POINT_ESTIMATES/INTERVALS完整保存。",
    "所有区间为医院内患者组配对2000次重抽样、seed20261017，条件于固定模型/校准/源阈值。这里没有重训bootstrap，不代表开发选择和新机构部署的全部不确定性。多重比较未校正，均为探索性，不能用是否跨0筛掉不利结果。",
    "同容量先分配整个人群复核名单；固定源阈值转移可能产生不同复核率，两者分开报告。TabPFN无授权的诚实源OOF阈值，固定源阈值指标留NA；不补造阈值。",
    "\n## 全部24个校准LLM结果",
    "|评价|模型|条件|时点|AUROC [95%区间]|AP|Brier|10%捕获/1000 [区间]|20%捕获/1000 [区间]|",
    "|---|---|---|---|---|---|---|---|---|"]
  def text_ci(line,name,metric):
   z=ci(line,name,metric);return f'{fmt(z["estimate"])} [{fmt(z["lower95"])}, {fmt(z["upper95"])}]'
  for line in c.LINES:
   for model in c.MODELS:
    for condition in ("before","after"):
     for stage in c.STAGES:
      name=f"llm_{model}_{condition}_{stage}_cal";point=data[line]["point"][name]
      report.append("|"+"|".join([line,SHORT[model],condition,stage,text_ci(line,name,"AUROC"),fmt(point["AP"]),fmt(point["Brier"]),
        text_ci(line,name,"capacity10_captured_per1000"),text_ci(line,name,"capacity20_captured_per1000")])+"|")
  report+=["\n## 全部比较与校准诊断",
    "每个评价集合27组预定比较：同检查点after-before、同条件术毕-术前、after与同信息主模型/TabPFN、以及配对信息增量的微调前后变化。两集合54组比较的所有指标/区间/有效重抽样数见PAIRED_DIFFERENCES.csv；不只报告方向有利的条目。",
    "原始及校准结果的校准截距/斜率是留出集描述性诊断，未用来改写预测。所有源校准系数和独立检查见SOURCE_CALIBRATION_INDEPENDENT.csv。负系数会反转风险排序，须作为比较行为解释，不按留出表现翻转或挑选比较器。",
    "|评价|模型|条件|时点|源校准截距|源校准斜率|负斜率|","|---|---|---|---|---|---|---|"]
  for x in cal:report.append("|"+"|".join([x["line"],SHORT[x["model"]],x["condition"],x["stage"],fmt(float(x["source_intercept"])),fmt(float(x["source_slope"])),x["negative_slope_preserved"]])+"|")
  report+=["\n## 来源与可解释范围",
    "BioMedLM固定官方3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f；MedGemma1.5-4B-it固定91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b。Llama3使用用户指定NVIDIA公开NGC1.0独立版本，导出张量和原Meta编码已资格验证；没有使用先前待授权HF8afb检查点，不能声称两者逐字节相同。ChatQA仓库只贡献来源明确的分词文件，不使用其模型权重/模板。",
    "医学域、模型大小、架构、原预训练及模板同时不同；不能由三者对比识别某一个因素的效应。仅使用同样的临床文字字段，没有影像、额外病例、自由生成风险百分比、合成推理监督或外部付费推理传输。",
    "正式比较属主评价已暴露后的追加探索，不能称新的盲外部验证。固定一次训练、原自然患病率与患者隔离，不按结果更改轮数、提示、阈值或赢家。资源比较另见RESOURCE_SCOPE_ZH.md。",
    "\n## 逐项待实际结果审读",
    "核对24行与完整原始诊断、全部54组配对对比及负/零/异质结果。对改善、无明显增益或不利的具体适用范围作实读记录，不用程序布尔值代替判断。六张新图的实际像素与计算器/既有图共同纳入最终视觉库存。",
    "本分析附件生成后仍须实际填写ANALYST_INTERPRETATION_AND_VISUAL_REVIEW.json，再由唯一所有者验收本补充项。全研究37项就绪后，另行执行正式统计—科学推断—整改复核；本附件不宣布该门槛通过。"]
  (c.O/"LLM_ANALYSIS_REPORT_ZH.md").write_text("\n\n".join(report)+"\n")
  for n in ("ALL_LLM_PLOT_DATA.csv","LLM_ANALYSIS_REPORT_ZH.md"):files[str(c.O/n)]=c.sha(c.O/n)
  c.save(c.O/"RENDER_AND_ANALYSIS_REPORT.json",{"status":"COMPLETE_RENDERED_ALL_LOCKED_LLM_RESULTS_ACTUAL_READING_REQUIRED",
    "time":c.now(),"inputs_sha256":refs,"code_lock_sha256":c.check_code(),"files":files,
    "figure_stems":figures,"figures":6,"actual_visual_review_complete":False,
    "actual_result_interpretation_required":True,"full_scientific_review_complete":False})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");a=p.parse_args()
 if not a.run:raise SystemExit("Explicit --run after numeric and independent checks required")
 run()
