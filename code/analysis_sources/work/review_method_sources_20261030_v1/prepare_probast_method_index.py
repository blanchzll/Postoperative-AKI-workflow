#!/usr/bin/env python3
"""Prepare a source-bound method index, without assessing study quality."""
from pathlib import Path
import csv,datetime,hashlib,json,os,re,resource,signal,time,xml.etree.ElementTree as ET
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
D=R/"outputs/review_method_sources_20261030_v1"
O=D/"probast_item_scope_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
 return h.hexdigest()
def save(p,v):
 with Path(p).open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n")
 os.chmod(p,0o600)
COMMON={
 "1.1":("来源资格","逐来源核对纠正源、手术记录与真实医院人群，公共来源单列适用范围。"),
 "1.2":("设计与时间结构","逐评价设计核对地理、时间、患者组边界及原评价暴露。"),
 "1.3":("纳排与代表性","核纳排、重复手术、缺失终点/化验和POD1条件人群，保留选择限制。"),
 "2.1":("字段与测量","核原字段码、单位、各中心及公共来源测量差异，不猜未知代码。"),
 "2.2":("预处理边界","核插补训练侧、筛选、变换、当前输入及缺失处理的一致实施。"),
 "2.3":("原测量与结局知识","区分原临床记录测量是否知晓结局与算法监督学习，不以无y输入替代测量盲法证据。"),
 "2.4":("使用时点","逐27/38、common6及POD1路线核输入可用时点；不把术后字段放入术前。"),
 "3.1":("终点定义","核KDIGO/真实RRT来源、纠正205/138、严格POD1与公共结局定义。"),
 "3.2":("终点测量一致性","核各中心及公共来源肌酐测量机会、终点缺失和条件分母。"),
 "3.3":("终点判定与预测信息","核判定公式、原记录及是否存在知晓预测字段的终点认定，不凭程序化公式推定临床盲法。"),
 "3.4":("预测与结局间隔","核预测锚点、POD1条件性与记录窗口；不把首次记录达标当生物起病。")
}
DEV={
 "4.1":("开发信息量","按具体模型核开发事件数、复杂度/约束和不确定性，不用单一事件数规则替代判断。"),
 "4.2":("变量表示","核原类别码、连续输入及样条，不以展示分位数或80g参照制造分组/推荐阈值。"),
 "4.3":("缺失与删失","核逐训练折插补、缺失选择及可评价终点；未知剂量不填0。"),
 "4.4":("开发患病率处理","核原家族的权重/重采样与后续开发校准；三LLM正式训练保留自然患病率。"),
 "4.5":("过拟合控制","核实际嵌套选择/折内流程、固定合同和评价暴露，不按留出结果追加搜索。")
}
EVAL={
 "4.1":("评价类型","逐路线区分开发表观、内部、地理、时间、公共和后验目标侧更新。"),
 "4.2":("评价精度","核评价事件数、配对患者区间及稀疏亚组，不把宽区间或阴性结果等同无效/等效。"),
 "4.3":("评价缺失与删失","核实际评价名单、终点可评价性、分母与时间掩码纠正，不丢未触发事件。"),
 "4.4":("评价患病率","核留出评价自然患病率；四开发患者工程测试不作性能证据。"),
 "4.5":("泄漏与隔离","核患者组、重复手术、所有训练步骤与留出边界；披露地理/时间评价重叠和暴露。"),
 "4.6":("重抽样对象","先核内部开发重抽样是否含完整拟合链，再区分固定模型留出患者bootstrap的条件推断范围。"),
 "4.7":("性能和用途","核区分度、校准、容量/源阈值、净获益、资源和不利结果，不由单一AUROC选择新赢家。")
}
def components(phase,item):
 if item.startswith("1."):return ["primary_A","final_evaluation","common6_source_and_public_review","POD1_results_review","four_issue_resolution"]
 if item.startswith("2."):return ["primary_A","model_lock","policy_lock","secondary_common6","secondary_pod1","llm_three_model_supplement","calculator"]
 if item.startswith("3."):return ["primary_A","secondary_outcomes","secondary_pod1","chronology_availability","stratified_and_subgroups"]
 if phase=="development":
  return ["primary_A","primary_B","primary_C","run_d","verify_d","run_e","model_lock","policy_lock","secondary_fixed_flow","secondary_pod1","secondary_common6","llm_three_model_supplement"]
 return ["final_evaluation","secondary_fixed_flow","common6_source_and_public_review","POD1_results_review","four_issue_independent_calculation","llm_three_model_supplement","decision_workload"]
def main():
 began=time.monotonic()
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
 if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 frozen={
  R/"AGENTS.md":"b550409c427153e92b13dfd92870ae4838e76de12808f84c4f90630e16c9bbf1",
  R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json":"afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3",
  R/"work/llm_supplement_20261028_v1/formal_benchmark_v1.py":"05040a02b79880f47ea7c480c34e4f4db62e85494cb89143ea5f3bbb79e2ddce",
  R/"outputs/llm_supplement_20261028_v1/formal_v1/FORMAL_TRAINING_CONTRACT.json":"eb27122860f1500b848efeea0ce1a19846e7d94aa5d0de7adbabc2c8db2041de",
  R/"outputs/post_reanalysis_process_review_v1/REQUIRED_COMPONENTS.json":"117c5287a0797aada358bc4727cd801e2d388e93a2b7457c541f1d3ad3da5690",
  D/"probast_ai_europepmc_fulltext.xml":"69f1f7f327b46ee945d7ba08b70326199153a48ea6a0823298aa7baa8638715d"
 }
 for p,h in frozen.items():assert sha(p)==h,str(p)
 protected=json.loads((R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json").read_text())["files"]
 for p,h in protected.items():assert sha(p)==(h["sha256"] if isinstance(h,dict) else h)
 required=json.loads((R/"outputs/post_reanalysis_process_review_v1/REQUIRED_COMPONENTS.json").read_text())
 valid={x["component"] for x in required["components"]};assert len(valid)==37
 root=ET.parse(D/"probast_ai_europepmc_fulltext.xml")
 ids={x.attrib.get("pub-id-type"):"".join(x.itertext()) for x in root.findall(".//article-id")}
 assert ids.get("doi")=="10.1136/bmj-2024-082505" and ids.get("pmid")=="40127903" and ids.get("pmcid")=="PMC11931409"
 date=root.find(".//article-meta/pub-date");assert [date.findtext(x) for x in ["year","month","day"]]==["2025","3","24"]
 phase=None;rows=[];source_cells=[]
 for tri,tr in enumerate(root.findall(".//table-wrap[@id='tbl2']/table//tr")):
  cells=[" ".join("".join(x.itertext()).split()) for x in tr if x.tag in ("td","th")]
  if cells==["Model development"]:phase="development";continue
  if cells==["Model evaluation"]:phase="evaluation";continue
  if phase is None:continue
  for ci,text in enumerate(cells):
   m=re.match(r"^([1-4]\.[1-9])\s+(.+)$",text)
   if not m:continue
   item=m.group(1);title,question=(COMMON if item in COMMON else DEV if phase=="development" else EVAL)[item]
   comps=components(phase,item);assert set(comps)<=valid
   locator=f"table-wrap[@id=tbl2]/table//tr[{tri}]/cell[{ci}] (zero-based index)"
   source_cells.append({"phase":phase,"item":item,"source_cell_locator":locator,"source_question_sha256":hashlib.sha256(m.group(2).encode()).hexdigest()})
   rows.append({"phase":phase,"item":item,"domain":item.split(".")[0],"review_topic":title,"project_evidence_question":question,"component_candidates":"|".join(comps),"source_table":"PROBAST+AI Table2","source_question_sha256":source_cells[-1]["source_question_sha256"],"study_assessment_status":"PENDING_ACTUAL_FINAL_REVIEW","answer":"","domain_judgment":"","evidence_locations":"","rationale":""})
 assert len(rows)==34 and sum(x["phase"]=="development" for x in rows)==16 and sum(x["phase"]=="evaluation" for x in rows)==18
 assert len({(x["phase"],x["item"]) for x in rows})==34
 O.mkdir(mode=0o700)
 with (O/"PROBAST_AI_PROJECT_ITEM_INDEX.csv").open("x",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 save(O/"SOURCE_TABLE_CELL_BINDINGS.json",{"article_xml_sha256":sha(D/"probast_ai_europepmc_fulltext.xml"),"cells":source_cells,"verbatim_questions_not_duplicated":True,"original_licensed_source_retained":True})
 old=R/"outputs/knowledge_guided_revision_20261029_v1/source_checks/PRIMARY_SOURCE_VERIFICATION.json"
 sections=["Assessment of quality (model development) versus risk of bias (model evaluation)","Steps for using PROBAST+AI","Multiple PROBAST+AI assessments and extending answers from model development to model evaluation"]
 save(O/"SOURCE_READING_DEPTH_ADDENDUM.json",{
  "status":"PROBAST_AI_MAIN_ARTICLE_AND_ALL_TABLE2_ITEMS_ACTUALLY_READ_NOT_STUDY_SCORED",
  "doi":ids["doi"],"pmid":ids["pmid"],"pmcid":ids["pmcid"],"publication_date_from_original_xml":"2025-03-24",
  "fulltext_source_url":"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11931409/fullTextXML",
  "publisher_url":"https://www.bmj.com/content/388/bmj-2024-082505",
  "old_source_depth_record":{"path":str(old.relative_to(R)),"sha256":sha(old),"old_record_preserved":True,"old_PROBAST_abstract_only_depth_superseded_for_this_named_source_only":True},
  "actual_reading":["Table2 all16 model-development and18 model-evaluation signalling items",*sections],
  "development_signalling_items":16,"evaluation_signalling_items":18,
  "supplementary_EE_and_instructions_read":False,
  "access_diagnostics":{"publisher_HTML":"403","PMC_HTML":"reCAPTCHA","NCBI_OA_metadata":"404","EuropePMC_JATS":"200 exactDOI/PMID/PMCID verified","EuropePMC_supplement_zip":"HTTP200 then finite read timeout; no PDF accepted or read"},
  "paper_search_identity_receipt_sha256":sha(D/"PAPER_SEARCH_IDENTITY_RECEIPT.json"),
  "paper_search_date_warning":"The CLI year-only January1 placeholder is not the publication date; original JATS day/month/year checked.",
  "paper_search_skill":{"path":"/home/lei/codex-workspaces/paper-retrieval-tools/paper-search-mcp/claude-code/SKILL.md","sha256":"affddf3a4dc463e42cc56714819109333665e813e5c46ba6085744a14ada6161"},
  "biomcp":"No enabled connector or usable server executable established; no BioMCP execution claimed.",
  "license":"Article source states CC BY-NC4.0. Internal method reference; original text/XML/PDF not automatically selected into a shared archive.",
  "study_scoring_started":False,"scientific_acceptance":False,"required_component_count_changed":False
 })
 doc="""# PROBAST+AI原文补读与本研究的审查入口

原文：Moons等，BMJ 2025；388:e082505，doi:10.1136/bmj-2024-082505。官方Europe PMC接口取得179719字节JATS全文，DOI/PMID/PMCID及2025-03-24日期核对一致。已实际读取表2全部16项开发问题、18项评价问题，以及开发质量/评价偏倚区别、四步使用和多模型分别评价的章节。此前只读摘要的历史记录保留，本增补仅更新这一来源的实际阅读深度。详细解释附件尚未取得或读到，不声称已完成正式PROBAST评分。

对后续研究过程审查的应用：
- 分开判断开发质量、性能估计偏倚、估计精度与适用范围。模型性能不佳不能单独证明其留出估计无效。
- 每个实质不同模型和评价情境应有明确对象；同源事实可复用，共同变量公共评价及POD1条件终点要另核人群、测量与时间。
- 单项不满足先说明其实际影响，再形成领域判断；不以问题数量机械否定整个研究。
- 审查4.6时先明确重抽样的任务。本项目对完整开发链的内部验证与对固定留出预测的患者bootstrap是不同推断对象；是否存在遗漏必须查代码和实际边界，不为套条目重跑有效拟合。此项为项目应用判断，须在最终审读中核实。
- 记录评价暴露、患者重叠、稀疏事件、模型/校准器冻结边界及不利结果；无信息按未知处理，不补造测量盲法或时间。

PROBAST_AI_PROJECT_ITEM_INDEX.csv按原表定位绑定全部34项，仅给出本研究待查问题与37组件候选入口；答案、领域判断、理由仍空，明确PENDING_ACTUAL_FINAL_REVIEW。它是已读方法资料和证据索引，不能充当研究审查结果。全部37实际就绪后才按既定snapshot/numeric及三轮审读执行。

访问记录如实保留：BMJ403、PMC网页验证、NCBI元数据404；合法公开的Europe PMC全文API成功。补充ZIP返回200后读取超时，未接受PDF，不绕过网页验证、不关闭TLS或新建环境。未更改数值方法、种子、模型、源、37门槛或保护稿件。
"""
 with (O/"METHOD_READING_AND_APPLICATION_ZH.md").open("x",encoding="utf-8") as f:f.write(doc)
 for p,h in frozen.items():assert sha(p)==h,str(p)
 for p,h in protected.items():assert sha(p)==(h["sha256"] if isinstance(h,dict) else h)
 outputs={x.name:sha(x) for x in O.iterdir() if x.is_file()}
 receipt={"status":"METHOD_SOURCE_AND_34_ITEM_INDEX_READY_NOT_STUDY_SCIENTIFIC_ACCEPTANCE","time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_sha256":sha(__file__),"source_xml_sha256":sha(D/"probast_ai_europepmc_fulltext.xml"),"outputs":outputs,"items":34,"study_answers_or_judgments_written":0,"original_frozen_files_unchanged":True,"protected92_hashes_unchanged":True,"candidate_analysis_executed":False,"new_fits":0,"source_download_and_reading_not_a37_component_acceptance":True,"elapsed_seconds":time.monotonic()-began,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 save(O/"METHOD_INDEX_PREPARATION_RECEIPT.json",receipt)
 print(json.dumps({"receipt":str(O/"METHOD_INDEX_PREPARATION_RECEIPT.json"),"sha256":sha(O/"METHOD_INDEX_PREPARATION_RECEIPT.json"),"items":34,"elapsed_seconds":receipt["elapsed_seconds"],"peak_rss_bytes":receipt["peak_rss_bytes"]}))
if __name__=="__main__":main()
