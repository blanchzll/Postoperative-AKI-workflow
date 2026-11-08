"""Write reviewed analytical findings from verified current tables, not manuscript prose."""
from pathlib import Path
import json,csv,hashlib,datetime
import pandas as pd
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1")
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=pd.read_csv(A/"tables/PRIMARY_EQUAL_CAPACITY20.csv");b=pd.read_csv(A/"tables/ALL_SIGNED_BOUNDARIES20.csv");mi=pd.read_csv(A/"mimic_calendar_v2/INTERVALS.csv");mp=pd.read_csv(A/"mimic_calendar_v2/POINT_ESTIMATES.csv");cuts=pd.read_csv(A/"mimic_calendar_v2/SOURCE_THRESHOLD_BURDEN.csv")
lines=[]
for r in g.itertuples():
 q=b[b.comparison_id==r.comparison_id].iloc[0]
 lines.append(f"| {r.comparison_id} | {r.n}/{r.events} | {int(r.base_captured_n)} → {int(r.extra_captured_n)} | {r.delta_captured_per1000:.3f} [{r.delta_captured_per1000_lower95:.3f}, {r.delta_captured_per1000_upper95:.3f}] | {q.signed_allowable_extra_cost_per1000:.3f} [{q.lower95:.3f}, {q.upper95:.3f}] |")
mrows=[]
for r in mp.itertuples():
 def val(m):
  q=mi[(mi.model==r.model)&(mi.metric==m)].iloc[0]
  return f"{q.estimate:.3f} [{q.lower95:.3f}, {q.upper95:.3f}]"
 mrows.append(f"| {r.model} | {val('AUROC')} | {val('Brier')} | {val('calibration_slope')} | {r.capacity20_captured_n}/59 |")
text="""# 本次新增分析结果与统计/科学审查
分析分支：novelty_strengthening_20261101_v1；公共源当前版本：mimic_calendar_v2。
本报告仅审分析、代码、来源、结果与图表，不审或改稿件正文。整体交付状态由同目录ACCEPTANCE.json决定，旧37项及四补充验收不能代替本次验收。

## 本次实际新增
在正式数值计算前冻结的问题是：同一人群、同一总复核当量预算下，增加信息/推迟评分与扩展基础复核名单如何改变记录事件覆盖。14对比、raw/cal两视图、6档预算、31档增量采集情景实际计算5208行、168个带符号等捕获边界；患者组医院内2000次配对bootstrap、seed20261017，核验沿用原索引流，原模型/插补/校准/阈值不改，没有新增预测模型拟合。

公共源不再以INSPIRE充作新的独立来源。MIMIC-IV3.1实际扫描6364488诊断行、859655手术代码行及158374764检验行；独立来源日历敏感性实际评分、统计和复核完成。eICU审过结构化定义和候选路径后不评分。完整原文依据、资格审核及失败/修前证据全部保留。

## 主要结果：20%容量和联合预算
a=0时是既有同容量比较的核对，不冒充新增独立验证。表中最后一列D才是新增函数。区间均为逐点95%百分位区间，非同时置信带。

| 对比 | 人数/事件 | 基础→更多信息捕获数 | 捕获差/1000 [95%CI] | D：复核当量/1000 [95%CI] |
|---|---:|---:|---:|---:|
"""+"\n".join(lines)+"""
D=(基础整数名单K−更多信息路径达到同一基础捕获数所需最短名单)/N×1000。它只要求达到相同捕获数，并不要求严格多捕获；因此D的正下界和捕获差区间严格大于0的网格不必相同。它可为负；基础捕获为0的格单独标为平凡目标，不解释为效益。采集成本a为每名合格者追加的复核当量，追加路径名单floor(K−aN)；aN>K不可行，不能填成0收益。采用整数精确网格运算，保留原概率、患者组及记录顺序处理并列。

主要9→27比较在地理和时间线的边界区间均跨0；地理27→38主系统的正边界较明确，时间线区间很宽并跨0。同家族支持结果进一步限制归因：地理27→38 LR为13.817[−97.369,118.007]，时间27→38 LR为−25.717[−321.472,100.119]、Voting为−68.249[−275.283,56.436]。时间9→27 LR为98.633[25.564,140.916]；它不能被换成新的主赢家。完整参考和不利结果均在表中。

在预先固定的20%网格，地理27→38主对比只有a=0、0.01、0.02、0.03、0.04、0.05这6格的捕获差逐点区间完全在0以上；另外三个主要对比没有这样的格。所有其他预算、视图、参考对比、不可行格也完整列出。这是对固定网格的描述，不能用其最大格构造临床推荐界值，D下界也不是同时保障的部署成本上限。

以上比较要求追加采集任务和复核任务可折入同一可替代资源池；基础信息的共有采集成本被视为既有。实际工时、成本、岗位可替代性没有测量，字段数未换算成分钟。术毕评分发生较晚，结果仅比较各阶段冻结系统的回顾性覆盖，不能把差异全部归因于字段，不能推断实际避免AKI或推荐等待术毕。

## 新的MIMIC独立来源日历敏感性
病例定义在看新结局/模型指标前锁定：同住院恶性胃或结直肠肿瘤与相应切除代码解剖匹配、成人、每人首个合格切除。诊断为回顾性出院编码分型，不进入六输入，也不构成前瞻性入组工具。
1231人候选中467无合格前基线Cr、5无day+1..+7记录Cr，纳入759人/759患者组、59例记录SCr比值事件；胃157、结直肠602。Alb缺465(61.3%)，BUN缺33，Hb缺40，按冻结源插补处理；不是完整病例删选。后窗Cr记录数范围1–17、中位4，不声称每人完整随访7天。

基线为手术账单日期前30日最新、且chart/store均在该日开始前可见的测量；结局是该日期后日历+1至+7记录SCr≥1.5倍基线，排除day0。不补造切皮/术毕时钟、不增加0.3mg/dL或RRT未资格结局、不称完全相同的原7日KDIGO外验。年龄匿名化代理、测量选择、异质检验、回顾性编码和源Alb高缺失均限制运输解释。

两个冻结源RF在同759人上各报告raw/cal，共4概率视图。没有目标侧预测训练或再校准；目标校准回归仅是诊断，不用于更新风险。

| 源系统/概率视图 | AUROC [95%CI] | Brier [95%CI] | 联合校准斜率 [95%CI] | 目标top20%捕获 |
|---|---:|---:|---:|---:|
"""+"\n".join(mrows)+"""
目标排名20%是152人；原源20%风险切点在新来源则实际标记279/759(36.8%，捕获33/59)和210/759(27.7%，捕获22/59)。源80%敏感性切点分别标记501/759与759/759；后者覆盖全部人因此不是高效覆盖。50%/80%逆向名单和全区间另表保留，它们使用目标结局作回顾性摘要，不是可直接部署的切点。

这些结果支持保留迁移失败模式和源阈值负担差异，不能声称校准/性能普遍保留。没有把两个源RF或四视图当成四个独立外部样本。MIMIC为当前中国/INSPIRE之外的独立来源，但未声称整个研究过程从未看过该数据、或恢复全新盲法验证。

eICU不能满足本次原手术时间锚和六术前输入资格，故没有性能结果。原初审1768是广义肿瘤候选条目，实际路径计数为1412 GI路径及356其他泌尿生殖路径；均不是严格合格的本研究手术队列。宽标签已追加澄清，原审计不抹去。

## 真正发现并修复的错误
新MIMIC原始单位换算后的float边界漏掉恰好1.5倍的观测，改变3个事件标签和另外2个首次达标时点。按同一原阈值用Decimal、独立Fraction重建，当前59例；759人、所有输入、冻结模型概率和阈值均未改变。修前24文件可恢复备份核哈希，旧mimic_calendar的56例只作历史，不得写入新结果。只重算本轮MIMIC受影响统计，没有重训或改临床阈值。
新增图实际像素审读发现校准CI下界被−0.10坐标截断，已扩展到−0.16并补1000点边距、标题留白；原PDF/代码保留，数据未改。静态导出检查最初不识别循环导出，改显式SVG/PDF/TIFF/PNG后14项通过，不把工程检查冒充科学审查。

## 独立复算与适用范围
资源比较独立脚本未导入原引擎，逐个展开重采样患者的记录，核28个对比视图全部2000次新函数/区间、5208格及168边界；319503断言，最大差5.684e−14。病例输入以Fraction重放全部25487条已提取原文检验，4554输入/身份检查、所有759标签和首次时点零分歧；没有伪称第二次全扫描1.58亿行。
MIMIC统计独立脚本使用sklearn指标及statsmodels IRLS，原脚本为原生秩统计/Scipy优化。4视图所有2000次重采样的指标、校准回归、156区间行、66源切点指标行和8逆向覆盖行全部重算；16920断言，校准区间最大差约2.351e−7，其余浮点差在1e−9内。模型参数没有据目标结果更新。
区间以患者组为单位、保留医院层。条件在固定源模型和可评价目标记录上，未包含新开发/重训不确定性；多重逐点分析不作全局阳性或最优宣称。旧地理/时间273记录/272组重叠及既往评价暴露保持披露，不池化两线充作独立复现。

## 最近先例与可保留的新增价值
[Lei2019](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2757251)已有围术期多阶段输入和最高风险人群容量分层；[CoAI](https://doi.org/10.1038/s41551-022-00872-8)已有基于成本/时间的特征选择。本次不主张新成本敏感算法或绝对首次。可保留的具体增量是：用冻结临床系统把增加信息与扩大名单放到一个共同预算问题中，给出包含阴性/不一致结果的条件边界，并以另一个真实公共来源显示可运输范围。它提供可核查的流程选择依据，成本本身与干预效果仍需不同证据。原文阅读/检索范围为定向最近先例核查，不等于排除了所有历史先例。

## 结论处置与交接
ISSUE_AND_REMEDIATION.csv记录13个具体问题、影响、修复/限定和复核；CLAIM_TABLE_VERSION_MATRIX.csv将10项主张绑定源表、版本和哈希。确定计算/显示/元数据问题均已修复或隔离；无法补足的时间锚、测量和资源事实明确限制相应主张。原有效主模型、LLM等既有可靠结论不因新来源表现较差被一并撤回。

旧δ7仅是原20%容量、0.5一档网格中首个两线共同满足值，按未舍入上界严格U<δ；本轮D的复核当量量纲不改变其每1000例事件损失量纲。没有重启白蛋白累计窗因果分析，不恢复临床剂量/安全界值主张。
本次允许写作对话使用新审后证据；未在此修改任何稿件正文/图注原文件。正式主补图的选择、全文中英文整合及JNO提交包更新仍由既有final写作对话负责。原37项、四补充及最新投稿基线保护与本扩展验收分别记录。
"""
(A/"ANALYSIS_RESULTS_AND_REVIEW_ZH.md").write_text(text)
issues=list(csv.DictReader((A/"ISSUE_AND_REMEDIATION.csv").open()));claims=list(csv.DictReader((A/"CLAIM_TABLE_VERSION_MATRIX.csv").open()))
review={"status":"SCIENTIFIC_REVIEW_COMPLETED_ACCEPT_NARROWED_ANALYTICAL_CLAIMS_WITH_EXPLICIT_LIMITS","reviewer":"Codex current analysis owner; actual source/code/results/plots review, not external expert signoff","review_scope":"Only newly authorized novelty strengthening; no manuscript prose review","statistics_review":"Paired patient/hospital design, integer budgets, full new functions and CIs independently verified, all pointwise grids/NA/negative results retained.","scientific_judgment":"Retain conditional workflow tradeoffs and independent-source six-input calendar transport sensitivity. No uniform winning information range, measured cost, clinical effect or exact full-input external validation.","alternative_explanations":[{"explanation":"Different source learners/preprocessing/calibration or scoring time drive a workflow difference rather than added fields alone.","discriminating_evidence":"Existing locked same-family LR/Voting reference results are inconsistent; the present data do not isolate every component."},{"explanation":"New-source lab observation/missingness, baseline definition and coded phenotype rather than general predictor failure contribute to poor transfer.","discriminating_evidence":"Documented467/5exclusions,465missingAlb and calendar endpoint; an exact-clock, comparably measured cohort is not available here."}],"numeric_evidence":{"resource":"resource_comparison/INDEPENDENT_NUMERICAL_VERIFICATION.json","MIMIC_input":"mimic_calendar_v2/INDEPENDENT_INPUT_ENDPOINT_REVIEW.json","MIMIC_metrics":"mimic_calendar_v2/INDEPENDENT_NUMERICAL_VERIFICATION.json","exact_rebuild":"mimic_calendar_v2/EXACT_REBUILD_REPLAY.json"},"issues":issues,"claims":claims,"unresolved_confirmed_computational_errors_current_chain":0,"residual_limits_not_erased":["No exact verified surgical clock fornewpublicvalidation","No full27/38orLLMnewexternalvalidation","No measured cost/staff fungibility/intervention efficacy","Selection/missingness/calendar outcome and fixed-model conditional inference","No all-field or field-wide absolute novelty proof"],"skills_applied":["phd-skills experiment-design/dataset-curation supporting modules","nature-statistics reporting/unit/multiplicity/figures/reviewer rules","rw-research-referee strongest claims, bias, alternatives, severity, remedy/review","academic-plotting/scientific-visualization/nature-figure Python and figure-library availability review"],"source_api_draft_not_expert_review":True,"not_clinical_deployment_approval":True,"date":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"reviewed_evidence_hashes":{f:sha(A/f) for f in ["ISSUE_AND_REMEDIATION.csv","CLAIM_TABLE_VERSION_MATRIX.csv","ANALYSIS_PROTOCOL_LOCK.json","EVIDENCE_SOURCE_TABLE_LEDGER.csv","PUBLIC_METHODS_AND_PRIOR_ART_REVIEW.json","ANALYSIS_RESULTS_AND_REVIEW_ZH.md"]}}
(A/"SCIENTIFIC_REVIEW.json").write_text(json.dumps(review,ensure_ascii=False,indent=2)+"\n")
print("Written actual analytical/scientific report and13issue10claimreview")
