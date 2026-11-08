#!/usr/bin/env python3
"""Package the final source review and accepted incremental table repair."""
from pathlib import Path
import csv,datetime,hashlib,json
R=Path(__file__).resolve().parents[1]
S=R.parent/'clinical_supplement_20261108_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
entries=[
('W01','Latest public review, not the eight November 6 reports','accepted_verified','inputs/WORKBUDDY_LATEST_SOURCE_RECEIPT.json; audit/LATEST_REVIEW_EXTRACTION_VERIFICATION.json','All 32 visible messages match the retrieved source. Latest substantive review is message 31.','analysis','closed'),
('W02','Do not expand hypothetical workflow simulation without new clinical information','accepted_no_redundant_analysis','PROTOCOL_LOCK.json; inputs/31_assistant.md','Use accepted S16 and November 7/8 resource analyses. No new arrival, staffing, latency, onset or efficacy assumptions.','analysis','closed'),
('W03','Retain and integrate the seven-question clinical supplement','accepted_existing_results','clinical_supplement_20261108_v1/ACCEPTANCE.json; audit/FROZEN_SOURCE_RECHECK.json','25 result tables, 16 reviewed claims, 3 figures and 8 panels already exist; 159 inputs and 90 outputs rehashed unchanged.','analysis_then_writing','scientific_closed_package_pending'),
('W04','Formal indications for individual diagnostic tests and operation-end triggers','qualified_to_supported_question','clinical_supplement_20261108_v1/results/POLICY_PAIRED_DIFFERENCES.csv; FIXED_TRIGGER_COMPARISONS.csv; FIELD_COMPLETION.csv','Report fixed-score information routing, equal-cost capture and source-threshold burden. Individual treatment/test indication is a different estimand.','analysis','closed'),
('W05','Actual daily or shift staffing quotas','qualified_to_cohort_demand','clinical_supplement_20261108_v1/results/DAILY_SUMMARY.csv; DAILY_TIME_SCENARIOS.csv','Report study-cohort date-indexed potential final reviews and assumed time scenarios, once per final selected operation; actual arrival and completion clocks absent.','analysis','closed'),
('W06','Performance in a new hospital','retain_transport_results_without_new_fitting','clinical_supplement_20261108_v1/results/CENTER_METRICS.csv; CENTER_SOURCE_COVERAGE.csv; MIMIC_BRIER_REVERIFIED.csv','Preserve fixed-model center/public results, source thresholds and sparse centers; no invented LOCO or local recalibration.','analysis','closed'),
('W07','Time from available prediction to complete KDIGO ascertainment','retain_distinct_recorded_clocks','clinical_supplement_20261108_v1/audit/ORIGINAL_COHORT_CLOCK_AUDIT.csv; MIMIC_KDIGO_CLOCK_AUDIT.json; INSPIRE_RECORD_INTERVAL_RECHECK.json','Preserve 148 measured SCr-component records versus 59 frozen calendar events (55 overlap), and 42.92-hour recorded surgery-start interval; do not convert either to warning lead time.','analysis','closed'),
('W08','System effects on mortality, RRT, hospitalization and other outcomes','retain_prognostic_analysis','clinical_supplement_20261108_v1/results/HARD_OUTCOME_DISCRIMINATION.csv; HARD_OUTCOME_ASSOCIATIONS.csv; LOS_ASSOCIATIONS.csv; audit/CAUSAL_IDENTIFIABILITY.json','Describe frozen AKI-score associations with recorded outcomes, including adverse/heterogeneous findings and endpoint-incorporated RRT. No implementation exposure/time-zero supports a causal contrast.','analysis','closed'),
('W09','Safety of lowering low-risk monitoring','retain_observation_tradeoff','clinical_supplement_20261108_v1/results/LOW_RISK_EVENTS.csv; MIMIC_LOW_RISK_EVENTS.csv; MIMIC_DELETION_SENSITIVITY.csv; audit/LOW_RISK_INTERVAL_UNIT_REVIEW.json','Report event-containing low-risk strata and losses of known events under fixed deletion schedules. False safety-identification flags mean unavailable evidence, not demonstrated harm.','analysis','closed'),
('W10','Claim of zero patient-level bootstrap in the November 8 supplement','corrected_review_inaccuracy','audit/BOOTSTRAP_PROVENANCE_CORRECTION.json','Original supplement: 2,000 patient-group samples per context; stored weight headers and independent replays verified. Current final source audit: zero new bootstrap draws.','analysis','closed'),
('W11','Repair mixed-metric eTable 4 from source Table_S19','confirmed_current_reporting_error_repair_supplied','results/ETABLE4_SOURCE_MAPPED.csv; docs/ETABLE4_REPLACEMENT.md; audit/ETABLE4_MAPPING_VERIFICATION.json','All 28 numeric rows match source rounding. Restore explicit metric/capacity/unit and correct title/footnote; no outcome estimate changes.','analysis_then_writing','mapping_closed_render_pending'),
('W12','Full selection-optimism correction or human expert panel','retain_actual_review_scope','clinical_supplement_20261108_v1/SCIENTIFIC_REVIEW.json; results/DEVELOPMENT_OPTIMISM_AUDIT.csv','Fixed-system uncertainty and fixed-choice refits remain distinct from full selection correction; numeric alternative implementations are not independent human experts.','analysis','closed'),
('W13','Convert the paper to an identifiability/methods paper','not_adopted_preserve_user_clinical_framing','inputs/13_assistant.md; inputs/31_assistant.md; user clinical-positioning instruction','Keep gastrointestinal cancer perioperative AKI information needs, assessment timing and review capacity as the clinical questions. Put material evidence limits once with the supported finding; do not make a list of unavailable estimands the main contribution.','writing','instruction_handed_off'),
('W14','Unconditionally require EHR extraction, Delphi, clinician experiments or historical quasi-experiments','not_added_inputs_unavailable','REQUEST.json; PROTOCOL_LOCK.json; inputs/31_assistant.md','These designs need data or participants absent from the authorized source. They are possible future routes, not unfinished mandatory computations. Do not invent experts or causal variables.','analysis','closed'),
('W15','Readable combined figures plus every independent panel PDF','confirmed_targeted_export_repair','writing owner figure PDF audit; clinical_supplement_20261108_v1/figures','Original 20 figures are at 183 mm with readable native fonts. New eFigures 16–18 shrink below 4.3 pt; repair their layout/fonts only, preserve numerical artists, export/recheck actual embedded sizes and standalone panels.','writing','repair_in_progress'),
('W16','Apply specified figure selection/library/nature/Python coding skills','authorized_supporting_workflow','REQUEST.json; writing owner skill/source/figure contract receipts','Use current-question figure decisions, ScientificFigureLibrary reference search, Python backend, Nature-style vector/editability/readability QA and ks-omics source/manifest checks. A named skill is not scientific validation.','writing','handoff_required'),
('W17','Finish IMRaD, legends/tables/supplements/cover letter','authorized_current_version_integration','REQUEST.json; writing owner existing complete package','Use a new writing version; preserve prior releases and all valid frozen estimates. Correct table mapping and necessary figure size issues, then review all manuscript parts together.','writing','in_progress'),
('W18','Direct, human-edited writing and unwanted watermark cleanup','authorized_faithful_editorial_review','REQUEST.json; anti-defensive-writing/humanizer/lieflat-less-ai-tone skills','Review final English with anti-defensive-writing and humanizer, Chinese where applicable with lieflat. Preserve numbers, citations, scientific uncertainty and truthful AI/author disclosures; check artifacts for unwanted watermark/DRAFT/tool-brand remnants.','writing','in_progress'),
('W19','Final package and complete accepted code upload','authorized_verify_new_release','REQUEST.json; existing publication receipts','Current previous upload is a baseline, not completion of this new request. Include this audit and new figure/reporting code, verify remote commit/content hashes and archive integrity. Exclude raw shared conversation, patient rows and exact surgery dates.','writing','pending_new_version'),
('W20','Author, ethics, funding, data access, AI responsibility and approval facts','retain_factual_author_slots','REQUEST.json; existing author-review documents','Use only verified supplied facts. Keep factual author decisions visible for author review; no invented confirmations and no journal submission claim.','writing','author_fact_items_retained')]
rows=[dict(zip(['id','recommendation','disposition','evidence','action_or_retained_result','owner','status'],x)) for x in entries]
csvout(R/'results/WORKBUDDY_DELTA_CLOSURE.csv',rows)
# These are quantitative findings already accepted, not newly estimated effects.
brief='''# 最终科学审读与写作交接

当前资料已完成可支持的临床分析；本轮新增工作是核对最新评审、纠正报告映射并审查图的刊用尺寸。保留胃肠肿瘤手术患者围术期AKI风险评估的临床定位。

## 可以直接写入稿件的结果

- 术前固定20%名单：9项/27项方案在地理评价分别捕获32/72和34/72事件，在时间评价分别捕获35/66和38/66事件。它们回答信息负担与名单捕获的关系，未设等效性检验。
- 既有选择性25%信息补充与同总成本扩容基础复核相比，9→27的每千例捕获差为−1.73（95%CI −6.88至4.32）和1.19（−1.19至3.57）；27→38为0.86（−1.73至2.60）和0.00（−2.97至2.97）。保留全部策略及代价敏感性结果；没有一致的增量支持把单一规则设成临床追加检查门槛。
- 术毕完整系统的增益随评价场景变化。eTable 4须分别标注AUROC、Brier、10%与20%捕获差；模型家族、开发和校准也随系统改变，不能把差异全部归给新增输入。
- 真实手术日期支持研究队列潜在复核需求：地理中心3的配对样本在690个有研究手术日产生438次最终名单复核，日均0.635、P95为2、最大3。人员工时仍是明确给定耗时下的资源情景，不是医院排班定额。
- 中心与公共数据评价提供固定模型迁移表现。小事件中心、阈值覆盖变化及MIMIC Brier不利结果均保留；不增加没有足够依据的目标医院重拟合。
- MIMIC严格测量时钟下的SCr组分阳性为148例，与冻结59例日历比值事件重叠55例；二者仍是不同终点。INSPIRE的42.92小时是手术开始到记录性肌酐标准的间隔。完整KDIGO及实际模型可用时钟未取得，故不另造预警提前量。
- 冻结AKI风险分数与死亡、RRT、再手术、再入院和住院日的预后关系已分析，稀疏及不利结果均在表中；RRT已纳入原AKI定义。系统实施效应没有可用的暴露及时零数据。
- 低风险组仍有事件。MIMIC每日仅留首次术后SCr删去522/3,586次检测（14.6%），损失7/59个已知事件（11.9%；患者bootstrap95%CI 4.3%–20.4%）。这是监测数量与记录性事件可见性的权衡；未测时段的事件和真实减测安全性没有被观测。

## 本轮必须落实的修复

1. eTable 4有28行，现表数值与Table_S19一致，但缺Metric/Capacity/Unit，标题脚注把所有行解释为20%捕获差。使用`results/ETABLE4_SOURCE_MAPPED.csv`和`docs/ETABLE4_REPLACEMENT.md`；逐行指标来自源字段，不按数值大小猜测。保留源CSV的精度，不再把3有效位展示误称为全精度。
2. eFigures16–18原生复合图宽约325–341mm，缩至183mm的最小字体约3.81–4.22pt。只修这3幅图的排版和字体，按最终Word/PDF置入宽度复核，不重绘原20图。单独panel PDF与复合图都要通过实际尺寸检查。
3. WorkBuddy称临床补充分析“0次新患者级bootstrap”有误：原4个评价各有2,000次患者分组重采样，NPZ头部为2001×N（含原点估计行），独立替代算法复算各2,000次。当前这轮来源/报告审计本身没有新bootstrap。主研究医院分层重采样与新增补充的患者分组重采样、每日7天区块、Wilson区间需分别按实际代码描述。

## 结果的组织方式

正文围绕术前信息需求、术毕重估和有限复核容量展开；先给观察结果和相应临床含义，再用一句说明该结果的适用范围。将必要的数据与实施限制集中在对应方法和Limitations，不把“没有证明什么”反复写成每一段的结尾。保留有利、阴性与场景间差异，不能把置信区间跨零解释为等效或完全无价值。

最新WorkBuddy最后不再建议扩展虚拟研究；本轮接受这一建议。到达时刻、实际耗时、信息延迟和AKI起病时刻没有新增资料，再增加分析者给定场景不会补足这些观测。正式Delphi、医生实验或准实验需要参与者/历史暴露，不能作为当前尚未运行的必做分析。

不采纳把全文转为“不可识别性边界方法学论文”的建议。用户已明确要求临床文章；现有有价值的结果足以围绕临床问题组织，边界服务解释。人工专家团、真实上线、临床安全或完整模型选择乐观度校正都没有新增证据，不得借技能或审计名称替代。

## 交付边界与剩余工作

本次来源和科学分析审读完成后，由`预测模型manuscript final`新建写作版本完成eTable4修复、3幅图排版、正文/附件/cover letter一致性审读、指定写作技能和水印检查、投稿包及完整代码远端上传。旧稿/旧发布保留。真实作者、伦理、资助和AI责任确认按已有可核实事实保留作者核对项；当前任务不把未知事实写成已确认，也不声称已实际向期刊投稿。
'''
(R/'docs/FINAL_SCIENTIFIC_HANDOFF_ZH.md').write_text(brief)
review={'status':'PASS_SOURCE_AND_SCIENTIFIC_ANALYSIS_REVIEW_WITH_TARGETED_REPORTING_REPAIRS','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'Codex; source matched audit and prior alternative numerical implementations; no human expert panel','source_request_sha256':sha(R/'REQUEST.json'),'protocol_sha256':sha(R/'PROTOCOL_LOCK.json'),'read_latest_share_messages':32,'axes_reviewed':{'data_and_units':'Corrected 3710/205 and paired2428/138 sources retained; operation rates and patient-group uncertainty remain distinct.','comparisons_and_inference':'Fixed-system, same-total-cost, source-threshold and cohort-capacity comparisons retained as different estimands. Four primary selective-capture intervals include zero, without equivalence inference.','measurement_and_time':'Recorded calendar endpoint, exact measured SCr components and complete KDIGO separated; no new biological or warning clocks imputed.','outcomes_and_safety':'Prognostic associations and measurement-deletion sensitivity do not estimate implementation effects; low-risk rates/rare strata retained.','review_provenance':'Original clinical supplement did use B2000; current audit uses no new resampling. Skills and alternative code do not constitute human reviewers.','reporting_repairs':'Current eTable4 mixed-metric error confirmed; unique source-based replacement supplied. Three oversize new figures need typography/layout correction.','writing_scope':'Keep clinical questions as the contribution; do not substitute an inventory of unavailable estimands for results.'},'quantitative_checks':json.loads((R/'audit/AUDIT_RUN.json').read_text()),'closed_analysis_issues':['Review bootstrap-count inaccuracy corrected against stored sampling headers and run/replay receipts.','28-row eTable4 mapping uniquely recovered without changing source estimates.','No additional simulated clocks/arrivals/efficacy scenarios warranted under existing evidence.'],'downstream_required':['Writing owner integrates source-mapped eTable4 and independently checks final render.','Writing owner repairs eFigures16–18 at actual embedding size and verifies all standalone panels.','Same final manuscript version receives substantive writing/statistics/source and release reviews.','Final package and accepted code include incremental audit/export fixes and verified remote release receipt.'],'new_fits':0,'new_bootstraps':0,'new_scenarios':0,'all_original_sources_preserved':True,'full_manuscript_review_in_this_analysis_branch':False,'final_submission_package_accepted_here':False,'GitHub_upload_by_this_analysis_branch':False}
save(R/'SCIENTIFIC_REVIEW.json',review)
print(json.dumps({'closure_rows':len(rows),'scientific_review_sha256':sha(R/'SCIENTIFIC_REVIEW.json'),'handoff_sha256':sha(R/'docs/FINAL_SCIENTIFIC_HANDOFF_ZH.md')},indent=2))
