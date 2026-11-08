from pathlib import Path
import json,re,csv,shutil
W=Path(__file__).resolve().parents[1];B=W.parent;R=W/'review';P=W/'package';A=P/'03_Author_review';T=W/'work/text'
report='''# 最后一轮科学与逻辑审读

本稿具有可保留的应用研究增量，创新性中等。主要价值是把信息采集、评分时点和有限复核容量组织为可量化的流程选择：在相同记录、相同容量下估计27项减为九项的捕获代价，并报告术毕更新、名单扩容及源阈值迁移的实际操作后果。现有结果支持这些固定流程在已观察记录上的条件性比较。

## 近邻研究与新增价值

| 最近先例 | 原始研究实际问题与方法 | 与本稿关系及新增内容 |
| --- | --- | --- |
| OAKS，2022，doi:10.1093/bjsopen/zrab150 | 六项术前变量；两个胃肠/腹部手术外部队列；区分度、校准及预定概率风险组。 | 精简变量已有成熟先例。本稿量化27项减为九项在相同名单容量下的成对捕获差。 |
| Skovsgaard，2026，doi:10.1093/bjsopen/zrag124 | 丹麦结直肠癌登记队列；术前逻辑回归、连续变量样条、插补及内部bootstrap。 | 与本稿共享临床预测场景。本稿主要增加固定容量的流程取舍与地理、时间两线结果。 |
| Lei，2019，doi:10.1001/jamanetworkopen.2019.16921 | 比较围术期阶段；按最高风险20%分层，并检查10%及30%。 | 固定比例捕获及术中更新均有直接先例。新增价值集中于减少字段的成对代价、容量边际变化及迁移名单与覆盖的联合报告。 |
| Kim，2021，doi:10.1213/ANE.0000000000005057 | 腹内手术内部开发/评价，加入术中资料，并使用平衡敏感度与特异度的风险阈值。 | 说明增加术中资料本身不能承担新颖性；本稿报告完整系统的成对操作差异及同家族支持结果。 |
| Do，2026，doi:10.1093/jamiaopen/ooag092 | INSPIRE的2/3期AKI；术前、术中及组合模型，并比较时序架构。 | 数据和术中信息整合均有先例。本稿公开数据部分关注源规则迁移后的名单规模、实际覆盖和编码测量条件。 |

来源读取：OAKS、Lei原始全文XML；Kim作者稿BioC全文；Skovsgaard、Do已保存原始全文XML。最近先例的核心方法和用途均进入比较。Zhuo（doi:10.1016/j.bja.2024.02.018）的外部校准、净获益评价按此前已核查的完整摘要范围使用。检索结合服务器paper-search-mcp的PubMed/Crossref及网页检索；完整查询与返回在literature/。本轮检索支持具体增量定位，未证明整个文献领域不存在更接近的研究。

可用于投稿定位的核心段落：本研究将术后AKI预测的输入需求与有限复核能力相连接，估计减少字段在相同记录、相同容量下的捕获代价，并将该结果与术毕更新和阈值迁移后的名单规模、事件覆盖并列呈现。相比已有精简模型、阶段比较和高风险分层，新增的是围绕同一服务配置问题的成对量化评价。实际采集省时及患者获益仍需在工作流实施中测量。

## 审读视角一：科学设计与统计

读者为围术期、肾脏病与临床预测方法研究者。优势是使用患者组划分及重抽样、相同权重的成对对比，并保留不利支持结果。核对评价代码后，固定容量使用ceil(qN)，按医院分层抽取患者组，每次重新排序/路由；操作指标方向与正文相符。

| 编号 | 定位与证据 | 判断与处理 |
| --- | --- | --- |
| S01 | Methods：评价边界；冻结分母与病例流程；273条/272组重叠。 | 两条评价线共享记录并有既往评价暴露，不能被解释为两次未触及的独立验证。正文原有事实保持，检查无“全新独立验证”主张。 |
| S02 | Results：δ网格；Table_S13_MARGIN_SENSITIVITY.csv。 | 原始上界6.914433880726012，严格U<δ；20%下每0.5一档的首个共同值为7。保留事后性质和原始界值，未改阈值或重新估计区间。 |
| S03 | 阶段比较及同家族LR结果；eTable 4。 | 完整系统比较有明确用途。将解释直接写成该估计对象，保留同家族、去重叠、ALT及时间不利结果。 |
| S04 | 公共编码九输入与六输入比较；eTables 6–7。 | 比较的是模型/预处理/编码测量共同构成的流程。源阈值下名单与覆盖同时报告。 |
| S05 | POD1、分配、稳定性及情景结果。 | 区分条件性终点、患者抽样区间、随机分配变异和假设情景。没有把计算探针通过解释为名单稳定或临床有效。 |

科学性结论：研究问题与主要估计对象能够对应，结论保留在实际证据范围。记录性终点、可评价性选择、既往暴露、固定流程区间及未测量实施效果是现有证据条件；本轮不以改写消除这些条件。

## 审读视角二：创新性与研究意义

读者为安排术后监测与复核资源的临床服务。最重要的修正是明确Lei原文已经使用最高风险20%并考察其他容量，避免把固定容量本身当作方法创新。OAKS等已有更少变量，九项输入的价值由同记录成对结果及本地采集条件决定。新增字段数、公开队列和模型组合不单独构成创新。

N01/N03在引言与讨论中补足该最近先例。N02将新增贡献具体写成在记录和容量固定时估计字段减少的代价。创新性评价为有意义的临床应用与操作评价增量；现有证据没有建立算法创新或已测量的经济/临床效益。

## 审读视角三：框架、逻辑与可读性

全文按“输入减少的捕获代价—术毕更新—容量与阈值操作后果”展开。L01明确采集字段与复核名单是不同工作量。L02使目标与估计对象逐项对应。L03将讨论开头改为实际捕获数量；L04/L05直接定义系统比较；L06使结论落在可核查的数量和本地决策用途。删除结果标题中的旧4.x层级，保留正式期刊小标题。

新建cover letter，合成包含eMethods、eResults、19个编号eTable和14个eFigure的单一补充Word/PDF，并独立提供全部17幅图注。两张主表保持可编辑三线表；原始图形导出逐字节复用。期刊eFigure/eTable编号与历史源文件标识通过交叉表连接。

## 综合判断与实改记录

科学主线成立，贡献定位已收紧到实际完成的成对流程评价。需修正的是贡献归属、目标与估计对象衔接、讨论结果顺序、期刊补充材料结构和交付完整性。CORRECTIONS.json保存12项主文修订；当前图表、披露、代码与语言处理另有验收记录。

防御性检查只检查实际文本一次：保留会改变结果解释的事实，去掉无信息的自证、泛化价值判断及假想反驳。未添加新的实验要求，未调整冻结结果，也未将内部三个审读视角包装为外部独立同行评审。

作者还需依据原始资料完成伦理、作者/单位/贡献、基金/冲突、数据与模型访问、中心/方案或注册、患者公众参与、主观结局/预测指标摘录人员及盲法（如适用）、AI版本和使用历史，以及投稿信中的原创性、相关稿件和作者批准事实。清单单独保存在AUTHOR_ITEMS_ZH.md。
'''
(R/'FINAL_SCIENTIFIC_REVIEW_ZH.md').write_text(report)
# Exact web query provenance complements the installed server retrieval logs.
webqueries=['postoperative acute kidney injury prediction fixed review capacity reduced inputs capture risk model','acute kidney injury resource constrained feature acquisition prediction perioperative','risk stratification postoperative acute kidney injury Lei 2019 top risk 20% 2019.16921','10.1093/jamiaopen/ooag092 intraoperative integration AKI','"acute kidney injury" "prediction" "capacity" "surgery" model','"acute kidney injury" "parsimonious" "gastrointestinal"','"acute kidney injury" "feature acquisition" model','"postoperative" "AKI" "workload" prediction']
(R/'literature/WEB_QUERY_PROVENANCE.json').write_text(json.dumps({'date_Asia_Shanghai':'2026-11-01','queries':webqueries,'closest_primary_articles':['https://doi.org/10.1093/bjsopen/zrab150','https://doi.org/10.1093/bjsopen/zrag124','https://doi.org/10.1001/jamanetworkopen.2019.16921','https://doi.org/10.1213/ANE.0000000000005057','https://doi.org/10.1093/jamiaopen/ooag092'],'coverage_limit':'Scoped novelty review, not a systematic review or evidence of absence; no first claim'},indent=2))
(R/'API_METHOD_REVIEW.json').write_text(json.dumps({'status':'REVIEWED_PUBLIC_METHODS_ASSISTANCE','receipt':str(next((W/'outputs/research_api_rewards').glob('*/receipt.json'))),'private_manuscript_or_patient_data_sent':False,'codex_review':['Read response and verified generic equal-capacity and fixed-threshold definitions','Primary Lei methods establish top20 with10/30 sensitivity; incorporated explicitly rather than inferring missing evaluation','OAKS probability-risk categories distinguished from top-fraction capacity','No novelty verdict delegated to API output'],'model_configuration_changed':False},indent=2))
# A concise reporting map supplements the original checklist, marking facts that remain genuinely unavailable.
items=[('1','Title','MANUSCRIPT title','Reported'),('2','Abstract','Structured Abstract','Reported'),('3','Context','Introduction; Discussion','Reported; inequalities not specifically evaluated'),('4','Objectives','Introduction final paragraph','Reported'),('5','Data sources and dates','Methods; eMethods 8','Reported'),('6','Eligibility and setting','Methods; eTable 2','Center identities pending'),('7','Data preparation','eMethods 1, 3, 8','Group fairness-specific QC not established'),('8','Outcome','Methods; eMethods 3, 8','Assessor/blinding facts pending'),('9','Predictors','Methods; eTable 1; source dictionary','Measurement assessors/blinding pending'),('10','Sample size','eMethods 8; cohort flow','Available eligible records; no new calculation'),('11','Missing data','eMethods 1, 8; eTable 10','Reported'),('12','Analysis methods','Methods; eMethods; code','Reported'),('13','Imbalance handling','eMethods 8','Natural prevalence retained'),('14','Fairness','eTable 16; stratified source tables','Subgroups retained; formal fairness analysis not established'),('15','Model outputs','eMethods; research calculator code','Clinical decision thresholds require local evaluation'),('16','Development/evaluation differences','Methods; Discussion','Reported'),('17','Ethics','Acknowledgment','Author documentation pending'),('18','Open-science declarations','Acknowledgment; code README','Funding, conflicts, protocol/registration and access pending'),('19','Public involvement','AUTHOR_ITEMS_ZH.md','Author fact pending'),('20','Participant description','Table 1; eTable 2','Reported'),('21','Analysis counts','Table 2; eTables; source CSV','Reported'),('22','Full model specification','Analysis source; environment records','Trained states/access arrangement pending'),('23','Performance','Figures 1–3; eTables; source CSV','Reported'),('24','Model updating','eMethods 2; eTable 14','Reported'),('25','Interpretation','Discussion','Reported'),('26','Limitations','Methods; Discussion','Reported'),('27','Intended use and next steps','Discussion; Conclusions','Reported')]
with (P/'01_Journal_files/REPORTING_CHECKLIST.csv').open('w',newline='') as f:
 out=csv.writer(f);out.writerow(['TRIPOD_AI_item','Short_topic','Location','Status']);out.writerows(items)
(P/'01_Journal_files/REPORTING_CHECKLIST_README.md').write_text('TRIPOD+AI reporting map, with pending items left visible. This is a study-specific location map, not a claim that every subitem is complete. Official expanded checklist: [TRIPOD+AI](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf).\n')
# Journal guidance summary is intentionally compact; exact official page is the controlling source.
(R/'JOURNAL_REQUIREMENTS.json').write_text(json.dumps({'journal':'JAMA Network Open','checked_Asia_Shanghai':'2026-11-01','source':'https://jamanetwork.com/journals/jamanetworkopen/pages/instructions-for-authors','article_type':'Original Investigation','main_text_limit':3000,'abstract_limit':350,'key_points_limit':100,'title_character_limit':100,'main_display_limit':5,'main_format':'Word, double-spaced, ragged right, 10–12 point; superscript references; numbered pages','supplement_format':'Single editable Word and accompanying PDF; eMethods/eTables/eFigures, separate supplementary references; 10-point Times New Roman single spacing','figure_format':'Separate accepted vector PDFs retained','cover_letter':'Corresponding author contact and related-manuscript facts','ai_disclosure':'Acknowledgment: tool, version, producer, dates, uses, author responsibility','file_size_recommendation':'Individual <=1 MB and combined <=5 MB; supplementary Word may exceed the recommended individual size; see final measured sizes'},indent=2))
# Reports are author-facing material, outside the set of manuscript-upload files.
for name in ['FINAL_SCIENTIFIC_REVIEW_ZH.md','CORRECTIONS.json','MANUSCRIPT_REVISIONS.diff','CHINESE_REVISIONS.diff','NUMBERING_CROSSWALK.json','ETABLE_SOURCE_MAP.json','LANGUAGE_PASS_RECEIPT.json','LESS_AI_COMPARISON.md','LANGUAGE_HIT_DISPOSITIONS.json','JOURNAL_REQUIREMENTS.json']:
 shutil.copy2(R/name,A/name)
for file in ['CITATION_AUDIT.md','CITATION_AUDIT.json']:
 shutil.copy2(B/file,A/('RECEIVED_'+file))
with (A/'AUTHOR_ITEMS_ZH.md').open('a') as f:f.write('\n- 按事实核实结局及需主观摘录的预测指标的评估人员、资格与盲法；未记录则如实说明。\n- 上述报告清单为位置核对，不将待补项标记为已完成。\n')
readme='''# 投稿材料使用说明

目标期刊：JAMA Network Open；稿件类型：Original Investigation。

所有本轮材料在本目录内。作者原始事实待补项列于03_Author_review/AUTHOR_ITEMS_ZH.md，投稿信内相应位置保留明确占位。

| 目录 | 内容与用途 |
| --- | --- |
| 01_Journal_files | Cover letter、英文正文Word、两张三线表（合并及单表）、完整figure legends、3幅主图PDF、单一补充Word/PDF、报告规范位置清单。正文已包含两表与主图图注，上传时依系统文件类型选择对应版本。 |
| 02_Code_and_aggregate_data | 分析源码、作图/文档代码、环境规格、完整聚合CSV与可执行聚合算术核对。 |
| 03_Author_review | 中文审阅稿、最终审读报告、修改与编号交叉表、语言处理记录、作者待补。PDF清稿校样也在此目录。 |
| 04_Editable_sources | 当前英文/中文Markdown、cover letter、图注、补充材料及14条BibTeX/RIS。 |
| 05_Figure_assets | 全部17幅图的68份原始PDF/SVG/PNG/TIFF导出，按期刊图号归组。 |

建议从01_Journal_files/MANUSCRIPT.docx与COVER_LETTER.docx开始。SUPPLEMENT_1.docx为可编辑补充文件，SUPPLEMENT_1.pdf为同版本LibreOffice导出的连续校样。FIGURE_LEGENDS.docx包含3幅主图与14幅补充图全部图注。两张主表可编辑，只有顶线、表头下线和底线。

原始图形导出与统计源保持原哈希；补充图仅按正文首次引用顺序改编号，未重画。代码和CSV沿用冻结源标识，03_Author_review/NUMBERING_CROSSWALK.json提供映射。01目录中的ANALYSIS_CODE.zip与AGGREGATE_TABLES.zip是对应源码/聚合数据的便携封装。

文字依次经Humanizer及lieflat-less-ai-tone处理。第二个技能的外语、引文、代码及必要方法列举排除规则保持；文字处理不改变数值、科学限定或AI使用披露。

作者确认伦理、署名、访问及投稿声明等事实后，再完成投稿系统填写。本目录未执行投稿或对外发布。
'''
(P/'README_ZH.md').write_text(readme)
print('Final science review, reporting map, author instructions saved')
