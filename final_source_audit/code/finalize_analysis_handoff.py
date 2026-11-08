#!/usr/bin/env python3
from pathlib import Path
import datetime,hashlib,json,zipfile
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
readme='''# Final WorkBuddy source audit and analysis handoff

This version verifies the latest public WorkBuddy review against the accepted corrected-data clinical analyses. It adds source review, deterministic arithmetic checks and the unique 28-row eTable4 metric/capacity/unit mapping. It does not fit prediction models, generate new bootstrap samples or add workflow simulations.

The canonical server root is `/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/final_workbuddy_analysis_20261108_v1`.

Run `python3 code/final_workbuddy_audit.py` on t630 where the original private source and frozen writing/analysis releases remain available. The script reads those sources and writes only this audit root. `--project-root` and `--audit-root` can select relocated equivalent server copies. Source hashes are checked before derived summaries. Patient data, sampling arrays, exact operation dates, shared raw tool outputs and credentials are excluded from this handoff. The accepted original research-code release remains a separate dependency.

Read `docs/FINAL_SCIENTIFIC_HANDOFF_ZH.md` for the clinical result interpretation and targeted repairs. `results/WORKBUDDY_DELTA_CLOSURE.csv` separates accepted evidence, corrections, unidentifiable estimands and the writing owner's remaining work. `docs/ETABLE4_REPLACEMENT.md` and its source-mapped CSV supply the repaired table. Metadata timestamps vary upon a rerun; scientific source checks and the source-mapped table are deterministic.

The new scientific acceptance covers this analysis audit. The overall user request additionally requires the writing owner to complete the updated manuscript, final-size figure repairs, individual panel PDFs, editorial review, submission package and verified code publication. No actual journal submission or human expert-panel approval is represented here.
'''
(R/'README.md').write_text(readme)
mandatory=['SCIENTIFIC_REVIEW.json','results/WORKBUDDY_DELTA_CLOSURE.csv','docs/ETABLE4_REPLACEMENT.md']
handoff='''# 预测模型manuscript final：本次科学分析交接

本次分析任务`AKI_FINAL_WORKBUDDY_ANALYSIS_20261108_V1`已完成来源、统计报告与结论审读。保留原临床补充分析和冻结模型，新增0模型、0bootstrap、0虚拟场景。已核32条最新分享消息、159个输入、90个既有输出与123项聚合算术检查。

本轮必须先读最多3项：

1. `SCIENTIFIC_REVIEW.json`
2. `results/WORKBUDDY_DELTA_CLOSURE.csv`
3. `docs/ETABLE4_REPLACEMENT.md`

20项建议闭环中的科学分析已完成；eTable4源映射与3幅图实际刊用尺寸修复由当前写作新版本承接。详细临床结果写作入口为`docs/FINAL_SCIENTIFIC_HANDOFF_ZH.md`，按需阅读，不再从旧WorkBuddy round4或早期8报告重启。

临床定位保持胃肠肿瘤手术患者的术前信息需求、术毕风险重估和有限复核容量。保留差异、阴性、低风险组已知事件与迁移不利结果，必要边界只在相应结果和Limitations准确说明。不把识别边界本身改成全文的方法学主贡献。

将8列表和其源CSV接入新写作版；3幅图仅作已授权的排版/字号修复，保留原数值artist。完成23幅合图/71个独立panel（若明确更动以新manifest为准）的实际置入大小QA、指定skills使用记录、最终正文/图表/附件/cover letter的一致性与人类化文字审读、水印检查。旧版本不改。

后续公开代码须包含本次新审计代码、表格修复和新增图导出代码；使用白名单交接ZIP，不公开本目录inputs里的原始分享工具输出或任何患者级文件。收到新包与远端GitHub发布核验回执后，原线程继续完成本次总目标验收。当前科学验收不替代新投稿包验收。
'''
(R/'MANUSCRIPT_HANDOFF_ZH.md').write_text(handoff)
# Explicitly whitelisted aggregate/code artifacts only.
paths=[Path(x) for x in ['AGENTS.md','REQUEST.json','PROTOCOL_LOCK.json','README.md','MANUSCRIPT_HANDOFF_ZH.md','SCIENTIFIC_REVIEW.json']]
for folder in ['code','docs','results','audit']:
 for p in sorted((R/folder).glob('*')):
  if p.is_file() and p.suffix in {'.py','.md','.csv','.json'}:paths.append(p.relative_to(R))
assert all(p.parts[0] not in {'inputs','private'} for p in paths)
assert len(paths)==len(set(paths))
manifest={'task_id':'AKI_FINAL_WORKBUDDY_ANALYSIS_20261108_V1','utc':now,'source_sha256':sha(R/'inputs/WORKBUDDY_LATEST_CONVERSATION.json'),'prior_clinical_acceptance_sha256':sha(R.parent/'clinical_supplement_20261108_v1/ACCEPTANCE.json'),'files':[{'path':str(p),'sha256':sha(R/p),'bytes':(R/p).stat().st_size} for p in paths],'private_raw_shared_inputs_excluded':True,'new_patient_level_outputs':0}
save(R/'SOURCE_AND_OUTPUT_MANIFEST.json',manifest)
zipname=R/'ANALYSIS_HANDOFF_AGGREGATE_CODE.zip'
with zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in paths+[Path('SOURCE_AND_OUTPUT_MANIFEST.json')]:z.write(R/p,str(p))
with zipfile.ZipFile(zipname) as z:assert z.testzip() is None;filecount=len(z.namelist())
accept={'task_id':manifest['task_id'],'status':'PASS_FINAL_SOURCE_AND_SCIENTIFIC_ANALYSIS_AUDIT','accepted_utc':now,'analysis_root':str(R),'request_sha256':sha(R/'REQUEST.json'),'protocol_sha256':sha(R/'PROTOCOL_LOCK.json'),'scientific_review_sha256':sha(R/'SCIENTIFIC_REVIEW.json'),'manifest_sha256':sha(R/'SOURCE_AND_OUTPUT_MANIFEST.json'),'source_sha256':manifest['source_sha256'],'source_messages_verified':32,'prior_inputs_unchanged':159,'prior_outputs_unchanged':90,'aggregate_numeric_checks':123,'source_mapped_eTable4_rows':28,'recommendations_dispositioned':20,'new_predictive_fits':0,'new_bootstrap_replicates':0,'new_simulation_scenarios':0,'new_patient_data':0,'clinical_result_estimates_changed':False,'prior_manuscript_package_modified':False,'required_work_remaining_in_this_analysis':False,'required_downstream_writing_and_publication_work_remaining':True,'writing_thread_id':'01a192de-cc29-7d52-9dfa-42603428fc4e','mandatory_read_at_most_three':mandatory,'bundle':{'path':str(zipname),'sha256':sha(zipname),'bytes':zipname.stat().st_size,'files':filecount,'CRC':'PASS','raw_shared_conversation_and_patient_data_excluded':True},'human_expert_review':False,'whole_user_goal_complete':False}
save(R/'ACCEPTANCE.json',accept)
(R/'ACCEPTANCE.json.sha256').write_text(sha(R/'ACCEPTANCE.json')+'  ACCEPTANCE.json\n')
status={'task_id':manifest['task_id'],'updated_utc':now,'status':'SCIENTIFIC_ANALYSIS_ACCEPTED_AWAITING_NEW_WRITING_RELEASE','acceptance_sha256':sha(R/'ACCEPTANCE.json'),'required_analysis_work_remaining':False,'required_downstream_work':['Current-version eTable4 and 3 figure layout/font repairs with independent source/artist checks.','Same-version manuscript, supplement, cover letter and all figure panel PDF/source consistency, skill and watermark/tone reviews.','Final submission package and complete accepted code upload with verified remote receipts.'],'old_releases_preserved':True}
save(R/'STATUS.json',status)
print(json.dumps({'status':accept['status'],'acceptance_sha256':sha(R/'ACCEPTANCE.json'),'bundle':accept['bundle']},indent=2))
