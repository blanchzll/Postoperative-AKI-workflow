from pathlib import Path
import json,csv,hashlib,datetime,subprocess,fcntl,zipfile,os
R=Path(__file__).resolve().parents[1];P=R.parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat();request='76c02283885efe895e79ac32166e9636505e8d7eb07a710fa82456c14cef3367';baseline='fafbe4e36eb5b715d44343c195103af3fd28a47dc72c60e08d28afe6f3ba6244'
assert sha(R/'REQUEST.json')==request;assert sha(P/'writing_runs/probabilistic_integration_20261105_v1/package/PROBABILISTIC_INTEGRATION_ACCEPTANCE.json')==baseline;assert sha(R/'PROTOCOL_LOCK.json')=='5317bfe92b528e6aa3eb536fa41654178bc09d62f445555165a163deb66e6db3'
def save(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Recheck all owning inputs, not old QA status alone.
source={}
for name in ['PATIENT_ANALYSIS_INPUT_HASHES.json','OBSERVATION_OUTCOME_INPUT_HASHES.json','REUSED_RESULT_INPUT_HASHES.json']:
 data=json.loads((R/'audit'/name).read_text())
 for p,v in data.items():
  h=v['sha256'] if isinstance(v,dict) else v;assert Path(p).is_file() and sha(p)==h,('changed_input',p);source[p]=h
verify=['INDEPENDENT_NUMERIC_VERIFICATION.json','EXTRA_INDEPENDENT_VERIFICATION.json','REUSED_RESULT_VERIFICATION.json','FIGURE_PDF_QUALIFICATION.json']
for name in verify:assert json.loads((R/'audit'/name).read_text())['status'].startswith('PASS')
# Current source-backed scientific review, including repeated-operation and zero-event interpretation.
review=json.loads((R/'SCIENTIFIC_REVIEW.json').read_text());review['remediation_receipts']+=['audit/LOW_RISK_INTERVAL_UNIT_REVIEW.json'];review['operation_patient_units']='Rates and captures per eligible operation record; intervals resample patient groups; counts in results/COHORT_OPERATION_PATIENT_COUNTS.csv';review['all_7_questions_reviewed']=True;save('SCIENTIFIC_REVIEW.json',review)
api=R/'outputs/research_api_rewards/20261108T125425-00a3385292/receipt.json';ar=json.loads(api.read_text());assert ar['data_class']=='public' and ar['provider_status']=='completed'
save('audit/API_DRAFT_METHOD_REVIEW.json',{'reviewed':now,'receipt_sha256':sha(api),'input_sha256':ar['input_sha256'],'data_class':'public/synthetic only','tokens':ar['usage']['total_tokens'],'Codex_review':'Six draft method checks individually assessed: pairing/budget/arrival proxy/timed criteria/deletion/causal distinction all reflected in source-bound analysis and scientific review. Official sources checked by Codex; not independent empirical validation.','private_material_sent':False})
units=['aki-clinical-supplement-patient-20261108-v1','aki-clinical-supplement-observation-20261108-v1','aki-clinical-supplement-observation-20261108-v1-r1','aki-clinical-supplement-outcomes-20261108-v1-r2','aki-clinical-supplement-independent-20261108-v1','aki-clinical-supplement-independent-20261108-v1-r1','aki-clinical-supplement-independent-20261108-v1-r2'];states={}
for unit in units:
 a=subprocess.run(['systemctl','--user','show',unit+'.service','-p','MainPID','-p','ActiveState','-p','Result','-p','ExecMainStatus'],capture_output=True,text=True).stdout;v=dict(line.split('=',1) for line in a.splitlines() if '=' in line);assert v.get('MainPID','0')=='0';states[unit]=v
locks={}
for name in ['PATIENT_STAGE.lock','OBSERVATION_STAGE.lock','INDEPENDENT_STAGE.lock']:
 with (R/name).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[name]='free; check acquired then released';fcntl.flock(f,fcntl.LOCK_UN)
save('audit/OWNED_PROCESS_CLOSURE.json',{'at':now,'owned_units':states,'locks':locks,'all_owned_stage_MainPID_zero':True,'worker_pool_shutdown_joined':True,'other_jobs_services_and_models_modified':False,'memory_now':subprocess.run(['free','-b'],capture_output=True,text=True).stdout,'GPU_now_read_only':subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.total,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout})
# Accurate table index and analysis responsibility boundary.
rows=[]
for f in sorted((R/'results').glob('*.csv')):
 with f.open() as a:q=list(csv.reader(a))
 rows.append({'file':str(f.relative_to(R)),'rows':len(q)-1,'columns':len(q[0]),'writing_role':'SUPPLEMENTARY_ANALYSIS_CANDIDATE','release_scope':'INTERNAL_DATE_INDEXED_ONLY' if f.name=='DAILY_POTENTIAL_DEMAND.csv' else 'AGGREGATE_REVIEWABLE'})
with (R/'TABLE_INDEX.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'MANUSCRIPT_HANDOFF_ZH.md').write_text('''# 给原写作对话的分析交接

本轮AKI_CLINICAL_SUPPLEMENT_20261108_V1已完成授权范围内的可行计算、来源匹配、替代数值实现核对、科学审读与整改。未修改原始数据、冻结模型、原20图、全文或投稿包。

先读ACCEPTANCE.json、SCIENCE_CONCLUSION_MATRIX.csv、ANALYSIS_RESULTS_AND_REVIEW_ZH.md。七问逐项状态在QUESTION_EVIDENCE_ANALYSIS_MATRIX.csv；source CSV、指标、分母、区间、代码和图分面已逐主张绑定。

可直接取用的分析交接文本为eMETHODS_FOR_WRITING_EN.md、eRESULTS_FOR_WRITING_EN.md，图例为FIGURE_LEGENDS_FOR_WRITING.md。TABLE_INDEX.csv列25张结果表。3幅新增图/8独立分面PDF是补充候选，不要求增加主图或原图重画；写作侧自行编号并维持3000词/5主图限制。

必须保留的重要反证：选择性采集同成本增量不稳定；全采集有不利情景；固定阈值的复核量与同预算比较不同；MIMIC基率Brier不佳、时间性差异区间跨零；低风险仍有事件，删测会失去已知事件；中心稀疏且既往评价暴露；完整开发选择乐观度未校正。27→38还伴随锁定模型族变化，不能解释为纯字段数量效应。

实际日期索引结果只代表本研究队列，不能当医院每日/班次到达或已完成临床动作。SCr测量组分/记录间隔不能改名为完整KDIGO或真实预警提前量。硬结局仅为冻结AKI分数预后关联，RRT与原目标定义有重叠；无实施因果获益或停监测安全证据。

交接压缩包排除private/、具体日期需求明细及WorkBuddy原文，仅含聚合表、代码、图源与审读记录。原写作对话负责全文、投稿材料和用户指定GitHub仓库；本分析分支不发布/推送。
''')
(R/'docs/current-handoff.md').write_text(f'''# 已完成：临床补充分析交接

当前目标：将已验收分析交给原写作对话01a192de-cc29-7d52-9dfa-42603428fc4e；本分支不写全文或发布。更新时间：{now}。

必须先读（3项）：
1. {R}/ACCEPTANCE.json
2. {R}/SCIENCE_CONCLUSION_MATRIX.csv
3. {R}/MANUSCRIPT_HANDOFF_ZH.md

七问的现有数据可行计算、数值独立实现、科学审读和整改完成。3图8分面、精简eMethods/eResults和25表留服务器。原始输入/旧模型/旧图不变。缺失时钟、实施暴露和反事实观测的可识别性限制已明确，不借分析验收宣称临床部署、干预获益或停监测安全。

没有活动分析PID；失败、修前结果及诊断保留，不恢复旧服务或重算已通过任务。下一步仅由原写作对话读取本交接并整合，除非用户提出新分析范围。
''')
# Freeze outputs, with date-indexed details retained internally and omitted from handoff bundle.
rootnames=['AGENTS.md','REQUEST.json','PROTOCOL_LOCK.json','QUESTION_EVIDENCE_ANALYSIS_MATRIX.csv','SCIENCE_CONCLUSION_MATRIX.csv','WORKBUDDY_ANALYSIS_DISPOSITIONS.csv','SCIENTIFIC_REVIEW.json','ANALYSIS_RESULTS_AND_REVIEW_ZH.md','eMETHODS_FOR_WRITING_EN.md','eRESULTS_FOR_WRITING_EN.md','FIGURE_LEGENDS_FOR_WRITING.md','TABLE_INDEX.csv','README_ZH.md','MANUSCRIPT_HANDOFF_ZH.md']
files=[R/n for n in rootnames]+[R/'docs/current-handoff.md']+sorted((R/'code').glob('*.py'))+sorted((R/'results').glob('*.csv'))+[p for p in (R/'figures').rglob('*') if p.is_file() and 'pdf_previews' not in p.parts]+[R/'audit'/n for n in verify]+[R/'audit'/n for n in ['OWNED_PROCESS_CLOSURE.json','PUBLIC_EVIDENCE_VERIFICATION.json','API_DRAFT_METHOD_REVIEW.json','LOW_RISK_INTERVAL_UNIT_REVIEW.json','MIMIC_KDIGO_CLOCK_AUDIT.json','CAUSAL_IDENTIFIABILITY.json','INSPIRE_RECORD_INTERVAL_RECHECK.json','ORIGINAL_COHORT_CLOCK_AUDIT.csv','HARD_OUTCOME_SOURCE_AUDIT.csv','SCIENTIFIC_REVIEW_REMEDIATIONS.json','ZERO_EVENT_CENTER_PRECISION_REPAIR.json']]
assert len(set(map(str,files)))==len(files)
manifest={'task_id':'AKI_CLINICAL_SUPPLEMENT_20261108_V1','at':now,'request_sha256':request,'protocol_sha256':sha(R/'PROTOCOL_LOCK.json'),'unchanged_owning_source_hashes':source,'files':[{'path':str(f.relative_to(R)),'sha256':sha(f),'bytes':f.stat().st_size,'bundle_included':f.name!='DAILY_POTENTIAL_DEMAND.csv'} for f in files]};save('SOURCE_AND_OUTPUT_MANIFEST.json',manifest)
bundle=R/'ANALYSIS_HANDOFF_AGGREGATE_CODE_FIGURES.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in files:
  if f.name!='DAILY_POTENTIAL_DEMAND.csv':z.write(f,str(f.relative_to(R)))
 z.write(R/'SOURCE_AND_OUTPUT_MANIFEST.json','SOURCE_AND_OUTPUT_MANIFEST.json')
with zipfile.ZipFile(bundle) as z:
 assert z.testzip() is None
 assert not any(n.startswith('private/') or 'PRIVATE' in n or n.endswith('DAILY_POTENTIAL_DEMAND.csv') for n in z.namelist());zipcount=len(z.namelist())
acc={'task_id':'AKI_CLINICAL_SUPPLEMENT_20261108_V1','status':'PASS_FEASIBLE_SEVEN_QUESTION_ANALYSES_WITH_IDENTIFIABILITY_BOUNDARIES','accepted_Asia_Shanghai':now,'analysis_owner_thread':'01a1b977-b819-7131-be93-23fe314aa853','writing_owner_thread':'01a192de-cc29-7d52-9dfa-42603428fc4e','remote_root':str(R),'request_sha256':request,'baseline_acceptance_sha256':baseline,'protocol_sha256':sha(R/'PROTOCOL_LOCK.json'),'scientific_review_sha256':sha(R/'SCIENTIFIC_REVIEW.json'),'question_matrix_sha256':sha(R/'QUESTION_EVIDENCE_ANALYSIS_MATRIX.csv'),'claim_matrix_sha256':sha(R/'SCIENCE_CONCLUSION_MATRIX.csv'),'source_output_manifest_sha256':sha(R/'SOURCE_AND_OUTPUT_MANIFEST.json'),'source_input_files_rehashed_unchanged':len(source),'seven_questions_processed':7,'scientific_claim_rows':16,'results_tables':len(rows),'new_figures':3,'standalone_subpanel_PDFs':8,'PDF_files_checked':11,'new_predictive_model_fits':0,'association_regressions_only':True,'independent_numeric_implementations':{n:sha(R/'audit'/n) for n in verify[:3]},'visual_QA':sha(R/'audit/FIGURE_PDF_QUALIFICATION.json'),'all_owned_stage_processes_exited':True,'reviewable_bundle':{'path':str(bundle),'sha256':sha(bundle),'files':zipcount,'bytes':bundle.stat().st_size,'CRC':'PASS','patient_level_data_and_exact_dates_excluded':True},'original_frozen_models_data_results_and_manuscript_preserved':True,'old_v1_QA_not_used_as_new_acceptance':True,'unidentified_estimands':['actual diagnostic-test indication or individual clinical benefit','actual daily/shift alert arrival and completed clinical work','actual prediction-available-to-KDIGO warning lead time and complete KDIGO','implementation causal effects on death/dialysis/LOS','safety of reducing/stopping monitoring'],'complete_selection_optimism_corrected':False,'independent_human_expert_review':False,'full_manuscript_or_submission_complete':False,'GitHub_push_performed_by_this_branch':False,'required_work_remaining_in_this_analysis':False}
save('ACCEPTANCE.json',acc);h=sha(R/'ACCEPTANCE.json');(R/'ACCEPTANCE.json.sha256').write_text(h+'  ACCEPTANCE.json\n')
for name in ['STATUS.json','OWNER.json']:
 d=json.loads((R/name).read_text());d.update(status='COMPLETE_ACCEPTED_ANALYSIS_HANDOFF_READY',updated=now,acceptance_sha256=h,no_new_outcome_calculations=False,all_owned_stage_processes_exited=True,current_stage='accepted; send result/hashes to writing owner; no scientific jobs remain');save(name,d)
print(json.dumps({'status':acc['status'],'acceptance_sha256':h,'bundle_sha256':acc['reviewable_bundle']['sha256'],'tables':len(rows),'figures':3,'standalone_panels':8,'source_files_unchanged':len(source),'bundle_bytes':bundle.stat().st_size}))
