"""Receipt/hash and numeric review for completed common6 development and public evaluation."""
from pathlib import Path
import json,hashlib,fcntl,os,csv
from datetime import datetime,timezone,timedelta
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');OUT=ROOT/'outputs/common6_public_evaluation_20261024_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
sources={}
current_source_sha256=sha(ROOT/'outputs/v4_development_inputs_20261018/SOURCE_LOCK.json')
assert current_source_sha256=='afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3'
for source_line in ['geography','temporal']:
 child=ROOT/'outputs/common6_development_20261024_v1/inputs'/source_line/'SOURCE_LOCK.json'
 assert read(child)['source_parent_sha256']==current_source_sha256
def check(p,h=None):
 p=Path(p);value=sha(p);assert h is None or value==h,(str(p),value,h)
 sources[str(p)]=value
 return read(p) if p.suffix=='.json' else None
def files(p):
 r=check(p)
 for name,h in r.get('files',{}).items():check(Path(p).parent/name,h)
 return r
for stage in ['common6_strict_continuation_20261024_v1','common6_internal_assembly_20261024_v1','common6_model_lock_20261024_v1','common6_public_evaluation_20261024_v1']:
 d=ROOT/'outputs'/stage;l=read(d/'LAUNCH.json');proc=Path('/proc')/str(l['pid'])/'stat'
 assert not proc.exists() or proc.read_text().split()[21]!=str(l['start_ticks'])
 assert not (d/'FAILURE.json').exists()
 with (d/'SUPERVISOR.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(h,fcntl.LOCK_UN)
 # Frozen code and dependency hashes, without executing modules.
 src=ROOT/'work'/stage
 locks=['EXECUTION_CODE_LOCK.json','EXECUTION_DEPENDENCIES.json'] if stage.startswith('common6_strict') else ['CODE_LOCK.json','DEPENDENCIES.json']
 for name in locks:
  dep=check(d/name)
  for f,h in dep.items():check((src/f) if name.endswith('CODE_LOCK.json') else Path(f),h)
bank=ROOT/'outputs/common6_strict_continuation_20261024_v1'
a=check(bank/'NESTED_BANK_ACCEPTANCE.json');jobs=read(bank/'JOB_MANIFEST.json')['jobs'];assert len(jobs)==len(a['receipts'])==8928
for j in jobs:
 p=Path(j['dest']);r=check(p,a['receipts'][j['id']])
 for name,h in r.get('files',{}).items():check(p.parent/name,h)
 if j['kind']=='coarse':
  assert r['status']=='PASS'
  check(bank/'scopes'/j['scope_id']/'SCOPE.json',r['source_scope_sha256'])
 else:
  if j['kind'] in ['search','fit']:
   assert r['signature']['task']['scope_sha256']==j['task']['scope_sha256']
   assert r['signature']['task']['stage']=='preop' and r['signature']['task']['branch']=='full'
assembly=ROOT/'outputs/common6_internal_assembly_20261024_v1'
stage=check(assembly/'STAGE_INTERNAL_ACCEPTANCE.json')
detail=files(assembly/'calibrated_internal/ACCEPTANCE.json')
assert sha(assembly/'calibrated_internal/ACCEPTANCE.json')==stage['detail_sha256']
assert detail['candidate_fold_metrics']==110 and detail['procedure_fold_metrics']==10 and detail['calibration_contexts']==144
lockdir=ROOT/'outputs/common6_model_lock_20261024_v1';locks=check(lockdir/'STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json')
check(assembly/'STAGE_INTERNAL_ACCEPTANCE.json',locks['internal_acceptance_sha256'])
models={}
for rec in locks['models']:
 m=check(rec['path'],rec['sha256']);models[m['identity']['line']]=m
 assert set(m['input_columns'])=={'Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb'} and m['actual_input_count']==6
 for x in list(m['base_components'].values())+[m['calibrator'],m['calibration_training_scores']]+([m['meta']] if m['meta'] else []):check(x['path'],x['sha256'])
 files(Path(rec['path']).parent/'ACCEPTANCE.json')
public=check(OUT/'STAGE_PUBLIC_ACCEPTANCE.json')
pre=check(OUT/'PRE_SCORE_LOCK.json',public['pre_score_lock_sha256'])
check(lockdir/'STAGE_COMMON6_MODEL_LOCK_ACCEPTANCE.json',pre['model_lock_receipt_sha256'])
assert pre['models']==locks['models'] and public['n']==6043 and public['events']==253
ev=check(OUT/'EVALUATION.json',read(OUT/'EVALUATION_ACCEPTANCE.json')['sha256'])
assert public['replicates']==2000 and not public['public_domain_calibration_fitted']
for k,p in ev['point'].items():
 assert p['n']==6043 and p['events']==253 and abs(p['event_rate']-253/6043)<1e-14
 for cap in ['10','20']:
  assert p['capacity'+cap+'_captured_n']+p['capacity'+cap+'_missed_n']==253
  assert p['capacity'+cap+'_captured_n']+p['capacity'+cap+'_false_alerts_n']==p['capacity'+cap+'_reviewed_n']
with (OUT/'POINT_ESTIMATES.csv').open() as f:
 for row in csv.DictReader(f):
  p=ev['point'][row['model']]
  for k,v in p.items():
   if v is None:assert row[k]==''
   else:assert abs(float(row[k])-float(v))<1e-10
with (OUT/'INTERVALS.csv').open() as f:
 rows=list(csv.DictReader(f))
 assert len(rows)==len(ev['intervals'])
 for row,d in zip(rows,ev['intervals']):
  for k,v in d.items():
   if v is None:assert row[k]==''
   elif isinstance(v,(int,float)):assert abs(float(row[k])-float(v))<1e-10
   else:assert row[k]==v
for name in ['SOURCE_THRESHOLD_BURDEN.csv','SEVERE_CAPTURE_DESCRIPTIVE.csv','CALIBRATION_CURVES.csv','EVALUATION_ACCEPTANCE.json']:check(OUT/name)
for line in ['geography','temporal']:
 a=ev['point'][line+'__raw'];b=ev['point'][line+'__calibrated']
 for metric in ['AUROC','AP','capacity10_captured_n','capacity20_captured_n']:assert a[metric]==b[metric]
def interval(model,metric):
 return next(d for d in ev['intervals'] if d['model']==model and d['metric']==metric)
def fmt(model,metric,digits=3):
 d=interval(model,metric);return f"{d['estimate']:.{digits}f} [{d['lower95']:.{digits}f}, {d['upper95']:.{digits}f}]"
# Fixed input source has no target-derived fitting state. No choice between these two using public results.
table=[];thresholdrows=[]
for line in ['geography','temporal']:
 k=line+'__calibrated';p=ev['point'][k]
 table.append('|'+ '|'.join([line,models[line]['family'],fmt(k,'AUROC'),fmt(k,'AP'),fmt(k,'Brier',4),fmt(k,'observed_expected',2),fmt(k,'calibration_slope',2),
 str(int(p['capacity20_captured_n']))+'/253',f"{p['capacity20_PPV']*100:.2f}%"] )+'|')
with (OUT/'SOURCE_THRESHOLD_BURDEN.csv').open() as f:burden=list(csv.DictReader(f))
for line in ['geography','temporal']:
 for rule in ['training_sensitivity80','training_capacity10','training_capacity20']:
  z={d['metric']:float(d['estimate']) for d in burden if d['source_line']==line and d['rule']==rule}
  thresholdrows.append(f"|{line}|{rule}|{int(z['reviewed_n'])} ({100*z['reviewed_n']/6043:.1f}%)|{int(z['captured_n'])}|{100*z['sensitivity']:.1f}%|{100*z['PPV']:.1f}%|")
report='\n'.join([
 '# 本轮共同六输入开发与INSPIRE结果：来源及数值核验','',
 '这是当前纠正源的计算／产物核验记录，不是稿件正文或最终科学审查。',
 '本轮中国源锁：'+current_source_sha256,
 '公共评价人数及事件：'+str(public['n'])+'/'+str(public['events'])+'；保存重采样次数：'+str(public['replicates'])+'。',
 '两条开发线的输入均为Age、Sex、PreopCr、PreopBUN、PreopAlb、PreopHb；最终家族分别为'+', '.join(line+':'+models[line]['family'] for line in ['geography','temporal'])+'。家族、校准及阈值以公共评分前锁为准，未按公共结果再挑选。','',
 '## 从当前结果生成的表','',
 '|源开发线|锁定家族|AUROC及区间|AP及区间|Brier及区间|O:E及区间|校准斜率及区间|20%容量捕获|PPV|',
 '|---|---|---|---|---|---|---|---|---|',
 *table,'',
 '|源开发线|源训练阈值规则|实际复核人数/比例|捕获AKI|敏感度|PPV|',
 '|---|---|---|---|---|---|',*thresholdrows,'',
 '## 核验边界与待严审事项','',
 '实际核对当前bank全部任务回执、文件哈希和输入签名；内部校准结果、锁定模型及基础状态、公共评分前锁、公共CSV/JSON点估计和区间逐项一致。模型或阈值均未在本程序重训。',
 '源训练固定数值阈值在迁移后不保证固定复核比例或敏感度；回顾性排序容量与单病例阈值负担应分开报告。',
 '公共只有共同六输入；不能把这项结果扩展为原9/27/38变量模型或端到端选择性升级的国际验证，也不能由预测结果主张临床干预效应。',
 '公共采样肌酐终点与中国临床AKI的观测和可评价范围有差异；年龄分箱、实验室记录精度、RRT/尿量可用性及监测覆盖须保留。',
 '候选家族外折成绩、完整内层选优程序外折成绩与留出评价各有范围；评价已有暴露，不能把追加核验称作新盲确认。',
 '病例分期的捕获和小时窗口须由各自已授权的新结果支持，本核验不复制历史数值。',
 '风险曲线是独立必需组件，未完成风险曲线不阻止本common6来源核验，但完整37项和最终科学严审仍不可跳过。',
 '本程序保留全部当前表，不根据已见性能决定保留哪条源模型。最终科学结论由全部分析齐备后的真实统计与推断审查决定。'])
(OUT/'COMMON6_CURRENT_SOURCE_QC_ZH.md').write_text(report)
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
write(OUT/'REPORT_SOURCES.json',{'verified_files':sources,'review_code_sha256':sha(__file__),'time_beijing':now})
write(OUT/'REVIEW_ACCEPTANCE.json',{'status':'PASS_common6_development_locks_and_public_results_reverified',
 'time_beijing':now,'current_source_lock_sha256':current_source_sha256,'scientific_review_performed':False,'manuscript_review_performed':False,'bank_tasks':len(jobs),'candidate_fold_metrics':110,'procedure_fold_metrics':10,'calibration_contexts':144,
 'locked_models':2,'public_records':6043,'public_events':253,'public_bootstrap_replicates':2000,
 'csv_json_points_intervals_agree':True,'completed_owned_processes_exited_and_locks_released':True,
 'files':{f:sha(OUT/f) for f in ['COMMON6_CURRENT_SOURCE_QC_ZH.md','REPORT_SOURCES.json']},
 'whole_project_complete':False})
# Keep process-stage receipts separate from whole-study interpretation.
for d,status in [(bank,'PASS_8928_common6_bank_receipts_reverified'),(assembly,'PASS_common6_strict_internal_artifacts_reverified'),(lockdir,'PASS_two_common6_model_locks_reverified')]:
 write(d/'REVIEW_ACCEPTANCE.json',{'status':status,'time_beijing':now,'public_review_receipt':str(OUT/'REVIEW_ACCEPTANCE.json'),
 'public_review_receipt_sha256':sha(OUT/'REVIEW_ACCEPTANCE.json'),'whole_project_complete':False})
print('PASS',len(sources),'files verified',*[models[l]['family'] for l in ['geography','temporal']])
