"""Persist engineering acceptance and current compute state; no clinical scoring."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys,os,json,subprocess,importlib.metadata as md
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
sys.path.insert(0,str(ROOT/'work/next_stage_preparation_20261022_v1'))
from prepare import sha,read,atomic
out=ROOT/'outputs/final_evaluation_engine_20261022_v1'
q=read(out/'FUNCTIONAL_QUALIFICATION.json')
assert q['status']=='PASS_synthetic_fixed_risk_evaluation_engine_only'
assert all(sha(ROOT/f)==h for f,h in q['source_files'].items())
assert all(sha(f)==h for f,h in q['skill_source_hashes'].items())
try:still_ours=b'work/final_evaluation_engine_20261022_v1/qualify.py' in (Path('/proc')/str(q['pid'])/'cmdline').read_bytes()
except FileNotFoundError:still_ours=False
assert not still_ours
monitor=max((ROOT/'outputs/ensemble_calibration_20261020_v1/monitoring').glob('heartbeat_*.json'),key=lambda p:p.stat().st_mtime)
mon=read(monitor)
# Saved receipt is nested; monitor's compact stdout uses a different flat presentation.
alive=mon['controller']['alive']
assert mon['state']=='running' and alive and mon['code_lock_verified'] and not mon['errors']
status=read(ROOT/'outputs/ensemble_calibration_20261020_v1/status_e.json')
resource_state=read(ROOT/'outputs/ensemble_speedup_20261021_v2/live_resources.json')
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True)
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
atomic(out/'CLOSEOUT_DIAGNOSTIC.json',{'historical_reporting_error':"KeyError controller_alive in an inline report writer",
 'cause':'Saved monitor schema uses controller.alive; compact stdout had controller_alive.',
 'correction':'Read and verify controller.alive from persisted receipt. No model task failed or reran.',
 'functional_metrics_source_unchanged':True})
environment={'time_beijing':now,'python_executable':sys.executable,'python_version':sys.version,
 'packages':{p:md.version(p) for p in ['numpy','scikit-learn','pandas','scipy','statsmodels']},
 'effective_CPU_affinity':len(os.sched_getaffinity(0)),'E_resource_snapshot':status['resources'],
 'dispatcher_resource_state':resource_state,'GPU_query':gpu.stdout.strip(),
 'original_model_requirements_sha256':sha(ROOT/'outputs/ensemble_calibration_20261020_v1/model_requirements.txt')}
atomic(out/'ENVIRONMENT_RESOURCE_RECEIPT.json',environment)
contract={
 'scope':'Fixed held-out patient probabilities only; no clinical evaluation performed.',
 'resampling':{'unit':'patient-group within each hospital','replicates':2000,'seed':20261017,
  'paired_models_share_identical_resamples':True,'all_group_records_share_multiplicity':True,
  'zero_event_samples_retained_for_defined_metrics':True,'per_metric_valid_repeats_reported':True,
  'intervals_conditional_on_metric_defined':True,'predictions_conditioned_as_fixed':True},
 'capacity':{'fractions':[.1,.2],'count':'ceil(n*fraction)',
  'ties':'descending risk then lexicographic patient-group and record key',
  'pool':'Population supplied to the function. Do not silently change to hospital-local ranking.',
  'hospital_breakdown':'Inherit global alerts or locked thresholds for the original policy. Label any hospital-local capacity ranking separately.'},
 'pairing_loader_pending':'Check each prediction source against exact record, group, hospital and outcome metadata; length equality alone does not establish lineage.',
 'batch_selective_policy_pending':'Reapply locked outcome-free upgrade routing in each bootstrap sample before mixing and applying the locked whole-policy calibrator. A frozen mixed-risk vector conditions on observed allocation and is not equivalent to fixed-upgrade-budget strategy evaluation.',
 'calibration_intervals_pending':True,
 'deployment_thresholds':'Training-only locked thresholds; test-cohort capacity is retrospective.',
 'clinical_scoring_prerequisites':['E accepted','route models locked','both strategies locked',
  'clinical loader qualified','strategy bootstrap adapter qualified'],
 'no_final_score_access_before_locks':True}
atomic(out/'METHOD_CONTRACT.json',contract)
atomic(out/'RESOURCE_PRIORITY_RECEIPT.json',{
 'time_beijing':now,'user_authorization':'This conversation may preferentially use all CPU and GPU resources if memory is sufficient.',
 'effective_CPU':len(os.sched_getaffinity(0)),'GPU_snapshot':gpu.stdout.strip(),
 'current_dispatch_evidence':resource_state,'available_memory_snapshot':status['resources'],
 'rule':'Use highest useful throughput under measured RAM, growth, ready dependencies and actual CPU contention; no arbitrary small core/device cap.',
 'unchanged_guards':'RAM/disk headroom and unrelated tasks/shared services preserved.',
 'current_E_backend':'Frozen CPU; measured CPU/GPU XGB predictions differ, active chain unchanged.',
 'new_task_backend':'Prefer supported and qualified GPU for suitable future stages, record numerical/backend provenance.',
 'no_claim_all_hardware_currently_utilized':True})
report=f"""# 评价接口准备与资源复核

时间（北京时间）：{now}

严格校准主任务继续运行。本轮完成固定预测评价模块的合成核查，未读取当前最终医院／时间模型效能。

## 本轮完成

模块实现AUROC、AP、Brier、观察／预期比，及10%／20%复核容量下的检出、漏诊、PPV和每千例负担。名额取ceil(n×比例)，平分按固定患者组／记录键处理。医院内按患者组配对重采样，同组多条记录共同抽样，各模型共用同一份样本。

零事件重复保留：可定义的Brier和负担仍计算；AP／敏感度等标记不可计算，各指标分别报告有效重复数。不会丢掉整次抽样后统一声称所有指标均有2,000次有效重复。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 12项功能测试 | 通过 | TEST.log |
| 加权AUROC／AP／Brier与scikit-learn对照 | 通过 | test_metrics.py及TEST.log |
| 与既定复核名额规则的30项对照 | 全部一致 | FUNCTIONAL_QUALIFICATION.json |
| 2,000次配对重采样 | 合成批次通过 | FUNCTIONAL_QUALIFICATION.json |
| 当前主任务 | 运行、代码锁一致、无错误 | {monitor.relative_to(ROOT)} |

对新接口的检查发现了索引前可能静默截去多余预测的问题，已增加预测长度／维度和缺失身份检查；旧模型和活动科学代码未改。

合成数据为241条、121个患者组、3个医院；2,000次重复耗时{q['synthetic_bootstrap_seconds']:.2f}秒，进程峰值RAM约{q['peak_RSS_bytes']/1024**2:.1f} MiB。批次正常退出。这是功能资格结果，不能作为临床效能。

## 接入条件

固定概率指标模块已通过功能核查；临床预测加载器、校准区间与选择性策略重采样适配仍待接入验收。每次加载须逐项核对患者、结局、分区和来源。

批次选择性策略须在每次重采样内按已锁定门控重新安排升级，再混合概率并应用已锁定的全策略校准器。不能把固定混合概率的区间默认为固定升级预算策略的完整评价。全局规则的逐院汇总保留全局预警名单或训练内阈值；院内重新排名的容量场景另行命名。

最终模型与两条策略均锁定前，不读取当前最终医院／时间效能。具体合同见METHOD_CONTRACT.json。

## 计算与资源

主批次完成{status['completed_tasks']:,}／{status['total_tasks']:,}项有限任务；不同任务耗时不同，该比例不是全研究完成百分比。

可用CPU亲和范围为{len(os.sched_getaffinity(0))}核；实际有效可用RAM快照约{status['resources']['available_bytes']/1024**3:.2f} GiB。GPU查询：{gpu.stdout.strip()}。

按最新授权以最高有效吞吐安排CPU／GPU，当前并发受实测内存、任务增长与就绪依赖约束。既定调度没有固定的小核心上限；内存／磁盘保护、共享服务和其他任务保持。严格校准沿用冻结CPU实现，此前真实检查已证实CPU／GPU XGB输出不同；适用的新任务优先资格核查后使用GPU。不会把空闲显存当系统RAM，也不以占用率代替有效加速。

## 核验范围与后续

本轮实际加载phd-skills入口、医学适配及phd-paper-verification模块和固定版上游方法，应用公式—代码对应和数字来源核验；没有进行全文或独立复核。来源哈希见FUNCTIONAL_QUALIFICATION.json。此前本地草拟路线连续两次超时，按已有授权由Codex完成实现和审查，未重复失败路径或更改共享模型服务。

原中国主队列3,710／171、配对2,428／122及9／27／38项路线保持。后续仍为E验收→模型锁定→策略开发与锁定→最终评价→必要补充及完整交付。研究整体未完成。
"""
r=out/'EVALUATION_READINESS_ZH.md'
with r.open('w') as f:f.write(report);f.flush();os.fsync(f.fileno())
atomic(out/'CLOSEOUT_ACCEPTANCE.json',{
 'status':'PASS_synthetic_evaluation_preparation_only','time_beijing':now,
 'files':{f:sha(out/f) for f in ['FUNCTIONAL_QUALIFICATION.json','METHOD_CONTRACT.json',
 'ENVIRONMENT_RESOURCE_RECEIPT.json','RESOURCE_PRIORITY_RECEIPT.json','EVALUATION_READINESS_ZH.md','CLOSEOUT_DIAGNOSTIC.json']},
 'closeout_source_sha256':sha(__file__),'finite_qualification_process_exited':True,
 'E_monitor_receipt':str(monitor),'E_controller_healthy':True,
 'final_model_or_policy_locked':False,'clinical_performance_evaluated':False,'entire_research_complete':False})
statepath=ROOT/'outputs/IMPLEMENTATION_STATE.json';state=read(statepath)
atomic(out/'IMPLEMENTATION_STATE_BEFORE.json',state)
state['updated_at_beijing']=now
state['stages']['final_evaluation_engine']='synthetic_fixed_prediction_metrics_qualified_clinical_loader_and_policy_adapter_pending'
state['final_evaluation_engine_preparation']=str(out/'CLOSEOUT_ACCEPTANCE.json')
state['latest_resource_priority_receipt']=str(out/'RESOURCE_PRIORITY_RECEIPT.json')
state['phd_skills']={'loaded_at_beijing':now,'entrypoint':'/home/lei/codex-workspaces/skill-management/skills/phd-skills/SKILL.md','module':'phd-paper-verification','scope':'formula and code qualification only'}
state['last_E_verification']={'state':mon['state'],'controller_alive':alive,'source_lock_verified':mon['code_lock_verified'],'errors':mon['errors'],'monitoring_receipt':str(monitor.relative_to(ROOT))}
atomic(statepath,state)
h=ROOT/'outputs/ensemble_continuation_20261018_v1/NEXT_EXECUTION_HANDOFF_ZH.md'
with h.open('a') as f:
 f.write(f"""
## 评价接口准备与最新资源授权（北京时间{now}）

work/final_evaluation_engine_20261022_v1与对应outputs已保存。CLOSEOUT_ACCEPTANCE.json验收12项测试、30项既定名额规则对照、241条/121组/3院/2000次合成配对重采样及进程退出。只验收固定概率指标/容量/bootstrap；不是临床加载器、校准区间或选择性策略bootstrap全套验收，不重复该功能批次充当进展。
METHOD_CONTRACT.json规定每指标有效重复数、保留零事件抽样、区分全局与院内容量。批次选择性升级在每次重采样中重新运行锁定门控，混合后用锁定全策略校准器；不能用固定混合分数默默替代。最终模型和策略锁定前不读最终效能。
用户允许本聊天优先使用所有CPU/GPU，内存足够时最大化有效吞吐。RESOURCE_PRIORITY_RECEIPT.json保存48CPU、RAM及P40空闲显存快照，既有内存/磁盘保护和共享服务保持；没有宣称硬件已满载。当前E健康且唯一，保持冻结CPU链。
已实际加载phd-skills、医学适配、phd-paper-verification及固定上游方法，本轮只作公式/代码核验；后续复用入口并按阶段路由。关闭报告曾误读监控扁平展示字段，已按实际嵌套controller.alive修正，诊断保存；无模型任务失败或重跑。
""");f.flush();os.fsync(f.fileno())
print(json.dumps({'status':'PASS_synthetic_evaluation_preparation_only','report':str(r),
 'E_state':status['state'],'completed_tasks':status['completed_tasks'],'total_tasks':status['total_tasks'],
 'effective_CPU':len(os.sched_getaffinity(0)),'available_RAM_GiB':status['resources']['available_bytes']/1024**3,
 'GPU':gpu.stdout.strip()},ensure_ascii=False))
