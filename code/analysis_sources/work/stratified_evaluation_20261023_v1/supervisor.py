"""Finite continuation of verified frozen prediction evaluation, no model development."""
from pathlib import Path
import sys,os,time,fcntl,subprocess,traceback,shutil
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/stratified_evaluation_20261023_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
OWNED=[];KINDS=('hospital','age','sex','diabetes','tumor_site','egfr')
def rss(pid):
 try:return int(next(x.split()[1] for x in Path('/proc',str(pid),'status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
 except (OSError,StopIteration):return 0
def report(receipts):
 content=['# 预设分层、逐院区间及严重AKI捕获',
 '',
 '四个正式留出评价分区使用相同锁定模型及策略，按医院、年龄、性别、糖尿病、肿瘤部位源编码和术前eGFR进行描述与区间估计。年龄界值65岁、eGFR界值60 mL/min/1.73m²；分层缺失单列，不用插补结果推定亚组。一次低eGFR不等于确诊CKD。',
 '',
 '每次医院内患者组重采样后，先在整个评价样本重新分配选择性升级及复核容量，再统计各组的实际复核和检出。各组的复核比例可以不同；没有在各组重新训练模型或单独调阈值。',
 '',
 '2／3期病例的捕获使用原任意临床AKI模型的相同风险及复核名单。严重AKI敏感度、阳性预测值、漏诊和每千例检出单独列出；没有重新开发严重AKI模型。',
 '',
 '每项指标采用2000次医院内患者组bootstrap及原种子20261017，保存有效重复数、空亚组重复及配对差值。校准截距、斜率仅描述预测，不更新风险。小事件组、零事件组及不可估计指标均保留；不同组是否显著不能直接当作异质性检验。',
 '',
 '输出按开发线／人群／分层维度保存POINT.csv、INTERVALS.csv、PAIRED_DIFFERENCES.csv、RESAMPLING_RECEIPT.json与COMPLETE.json。医院和时间评价人群重叠，分层人群也重叠，不能累加事件数冒充新增独立样本。',
 '',
 'eGFR公式已核对美国国家肾脏基金会原始说明：[2021 CKD-EPI creatinine equation](https://www.kidney.org/ckd-epi-creatinine-equation-2021)。肌酐换算为mg/dL，女0／男1沿用源编码；年龄／性别／肌酐缺失或不适用成人公式者列unknown。肿瘤部位保留源0／1／2编码，未凭推测更改标签。',
 '',
 '此批次仅完成分层与严重AKI捕获，公共共同六输入模型、稳定性重拟合、敏感性／SHAP、决策情景、POD1／白蛋白、研究工具与完整稿件仍按方案继续。']
 (OUT/'STRATIFIED_EVALUATION_REPORT_ZH.md').write_text('\n'.join(content)+'\n')
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 code=base.read(OUT/'CODE_LOCK.json')
 dependency=base.read(OUT/'EXECUTION_CONTRACT.json')['dependencies']['formal_engine_code_lock_sha256']
 assert base.sha(base.OUT/'CODE_LOCK.json')==dependency
 for n,h in base.read(base.OUT/'CODE_LOCK.json').items():assert base.sha(ROOT/'work/formal_evaluation_20261023_v1'/n)==h
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_subgroup_allocation_and_severe_capture'
 if (OUT/'STAGE_STRATIFIED_ACCEPTANCE.json').exists():return
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 while not (base.OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json').exists():
  for n,h in code.items():assert base.sha(CODE/n)==h
  if (base.OUT/'FAILURE.json').exists():raise RuntimeError('Formal evaluation failure; diagnose before continuation')
  base.atomic(OUT/'status.json',{'state':'waiting_formal_evaluation_acceptance','pid':os.getpid(),'time_UTC':base.stamp(),'owned_workers':0})
  time.sleep(30)
 base.gate()
 assert base.read(base.OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json')['status']=='PASS_four_heldout_partitions_fixed_models_and_locked_policies'
 prior=(base.OUT/'SUPERVISOR.lock').open('a')
 while True:
  try:fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 jobs=[(line,cohort,kind) for line,cohort in base.EXPECTED for kind in KINDS]
 live={};done=[];peak={};failed=[];pilot_ok=False;per_task=768*1024**2
 while jobs or live:
  for key,item in list(live.items()):
   process,log,job=item;peak[key]=max(peak.get(key,0),rss(process.pid));rc=process.poll()
   if rc is not None:
    log.close();del live[key]
    if rc:failed.append({'job':job,'returncode':rc})
    else:
     line,cohort,kind=job;d=OUT/(line+'_'+cohort)/kind;acc=base.read(d/'COMPLETE.json')
     assert acc['status']=='PASS_prespecified_stratum_intervals_and_severe_capture'
     for f,h in acc['files'].items():assert base.sha(d/f)==h
     done.append(job);pilot_ok=True;per_task=max(per_task,2*peak[key])
  if failed and not live:raise RuntimeError('Stratified worker failed; no unchanged retry '+str(failed))
  available=base.mem();head=max(4*1024**3,int(.25*available));growth=sum(max(0,per_task-rss(p.pid)) for p,l,j in live.values())
  for n,h in code.items():assert base.sha(CODE/n)==h
  max_workers=len(os.sched_getaffinity(0)) if pilot_ok else 1
  while jobs and not failed and len(live)<max_workers and available-head-growth>=per_task and shutil.disk_usage(OUT).free>60*1024**3:
   job=jobs.pop(0);key='_'.join(job);log=(OUT/(key+'.log')).open('ab')
   process=subprocess.Popen([str(base.PYTHON),'-B',str(CODE/'stratified.py'),*job],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   live[key]=(process,log,job);OWNED.append((process,log));peak[key]=0;growth+=per_task
  base.atomic(OUT/'status.json',{'state':'running_stratified_evaluation' if live else 'waiting_memory','pid':os.getpid(),'time_UTC':base.stamp(),
   'running':{k:v[0].pid for k,v in live.items()},'completed':done,'pending':jobs,'failed':failed,'pilot_accepted':pilot_ok,
   'available_bytes':available,'growth_reserved_bytes':growth,'per_task_reservation_bytes':per_task,'peak_RSS_bytes':peak,'entire_research_complete':False})
  if jobs or live:time.sleep(10)
 assert len(done)==24
 receipts={ '/'.join(job):base.sha(OUT/(job[0]+'_'+job[1])/job[2]/'COMPLETE.json') for job in done}
 report(receipts)
 base.atomic(OUT/'STAGE_STRATIFIED_ACCEPTANCE.json',{'status':'PASS_24_prespecified_stratified_evaluations','time_UTC':base.stamp(),
  'receipts':receipts,'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'report_sha256':base.sha(OUT/'STRATIFIED_EVALUATION_REPORT_ZH.md'),
  'owned_workers_remaining':0,'severe_capture_of_locked_any_AKI_models':True,'entire_research_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_stratified_evaluation_and_severe_capture','time_UTC':base.stamp(),'owned_workers_remaining':0,'entire_research_complete':False})
 prior.close();guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'owned_running_pids':[p.pid for p,l in OWNED if p.poll() is None]})
  for p,log in OWNED:
   p.wait()
   if not log.closed:log.close()
  raise
