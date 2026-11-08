"""Finite owner: wait for locks, dispatch four independent evaluation partitions, release."""
from common import *
OWNED=[]
import fcntl,subprocess,time,traceback,shutil
def rss(pid):
 try:
  d=Path('/proc')/str(pid)/'status'
  return int(next(x.split()[1] for x in d.read_text().splitlines() if x.startswith('VmRSS:')))*1024
 except (OSError,StopIteration):return 0
def report(accepted):
 text=['# 五中心地理、时间与同患者信息增量：正式评价',
 '本报告使用已锁定V1模型及选择性策略。所有新评分发生于两类锁定验收之后。地理与时间线是分别训练的模型版本；病例及旧版成绩曾被研究团队查看，本次为回顾性重建评价。',
 '','## 已完成范围','',
 '26个已锁定模型包、24种预设选择性策略版本；四个留出评价分区。每个分区进行医院内患者组2000次重采样，配对模型共用同一次抽样；批次升级策略每次重新分配升级名额。含原始混合概率与校准后概率，按指标保留有效重复次数。',
 '','|开发线|评价人群|模型|N／AKI|AUROC|AP|Brier|复核20%检出／漏诊|','|---|---|---|---|---|---|---|---|']
 for line,cohort in EXPECTED:
  d=OUT/(line+'_'+cohort);res=read(d/'EVALUATION.json')
  for model,q in res['point'].items():
   if '_champion_' not in model:continue
   def f(v):return '不可估计' if v is None else f'{v:.4f}'
   text.append(f"|{line}|{cohort}|{model}|{q['n']}／{int(q['events'])}|{f(q['AUROC'])}|{f(q['AP'])}|{f(q['Brier'])}|{q['capacity20_captured_n']}／{q['capacity20_missed_n']}|")
 text+=['','## 如何读取增量','',
  '每分区的PAIRED_DIFFERENCES.csv给出同患者、相同复核容量的增量及95%区间，包括主模型、获胜家族并集的同算法对照、选择性升级对基础／全部升级／同人数随机升级。正的captured_per1000为多检出，负的missed_per1000为少漏诊；区间跨零不等于等效。',
  '主情景为升级50%、复核20%；支持情景升级25%、复核10%，全部预设组合保留。individual为固定距离规则，报告实际升级人数；capacity为批次相对人数规则。追加整合人数仅是信息负担代理，不代表已实测成本节省。',
  '校准截距采用固定斜率1的offset模型；另列联合截距和斜率。仅为评价诊断，没有更新V1概率。校准无法估计及零事件重采样均保留为指标缺失。',
  'HOSPITAL_YEAR_DESCRIPTIVE.csv是继承全评价队列复核名单的逐院／年度描述，并非各院重新选最高20%。其中分层区间尚待补齐，小事件医院不能作精确校准结论。',
  '','## 尚未完成的项目','',
  '逐院区间、预设临床亚组与严重AKI捕获、必要敏感性／SHAP、固定流程200次稳定性重拟合、INSPIRE共同六输入模型与公共评价、冻结决策情景、POD1及白蛋白探索、研究计算器／复现包和完整稿件均须单独验收。本报告不表示整个项目完成。',
  '','## 工件','',
  '每个分区目录保存POINT_ESTIMATES.csv、INTERVALS.csv、PAIRED_DIFFERENCES.csv、LOCKED_THRESHOLD_BURDEN.csv、CALIBRATION_CURVES.csv及COMPLETE.json。患者级预测仅留服务器私有目录，不公开上传。']
 (OUT/'FORMAL_EVALUATION_REPORT_ZH.md').write_text('\n'.join(text)+'\n')
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 expected=read(OUT/'CODE_LOCK.json')
 for n,h in expected.items():assert sha(CODE/n)==h
 assert read(OUT/'FUNCTIONAL_QUALIFICATION.json')['status']=='PASS_synthetic_metrics_calibration_policy_bootstrap'
 assert read(OUT/'CLINICAL_LOADER_QUALIFICATION.json')['status']=='PASS_all_current_locked_routes_development_only'
 if (OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json').exists():return
 atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],
    'time_UTC':stamp(),'code_lock_sha256':sha(OUT/'CODE_LOCK.json'),'scope':'four fixed model and locked policy evaluation batches'})
 while not (POLICIES/'STAGE_POLICY_LOCK_ACCEPTANCE.json').exists():
  if (POLICIES/'FAILURE.json').exists() or (ROOT/'outputs/selective_search_20261023_v1/FAILURE.json').exists():
   raise RuntimeError('Upstream failed; preserve ownership and diagnose before continuation')
  atomic(OUT/'status.json',{'state':'waiting_model_and_policy_lock_acceptance','pid':os.getpid(),'time_UTC':stamp(),'owned_workers':0,'final_scores_accessed':False})
  time.sleep(30)
 gate()
 prior=(POLICIES/'SUPERVISOR.lock').open('a')
 while True:
  try:fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(10)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 jobs=list(EXPECTED);live={};done=[];failures=[];peak={}
 while jobs or live:
  for key,item in list(live.items()):
   process,log=item;peak[key]=max(peak.get(key,0),rss(process.pid));rc=process.poll()
   if rc is not None:
    log.close();del live[key]
    if rc:failures.append({'key':key,'returncode':rc})
    else:
     rec=read(OUT/key/'COMPLETE.json');assert rec['status']=='PASS_fixed_models_and_locked_policy_heldout_evaluation'
     for f,h in rec['files'].items():assert sha(OUT/key/f)==h
     done.append(key)
  for n,h in expected.items():assert sha(CODE/n)==h
  if failures and not live:raise RuntimeError('Evaluation worker failed; remaining work not launched: '+str(failures))
  available=mem();headroom=max(4*1024**3,int(available*.25))
  reserved_growth=sum(max(0,max(3*1024**3,peak.get(k,0)*2)-rss(v[0].pid)) for k,v in live.items())
  freecpu=len(os.sched_getaffinity(0))-len(live)
  while jobs and not failures and freecpu>0 and available-headroom-reserved_growth>=3*1024**3 and shutil.disk_usage(OUT).free>60*1024**3:
   line,cohort=jobs.pop(0);key=line+'_'+cohort
   log=(OUT/(key+'.log')).open('ab')
   process=subprocess.Popen([str(PYTHON),'-B',str(CODE/'clinical.py'),line,cohort],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   live[key]=(process,log);OWNED.append((process,log));peak[key]=0
   reserved_growth+=3*1024**3;freecpu-=1
  atomic(OUT/'status.json',{'state':'running_heldout_evaluation' if live else 'waiting_memory',
    'pid':os.getpid(),'time_UTC':stamp(),'running':{k:v[0].pid for k,v in live.items()},'complete':done,'pending':jobs,
    'failed':failures,'available_bytes':available,'growth_reserved_bytes':reserved_growth,'peak_RSS_bytes':peak,
    'final_scores_accessed':bool(live or done),'entire_research_complete':False})
  if jobs or live:time.sleep(10)
 accepted={k:sha(OUT/k/'COMPLETE.json') for k in done};assert len(accepted)==4
 report(accepted)
 atomic(OUT/'STAGE_FINAL_EVALUATION_ACCEPTANCE.json',{'status':'PASS_four_heldout_partitions_fixed_models_and_locked_policies',
  'time_UTC':stamp(),'partitions':accepted,'report_sha256':sha(OUT/'FORMAL_EVALUATION_REPORT_ZH.md'),
  'code_lock_sha256':sha(OUT/'CODE_LOCK.json'),'task_owned_workers_remaining':0,
  'hospital_year_descriptive_only_intervals_pending':True,'subgroup_sensitivity_public_scenarios_tools_manuscript_pending':True,'entire_research_complete':False})
 atomic(OUT/'status.json',{'state':'complete_four_heldout_evaluations','time_UTC':stamp(),'task_owned_workers_remaining':0,'entire_research_complete':False})
 prior.close();guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  atomic(OUT/'FAILURE.json',{'time_UTC':stamp(),'traceback':traceback.format_exc(),'owned_running_pids':[p.pid for p,l in OWNED if p.poll() is None]})
  for process,log in OWNED:
   process.wait()
   if not log.closed:log.close()
  raise
