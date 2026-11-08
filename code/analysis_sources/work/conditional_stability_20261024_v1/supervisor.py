from pathlib import Path
import sys,os,time,fcntl,subprocess,traceback,shutil
from stability import ROOT,CODE,OUT,base,np,pd,rt
OWNED=[]
def rss(pid):
 try:return int(next(l.split()[1] for l in Path('/proc',str(pid),'status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
 except (OSError,StopIteration):return 0
def verify(route,rep):
 d=OUT/'replicates'/route/f'{rep:03d}';c=base.read(d/'COMPLETE.json')
 assert c['signature']['code_lock']==base.sha(OUT/'FIT_IMPLEMENTATION_CODE_LOCK.json')
 for f,h in c['files'].items():assert base.sha(d/f)==h
 if c['status']=='PASS_conditional_fixed_flow_refit':
  for f,h in c['fold_receipts'].items():assert base.sha(d/f'fold_{f}'/'COMPLETE.json')==h
  assert base.sha(d/'full/COMPLETE.json')==c['full_receipt_sha256']
  for dirname in [f'fold_{f}' for f in range(3)]+['full']:
   for f,h in base.read(d/dirname/'COMPLETE.json')['files'].items():assert base.sha(d/dirname/f)==h
 else:assert c['status']=='INVALID_planned_sample_retained_no_replacement'
 return c
def summarize(routes):
 summary=[];receipts={}
 for route in routes:
  preds=[];raw=[];stats=[];invalid=[];ids=None
  for rep in range(200):
   d=OUT/'replicates'/route/f'{rep:03d}';c=verify(route,rep);receipts[route+'/'+str(rep)]=base.sha(d/'COMPLETE.json')
   if c['status'].startswith('INVALID'):invalid.append({'replicate':rep,'reason':c['reason']});continue
   z=np.load(d/'PREDICTIONS_PRIVATE.npz')
   if ids is None:ids=z['excel_row']
   else:np.testing.assert_array_equal(ids,z['excel_row'])
   preds.append(z['calibrated']);raw.append(z['raw']);stats.append(base.read(d/'RESULT.json')['metrics'])
  dest=OUT/'summaries'/route;dest.mkdir(parents=True,exist_ok=True)
  pd.DataFrame(stats).to_csv(dest/'REPLICATE_METRICS.csv',index=False)
  if preds:
   a=np.array(preds);r=np.array(raw)
   pd.DataFrame({'excel_row':ids,'calibrated_mean':a.mean(0),'calibrated_p025':np.quantile(a,.025,axis=0),'calibrated_p975':np.quantile(a,.975,axis=0),'raw_mean':r.mean(0),'raw_p025':np.quantile(r,.025,axis=0),'raw_p975':np.quantile(r,.975,axis=0)}).to_csv(dest/'INDIVIDUAL_RISK_RANGES_PRIVATE.csv',index=False)
  diagnostics=[base.read(p) for p in (OUT/'replicates'/route).glob('*/full/INFERENCE_DIAGNOSTIC.json')]
  q={'route':route,'planned':200,'valid':len(preds),'invalid':len(invalid),'strict_probe_parity_failures':sum(not z['strict_single_batch_reorder_tolerance_pass'] for z in diagnostics),'recorded_max_probe_difference':max([z['max_probe_abs_difference'] for z in diagnostics],default=None)}

  if stats:
   for k in stats[0]:
    v=np.array([z[k] for z in stats]);q[k+'_mean']=float(v.mean());q[k+'_p025']=float(np.quantile(v,.025));q[k+'_p975']=float(np.quantile(v,.975))
  summary.append(q);base.atomic(dest/'INVALID.json',invalid)
 pd.DataFrame(summary).to_csv(OUT/'ROUTE_STABILITY_SUMMARY.csv',index=False)
 (OUT/'CONDITIONAL_STABILITY_REPORT_ZH.md').write_text('# 固定流程的条件性稳定性检验\n\n10条原锁定路线各计划200次开发患者组bootstrap，保留重复次数及组内手术记录。变量（包括各集成成员自己的输入）、插补方法、家族、参数及成员固定，重新拟合插补／转换／模型及患者组隔离3折OOF校准。没有重做选模。复制病例在同一内折，插补拟合保留复制权重，但同一原病例的复制使用相同随机完成值。\n\n原V1风险与重拟合风险在相同开发参考人群比较，报告原始／校准风险偏差、固定原阈值与重拟合阈值的翻转、最高10%／20%名单重合。个体2.5%—97.5%范围是条件性经验分布，不是真实风险置信区间，也不代表无偏预测效能。无效抽样保留，不替换为有利重复。\n\n主表ROUTE_STABILITY_SUMMARY.csv；各路线逐重复指标及私有个体范围在summaries/。部分重拟合KNN组件单例与批量在原1e-12标准下不一致，保留失败标记和最大差异；完整批次重载严格一致。此批仅固定批次风险稳定性诊断，不将这些副本称为已通过个例部署资格。此结果不替换原模型，不以稳定性诊断或最终留出表现重新选赢家。公共、敏感性、SHAP、决策情景与研究工具另行验收。\n')
 base.atomic(OUT/'STAGE_STABILITY_ACCEPTANCE.json',{'status':'PASS_200_planned_conditional_bootstraps_per_10_routes','receipts':receipts,'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),'summary_sha256':base.sha(OUT/'ROUTE_STABILITY_SUMMARY.csv'),'report_sha256':base.sha(OUT/'CONDITIONAL_STABILITY_REPORT_ZH.md'),'owned_workers_remaining':0,'refit_probe_parity_failures_retained_as_diagnostics_not_deployment_qualified':True,'entire_research_complete':False})
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 base.gate();code=base.read(OUT/'CODE_LOCK.json')
 for n,h in code.items():assert base.sha(CODE/n)==h
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_group_bootstrap_multiplicity_imputation_boundaries'
 if (OUT/'STAGE_STABILITY_ACCEPTANCE.json').exists():return
 routes=base.read(OUT/'ROUTES.json');assert len(routes)==10
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 pilots=[];jobs=[];live={};done=[];failed=[];peaks={};reservations={r:2*1024**3 for r in routes};accepted=[]
 for r in routes:
  available={i for i in range(200) if (OUT/'replicates'/r/f'{i:03d}'/'COMPLETE.json').exists()}
  for i in sorted(available):verify(r,i);done.append(r+'_'+str(i))
  if 0 in available:
   c=verify(r,0);assert c['status']=='PASS_conditional_fixed_flow_refit'
   accepted.append(r);reservations[r]=max(768*1024**2,2*c['peak_rss_bytes'])
   jobs.extend((r,i) for i in range(1,200) if i not in available)
  else:pilots.append((r,0))
 while pilots or jobs or live:
  for key,(p,log,r,b) in list(live.items()):
   peaks[key]=max(peaks.get(key,0),rss(p.pid));rc=p.poll()
   if rc is not None:
    log.close();del live[key]
    if rc:failed.append({'route':r,'replicate':b,'returncode':rc})
    else:
     c=verify(r,b);done.append(key)
     if b==0:
      if c['status']!='PASS_conditional_fixed_flow_refit':failed.append({'route':r,'reason':'pilot invalid; diagnose no substitution'});continue
      reservations[r]=max(768*1024**2,2*c['peak_rss_bytes'],2*peaks[key]);accepted.append(r)
      base.atomic(OUT/(r+'_PILOT_ACCEPTANCE.json'),{'status':'PASS_first_planned_replicate_reusable','receipt_sha256':base.sha(OUT/'replicates'/r/'000/COMPLETE.json'),'resource_reservation_bytes':reservations[r]})
      jobs.extend((r,i) for i in range(1,200))
  if failed and not live:raise RuntimeError('Stability worker failure; retain checkpoints; diagnose before retry '+str(failed))
  available=base.mem();head=max(4*1024**3,int(available*.25))
  growth=sum(max(0,reservations[r]-rss(p.pid)) for p,l,r,b in live.values())
  for n,h in code.items():assert base.sha(CODE/n)==h
  while (pilots or jobs) and not failed and len(live)<len(os.sched_getaffinity(0)):
   pending=pilots if pilots else jobs
   r,b=pending[0]
   if available-head-growth<reservations[r] or shutil.disk_usage(OUT).free<100*1024**3:break
   pending.pop(0);key=r+'_'+str(b);logdir=OUT/'logs'/r;logdir.mkdir(parents=True,exist_ok=True)
   log=(logdir/(str(b)+'.log')).open('ab')
   p=subprocess.Popen([str(base.PYTHON),'-B',str(CODE/'stability.py'),r,str(b)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   live[key]=(p,log,r,b);OWNED.append((p,log));peaks[key]=0;growth+=reservations[r]
  base.atomic(OUT/'status.json',{'state':'running_conditional_stability' if live else 'waiting_resources','pid':os.getpid(),'time_UTC':base.stamp(),'running':{k:v[0].pid for k,v in live.items()},'completed':len(done),'total':2000,'pending_pilots':len(pilots),'accepted_pilots':accepted,'pending_replicates':len(jobs),'failures':failed,'available_bytes':available,'growth_reserved_bytes':growth,'reservations_bytes':reservations,'entire_research_complete':False})
  if pilots or jobs or live:time.sleep(15)
 assert len(done)==2000;summarize(routes)
 base.atomic(OUT/'status.json',{'state':'complete_conditional_stability','time_UTC':base.stamp(),'completed':2000,'owned_workers_remaining':0,'entire_research_complete':False})
 guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'owned_running_pids':[p.pid for p,l in OWNED if p.poll() is None]})
  for p,l in OWNED:
   p.wait()
   if not l.closed:l.close()
  raise
