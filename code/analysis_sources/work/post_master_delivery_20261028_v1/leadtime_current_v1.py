"""Fixed preop-landmark to recorded SCr-criterion interval; not live clinical lead time."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback,datetime,math,resource,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
OUT=ROOT/'outputs/benchmark_and_leadtime_20261024_v1'
PUB=ROOT/'outputs/public_eligibility_refinement_20261022_v1'
EVAL=ROOT/'outputs/common6_public_evaluation_20261024_v1'
sys.path.insert(0,str(ROOT/'work/common6_public_evaluation_20261024_v1'))
import evaluate_public as parent
stats=parent.stats
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):parent.base.atomic(p,v)
def read(p):return json.loads(Path(p).read_text())
def metrics(y,lead,alert):
 y=np.asarray(y,dtype=int);lead=np.asarray(lead,float);alert=np.asarray(alert,bool)
 result=stats.counts(y,alert);captured=lead[(y==1)&alert]
 for q,name in [(.25,'Q1'),(.5,'median'),(.75,'Q3')]:
  result['detected_record_interval_hours_'+name]=float(np.quantile(captured,q)) if len(captured) else None
 for h in [12,24,48]:
  n=int(((y==1)&alert&(lead>=h)).sum())
  result[f'all_AKI_detected_atleast{h}h_n']=n
  result[f'all_AKI_detected_atleast{h}h_fraction']=n/int(y.sum()) if y.sum() else None
 return result
def qualify():
 y=np.array([1,1,1,0,0]);lead=np.array([6.,24.,72.,np.nan,np.nan]);alert=np.array([1,0,1,1,0],bool)
 m=metrics(y,lead,alert)
 assert m['captured_n']==2 and m['false_alerts_n']==1 and m['missed_n']==1
 assert m['all_AKI_detected_atleast12h_n']==1 and m['all_AKI_detected_atleast24h_n']==1 and m['all_AKI_detected_atleast48h_n']==1
 assert m['all_AKI_detected_atleast24h_fraction']==1/3
 assert m['detected_record_interval_hours_median']==39
 assert metrics(y,lead,np.zeros(5,bool))['detected_record_interval_hours_median'] is None
 assert metrics(np.zeros(5,int),lead,alert)['all_AKI_detected_atleast24h_fraction'] is None
 # Reordering cannot change fixed-threshold event-level statistics.
 j=np.array([3,0,4,2,1]);assert metrics(y[j],lead[j],alert[j])==m
 save(OUT/'LEADTIME_FUNCTIONAL_QUALIFICATION.json',{'status':'PASS_synthetic_earliness_denominators_misses_and_reordering','code_sha256':sha(__file__)})
SHARED=None
def score_repeat(rep):
 weights,groups,records,scorer=SHARED
 idx,g,r=stats.expanded(weights[rep],groups,records)
 result=scorer(idx,g,r)
 return [{'replicate':rep,'rule':name,**value} for name,value in result.items()]
def execute():
 global SHARED
 begin=time.time();c=read(OUT/'LEADTIME_CONTRACT.json')
 assert sha(__file__)==c['code_sha256']
 for p,h in c['sources'].items():assert sha(ROOT/p)==h,p
 assert read(OUT/'LEADTIME_FUNCTIONAL_QUALIFICATION.json')['code_sha256']==sha(__file__)
 x,m=parent.source();receipt,locks=parent.locks()
 # Join exact already-evaluated cohort. Never relabel/replace Chinese or public endpoints.
 operations=pd.read_csv(PUB/'PRIVATE_FIRST_OPERATION_PROFILE.csv',dtype={'op_id':str,'subject_id':str})
 z=m.reset_index(drop=True).merge(operations[['op_id','subject_id','opstart_time','opend_time','discharge_time']],
       on=['op_id','subject_id'],validate='one_to_one',how='left').set_index('record_key').loc[m.index]
 assert len(z)==6043 and z.y_scr.sum()==253
 assert np.isfinite(z[['opstart_time','opend_time','discharge_time']].to_numpy()).all()
 event=z.y_scr.to_numpy()==1
 assert z.loc[event,'first_observed_criterion_time'].notna().all()
 assert (z.loc[event,'first_observed_criterion_time']>z.loc[event,'opend_time']).all()
 assert (z.opend_time>z.opstart_time).all()
 # t0 is a scheduled retrospective landmark; it is NOT claimed to be a measured alert or result-release time.
 lag=(z.first_observed_criterion_time-z.opstart_time).to_numpy(float)/60
 pred={}
 for line in ['geography','temporal']:
  r=read(EVAL/line/'PREDICTION_ACCEPTANCE.json');p=EVAL/line/'PRIVATE_PREDICTIONS.csv'
  assert sha(p)==r['prediction_sha256']
  v=pd.read_csv(p,dtype={'record_key':str,'group_hash':str},float_precision='round_trip').set_index('record_key').loc[m.index]
  assert np.array_equal(v.y_scr.to_numpy(),m.y_scr.to_numpy())
  assert v.group_hash.tolist()==x.group_hash.tolist()
  pred[line]=v.prob_calibrated.to_numpy()
 yy,gg,rr,cc,strata,models=stats._design(m.y_scr.to_numpy(),x.group_hash.astype(str).to_numpy(),
      x.record_key.astype(str).to_numpy(),['INSPIRE']*len(m),pred)
 order=m.index.get_indexer(rr);assert (order>=0).all();lead=lag[order]
 thresholds={line:locks[line]['training_thresholds']['training_sensitivity80'] for line in models}
 points={}
 def score(idx,groups,records):
  result={}
  for line,model in models.items():
   p=model.p[idx];y=yy[idx];ll=lead[idx]
   result[line+'__source_sensitivity80']=metrics(y,ll,p>=thresholds[line])
   for cap in [.1,.2]:
    f=stats.FixedRisk(y,p,groups,records);mask=f.review_counts(cap).astype(bool)
    result[line+f'__retrospective_capacity{int(cap*100)}']=metrics(y,ll,mask)
  return result
 points=score(np.arange(len(yy)),gg,rr)
 # Duplicate subject identity is not an extra patient. Source selection is one operation per subject.
 assert x.group_hash.is_unique and m.subject_id.is_unique
 rng=np.random.default_rng(20261017);proof=hashlib.sha256();weights=[]
 for rep in range(2000):
  w=stats.group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes());weights.append(w)
 weights=np.asarray(weights,dtype=np.int16)
 expected=read(EVAL/'EVALUATION.json')['receipt']
 assert proof.hexdigest()==expected['resample_weight_sha256'],'Exact original resampling receipt mismatch'
 SHARED=(weights,gg,rr,score)
 serial0=score_repeat(0);save(OUT/'LEADTIME_BOOTSTRAP_0000.json',serial0)
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 # Execution-only admission: own finite cgroup plus other AKI anonymous growth and OS.
 reserve=max(4*2**30,int(.25*available));worker_bytes=max(450*2**20,int(1.4*peak))
 cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
 maximum=(cg/'memory.max').read_text().strip();assert maximum!='max'
 own_remaining=max(0,int(maximum)-int((cg/'memory.current').read_text()))
 branch=cg.parent
 while branch!=Path('/sys/fs/cgroup'):
  mx=(branch/'memory.max').read_text().strip();used=int((branch/'memory.current').read_text())
  if mx!='max':available=min(available,max(0,int(mx)-used))
  branch=branch.parent
 extra=0;other=[]
 units=__import__('subprocess').run(['systemctl','--user','list-units','--state=active','--plain','--no-legend','aki-postslot-*'],capture_output=True,text=True,check=True).stdout.splitlines()
 for row in units:
  name=row.split()[0]
  if name in Path('/proc/self/cgroup').read_text():continue
  props=__import__('subprocess').run(['systemctl','--user','show',name,'-p','MemoryMax','-p','ControlGroup'],capture_output=True,text=True,check=True).stdout
  prop=dict(v.split('=',1) for v in props.splitlines() if '=' in v)
  assert prop['MemoryMax'].isdigit(),'Other active AKI unit needs a bounded resource budget'
  stat=Path('/sys/fs/cgroup')/prop['ControlGroup'].lstrip('/')/'memory.stat'
  mem=dict(v.split() for v in stat.read_text().splitlines())
  anonymous=sum(int(mem.get(k,0)) for k in ['anon','kernel','shmem'])
  growth=max(0,int(prop['MemoryMax'])-anonymous);extra+=growth;other.append({'unit':name,'growth_reserved_bytes':growth})
 usable=min(own_remaining,max(0,available-reserve-extra))
 assert usable>=worker_bytes+2**30,'Insufficient current resources for a safe finite bootstrap launch'
 workers=max(1,min(len(os.sched_getaffinity(0)),int((usable-2**30)//worker_bytes)))
 save(OUT/'LEADTIME_RESOURCE_ALLOCATION.json',{'workers':workers,'MemAvailable':available,'reserve':reserve,'measured_parent_peak':peak,'budget_per_worker':worker_bytes,'backend':'CPU resampling; no new model or GPU precision migration'})
 pending=[rep for rep in range(1,2000) if not (OUT/f'LEADTIME_BOOTSTRAP_{rep:04d}.json').exists()]
 completed=2000-len(pending)
 with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
  assert pool.submit(score_repeat,0).result()==serial0,'Fork/serial mismatch on a real fixed-cohort repeat'
  future={pool.submit(score_repeat,rep):rep for rep in pending}
  for f in as_completed(future):
   rep=future[f];save(OUT/f'LEADTIME_BOOTSTRAP_{rep:04d}.json',f.result());completed+=1
   if completed%100==0:save(OUT/'LEADTIME_STATUS.json',{'state':'running','completed':completed,'total':2000,'workers':workers})
 rows=[]
 for rep in range(2000):rows.extend(read(OUT/f'LEADTIME_BOOTSTRAP_{rep:04d}.json'))
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'LEADTIME_BOOTSTRAPS.csv',index=False)
 pframe=pd.DataFrame([{'rule':name,**value} for name,value in points.items()])
 pframe.to_csv(OUT/'LEADTIME_POINT_ESTIMATES.csv',index=False)
 intervals=[]
 for rule,q in frame.groupby('rule',sort=True):
  for key in q.columns:
   if key in ['rule','replicate']:continue
   v=pd.to_numeric(q[key],errors='coerce').dropna().to_numpy()
   ci=np.quantile(v,[.025,.975]) if len(v) else [None,None]
   intervals.append({'rule':rule,'metric':key,'estimate':points[rule][key],
     'lower95':ci[0],'upper95':ci[1],'valid_replicates':len(v),'planned':2000})
 pd.DataFrame(intervals).to_csv(OUT/'LEADTIME_INTERVALS.csv',index=False)
 # Verify basic burden and capture exactly agree with the previously accepted public evaluation.
 oldpoint=pd.read_csv(EVAL/'POINT_ESTIMATES.csv').set_index('model')
 for line in ['geography','temporal']:
  for cap in [10,20]:
   v=points[line+f'__retrospective_capacity{cap}']
   for field in ['reviewed_n','captured_n','missed_n','false_alerts_n']:
    assert int(v[field])==int(oldpoint.loc[line+'__calibrated',f'capacity{cap}_{field}'])
 # Private join only, no patient data released.
 z['scheduled_landmark_to_recorded_criterion_hours']=lag;z.to_csv(OUT/'PRIVATE_LEADTIME_JOIN.csv')
 result={'status':'FITTED_REQUIRES_REVIEW_fixed_landmark_recorded_SCr_intervals','cohort':6043,'events':253,
 'no_retrained_model_or_redefined_endpoint':True,'no_measured_realworld_alarm_time_claimed':True,
 'public_resampling_proof_matches':proof.hexdigest(),'source_thresholds':thresholds,'seconds':time.time()-begin,
 'code_sha256':sha(__file__),'contract_sha256':sha(OUT/'LEADTIME_CONTRACT.json'),
 'files':{f:sha(OUT/f) for f in ['LEADTIME_POINT_ESTIMATES.csv','LEADTIME_INTERVALS.csv','LEADTIME_BOOTSTRAPS.csv','PRIVATE_LEADTIME_JOIN.csv']}}
 save(OUT/'LEADTIME_FITTING_ACCEPTANCE.json',result)
 save(OUT/'LEADTIME_STATUS.json',{'state':'fitted_requires_review','completed':2000,'total':2000})
 print(json.dumps(result))
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1]=='qualify':qualify()
  else:execute()
 except Exception:save(OUT/'LEADTIME_FAILURE.json',{'traceback':traceback.format_exc(),'code_sha256':sha(__file__)});raise
