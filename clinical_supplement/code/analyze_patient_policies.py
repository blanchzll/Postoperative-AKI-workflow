from pathlib import Path
import os,sys,json,hashlib,time,resource,csv,datetime,platform
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,pandas as pd
from scipy.special import logit,expit
from scipy.optimize import brentq,minimize
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint
P=Path('/home/lei/codex-workspaces/aki-prediction-manuscript'); C=P/'correction_runs/full_data_20261025_postslot'; O=C/'outputs'; R=Path(__file__).resolve().parents[1]
H={}; B=2000; SEED=20261108; started=time.time()
def sha(p):
 p=Path(p);h=hashlib.sha256(p.read_bytes()).hexdigest();H[str(p)]={'sha256':h,'bytes':p.stat().st_size,'resolved':str(p.resolve())};return h
def js(p):sha(p);return json.loads(Path(p).read_text())
def read(p):sha(p);return pd.read_csv(p,float_precision='round_trip',dtype={'group_hash':str})
def save(n,d): (R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')
def table(n,rows):pd.DataFrame(rows).to_csv(R/n,index=False)
def ci(v):
 a=np.asarray(v);a=a[np.isfinite(a)];return list(map(float,np.quantile(a,[.025,.975]))) if len(a) else [None,None]
def weights(groups,seed):
 un,inv=np.unique(groups,return_inverse=True);w=np.random.default_rng(seed).multinomial(len(un),np.full(len(un),1/len(un)),size=B).astype(np.int16)[:,inv]
 return np.vstack([np.ones((1,len(groups)),np.int16),w])
def order(p,g,k,ascending=False):return np.array(sorted(range(len(p)),key=lambda i:((p[i] if ascending else -p[i]),str(g[i]),str(k[i]))))
def choose(w,idx,k):
 v=w[:,idx];cum=v.cumsum(1);sel=np.minimum(v,np.maximum(np.asarray(k)[:,None]-(cum-v),0));out=np.zeros_like(w);out[:,idx]=sel;return out

def audit_start():
 lock=js(R/'PROTOCOL_LOCK.json');assert lock['status'].startswith('LOCKED')
 assert sha(R/'REQUEST.json')==lock['request_sha256'];assert sha(P/'writing_runs/probabilistic_integration_20261105_v1/package/PROBABILISTIC_INTEGRATION_ACCEPTANCE.json')==lock['baseline_acceptance_sha256']
 source=O/'v4_development_inputs_20261018';sl=js(source/'SOURCE_LOCK.json');assert sha(source/'SOURCE_LOCK.json')==lock['source_sha256']
 for f,h in sl['input_sha256'].items():assert sha(source/f)==h
 src=js(P/'outputs/full_data_audit_20261025_v1/candidate_release/MANIFEST.json')
 for key in ['clinical/preop_cohort.csv','clinical/paired_cohort.csv','SOURCE_VALUE_OVERLAY.csv']:
  assert sha(P/'outputs/full_data_audit_20261025_v1/candidate_release'/key)==src['files'][key]
 mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()};assert int(mem['MemAvailable'].split()[0])>8*1024**2
 cg={}
 for file in ['memory.max','memory.current','cpu.max','cpuset.cpus.effective']:
  f=Path('/sys/fs/cgroup')/file;cg[file]=f.read_text().strip() if f.exists() else 'unavailable'
 save('audit/ENVIRONMENT_AND_RESOURCES.json',{'python':sys.executable,'python_version':platform.python_version(),'affinity':sorted(os.sched_getaffinity(0)),'MemAvailable':mem['MemAvailable'],'cgroup_root':cg,'versions':{m:__import__(m).__version__ for m in ['numpy','pandas','scipy','sklearn','statsmodels']},'expected_peak_GiB':4,'GPU_reason':'Small fixed-score sorts and resampling; CPU NumPy avoids transfer and precision changes','threads':1,'threads_reason':'No model fitting; short arrays, vectorized over2000 draws; parallel BLAS overhead has no useful computation here'})
 wb=P/'outputs/workbuddy_revision_20261106_v1/inputs';wbm=js(wb/'REPORT_SHA256.json');assert all(sha(wb/'workbuddy_reports'/k)==h for k,h in wbm.items())
 save('audit/WORKBUDDY_HASH_CHECK.json',{'status':'PASS','reports':8,'manifest_sha256':sha(wb/'REPORT_SHA256.json')})
 return source

def routes(line,cohort,source):
 acc=js(O/'model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json');locks={}
 for ref in acc['models']:
  l=js(ref['path']);i=l['identity']
  if (i['line'],i['cohort'])!=(line,cohort) or not l['primary_route_winner']:continue
  if cohort=='preop' and i['branch'] not in ['compact','full']:continue
  if cohort=='paired' and i['branch']!='full':continue
  assert sha(ref['path'])==ref['sha256'];assert l['input_source_sha256']==sha(source/'SOURCE_LOCK.json')
  key=('base' if (cohort=='preop' and i['branch']=='compact') or (cohort=='paired' and i['stage']=='preop') else 'upgrade')
  folder=O/'formal_evaluation_20261023_v1'/f'{line}_{cohort}'/'private_predictions'/'_'.join(i[t] for t in ['stage','branch','role','family'])
  rec=js(folder/'COMPLETE.json');assert rec['signature']['model_lock']==ref['sha256'];assert sha(folder/'PREDICTIONS.csv')==rec['sha256']
  d=read(folder/'PREDICTIONS.csv').set_index('excel_row',drop=False).sort_index();m=read(source/(cohort+'_metadata.csv')).set_index('excel_row',drop=False)
  assert set(d.index)==set(m.index[m[line+'_partition_v2'].eq('validation')]);np.testing.assert_array_equal(d.y,m.loc[d.index,'y']);np.testing.assert_array_equal(d.group_hash,m.loc[d.index,'group_hash'])
  dev=read(l['calibration_training_scores']['path']);assert sha(l['calibration_training_scores']['path'])==l['calibration_training_scores']['sha256'];assert set(dev.group_hash).isdisjoint(d.group_hash)
  locks[key]=(l,d,dev)
 assert set(locks)=={'base','upgrade'}
 np.testing.assert_array_equal(locks['base'][1].index,locks['upgrade'][1].index)
 return locks

def main():
 source=audit_start();metrics=[];policies=[];diffs=[];trans=[];fields=[];randomrows=[];lowrows=[];dailyrows=[];dailysum=[];timecost=[];trigrows=[];provenance=[]
 for ix,(line,cohort) in enumerate([(a,b) for a in ['geography','temporal'] for b in ['preop','paired']]):
  tag=f'{line}_{cohort}';print('CONTEXT',tag,flush=True)
  lock=routes(line,cohort,source);bl,m,bd=lock['base'];ul,um,ud=lock['upgrade'];p=m.prob_calibrated.to_numpy();up=um.prob_calibrated.to_numpy();y=m.y.to_numpy(int);n=len(m);keys=m.excel_row.to_numpy();groups=m.group_hash.to_numpy(); w=weights(groups,SEED+ix);nn=w.sum(1);ey=w@y
  assert np.isfinite(p).all() and np.isfinite(up).all();assert np.all((p>=0)&(p<=1)) and np.all((up>=0)&(up<=1))
  np.savez_compressed(R/'private'/f'{tag}_RESAMPLING.npz',weights=w,records=keys,groups=groups,y=y,pbase=p,pupgrade=up)
  plp=O/f'selective_policy_lock_20261023_v1/locked_policies/{line}/{cohort}/0.25/capacity/selective/POLICY_LOCK.json';pl=js(plp);gate=pl['gate'];gate_dist=np.abs(logit(np.clip(p,gate['clip'],1-gate['clip']))-logit(gate['threshold']))
  existing_order=order(gate_dist,groups,keys,True);high_order=order(p,groups,keys);near_order=order(np.abs(p-bl['training_thresholds']['training_capacity20']),groups,keys,True)
  ran=np.array([int(hashlib.sha256(f'{SEED}|{g}|{k}'.encode()).hexdigest()[:15],16) for g,k in zip(groups,keys)],np.int64);random_order=order(ran,groups,keys,True)
  qk=np.ceil(.25*nn).astype(int);review=np.ceil(.2*nn).astype(int)
  upgrades={'no_collection':np.zeros_like(w),'existing_selective25':choose(w,existing_order,qk),'highest_base_risk25':choose(w,high_order,qk),'near_training_cutoff25':choose(w,near_order,qk),'random25':choose(w,random_order,qk),'all_collection':w.copy()}
  old=read(O/f'formal_evaluation_20261023_v1/{tag}/policy_0.25_capacity_selective_PRIVATE.csv').set_index('excel_row').loc[keys]
  np.testing.assert_array_equal(upgrades['existing_selective25'][0].astype(bool),old.upgraded.to_numpy(bool));np.testing.assert_allclose(np.where(old.upgraded,up,p),old.prob_raw_mixture,rtol=1e-13,atol=1e-13)
  # Original final policy calibration is monotone: all rank comparisons may use its saved raw mixture.
  sort=np.argsort(old.prob_raw_mixture.to_numpy(),kind='stable');assert np.all(np.diff(old.prob_calibrated.to_numpy()[sort])>=-1e-13)
  poolp=np.r_[p,up];pooly=np.r_[y,y];poolkeys=np.r_[keys,keys];poolg=np.r_[groups,groups];poolorder=order(poolp,poolg,poolkeys)
  outmeta=m.drop(columns=['prob_raw','prob_calibrated']).copy();outmeta['pbase']=p;outmeta['pupgrade']=up
  captured={};reviewm={}
  for cost in [0,.05,.1,.25]:
   total_budget=review+cost*qk
   for mode,u in upgrades.items():
    k=np.maximum(np.floor(total_budget-cost*u.sum(1)+1e-9),0).astype(int);k=np.minimum(k,nn)
    cc=np.concatenate([w-u,u],axis=1);sel=choose(cc,poolorder,k);tp=sel@pooly;cap=np.divide(tp,nn)*1000
    # No collection at original20% capacity is retained separately from same-budget expanded base.
    result={'context':tag,'comparison':'9to27' if cohort=='preop' else '27to38','policy':mode,'cost_ratio':cost,'N':n,'groups':len(np.unique(groups)),'events':int(y.sum()),'upgraded_n':int(u[0].sum()),'reviewed_n':int(sel[0].sum()),'total_review_equivalent_budget':float(total_budget[0]),'used_budget':float(sel[0].sum()+cost*u[0].sum()),'captured_events':int(tp[0]),'missed_events':int(y.sum()-tp[0]),'captured_per1000':float(cap[0]),'captured_per1000_lo':ci(cap[1:])[0],'captured_per1000_hi':ci(cap[1:])[1],'sensitivity':float(tp[0]/y.sum()),'sensitivity_lo':ci(tp[1:]/ey[1:])[0],'sensitivity_hi':ci(tp[1:]/ey[1:])[1],'CI_condition':'patient groups; fixed fits; rerouted acquisition and review each draw; posthoc'}
    policies.append(result);captured[(cost,mode)]=cap;reviewm[(cost,mode)]=sel
    if cost==.1:
     outmeta['upgrade_'+mode]=u[0];outmeta['review_'+mode]=sel[0,:n]+sel[0,n:]
   basecap=captured[(cost,'no_collection')]
   for mode in upgrades:
    dd=captured[(cost,mode)]-basecap;diffs.append({'context':tag,'cost_ratio':cost,'policy':mode,'comparator':'no_collection_same_budget_base_expansion','delta_captured_per1000':float(dd[0]),'lo':ci(dd[1:])[0],'hi':ci(dd[1:])[1]})
  base20=choose(w,high_order,review);base20tp=base20@y
  policies.append({'context':tag,'policy':'no_collection_original20','cost_ratio':0,'N':n,'groups':len(np.unique(groups)),'events':int(y.sum()),'upgraded_n':0,'reviewed_n':int(base20[0].sum()),'captured_events':int(base20tp[0]),'missed_events':int(y.sum()-base20tp[0]),'captured_per1000':float(base20tp[0]/n*1000),'captured_per1000_lo':ci(base20tp[1:]/nn[1:]*1000)[0],'captured_per1000_hi':ci(base20tp[1:]/nn[1:]*1000)[1]})
  for mode in upgrades:
   ss=reviewm[(.1,mode)];ss=(ss[0,:n]+ss[0,n:]).astype(bool);bs=base20[0].astype(bool)
   trans.append({'context':tag,'policy':mode,'cost_ratio':.1,'base_review_n':int(bs.sum()),'review_n':int(ss.sum()),'newly_captured_known_events':int(np.sum(ss&~bs&(y==1))),'lost_previously_captured_events':int(np.sum(bs&~ss&(y==1))),'retained_captured_events':int(np.sum(bs&ss&(y==1))),'persistently_missed_events':int(np.sum(~bs&~ss&(y==1)))})
  for seed in range(200):
   rr=np.random.default_rng(SEED+10000+seed).permutation(n);uu=np.zeros(n,int);uu[rr[:int(np.ceil(.25*n))]]=1;ss=np.where(uu,up,p);oo=order(ss,groups,keys);kk=int(np.floor(np.ceil(.2*n)+.1*np.ceil(.25*n)-.1*uu.sum()+1e-9));tp=int(y[oo[:kk]].sum());randomrows.append({'context':tag,'seed':SEED+10000+seed,'upgraded_n':int(uu.sum()),'reviewed_n':kk,'captured_events':tp})
  # Field presence is distinct from assay need. All values are recorded historical availability, not release timestamps.
  feat=read(source/(cohort+'_'+('preop' if cohort=='preop' else 'endop')+'_features.csv')).set_index('excel_row').loc[keys]
  add=[f for f in ul['input_columns'] if f not in bl['input_columns']];labs={'PreopWBC','PreopHb','PreopAlb','PreopALT','PreopBUN','PreopCr','PreopGlucose'};missing=feat[add].isna()
  for mode,u in upgrades.items():
   chosen=u[0]>0
   for field in add:fields.append({'context':tag,'policy':mode,'field':field,'data_type':'laboratory' if field in labs else 'history_or_operation_record','selected_records':int(chosen.sum()),'already_recorded':int((~missing[field]&chosen).sum()),'not_recorded':int((missing[field]&chosen).sum()),'actual_order_or_release_clock_available':False})
  # Center calibration at two frozen rules, no local fitting deployed.
  scoremap={'base':(p,bl),'upgrade':(up,ul)}
  for role,(pp,ll) in scoremap.items():
   for center in ['all']+sorted(m.center.astype(str).unique()):
    mask=np.ones(n,bool) if center=='all' else m.center.astype(str).eq(center).to_numpy(); yy=y[mask];pr=pp[mask]; ww=w[:,mask];den=ww.sum(1);events=ww@yy;pred=ww@pr;ee=int(yy.sum()); z=logit(np.clip(pr,1e-12,1-1e-12));bb=ww@((yy-pr)**2)/np.maximum(den,1); oe=events/np.maximum(pred,1e-12)
    a=float(brentq(lambda a:expit(z+a).sum()-ee,-100,100)) if 0<ee<len(yy) else None
    slope=None;status='single_class' if ee in [0,len(yy)] else 'estimated'
    if a is not None:
     X=np.column_stack([np.ones(len(z)),z]);res=minimize(lambda b:np.logaddexp(0,X@b).sum()-yy@(X@b),[a,1.],jac=lambda b:X.T@(expit(X@b)-yy),method='BFGS')
     if res.success:slope=float(res.x[1])
     else:status='optimizer_not_converged'
    for rule in ['training_capacity20','training_sensitivity80']:
     alert=pr>=ll['training_thresholds'][rule];tp=ww@(alert*yy);rr=ww@alert;rv=rr/np.maximum(den,1);sens=np.divide(tp,events,out=np.full(len(tp),np.nan),where=events>0)
     metrics.append({'context':tag,'role':role,'inputs':ll['actual_input_count'],'center':center,'N':len(yy),'groups':len(np.unique(groups[mask])),'events':ee,'expected_events':float(pr.sum()),'prevalence':ee/len(yy),'AUROC':float(roc_auc_score(yy,pr)) if 0<ee<len(yy) else None,'Brier':float(np.mean((yy-pr)**2)),'Brier_lo':ci(bb[1:])[0],'Brier_hi':ci(bb[1:])[1],'O_E':ee/float(pr.sum()),'O_E_lo':ci(oe[1:])[0],'O_E_hi':ci(oe[1:])[1],'intercept_offset':a,'slope':slope,'slope_status':status,'small_events':ee<20,'rule':rule,'threshold':ll['training_thresholds'][rule],'reviewed_n':int(alert.sum()),'captured_events':int((alert*yy).sum()),'review_fraction':float(alert.mean()),'review_fraction_lo':ci(rv[1:])[0],'review_fraction_hi':ci(rv[1:])[1],'sensitivity':float((alert*yy).sum()/ee) if ee else None,'sensitivity_lo':ci(sens[1:])[0],'sensitivity_hi':ci(sens[1:])[1],'local_adaptation':False})
   for rule in ['training_sensitivity80','training_capacity20']:
    low=pp<ll['training_thresholds'][rule];dn=w@low;ev=w@(low*y);rate=np.divide(ev,dn,out=np.full(len(ev),np.nan),where=dn>0);e=int((low*y).sum());ln=int(low.sum());wil=proportion_confint(e,ln,method='wilson') if ln else [None,None]
    lowrows.append({'context':tag,'role':role,'inputs':ll['actual_input_count'],'rule':rule,'threshold':ll['training_thresholds'][rule],'N':n,'events_total':int(y.sum()),'low_risk_n':ln,'low_risk_events':e,'observed_low_risk_event_rate':e/ln if ln else None,'wilson_lo':wil[0],'wilson_hi':wil[1],'patient_boot_lo':ci(rate[1:])[0],'patient_boot_hi':ci(rate[1:])[1],'events_missed_fraction':e/int(y.sum()),'monitoring_reduction_safety_identified':False})
  # Fixed per-case candidate and existing individual gate at operation-date indexing.
  individual=read(O/f'formal_evaluation_20261023_v1/{tag}/policy_0.25_individual_selective_PRIVATE.csv').set_index('excel_row').loc[keys];ipl=js(O/f'selective_policy_lock_20261023_v1/locked_policies/{line}/{cohort}/0.25/individual/selective/POLICY_LOCK.json')
  uni=gate_dist<=gate['cutoffs']['0.25'];np.testing.assert_array_equal(uni,individual.upgraded.to_numpy(bool))
  pp=individual.prob_calibrated.to_numpy(); th=ipl['training_thresholds']['training_capacity20'];a=p>=bl['training_thresholds']['training_capacity20'];b=pp>=th
  for mode,score,cut,updated in [('no_update',p,bl['training_thresholds']['training_capacity20'],np.zeros(n,bool)),('all_update',up,ul['training_thresholds']['training_capacity20'],np.ones(n,bool)),('existing_individual25',pp,th,uni)]:
   alert=score>=cut;tp=w@(alert*y);rv=w@alert
   trigrows.append({'context':tag,'trigger':mode,'N':n,'events':int(y.sum()),'updated_n':int(updated.sum()),'review_n':int(alert.sum()),'captured_events':int(tp[0]),'missed_events':int(y.sum()-tp[0]),'capture_per1000':float(tp[0]/n*1000),'capture_per1000_lo':ci(tp[1:]/nn[1:]*1000)[0],'capture_per1000_hi':ci(tp[1:]/nn[1:]*1000)[1],'threshold':cut,'threshold_origin':'unchanged development lock; exploratory retrospective evaluation'})
  outmeta['individual_updated']=uni;outmeta['base_fixed_alert']=a;outmeta['policy_fixed_alert']=b;outmeta['policy_probability']=pp;outmeta.to_csv(R/'private'/f'{tag}_DECISIONS.csv',index=False)
  # Daily rows: observed surgical cohort, conditional demands, not completed work or alert-arrival.
  for center in sorted(m.center.astype(str).unique()):
   mask=m.center.astype(str).eq(center).to_numpy();sub=m.loc[mask].copy();dates=pd.to_datetime(sub.surgery_date,format='ISO8601').dt.normalize();frame=pd.DataFrame({'date':dates,'group':groups[mask],'base_alert':a[mask],'policy_alert':b[mask],'updated':uni[mask],'new_alert':(~a&b)[mask],'repeat_alert':(a&b)[mask]})
   missing_lab=missing.loc[sub.index,[f for f in add if f in labs]].sum(axis=1).to_numpy();missing_chart=missing.loc[sub.index,[f for f in add if f not in labs]].sum(axis=1).to_numpy()
   frame['missing_lab_requests']=missing_lab*uni[mask];frame['missing_chart_requests']=missing_chart*uni[mask]
   frame['review_requests']=frame.base_alert.astype(int)+frame.policy_alert.astype(int);frame['unique_requested']=(frame.base_alert|frame.policy_alert)
   gg=frame.groupby('date');daily=gg[['base_alert','policy_alert','updated','new_alert','repeat_alert','missing_lab_requests','missing_chart_requests','review_requests']].sum();daily['operations']=gg.size();daily['unique_review_people']=gg.apply(lambda x:x.loc[x.unique_requested,'group'].nunique(),include_groups=False);daily=daily.reindex(pd.date_range(dates.min(),dates.max(),freq='D'),fill_value=0)
   for day,row in daily.iterrows():dailyrows.append({'context':tag,'center':center,'surgery_date':str(day.date()),**row.astype(int).to_dict(),'time_basis':'surgery-date index; potential requests; actual arrival/completion unknown'})
   for denominator in ['all_calendar_days','active_operation_days']:
    dd=daily if denominator=='all_calendar_days' else daily.loc[daily.operations>0]
    for field in ['operations','base_alert','policy_alert','updated','unique_review_people','repeat_alert','review_requests']:
     v=dd[field].to_numpy();rv=daily[field].to_numpy();L=len(rv);rng=np.random.default_rng(SEED+ix);nb=int(np.ceil(L/7));starts=rng.integers(0,L,size=(B,nb));inds=(starts[:,:,None]+np.arange(7))%L;bs=rv[inds.reshape(B,-1)[:,:L]]
     if denominator=='active_operation_days':
      active=daily.operations.to_numpy()[inds.reshape(B,-1)[:,:L]]>0;means=np.sum(bs*active,axis=1)/np.maximum(active.sum(1),1)
     else:means=bs.mean(1)
     dailysum.append({'context':tag,'center':center,'denominator':denominator,'days':len(dd),'metric':field,'total':int(v.sum()),'mean':float(v.mean()),'mean_block7_lo':ci(means)[0],'mean_block7_hi':ci(means)[1],'p50':float(np.quantile(v,.5)),'p95':float(np.quantile(v,.95)),'max':int(v.max()),'overload_above5_fraction':float((v>5).mean()),'overload_above10_fraction':float((v>10).mean()),'overload_above20_fraction':float((v>20).mean()),'actual_completed_actions_known':False})
   for reviewmin in [5,10,15]:
    for fieldmin in [.25,1]:
     for labmin in [0,5,10]:
      minutes=daily.review_requests*reviewmin+daily.missing_chart_requests*fieldmin+daily.missing_lab_requests*labmin
      timecost.append({'context':tag,'center':center,'review_minutes':reviewmin,'missing_chart_field_minutes':fieldmin,'missing_lab_minutes':labmin,'total_minutes':float(minutes.sum()),'mean_calendar_day_minutes':float(minutes.mean()),'p95_calendar_day_minutes':float(minutes.quantile(.95)),'max_day_minutes':float(minutes.max()),'fraction_days_over60minutes':float((minutes>60).mean()),'staff_substitution':'not measured; chart completion potentially delegable, clinical review distinct','assumption':'historical absent field does not prove new test required; no observed staffing times'})
  provenance.append({'context':tag,'N':n,'events':int(y.sum()),'base_family':bl['family'],'upgrade_family':ul['family'],'input_counts':f"{bl['actual_input_count']}to{ul['actual_input_count']}",'dev_n':bl['training_n'],'evaluation_exposure':'previously exposed','prediction_alignment':'exact labels/groups/rows; same cohort','feature_release_clocks_available':False})
 for filename,rows in [('POLICY_EQUAL_BUDGET.csv',policies),('POLICY_PAIRED_DIFFERENCES.csv',diffs),('CAPTURE_TRANSITIONS.csv',trans),('FIELD_COMPLETION.csv',fields),('RANDOMIZATION_200.csv',randomrows),('CENTER_METRICS.csv',metrics),('LOW_RISK_EVENTS.csv',lowrows),('DAILY_POTENTIAL_DEMAND.csv',dailyrows),('DAILY_SUMMARY.csv',dailysum),('DAILY_TIME_SCENARIOS.csv',timecost),('FIXED_TRIGGER_COMPARISONS.csv',trigrows),('PREDICTION_SOURCE_AUDIT.csv',provenance)]:table('results/'+filename,rows)
 sha(Path(__file__));save('audit/PATIENT_ANALYSIS_INPUT_HASHES.json',H);save('audit/PATIENT_ANALYSIS_RUN.json',{'status':'COMPUTED_REQUIRES_INDEPENDENT_REVIEW','seconds':time.time()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'pid':os.getpid(),'B':B,'seed':SEED,'contexts':4,'source_inputs_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v['sha256'] for p,v in H.items()),'no_new_model_fits':True})
 print('PATIENT_ANALYSES_COMPUTED',time.time()-started,flush=True)
if __name__=='__main__':main()
