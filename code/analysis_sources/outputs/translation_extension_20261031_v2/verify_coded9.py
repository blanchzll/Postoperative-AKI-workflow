from pathlib import Path
import os,sys,json,hashlib,math,time,resource,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');O=R/'outputs';T=O/'translation_extension_20261031_v2'
import numpy as np,pandas as pd,statsmodels.api as sm
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from scipy.special import expit
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(T/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
checks=0;errmax=0
def close(a,b,tol=1e-10):
 global checks,errmax
 if a is None and (b is None or not np.isfinite(float(b))):return
 diff=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));assert np.isfinite(diff) and diff<=tol,(a,b,diff,tol)
 checks+=1;errmax=max(errmax,diff)
def quantile(v,q):
 s=sorted(float(x) for x in v if np.isfinite(x))
 if not s:return None
 f=(len(s)-1)*q;a=int(f);b=min(a+1,len(s)-1);return s[a]+(f-a)*(s[b]-s[a])
def design():
 m=pd.read_csv(O/'public_eligibility_refinement_20261022_v1/PRIVATE_COMMON6_METADATA.csv',dtype={'record_key':str})
 x=pd.read_csv(T/'PRIVATE_CODED_NINE_FEATURES.csv',dtype={'record_key':str,'group_hash':str})
 m=m.merge(x[['record_key','group_hash']],on='record_key',validate='one_to_one').sort_values(['group_hash','record_key'])
 assert len(m)==6043 and m.group_hash.is_unique
 pred={}
 for line,family in [('geography','LR'),('temporal','Voting')]:
  for label,p in [('six_RF',O/'common6_public_evaluation_20261024_v1'/line/'PRIVATE_PREDICTIONS.csv'),('coded9_'+family,T/line/'PRIVATE_PREDICTIONS.csv')]:
   z=pd.read_csv(p,dtype={'record_key':str,'group_hash':str},float_precision='round_trip').set_index('record_key').loc[m.record_key]
   assert z.y_scr.tolist()==m.y_scr.tolist() and z.group_hash.tolist()==m.group_hash.tolist()
   for mode in ['raw','calibrated']:pred[line+'__'+label+'__'+mode]=z['prob_'+mode].to_numpy()
 return m,pred
def cache(y,p,g,r):
 order=np.lexsort((np.asarray(r,str),np.asarray(g,str),-p));pp=p[order];starts=np.flatnonzero(np.r_[True,pp[1:]!=pp[:-1]])
 return (order,starts,p,(y-p)**2)
def score(y,c,w):
 order,starts,p,err=c;ws=w[order];ys=y[order];n=int(w.sum());e=float(w@y);ne=n-e
 gp=np.add.reduceat(ws*ys,starts);gn=np.add.reduceat(ws*(1-ys),starts);cp=np.cumsum(gp);cn=np.cumsum(gn)
 auc=float(np.sum(gp*(ne-cn+.5*gn))/(e*ne)) if e and ne else None
 prec=np.divide(cp,cp+cn,out=np.zeros_like(cp),where=cp+cn>0);ap=float(np.sum(prec*gp)/e) if e else None
 out={'AUROC':auc,'AP':ap,'Brier':float(w@err/n),'observed_expected':float(e/(w@p))}
 before=np.r_[0,np.cumsum(ws)[:-1]]
 for frac in [.1,.2]:
  k=math.ceil(n*frac);sel=np.minimum(ws,np.maximum(0,k-before));tp=float(sel@ys);head='capacity'+str(int(frac*100))+'_'
  out.update({head+'captured_per1000':1000*tp/n,head+'missed_per1000':1000*(e-tp)/n,head+'false_alerts_per1000':1000*(k-tp)/n,head+'PPV':tp/k,head+'captured_n':tp,head+'reviewed_n':k})
 return out
def diagnosis():
 pre=O/'public_eligibility_refinement_20261022_v1'
 f=pd.read_csv(T/'PRIVATE_CODED_NINE_FEATURES.csv',dtype={'record_key':str}).set_index('record_key')
 m=pd.read_csv(pre/'PRIVATE_COMMON6_METADATA.csv',dtype={'record_key':str,'subject_id':str,'op_id':str})
 profile=pd.read_csv(pre/'PRIVATE_FIRST_OPERATION_PROFILE.csv',dtype={'subject_id':str,'op_id':str})
 z=m.merge(profile[['subject_id','op_id','opstart_time']],on=['subject_id','op_id'],validate='one_to_one').set_index('subject_id')
 codes=read(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json')['coded_history_rule']['category_lists'];positive={k:set() for k in codes};redacted=['I27','I42','I47','I49'];redcounts={k:0 for k in redacted};qcount=0;nr=0
 for chunk in pd.read_csv('/home/lei/physionet.org/files/inspire/1.4.2/diagnosis.csv.gz',dtype={'subject_id':str,'icd10_cm':str},chunksize=150000):
  nr+=len(chunk);qcount+=int(chunk.icd10_cm.str.startswith('Q').sum())
  for code in redacted:redcounts[code]+=int(chunk.icd10_cm.eq(code).sum())
  a=chunk[chunk.subject_id.isin(z.index)].join(z[['opstart_time']],on='subject_id')
  a=a[a.chart_time<a.opstart_time]
  for k,cs in codes.items():positive[k].update(a.loc[a.icd10_cm.isin(cs),'subject_id'])
 counts={}
 for k,ids in positive.items():
  got=m.subject_id.isin(ids).to_numpy(int);saved=f.loc[m.record_key,k].to_numpy(int);close(got,saved,0);counts[k]=int(got.sum())
 assert redcounts=={k:0 for k in redacted} and qcount==0
 china=pd.read_csv(O/'v4_development_inputs_20261018/preop_preop_features.csv');ch={'n':len(china),'Hypertension1':int(china.Hypertension.eq(1).sum()),'HeartDisease1':int(china.HeartDisease.eq(1).sum()),'Hypertension1_HeartDisease0':int((china.Hypertension.eq(1)&china.HeartDisease.eq(0)).sum())};assert ch=={'n':3710,'Hypertension1':1148,'HeartDisease1':313,'Hypertension1_HeartDisease0':974}
 return {'raw_diagnosis_rows':nr,'proxy_one_counts_verified':counts,'source_removed_code_zero_rows_verified':redcounts,'source_Q_rows':qcount,'China_existing_field_counts':ch,'China_recode_required':False}
def main():
 start=time.time();assert (T/'COMPUTATION_ACCEPTANCE.json').exists()
 m,pred=design();y=m.y_scr.to_numpy(float);g=m.group_hash.astype(str).to_numpy();r=m.record_key.astype(str).to_numpy();n=len(y)
 old=read(T/'PUBLIC_EVALUATION.json');points=old['point'];caches={k:cache(y,p,g,r) for k,p in pred.items()};fields=['AUROC','AP','Brier','observed_expected','capacity10_captured_per1000','capacity20_captured_per1000','capacity10_PPV','capacity20_PPV','capacity10_missed_per1000','capacity20_missed_per1000','capacity10_false_alerts_per1000','capacity20_false_alerts_per1000']
 for k,p in pred.items():
  close(roc_auc_score(y,p),points[k]['AUROC']);close(average_precision_score(y,p),points[k]['AP']);close(brier_score_loss(y,p),points[k]['Brier'])
  z=score(y,caches[k],np.ones(n,int))
  for f in fields:close(z[f],points[k][f])
  logit=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
  fit=sm.GLM(y,sm.add_constant(logit),family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
  close(fit.params[1],points[k]['calibration_slope'],1e-4);close(fit.params[0],points[k]['calibration_joint_intercept'],1e-4)
  offset=sm.GLM(y,np.ones((n,1)),offset=logit,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10);close(offset.params[0],points[k]['calibration_intercept'],1e-6)
 names=list(pred);cols=[(nm,f) for nm in names for f in fields];samples=np.full((2000,len(cols)),np.nan);rng=np.random.default_rng(20261017);proof=hashlib.sha256()
 # Each public record is one distinct patient; independent integer-draw implementation.
 for rep in range(2000):
  drawn=rng.integers(0,n,n);w=np.bincount(drawn,minlength=n).astype(np.int64);proof.update(w.astype('<i8').tobytes())
  for j,nm in enumerate(names):
   vals=score(y,caches[nm],w);samples[rep,j*len(fields):(j+1)*len(fields)]=[vals[f] for f in fields]
  if rep<5:
   for nm,p in pred.items():
    vals=score(y,caches[nm],w)
    close(roc_auc_score(y,p,sample_weight=w),vals['AUROC']);close(average_precision_score(y,p,sample_weight=w),vals['AP'])
 assert proof.hexdigest()==old['receipt']['resample_weight_sha256']
 for row in old['intervals']:
  if row['metric'] not in fields:continue
  v=samples[:,cols.index((row['model'],row['metric']))];close(quantile(v,.025),row['lower95']);close(quantile(v,.975),row['upper95'])
 for row in old['paired_differences']:
  if row['metric'] not in fields:continue
  a,b=row['model'].split(' MINUS ');v=samples[:,cols.index((a,row['metric']))]-samples[:,cols.index((b,row['metric']))]
  close(quantile(v,.025),row['lower95']);close(quantile(v,.975),row['upper95'])
 curve=pd.read_csv(T/'FULL_CAPTURE_REVIEW_CURVE.csv')
 for nm,p in pred.items():
  order=caches[nm][0];cum=np.r_[0,np.cumsum(y[order])];z=curve[curve.model==nm].sort_values('reviewed_n');close(z.reviewed_n,np.arange(n+1),0);close(z.captured_n,cum,0)
 threshold=pd.read_csv(T/'NINE_SOURCE_THRESHOLD_BURDEN.csv');cfg=[]
 pins=read(T/'CODE_HISTORY_AND_MODEL_PRELOCK.json')
 for it in pins['source_models']:
  model=read(it['model_lock']);nm=it['line']+'__coded9_'+model['family']+'__calibrated'
  for rule,cut in model['training_thresholds'].items():
   if rule.startswith('training_'):cfg.append((it['line'],nm,rule,cut))
 fs=['reviewed_n','captured_n','missed_n','false_alerts_n','sensitivity','PPV','specificity','captured_per1000','missed_per1000','reviewed_per1000','false_alerts_per1000'];ts=np.full((2000,len(cfg),len(fs)),np.nan);rng=np.random.default_rng(20261017)
 for rep in range(2000):
  w=np.bincount(rng.integers(0,n,n),minlength=n);e=float(w@y)
  for j,(line,nm,rule,cut) in enumerate(cfg):
   a=pred[nm]>=cut;rv=float(w@a);tp=float(w@(a*y));fp=rv-tp
   d={'reviewed_n':rv,'captured_n':tp,'missed_n':e-tp,'false_alerts_n':fp,'sensitivity':tp/e if e else np.nan,'PPV':tp/rv if rv else np.nan,'specificity':1-fp/(n-e),'captured_per1000':1000*tp/n,'missed_per1000':1000*(e-tp)/n,'reviewed_per1000':1000*rv/n,'false_alerts_per1000':1000*fp/n}
   ts[rep,j]=[d[f] for f in fs]
 for _,row in threshold.iterrows():
  j=next(j for j,c in enumerate(cfg) if c[0]==row.source_line and c[2]==row.rule);v=ts[:,j,fs.index(row.metric)];close(quantile(v,.025),row.lower95);close(quantile(v,.975),row.upper95)
 dg=diagnosis();np.savez_compressed(T/'INDEPENDENT_NUMERIC_SAMPLES_PRIVATE.npz',values=samples,columns=np.asarray(cols),threshold_values=ts)
 for it in pins['source_models']:
  assert sha(it['model_lock'])==it['model_lock_sha256']
  for p,h in it['components'].items():assert sha(p)==h
 for p,d in read(O/'post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json')['files'].items():assert sha(p)==d['sha256']
 save('INDEPENDENT_NUMERICAL_VERIFICATION.json',{'status':'PASS_INDEPENDENT_CODED_HISTORY_AND_ALL2000_KEY_METRIC_RECALCULATION','same_Codex_different_implementation_not_external_expert':True,'numeric_assertions':checks,'max_abs_difference_including_separate_calibration_optimizer':errmax,'views':8,'independent_patient_bootstrap_replicates':2000,'sampling_sha256':proof.hexdigest(),'verified_fields':fields,'source_threshold_interval_fields':fs,'all_curve_rows_rebuilt':len(curve),'diagnosis_verification':dg,'original_models_and_protected92_unchanged':True,'new_model_fits':0,'elapsed_seconds':time.time()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
 print(json.dumps({'status':'PASS','assertions':checks,'max_difference':errmax,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save('INDEPENDENT_CHECK_FAILURE.json',{'traceback':traceback.format_exc()});raise
