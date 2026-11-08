from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');OUT=Path(__file__).resolve().parent
for rel in ['model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json', 'formal_evaluation_20261023_v1/STAGE_FINAL_EVALUATION_ACCEPTANCE.json', 'fixed_flow_sensitivity_evaluation_20261024_v1/STAGE_EVALUATION_ACCEPTANCE.json']:
 if not (ROOT/"outputs"/rel).exists():raise RuntimeError("Supplementary TabPFN statistics deferred until main pipeline accepted: "+rel)
sys.path.insert(0,str(ROOT/'work/final_evaluation_engine_20261022_v1'))
from fixed_prediction_metrics import FixedRisk,_design,group_weights
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
from evaluation_stats import calibration

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def score(f,w=None):
 s=f.score(w)
 for frac in [.1,.2]:
  c=f.capacity(frac,w,s)
  for k in ['reviewed_n','captured_n','missed_n','PPV','sensitivity','captured_per1000']:
   s[f'capacity{int(frac*100)}_'+k]=c[k]
 for t in [.01,.02,.03,.05,.075,.1,.15,.2]:s[f'net_benefit_{t}']=f.net_benefit(t,w)['net_benefit']
 return s

def run(line,cohort):
 name=line+'_'+cohort;dest=OUT/'statistics'/name;dest.mkdir(parents=True,exist_ok=True)
 tasks=[('preop','compact'),('preop','full')] if cohort=='preop' else [('preop','full'),('endop','full')]
 if line=='temporal' and cohort=='paired':tasks += [('preop_noALT','full'),('endop_noALT','full')]
 pred={};meta=None;sources={}
 for stage,branch in tasks:
  d=OUT/'routes'/'_'.join([line,cohort,stage,branch]);receipt=json.loads((d/'COMPLETE.json').read_text());f=d/'predictions.csv'
  assert sha(f)==receipt['predictions_sha256']
  q=pd.read_csv(f,dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False)
  if meta is None:meta=q[['excel_row','group_hash','center','y']].copy()
  q=q.loc[meta.index];assert q[['excel_row','group_hash','center','y']].equals(meta)
  pred['tabpfn_'+stage+'_'+branch]=q.prob_tabpfn.to_numpy();sources[str(f)]=sha(f)
 locks=json.loads((ROOT/'outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json').read_text())['models']
 for ref in locks:
  i=ref['identity']
  if not ref['primary_route_winner'] or (i['line'],i['cohort'])!=(line,cohort) or (i['stage'],i['branch']) not in tasks:continue
  assert sha(ref['path'])==ref['sha256']
  mn='_'.join(i[k] for k in ['stage','branch','role','family']);d=ROOT/'outputs/formal_evaluation_20261023_v1'/name/'private_predictions'/mn;f=d/'PREDICTIONS.csv'
  assert sha(f)==json.loads((d/'COMPLETE.json').read_text())['sha256']
  q=pd.read_csv(f,dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False).loc[meta.index]
  assert q[['excel_row','group_hash','center','y']].equals(meta)
  pred['main_raw_'+i['stage']+'_'+i['branch']]=q.prob_raw.to_numpy()
  pred['main_cal_'+i['stage']+'_'+i['branch']]=q.prob_calibrated.to_numpy();sources[str(f)]=sha(f)
 yy,gg,rr,cc,strata,models=_design(meta.y,meta.group_hash,meta.excel_row.astype(str),meta.center.astype(str),pred)
 points={k:score(m) for k,m in models.items()}
 for k,m in models.items():
  assert abs(points[k]['AUROC']-roc_auc_score(yy,m.p))<1e-12
  assert abs(points[k]['AP']-average_precision_score(yy,m.p))<1e-12
  assert abs(points[k]['Brier']-brier_score_loss(yy,m.p))<1e-12
  points[k].update(calibration(yy,m.p))
 pairs={}
 base='preop_compact' if cohort=='preop' else 'preop_full';up='preop_full' if cohort=='preop' else 'endop_full'
 for v in ['tabpfn_','main_raw_','main_cal_']:pairs[v+'information_increment']={v+up:1,v+base:-1}
 for stage,branch in tasks:
  suffix=stage+'_'+branch
  for v in ['raw','cal']:
   if 'main_'+v+'_'+suffix in models:pairs['tabpfn_minus_main_'+v+'_'+suffix]={'tabpfn_'+suffix:1,'main_'+v+'_'+suffix:-1}
 for v in ['raw','cal']:
  pairs['change_in_increment_vs_main_'+v]={'tabpfn_'+up:1,'tabpfn_'+base:-1,'main_'+v+'_'+up:-1,'main_'+v+'_'+base:1}
 if line=='temporal' and cohort=='paired':
  pairs['tabpfn_noALT_information_increment']={'tabpfn_endop_noALT_full':1,'tabpfn_preop_noALT_full':-1}
  pairs['tabpfn_noALT_change_in_increment']={'tabpfn_endop_noALT_full':1,'tabpfn_preop_noALT_full':-1,'tabpfn_endop_full':-1,'tabpfn_preop_full':1}
 fields=['AUROC','AP','Brier','observed_expected','capacity10_captured_per1000','capacity20_captured_per1000','capacity20_PPV','capacity20_sensitivity']
 def linear(sc,terms,f):
  vals=[sc[k][f] for k in terms]
  return sum(sc[k][f]*v for k,v in terms.items()) if all(x is not None and np.isfinite(x) for x in vals) else None
 samples={k:{f:[] for f in fields} for k in models};deltas={k:{f:[] for f in fields} for k in pairs};rng=np.random.default_rng(20261017);proof=hashlib.sha256();zero=0
 for b in range(2000):
  w=group_weights(strata,len(yy),rng);proof.update(w.astype('<i8').tobytes());zero+=int(w@yy==0)
  sc={k:score(m,w) for k,m in models.items()}
  for k in models:
   for f in fields:
    v=sc[k][f]
    if v is not None and np.isfinite(v):samples[k][f].append(float(v))
  for k,terms in pairs.items():
   for f in fields:
    v=linear(sc,terms,f)
    if v is not None:deltas[k][f].append(float(v))
  if (b+1)%500==0:print(name,'BOOTSTRAP',b+1,flush=True)
 def cis(d,delta=False):
  rows=[]
  for k,fields0 in d.items():
   for f,v in fields0.items():
    lo,hi=np.quantile(v,[.025,.975]).tolist() if v else (None,None)
    rows.append({'comparison' if delta else 'model':k,'metric':f,'estimate':linear(points,pairs[k],f) if delta else points[k][f],'lower95':lo,'upper95':hi,'valid_replicates':len(v)})
  return rows
 result={'cohort':name,'point':points,'intervals':cis(samples),'paired_differences':cis(deltas,True),'receipt':{'source_hashes':sources,'replicates':2000,'seed':20261017,'resample_weight_sha256':proof.hexdigest(),'zero_event_replicates':zero,'group_unit_within_hospital':True,'fixed_models_conditional_CI':True,'point_AUROC_AP_Brier_independently_checked_sklearn':True,'raw_and_calibrated_main_separate':True,'not_new_blind_validation':True,'statistics_code_sha256':sha(__file__)}}
 save(dest/'RESULTS.json',result)
 pd.DataFrame([{'model':k,**v} for k,v in points.items()]).to_csv(dest/'POINT_ESTIMATES.csv',index=False)
 pd.DataFrame(result['paired_differences']).to_csv(dest/'PAIRED_DIFFERENCES.csv',index=False)
 save(dest/'COMPLETE.json',{'files':{f:sha(dest/f) for f in ['RESULTS.json','POINT_ESTIMATES.csv','PAIRED_DIFFERENCES.csv']}})
 print('STATISTICS_COMPLETE',name,flush=True)
if __name__=='__main__':
 for s in sys.argv[1:] or ['geography_paired','temporal_paired','geography_preop','temporal_preop']:
  line,cohort=s.split('_');run(line,cohort)
