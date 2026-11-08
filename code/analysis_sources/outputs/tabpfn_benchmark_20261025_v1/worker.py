import os
for k,v in {'TABPFN_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'8','OMP_NUM_THREADS':'8'}.items():os.environ[k]=v
import sys,json,time,hashlib,resource,pathlib,socket,gc,warnings
import numpy as np,pandas as pd,torch
from tabpfn import TabPFNClassifier
ROOT=pathlib.Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');OUT=pathlib.Path(__file__).resolve().parent;SRC=ROOT/'outputs/v4_development_inputs_20261018'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False));q.replace(p)
def denied(*a,**kw):raise RuntimeError('Network disabled for model computation')
socket.socket.connect=denied
torch.set_num_threads(8);torch.set_num_interop_threads(1)
route=sys.argv[1];line,cohort,stage,branch=route.split('|')
name='_'.join([line,cohort,stage,branch]);dest=OUT/'routes'/name;dest.mkdir(parents=True,exist_ok=True,mode=0o700)
protocol=json.loads((OUT/'PROTOCOL.json').read_text());weight=json.loads((OUT/'WEIGHT_RECEIPT.json').read_text())
assert sha(weight['download_path'])==protocol['checkpoint_sha256']
assert json.loads((OUT/'SMOKE.json').read_text())['status']=='PASS_SYNTHETIC_ONLY'
lock=json.loads((SRC/'SOURCE_LOCK.json').read_text())
for f,h in lock['input_sha256'].items():assert sha(SRC/f)==h
spec=json.loads((SRC/'FEATURE_SPECS.json').read_text())[stage]
compact=['Age','Sex','PreopCr','PreopBUN','PreopAlb','PreopHb','Diabetes','Hypertension','HeartDisease']
cols=compact if branch=='compact' else spec['num']+spec['cat']
cat=[c for c in cols if c in spec['cat']]
meta=pd.read_csv(SRC/(cohort+'_metadata.csv'),dtype={'group_hash':str},float_precision='round_trip').set_index('excel_row',drop=False)
tr=meta[meta[line+'_partition_v2']=='development'];te=meta[meta[line+'_partition_v2']=='validation']
assert len(set(tr.group_hash)&set(te.group_hash))==0
x=pd.read_csv(SRC/(cohort+'_'+stage+'_features.csv'),dtype={c:'string' for c in cat},float_precision='round_trip').set_index('excel_row')[cols]
for c in cat:
 mapping={k:i for i,k in enumerate(spec['cat_levels'][c])}
 assert set(x[c].dropna().unique())<=set(mapping),'Unknown category '+c
 x[c]=x[c].map(mapping).astype(float)
Xtr=x.loc[tr.index].to_numpy(dtype=np.float64);Xte=x.loc[te.index].to_numpy(dtype=np.float64)
assert not np.isinf(Xtr).any() and not np.isinf(Xte).any()
assert len(cols)==(9 if branch=='compact' else {'preop':27,'endop':38,'preop_noALT':26,'endop_noALT':37}[stage])
sig={'protocol_sha256':sha(OUT/'PROTOCOL.json'),'weight_sha256':weight['sha256'],'source_sha256':sha(SRC/'SOURCE_LOCK.json'),'code_sha256':sha(__file__),'route':route,'input_columns':cols,'categorical_indices':[cols.index(c) for c in cat],'training_n':len(tr),'evaluation_n':len(te),'training_groups':tr.group_hash.nunique(),'evaluation_groups':te.group_hash.nunique(),'training_evaluation_overlap':0,'training_rows_sha256':hashlib.sha256(tr.index.to_numpy(dtype='<i8').tobytes()).hexdigest(),'native_preprocessing':True}
if (dest/'COMPLETE.json').exists():
 old=json.loads((dest/'COMPLETE.json').read_text());assert old['signature']==sig and sha(dest/'predictions.csv')==old['predictions_sha256'];print('RESUME_SKIP',name,flush=True);sys.exit(0)
save(dest/'INPUT_LOCK.json',sig)
m=TabPFNClassifier(n_estimators=8,categorical_features_indices=sig['categorical_indices'],softmax_temperature=.9,balance_probabilities=False,average_before_softmax=False,model_path=weight['download_path'],device='cuda',inference_precision=torch.float32,fit_mode='fit_preprocessors',memory_saving_mode='auto',random_state=42,n_jobs=1)
start=time.time();save(dest/'STATUS.json',{'state':'fitting_training_context','pid':os.getpid()});print('START',name,len(tr),len(te),len(cols),flush=True)
with warnings.catch_warnings(record=True) as ww:
 warnings.simplefilter('always');m.fit(Xtr,tr.y.to_numpy());fitsec=time.time()-start
 save(dest/'STATUS.json',{'state':'predicting','pid':os.getpid(),'fit_seconds':fitsec});print('FIT_DONE',name,fitsec,flush=True)
 prob=m.predict_proba(Xte)[:,1]
assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
pr=te[['excel_row','group_hash','center','y']].copy();pr['prob_tabpfn']=prob
pr.to_csv(dest/'predictions.csv',index=False)
rec={'signature':sig,'predictions_sha256':sha(dest/'predictions.csv'),'fit_seconds':fitsec,'total_seconds':time.time()-start,'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'RSS_peak_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'warnings':sorted(set(str(w.message)[:300] for w in ww)),'native_probabilities_not_recalibrated':True,'evaluation_labels_not_used_for_fit':True,'no_model_selection':True,'network_disabled':True}
save(dest/'COMPLETE.json',rec);save(dest/'STATUS.json',{'state':'complete','pid':os.getpid()})
print('COMPLETE',name,'seconds',rec['total_seconds'],'RSS_GiB',rec['RSS_peak_GiB'],'GPU_GiB',rec['gpu_peak_GiB'],flush=True)
del m,Xtr,Xte,x,pr;gc.collect();torch.cuda.empty_cache()
