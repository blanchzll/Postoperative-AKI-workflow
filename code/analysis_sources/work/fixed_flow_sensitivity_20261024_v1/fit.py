"""Fixed-choice matched refits for ALT omission and ALT omission and the now-resolved prior high-baseline exclusion target."""
from pathlib import Path
import sys,os,json,copy,gc
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/fixed_flow_sensitivity_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
st=base.module('fixed_sensitivity_reused_engine',ROOT/'work/conditional_stability_20261024_v1/stability.py')
rt=st.rt
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
SCENARIOS=['matched_control','omit_ALT','exclude_high_baseline_Cr']
HIGH=set(pd.read_csv(ROOT/'outputs/cohort_v2_rrt_days_20261017/UNVERIFIED_HIGH_BASELINE_SCR.csv').excel_row.astype(int))
def check():
 for f,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/f)==h
 for f,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(f)==h
def fill_training(raw,meta,method,sp):
 assert meta.excel_row.is_unique
 imp=rt.StableMixedImputer(sp,method,seed=42,iterations=5,trees=200,neighbors=5,threads=1)
 tx=imp.fit_transform(raw,rt.keys(meta))
 assert not tx.isna().any().any() and list(tx)==sp['num']+sp['cat']
 for col in tx:
  mask=raw[col].notna()
  if col in sp['num']:np.testing.assert_array_equal(raw.loc[mask,col],tx.loc[mask,col])
  else:assert raw.loc[mask,col].tolist()==tx.loc[mask,col].tolist()
 return imp,tx
st.fill_training=fill_training
def adjust(defs,scenario):
 z=copy.deepcopy(defs)
 if scenario=='omit_ALT':
  for d in z.values():
   d['columns']=[x for x in d['columns'] if x!='PreopALT']
   d['spec']['num']=[x for x in d['spec']['num'] if x!='PreopALT']
   assert 'PreopALT' not in d['columns'] and 'PreopALT' not in d['spec']['num']
 return z
def heldout(lock):
 i=lock['identity'];sp=rt.source_spec(i['stage'],lock['input_columns'])
 m=pd.read_csv(base.SOURCE/(i['cohort']+'_metadata.csv'),dtype={'group_hash':str},float_precision='round_trip')
 m=m.loc[m[i['line']+'_partition_v2'].eq('validation')].copy()
 m=m.sort_values(['group_hash','excel_row'],kind='stable').reset_index(drop=True)
 x=pd.read_csv(base.SOURCE/(i['cohort']+'_'+i['stage']+'_features.csv'),dtype={c:'string' for c in sp['cat']},float_precision='round_trip').set_index('excel_row')
 return x.loc[m.excel_row,lock['input_columns']].reset_index(drop=True),m
def blueprint(route,scenario):
 ref=base.read(OUT/'ROUTES.json')[route];assert base.sha(ref['path'])==ref['sha256']
 lock=base.read(ref['path']);x,m=st.source_data(lock);defs=adjust(st.definitions(lock),scenario)
 # Original group assignments retained, then filter confirmed-unverifiable records.
 fold=np.full(len(m),-1,dtype=int)
 for k,(_,va) in enumerate(st.grouped_folds(m)):fold[va]=k
 keep=~m.excel_row.isin(HIGH).to_numpy() if scenario=='exclude_high_baseline_Cr' else np.ones(len(m),dtype=bool)
 x=x.loc[keep].reset_index(drop=True);m=m.loc[keep].reset_index(drop=True);fold=fold[keep]
 assert pd.DataFrame({'g':m.group_hash,'f':fold}).groupby('g').f.nunique().eq(1).all()
 for k in range(3):assert m.loc[fold==k].y.nunique()==2 and m.loc[fold!=k].y.nunique()==2
 vx,vm=heldout(lock);assert set(m.group_hash).isdisjoint(vm.group_hash)
 columns=set(v for d in defs.values() for v in d['columns'])
 if scenario=='omit_ALT':assert 'PreopALT' not in columns
 sig={'route_lock':ref['sha256'],'definitions':defs,'train_rows':m.excel_row.tolist(),
  'fold_assignment':fold.tolist(),'assessment_rows':vm.excel_row.tolist(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json'),
  'fixed_choice_calibration':'three-fold grouped honest OOF, no family/variable reselection','seed':42}
 key=rt.digest(sig)[:24]
 return lock,defs,x,m,fold,vx,vm,sig,key
def run(route,scenario):
 check();base.gate();assert scenario in SCENARIOS
 lock,defs,x,m,fold,vx,vm,sig,key=blueprint(route,scenario)
 cache=OUT/'fit_cache'/key;cache.mkdir(parents=True,exist_ok=True)
 import fcntl
 with (cache/'FIT.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX)
  if not (cache/'COMPLETE.json').exists():
   oof=np.full(len(m),np.nan);fold_receipts={}
   for k in range(3):
    tr=np.flatnonzero(fold!=k);va=np.flatnonzero(fold==k)
    p=st.fit_flow(lock,defs,x.iloc[tr].reset_index(drop=True),m.iloc[tr].reset_index(drop=True),
     x.iloc[va].reset_index(drop=True),m.iloc[va].reset_index(drop=True),cache/('fold_'+str(k)),{'fit':sig,'fold':k})
    oof[va]=p;fold_receipts[str(k)]=base.sha(cache/('fold_'+str(k))/'COMPLETE.json')
   assert np.isfinite(oof).all()
   cal=st.calibrator(pd.DataFrame({'y':m.y,'prob_raw':oof}));joblib.dump(cal,cache/'calibrator.joblib',compress=3)
   raw=st.fit_flow(lock,defs,x,m,vx,vm,cache/'full',{'fit':sig,'scope':'full_train_predict_original_holdout'},True)
   p=st.apply_calibration(cal,raw)
   np.testing.assert_allclose(p,st.apply_calibration(joblib.load(cache/'calibrator.joblib'),raw),rtol=1e-12,atol=1e-12)
   traincal=st.apply_calibration(cal,oof)
   cuts=rt.thresholds(m.y,traincal)
   z=vm.copy();z['prob_raw']=raw;z['prob_calibrated']=p;z.to_csv(cache/'PRIVATE_HELDOUT_PREDICTIONS.csv',index=False)
   z=m.copy();z['fold']=fold;z['prob_raw_oof']=oof;z['prob_calibrated_training']=traincal;z.to_csv(cache/'PRIVATE_TRAINING_OOF.csv',index=False)
   base.atomic(cache/'TRAINING_THRESHOLDS.json',cuts)
   base.atomic(cache/'COMPLETE.json',{'status':'PASS_fixed_choice_fit_and_training_only_calibration','signature':sig,
    'files':{f:base.sha(cache/f) for f in ['calibrator.joblib','PRIVATE_HELDOUT_PREDICTIONS.csv','PRIVATE_TRAINING_OOF.csv','TRAINING_THRESHOLDS.json']},
    'fold_receipts':fold_receipts,'full_receipt':base.sha(cache/'full/COMPLETE.json'),'training_n':len(m),'training_events':int(m.y.sum()),
    'heldout_predictions_not_yet_evaluated':True,'deployment_qualification_not_implied':True})
  rec=base.read(cache/'COMPLETE.json');assert rec['signature']==sig
  for f,h in rec['files'].items():assert base.sha(cache/f)==h
 d=OUT/'routes'/route/scenario;d.mkdir(parents=True,exist_ok=True)
 base.atomic(d/'COMPLETE.json',{'status':'PASS_fixed_flow_sensitivity_route_fitted','route':route,'scenario':scenario,'fit_cache':str(cache),
  'fit_receipt_sha256':base.sha(cache/'COMPLETE.json'),'exclude_evaluation_rows':sorted(HIGH) if scenario=='exclude_high_baseline_Cr' else [],
  'evaluation_pending':True,'whole_project_complete':False})
 gc.collect()
if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],sys.argv[2])
