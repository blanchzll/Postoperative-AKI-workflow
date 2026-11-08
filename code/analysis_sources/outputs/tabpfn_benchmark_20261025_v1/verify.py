from pathlib import Path
import json,hashlib,sys,math
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/final_evaluation_engine_20261022_v1'))
from fixed_prediction_metrics import FixedRisk,_design,group_weights

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];maximum=0.;ci_n=0;bootstrap_weight_matches=0
for name in ['geography_paired','temporal_paired','geography_preop','temporal_preop']:
 new=json.loads((OUT/'statistics'/name/'RESULTS.json').read_text())
 old=json.loads((ROOT/'outputs/formal_evaluation_20261023_v1'/name/'EVALUATION.json').read_text())
 assert new['receipt']['resample_weight_sha256']==old['receipt']['resample_weight_sha256'];bootstrap_weight_matches+=1
 # Compare the exact same frozen primary information contrast against this corrected-source main analysis.
 paired=name.endswith('paired')
 first='endop_full_champion_' if paired else 'preop_full_champion_'
 second='preop_full_champion_' if paired else 'preop_compact_champion_'
 oldmap={x['metric']:x for x in old['paired_differences'] if x['model'].startswith(first) and (' MINUS '+second) in x['model']}
 newmap={x['metric']:x for x in new['paired_differences'] if x['comparison']=='main_cal_information_increment'}
 for metric in set(oldmap)&set(newmap):
  for f in ['estimate','lower95','upper95']:
   a=oldmap[metric][f];b=newmap[metric][f]
   if a is None or b is None:assert a is None and b is None
   else:
    dif=abs(a-b);maximum=max(maximum,dif);assert dif<1e-10,(name,metric,f,a,b)
   ci_n+=1
 for model,point in new['point'].items():
  if not model.startswith('tabpfn_'):continue
  stagebranch=model[len('tabpfn_'):];d=OUT/'routes'/(name+'_'+stagebranch);rec=json.loads((d/'COMPLETE.json').read_text())
  assert sha(d/'predictions.csv')==rec['predictions_sha256']
  a=pd.read_csv(d/'predictions.csv',dtype={'group_hash':str},float_precision='round_trip')
  y=a.y.to_numpy();prob=a.prob_tabpfn.to_numpy()
  assert abs(roc_auc_score(y,prob)-point['AUROC'])<1e-12
  assert abs(average_precision_score(y,prob)-point['AP'])<1e-12
  assert abs(brier_score_loss(y,prob)-point['Brier'])<1e-12
  # Independent ordering/count implementation, preserving specified patient/record tie-breaks.
  order=np.lexsort((a.excel_row.astype(str).to_numpy(),a.group_hash.to_numpy(),-prob))
  for fraction in [.1,.2]:
   chosen=order[:math.ceil(fraction*len(y))]
   assert int(y[chosen].sum())==point[f'capacity{int(fraction*100)}_captured_n']
  # Verify hospital-stratified group bootstrap metrics against explicitly expanded records.
  yy,gg,rr,cc,strata,models=_design(y,a.group_hash,a.excel_row.astype(str),a.center.astype(str),{'p':prob})
  rng=np.random.default_rng(20261017);f=models['p']
  for b in range(25):
   w=group_weights(strata,len(yy),rng);idx=np.repeat(np.arange(len(yy)),w);z=f.score(w)
   assert abs(roc_auc_score(yy[idx],f.p[idx])-z['AUROC'])<1e-12
   assert abs(average_precision_score(yy[idx],f.p[idx])-z['AP'])<1e-12
  checks.append({'route':name+'_'+stagebranch,'point_metrics_sklearn':True,'capacity_independent_lexsort':True,'bootstrap_expansion_checks':25})
assert len(checks)==10
for f,h in json.loads((OUT/'CODE_LOCK.json').read_text()).items():assert sha(OUT/f)==h
r={'status':'PASS_NUMERICAL_AND_SOURCE_VERIFICATION','routes':checks,'frozen_reference_CI_values_checked':ci_n,'frozen_reference_max_abs_difference':maximum,'bootstrap_weight_hash_matches':bootstrap_weight_matches,'not_independent_human_review':True,'same_agent_alternative_implementations':True,'verification_code_sha256':sha(__file__)}
(OUT/'VERIFICATION.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='routes'}))
