"""Policy probability routing. Component risks must come from its exact locked models."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
sys.path.insert(0,str(ROOT/'work/selective_information_20261021_v1'))
import policy_core as core
import numpy as np,joblib
def risk(manifest,base,upgraded,groups,keys,calibrator=None):
 i=manifest['identity']
 mask=core.upgrade_mask(base,groups,keys,manifest['gate'],i['upgrade_fraction'],i['mode'],i['kind']=='random')
 raw=core.mix_predictions(base,upgraded,mask)
 if calibrator is None:
  p=Path(manifest['calibrator']['path'])
  assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['calibrator']['sha256']
  calibrator=joblib.load(p)
 p=np.clip(np.asarray(raw,dtype=np.float64),1e-6,1-1e-6)
 logit=np.log(p/(1-p)).reshape(-1,1)
 calibrated=calibrator.predict_proba(logit)[:,1]
 alerts={k:calibrated>=v for k,v in manifest['training_thresholds'].items() if k.startswith('training_') and isinstance(v,(int,float))}
 return {'prob_raw_mixture':raw,'prob_calibrated':calibrated,'upgraded':mask,'training_threshold_alerts':alerts}
