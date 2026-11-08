from pathlib import Path
import sys,json
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'engine'))
from core import *
from model_engine import one_fit
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
line=sys.argv[1];sp=spec_for('preop')
assert len(sp['num']+sp['cat'])==6 and set(COMPACT)-set(sp['num']+sp['cat'])=={'Diabetes','Hypertension','HeartDisease'}
try:spec_for('preop',['Diabetes']);raise RuntimeError('extra input accepted')
except AssertionError:pass
rng=np.random.default_rng(42);n=100
x=pd.DataFrame({c:rng.normal(size=n) for c in sp['num']});x['Sex']=pd.Series([str(i%2) for i in range(n)],dtype='string')
m=pd.DataFrame({'y':np.array([i%4==0 for i in range(n)],int)})
checks=[]
with threadpool_limits(limits=1):
 for params in [{'variant':'linear','C':.1},{'variant':'spline','C':.1,'n_knots':3}]:
  pipe,p,notes,fallback=one_fit(x.iloc[:80],m.iloc[:80],x.iloc[80:],sp,'LR',params,1)
  q=pipe.predict_proba(x.iloc[80:].iloc[::-1])[:,1][::-1]
  np.testing.assert_allclose(p,q,rtol=1e-12,atol=1e-12)
  assert len(pipe.feature_names_in_)==6 and set(pipe.feature_names_in_)==set(sp['num']+sp['cat'])
  checks.append({'variant':params['variant'],'six_columns_only':True,'reverse_equal':True,'finite':bool(np.isfinite(p).all())})
assert 'old=ROOT/' not in (CODE/'coarse.py').read_text()
from preprocess import imported_cache
assert imported_cache({'context':'final','branch':'full','cohort':'preop','stage':'preop'},None,None) is None
assert 'old=ROOT/' not in (CODE/'preprocess.py').read_text()
assert 'shutil.copyfile(old/' not in (CODE/'preprocess.py').read_text()
atomic(OUT/'ENGINE_QUALIFICATION.json',{'status':'PASS_synthetic_six_input_linear_and_nonlinear_LR','line':line,'checks':checks,'extra_input_rejected':True,'old27_result_shortcut_removed':True,'clinical_models_fitted':False})
