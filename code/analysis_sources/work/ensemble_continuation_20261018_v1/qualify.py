"""Real patient, leakage, ensemble and persistence acceptance checks."""
from core import *
import subprocess,importlib.util,joblib
import numpy as np,pandas as pd
from ensemble_math import *
def main():
 man=json.loads((OUT/"SCOPE_MANIFEST.json").read_text())
 s=man["scopes"][0];env=os.environ.copy();env.update(AKI_SCOPE_OUT=s["output"],AKI_SCOPE_INPUT=s["input"],OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 # Qualified real compact input imputer/model roundtrip in the actual nested scope.
 program="""
from core import *
from preprocess import impute
from model_engine import task_for,one_fit
import numpy as np,joblib
s=json.loads(Path(sys.argv[1]).read_text())
p={'fold':0,'train_excel_rows':s['train_rows'],'assessment_excel_rows':s['assessment_rows']}
t={'line':s['line'],'cohort':s['cohort'],'stage':'preop','branch':'compact','context':'qualification','method':'bayes'}
ct=task_for(t,p);impute(ct,2)
tx,tm,vx,vm,sp=read_cache(ct)
assert set(tx)==set(COMPACT) and set(vx)==set(COMPACT)
model,pr,w,f=one_fit(tx,tm,vx,sp,'NB',{'var_smoothing':1e-9},2)
d=OUT/'qualification';d.mkdir(exist_ok=True)
joblib.dump(model,d/'nb.joblib')
np.testing.assert_allclose(pr,joblib.load(d/'nb.joblib').predict_proba(vx)[:,1],atol=1e-12,rtol=1e-12)
assert set(tm.group_hash).isdisjoint(vm.group_hash)
bad=dict(ct);bad['train_rows']=ct['train_rows'][:10];bad['assessment_rows']=ct['train_rows'][:1]
try:frame_and_meta(bad,sp)
except ValueError:pass
else:raise AssertionError('Leakage guard failed')
atomic(d/'ACCEPTANCE.json',{'status':'PASS','compact_input_count':len(tx.columns),'train_n':len(tm),'assessment_n':len(vm),'raw_probability_roundtrip':True,'overlap_rejected':True})
"""
 q=OUT/"qualification_runtime.py";q.write_text(program)
 # core is resolved from code path; no patient data cross environment.
 env["PYTHONPATH"]=str(CODE)
 subprocess.run([str(PYTHON),"-B",str(q),str(OUT/"scopes"/s["id"]/"SCOPE.json")],env=env,cwd=ROOT,check=True)
 # Engineering exercise on real completed outer predictions only; no clinical claims.
 old=ROOT/"outputs/formal_development_20261018_v1/outer_raw/geography/preop/preop/compact"
 frames={f:pd.read_csv(old/"outer_0"/f/"predictions.csv",dtype={"group_hash":str}) for f in FAMILIES}
 m=matrix(frames).assign(fold=0)
 frames2={f:pd.read_csv(old/"outer_1"/f/"predictions.csv",dtype={"group_hash":str}) for f in FAMILIES}
 v=matrix(frames2).assign(fold=1)
 p,model,members=infer(m,v,"Stacking",3,.1)
 rev,_,_=infer(m,v.iloc[::-1],"Stacking",3,.1)
 np.testing.assert_allclose(p,rev[::-1],atol=1e-12,rtol=1e-12)
 vp,_,_=infer(m,v,"Voting",3)
 np.testing.assert_allclose(vp,v[rank_families(m)[:3]].mean(axis=1),atol=1e-12)
 bad=frames.copy();bad["LR"]=pd.concat([bad["LR"],bad["LR"].iloc[:1]])
 try:matrix(bad)
 except ValueError:pass
 else:raise AssertionError("Duplicate row accepted")
 try:infer(m,m,"Stacking",3,.1)
 except ValueError:pass
 else:raise AssertionError("Meta group overlap accepted")
 # Mismatched metadata must not silently align by array order.
 bad={f:x.copy() for f,x in frames.items()};bad["RF"].loc[0,"y"]=1-int(bad["RF"].loc[0,"y"])
 try:matrix(bad)
 except ValueError:pass
 else:raise AssertionError("Conflicting outcome accepted")
 c=calibrator(m[["y","LR"]].rename(columns={"LR":"prob_raw"}));cp=apply_calibration(c,v.LR)
 assert np.isfinite(cp).all() and np.all((cp>=0)&(cp<=1))
 atomic(OUT/"FUNCTIONAL_QUALIFICATION.json",{"status":"PASS","scope_count":len(man["scopes"]),"roots":len(man["roots"]),"engineering_only_no_new_scientific_model_claim":True,"real_compact_only_9_imputer_model_roundtrip":True,"patient_overlap_rejected":True,"duplicate_rows_rejected":True,"outcome_conflicts_rejected":True,"stacking_order_stable":True,"voting_arithmetic_verified":True,"base_calibration_finite":True,"source_manifest_sha256":sha(OUT/"SCOPE_MANIFEST.json")})
 print("QUALIFICATION_PASS")
if __name__=="__main__":main()
