from pathlib import Path
import json,hashlib,sys,math
import numpy as np
import pandas as pd
from scipy.special import expit
from core import *
OUT=ROOT/"outputs/published_benchmark_comparison_20261024_v1"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads((OUT/"ANALYSIS_CONTRACT.json").read_text())
for p,h in contract["sources_sha256"].items():assert sha(ROOT/p)==h,p
params=json.loads((ROOT/"outputs/published_benchmark_SIM_AKI_20261024_v1/PUBLISHED_COEFFICIENTS.json").read_text())
# Independent sex-specific piecewise equations rather than vector implementation.
for year in [2009,2021]:
 for male in [0,1]:
  for age in [18.,40.,85.]:
   for scr in [.4,.7,.9,2.,4.]:
    if year==2009:
     k=.9 if male else .7;power=-.411 if male else -.329
     expected=141*(scr/k)**(power if scr<=k else -1.209)*.993**age*(1 if male else 1.018)
    else:
     k=.9 if male else .7;power=-.302 if male else -.241
     expected=142*(scr/k)**(power if scr<=k else -1.2)*.9938**age*(1 if male else 1.012)
    np.testing.assert_allclose(egfr([age],[male],[scr*88.4],year),[expected],rtol=1e-14)
f=pd.DataFrame({"Age":[40.,70.],"Sex":[0.,1.],"PreopCr":[61.88,106.08],"ASA":[2.,3.],
 "Diabetes":[0.,1.],"PreopAlb":[40.,30.],"PreopHb":[120.,129.],
 "IntraopTransfusion":[0.,400.],"OperationTime":[60.,180.],
 "Hypertension":[0.,1.],"SurgicalApproach":["1","1"]})
x,g=sim_matrix(f)
assert x[0].tolist()==[40.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,0.,1.]
p,pp,_=sim_risk(f,params)
etas=x@np.array(params["beta"]).T+params["intercept"]
manual=np.array([[1,math.exp(a),math.exp(b)] for a,b in etas]);manual/=manual.sum(axis=1,keepdims=True)
np.testing.assert_allclose(pp,manual,rtol=1e-14)
np.testing.assert_allclose(p,1-manual[:,0],atol=1e-15)
f.loc[0,"Age"]=60
s=chen_score(f)
expected=.32*70-.39*30-.27*129+2.06+1.42+2.63+1.32+1.37
assert abs(s[1]-expected)<1e-12
try:chen_score(f.assign(SurgicalApproach=["2","1"]))
except ValueError:pass
else:raise AssertionError("Unknown gas duration accepted")
y=np.array([0,1,1,0,1,0]);score=np.array([-3.,1.,1.,2.,2.,0.])
groups=["b","a","c","d","e","f"];records=list("123456")
rank=rank_summary(y,score,groups,records)
prob=es.summary(y,expit(score),groups,records,with_calibration=False)
for k,v in rank.items():
 if k in prob and finite_number(v):np.testing.assert_allclose(v,prob[k],atol=1e-14)
order=[5,2,3,0,4,1]
q=rank_summary(y[order],score[order],np.array(groups)[order],np.array(records)[order])
assert rank==q
# Patient duplicates stay in one stratum and expand together, with unique copy keys.
d=pd.DataFrame({"excel_row":[4,3,2,1],"group_hash":["x","x","y","z"],"center":[1,1,1,2],"y":[0,1,0,1]})
c=canonical_context(d,{"a":np.array([.2,.8,.1,.7])},"prob")
weights=es.group_weights(c["strata"],len(c["y"]),np.random.default_rng(11))
ix,gg,rr=es.expanded(weights,c["g"],c["r"])
assert len(set(zip(gg,rr)))==len(ix)
ixx=np.flatnonzero(c["g"]=="x");assert len(set(weights[ixx]))==1
r=context_scores(c,ix,gg,rr)
assert "Brier" in r["a"] and "Brier" not in rank
result={"status":"PASS_source_hashes_equations_rank_capacity_and_group_resampling","checks":[
 "all locked sources sha256","independent scalar piecewise eGFR2009/2021 across sex and cutoffs",
 "source multinomial zero-reference stable-softmax independent hand exponentials",
 "RBC binary and minutes-to-hours mapping","Chen Table4 hand score and gas-unknown rejection",
 "AUROC AP and deterministic tied-capacity against existing probability evaluator",
 "row reorder","patient-group duplicate expansion"],"core_sha256":sha(Path(__file__).with_name("core.py")),
 "qualify_sha256":sha(__file__),"contract_sha256":sha(OUT/"ANALYSIS_CONTRACT.json")}
(OUT/"QUALIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
