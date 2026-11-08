"""Synthetic IO/cache qualification; never a clinical comparison or training job."""
from pathlib import Path
import json,hashlib,time,os
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
import comparison as c,bootstrap as b
def main():
 b.check_code_lock()
 blocked=False
 try:c.require_closed_formal()
 except AssertionError as e:
  assert "Formal controller" in str(e);blocked=True
 assert blocked,"Qualification expects active formal benchmark and must not advance into real metrics"
 T=c.L/"statistics_engineering_20261029_v1";T.mkdir(mode=0o700,exist_ok=True)
 fixture=T/"synthetic_chunk_io";assert not fixture.exists();fixture.mkdir(mode=0o700)
 line="synthetic_operator_only";d=fixture/line;d.mkdir(mode=0o700)
 rng=np.random.default_rng(291028);groups=np.array([f"fixture_patient_{i//2:03d}" for i in range(40)]);records=np.array([f"fixture_operation_{i:03d}" for i in range(40)]);centers=np.array(["SYNTHETIC_A" if i<20 else "SYNTHETIC_B" for i in range(40)]);y=np.array([i%9 in [0,1] for i in range(40)],dtype=int)
 pred={f"synthetic_model_{i:02d}":np.round(rng.uniform(.01,.99,40),1) for i in range(30)}
 yy,gg,rr,cc,strata,models=c._design(y,groups,records,centers,pred);weights=[];proof=hashlib.sha256()
 for _ in range(2000):
  w=c.group_weights(strata,len(yy),rng);weights.append(w);proof.update(w.astype("<i8").tobytes())
 weights=np.stack(weights);names=sorted(models);cuts={n:c.thresholds(yy,models[n].p) for n in names[:2]}
 inp=d/"PRIVATE_FIXED_EVALUATION_INPUTS.npz"
 with inp.open("wb") as f:np.savez_compressed(f,y=yy,group_hash=gg,records=rr,centers=cc,weights=weights,**{n:m.p for n,m in models.items()})
 c.save(d/"SOURCE_THRESHOLDS.json",cuts);c.save(d/"COMPARISONS.json",{"fixture_difference":{names[0]:1,names[1]:-1}})
 c.save(d/"PREPARATION_ACCEPTANCE.json",{"status":"SYNTHETIC_OPERATOR_FIXTURE_ONLY_NO_CLINICAL_DATA_OR_RESULTS","group_weights_sha256":proof.hexdigest(),"input_files_sha256":{n:c.sha(d/n) for n in ["PRIVATE_FIXED_EVALUATION_INPUTS.npz","SOURCE_THRESHOLDS.json","COMPARISONS.json"]}})
 b.O=fixture;start=time.monotonic();rec=b.chunk_worker((line,0,25))
 saved=d/"bootstrap_chunks/0000_0025.npz";stamp=saved.stat().st_mtime_ns;digest=c.sha(saved)
 rec2=b.chunk_worker((line,0,25));assert rec==rec2 and saved.stat().st_mtime_ns==stamp and c.sha(saved)==digest
 with np.load(saved,allow_pickle=False) as z:
  assert z["replicate"].tolist()==list(range(25));v=z["values"];fields=z["fields"].tolist();gotnames=z["models"].tolist();assert gotnames==names
  checks=0;worst=0.
  for rep in range(25):
   w=weights[rep]
   for j,n in enumerate(names):
    p=models[n].p
    independent={"AUROC":roc_auc_score(yy,p,sample_weight=w),"AP":average_precision_score(yy,p,sample_weight=w),"Brier":brier_score_loss(yy,p,sample_weight=w)}
    for metric,value in independent.items():
     delta=abs(v[rep,j,fields.index(metric)]-value);assert delta<1e-12;worst=max(worst,delta);checks+=1
    for frac in [.1,.2]:
     k=fields.index(f"capacity{int(frac*100)}_reviewed_n");assert v[rep,j,k]==np.ceil(w.sum()*frac);checks+=1
   # Undefined source-threshold metrics for comparators stay missing, never invented.
   for j,n in enumerate(names[2:],start=2):assert np.isnan(v[rep,j,fields.index("training_capacity20_reviewed_n")]);checks+=1
 result={"status":"PASS_SYNTHETIC_CHUNK_SERIALIZATION_CACHE_REUSE_METRIC_PARITY_AND_ACTIVE_FORMAL_GUARD","time":c.now(),"independent_checks":checks,"maximum_absolute_discrepancy":worst,"cached_chunk_reused_without_recompute":True,"real_comparison_blocked_while_formal_active":True,"synthetic_models":30,"synthetic_records":40,"patient_group_bootstrap_fixture":True,"actual_LLM_performance_accessed":False,"real_LLM_calibration_fitted":False,"formal_model_or_contract_modified":False,"code_lock_sha256":c.sha(Path(__file__).parent/"STATISTICS_CODE_LOCK.json"),"comparison_code_sha256":c.sha(Path(c.__file__)),"bootstrap_code_sha256":c.sha(Path(b.__file__)),"qualifier_code_sha256":c.sha(__file__),"seconds":time.monotonic()-start}
 c.save(T/"CHUNK_IO_AND_GUARD_QUALIFICATION.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()
