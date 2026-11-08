from secondary_runtime import *
import subprocess,os
tab=R/"outputs/tabpfn_benchmark_20261025_v1"
py="/home/lei/codex-workspaces/shared-python-envs/envs/tabular-foundation-py312-torch260-cu118-tabpfn221-v1/bin/python"
assert read(tab/"STATUS.json")["state"]=="predictions_complete_pending_statistics"
for rec in tab.glob("routes/*/COMPLETE.json"):
 x=read(rec);assert x["signature"]["source_sha256"]==sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")
 assert sha(rec.parent/"predictions.csv")==x["predictions_sha256"]
save(tab/"CODE_LOCK.json",{p.name:sha(p) for p in tab.glob("*.py")})
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
# Four disjoint statistical outputs, each with the unchanged per-cohort bootstrap seed.
live=[]
for name in ["geography_paired","temporal_paired","geography_preop","temporal_preop"]:
 log=(tab/(name+".statistics.log")).open("a")
 proc=subprocess.Popen([py,"-B",str(tab/"analyze.py"),name],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT);live.append((proc,log))
codes=[]
for proc,log in live:codes.append(proc.wait());log.close()
assert codes==[0]*4,codes
subprocess.run([py,"-B",str(tab/"verify.py")],env=env,cwd=R,check=True)
save(tab/"COMPARISON_ACCEPTANCE.json",{"status":"PASS_corrected_source_TabPFN_supplement_after_main_evaluation","verification_sha256":sha(tab/"VERIFICATION.json"),"all_ten_routes_retained":True,"main_model_winner_not_reselected":True,"whole_reanalysis_complete":False})
