"""Assemble versioned extension tables by selecting/copying verified results; no metric re-estimation."""
from pathlib import Path
import os,json,hashlib,csv,shutil
import pandas as pd,numpy as np
A=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot/outputs/novelty_strengthening_20261101_v1");O=A.parent;T=A/"tables";T.mkdir(mode=0o700,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert read(A/"resource_comparison/INDEPENDENT_NUMERICAL_VERIFICATION.json")["status"].startswith("PASS_INDEPENDENT")
assert read(A/"mimic_calendar_v2/INDEPENDENT_NUMERICAL_VERIFICATION.json")["status"].startswith("PASS_INDEPENDENT")
g=pd.read_csv(A/"resource_comparison/RESOURCE_BUDGET_GRID.csv");b=pd.read_csv(A/"resource_comparison/BREAK_EVEN_BOUNDARIES.csv")
g[(g.budget_percent==20)&(g["mode"]=="calibrated")&(g.extra_cost_per_person_review_equivalents==0)&(g.role=="primary")].to_csv(T/"PRIMARY_EQUAL_CAPACITY20.csv",index=False)
b[(b.budget_percent==20)&(b["mode"]=="calibrated")].to_csv(T/"ALL_SIGNED_BOUNDARIES20.csv",index=False)
region=g[["comparison_id","context","role","mode","budget_percent","extra_cost_per_person_review_equivalents","n","events","feasible_at_point","delta_captured_per1000","delta_captured_per1000_lower95","delta_captured_per1000_upper95","delta_captured_per1000_valid_replicates"]].copy()
region["pointwise_region"]=np.select([region.feasible_at_point!=True,region.delta_captured_per1000_lower95>0,region.delta_captured_per1000_upper95<0],["infeasible_at_observed_population","extra_workflow_positive_pointwise_interval","baseline_workflow_positive_pointwise_interval"],default="interval_includes_zero")
region["simultaneous_inference_or_clinical_threshold"]=False;region.to_csv(T/"ALL_POINTWISE_CHOICE_REGIONS.csv",index=False)
sumrows=[]
for key,d in region.groupby(["comparison_id","mode","budget_percent"],sort=False):
 for state in ["infeasible_at_observed_population","extra_workflow_positive_pointwise_interval","baseline_workflow_positive_pointwise_interval","interval_includes_zero"]:
  dd=d[d.pointwise_region==state]
  sumrows.append(dict(zip(["comparison_id","mode","budget_percent"],key),region=state,grid_cells=len(dd),cost_grid_values="|".join(format(x,".2f") for x in dd.extra_cost_per_person_review_equivalents),interpretation="Complete prelocked grid description; not a selected recommended cutoff or simultaneous band."))
pd.DataFrame(sumrows).to_csv(T/"COMPLETE_GRID_REGION_SUMMARY.csv",index=False)
ledger=[]
def keep(p,label,kind):
 assert p.exists(),p
 dest=T/label;shutil.copy2(p,dest);assert sha(dest)==sha(p)
 ledger.append({"analysis_axis":kind,"source":str(p),"source_sha256":sha(p),"delivered_table":str(dest.relative_to(A)),"delivered_sha256":sha(dest),"recomputed_in_this_task":kind=="new_MIMIC_calendar_v2"})
for context in ["geography_preop","geography_paired","temporal_preop","temporal_paired","INSPIRE_common6"]:
 for suffix in ["INCREMENT_POINTS.csv","INCREMENT_AND_INVERSE_INTERVALS.csv","INVERSE_CAPTURE_REVIEW_CURVE.csv"]:
  p=O/"translation_extension_20261031_v1"/(context+"_"+suffix)
  keep(p,"REUSED_"+p.name,"reused_accepted_capacity_and_inverse")
keep(O/"translation_extension_20261031_v1/SOURCE_THRESHOLD_TRANSFER_REUSED.csv","REUSED_SOURCE_THRESHOLD_TRANSFER.csv","reused_accepted_source_thresholds")
for name in ["POINT_ESTIMATES.csv","INTERVALS.csv","SOURCE_THRESHOLD_BURDEN.csv","INVERSE_COVERAGE.csv"]:
 keep(A/"mimic_calendar_v2"/name,"MIMIC_V2_"+name,"new_MIMIC_calendar_v2")
for name in ["POINT_ESTIMATES.csv","INTERVALS.csv","PAIRED_DIFFERENCES_VS_SIX_RF.csv"]:
 keep(O/"translation_extension_20261031_v2"/name,"REUSED_INSPIRE_CODED9_"+name,"reused_accepted_exposed_INSPIRE")
pd.DataFrame(ledger).to_csv(A/"EVIDENCE_SOURCE_TABLE_LEDGER.csv",index=False)
# Figure region summaries never turn an NA/infeasible scenario into a zero.
assert len(region)==5208 and len(b)==168
r=read(A/"resource_comparison/NUMERICAL_COMPUTATION_ACCEPTANCE.json")
m=read(A/"mimic_calendar_v2/INPUT_QUALIFICATION.json")
(A/"EVIDENCE_ASSEMBLY_ACCEPTANCE.json").write_text(json.dumps({"status":"PASS_TABLE_ASSEMBLY_NO_OLD_RESULT_RECOMPUTATION","new_resource_grid_rows":len(g),"new_boundaries":len(b),"choice_region_rows":len(region),"reused_and_new_table_sources":len(ledger),"calendar_n":m["n"],"calendar_events":m["events"],"source_ledger_sha256":sha(A/"EVIDENCE_SOURCE_TABLE_LEDGER.csv"),"table_hashes":{p.name:sha(p) for p in T.glob("*.csv")},"code_sha256":sha(__file__)},indent=2)+"\n")
print("Assembled",len(ledger),"source tables and complete choice regions")
