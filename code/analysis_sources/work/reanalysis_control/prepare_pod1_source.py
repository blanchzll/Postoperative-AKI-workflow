from pathlib import Path
import os,sys,json,hashlib,copy,subprocess
R=Path(__file__).resolve().parents[2];P=R.parents[1];OUT=R/"outputs/pod1_update_20261017"
PY=R/"envs/prediction_gpu/bin/python"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def main():
 import pandas as pd
 OUT.mkdir(parents=True,exist_ok=True)
 source=R/"outputs/alt_raw_units_20261018/preop_cohort.csv"
 data=pd.read_csv(source,dtype={"group_hash":str},float_precision="round_trip")
 # Recompute eligibility and sampled endpoint from corrected observed concentrations.
 observed=data[["base","pod1","pod23"]].notna().all(axis=1)&data["base"].gt(0)
 eligible=observed&data.pod1.div(data.base).lt(1.5)&data.pod1.sub(data.base).lt(26.5)
 assert eligible.equals(data.strict_POD1_secondary_cohort)
 assert int(eligible.sum())==2947 and int(data.loc[eligible,"pod23"].div(data.loc[eligible,"base"]).ge(1.5).sum())==39
 proto=json.loads((P/"outputs/pod1_update_20261017/protocol.json").read_text())
 proto.update(source=str(source.relative_to(R)),source_sha256=sha(source),source_lock_sha256=sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),row983="Confirmed corrected observed values; eligible sampled secondary cohort, endpoint negative.",folds="Restrict newly rebuilt corrected-source main 5outer/3inner grouped manifests to current observed POD1 eligibility; preserve hospital and 2022 rules and seeds.",max_workers=len(os.sched_getaffinity(0)),threads_per_worker=1,gpu_worker_limit=len(os.sched_getaffinity(0)))
 specs=json.loads((R/"outputs/v4_development_inputs_20261018/FEATURE_SPECS.json").read_text())
 for branch,sp in proto["branches"].items():
  for cat in sp["cat"]:sp["cat_levels"][cat]=specs["endop"]["cat_levels"][cat]
 for line in ["geography","temporal"]:
  method=json.loads((R/"outputs/formal_development_20261018_v1/selection"/line/"preop/final/METHOD_SELECTION.json").read_text())["selected_method"]
  proto["imputation"][line]=method
 proto["imputation"]["rationale"]="Apply the same prespecified upstream development-line method rule using the newly rerun corrected-source main selection. Refit stable encounter-keyed imputation in every POD1 training split; no new endpoint-specific 6x9 search."
 for f in [OUT/"protocol.json",R/"work/pod1_update_20261017/protocol.json"]:save(f,proto)
 allcounts=[]
 meta=pd.read_csv(R/"outputs/v4_development_inputs_20261018/preop_metadata.csv",dtype={"group_hash":str},float_precision="round_trip").set_index("excel_row",drop=False)
 for line,legacy in [("geography","preop_outcomes_python_20261017"),("temporal","preop_temporal_python_20261017")]:
  template=json.loads((P/"outputs"/legacy/"fold_inputs/fold_manifest.json").read_text())
  parts=json.loads((R/"outputs/v4_development_inputs_20261018"/(line+"_preop_resampling.json")).read_text())
  folder=R/"outputs"/legacy/"fold_inputs";folder.mkdir(parents=True,exist_ok=True);pairs=[]
  for old in template["pairs"]:
   pair={k:old[k] for k in ["name","role","outer","inner"]};role=pair["role"]
   if role.startswith("heldout"):
    rows={"train":meta.index[meta[line+"_partition_v2"].eq("development")].tolist(),"valid":meta.index[meta[line+"_partition_v2"].eq("validation")].tolist()}
   else:
    if role=="development_outer":z=next(z for z in parts if z["scheme"]=="outer" and z["fold"]==pair["outer"])
    elif role=="development_inner":z=next(z for z in parts if z["scheme"]=="inner" and z["parent_outer"]==pair["outer"] and z["fold"]==pair["inner"])
    else:raise ValueError(role)
    rows={"train":z["train_excel_rows"],"valid":z["assessment_excel_rows"]}
   assert set(meta.loc[rows["train"],"group_hash"]).isdisjoint(meta.loc[rows["valid"],"group_hash"])
   pair["files"]={}
   for side,rr in rows.items():
    frame=meta.loc[rr];f=folder/(pair["name"]+"_"+side+"_meta.csv");f.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(f,index=False)
    pair["files"][side+"_meta"]={"path":str(f.relative_to(R)),"sha256":sha(f)}
    pair[side+"_n"]=len(frame)
   pairs.append(pair)
  save(folder/"fold_manifest.json",{"pairs":pairs,"source_lock_sha256":sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json"),"old_template_used_for_names_only":True,"source_resampling_sha256":sha(R/"outputs/v4_development_inputs_20261018"/(line+"_preop_resampling.json"))})
  allcounts.append({"line":line,"pair_count":len(pairs)})
 save(OUT/"FRESH_CORRECTED_SOURCE_PREPARATION.json",{"status":"PASS_corrected_POD1_source_and_rebuilt_main_topology","eligible":2947,"events":39,"source_sha256":sha(source),"folds":allcounts,"no_old_models_or_fold_members_imported":True,"candidate_settings_preserved":True})
 print("POD1 corrected source and new fold metadata prepared",allcounts)
if __name__=="__main__":main()
