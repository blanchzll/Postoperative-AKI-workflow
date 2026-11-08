from secondary_runtime import *
import copy
import pandas as pd
stage="common6_development_20261024_v1";o=R/"outputs"/stage;o.mkdir(parents=True,exist_ok=True)
source=R/"outputs/v4_development_inputs_20261018"
if not ((o/"A_ACCEPTANCE.json").exists() and (o/"B_ACCEPTANCE.json").exists()):
 source=R/"outputs/v4_development_inputs_20261018"
 spec=read(source/"FEATURE_SPECS.json")
 six=["Age","PreopCr","PreopBUN","PreopAlb","PreopHb","Sex"]
 sp={"num":six[:-1],"cat":["Sex"],"cat_levels":{"Sex":spec["preop"]["cat_levels"]["Sex"]}}
 m=pd.read_csv(source/"preop_metadata.csv",dtype={"group_hash":str},float_precision="round_trip")
 x=pd.read_csv(source/"preop_preop_features.csv",float_precision="round_trip")
 checks={}
 contract={"features":six,"source_lock_sha256":sha(source/"SOURCE_LOCK.json"),"new_training_all_contexts":True,"no_foreign_cache":True,"expected_development":{"geography":[2551,133],"temporal":[2024,139]}}
 save(R/"outputs/next_stage_preparation_20261022_v1/COMMON6_SOURCE_INPUT_CONTRACT.json",contract)
 for line in ["geography","temporal"]:
  inp=o/"inputs"/line;inp.mkdir(parents=True,exist_ok=True)
  mm=m.loc[m[line+"_partition_v2"].eq("development")].copy()
  assert [len(mm),int(mm.y.sum())]==contract["expected_development"][line]
  xx=x.set_index("excel_row",drop=False).loc[mm.excel_row,["excel_row",*sp["num"],"Sex"]]
  mm.to_csv(inp/"preop_metadata.csv",index=False);xx.to_csv(inp/"preop_preop_features.csv",index=False)
  save(inp/"FEATURE_SPECS.json",{"preop":sp,"endop":copy.deepcopy(sp)})
  name=line+"_preop_resampling.json";save(inp/name,read(source/name))
  save(inp/"INPUT_ACCEPTANCE.json",{"status":"PASS_six_input_corrected_source_only","n":len(mm),"events":int(mm.y.sum())})
  lockfiles={f.name:sha(f) for f in inp.iterdir() if f.is_file() and f.name!="SOURCE_LOCK.json"}
  save(inp/"SOURCE_LOCK.json",{"input_sha256":lockfiles,"source_parent_sha256":sha(source/"SOURCE_LOCK.json"),"features":six})
  env={"AKI_SCOPE_INPUT":str(inp),"AKI_SCOPE_OUT":str(o/"runs"/line)}
  (o/"runs"/line).mkdir(parents=True,exist_ok=True)
  run(stage+"/build.py",(line,),env_extra=env)
  run(stage+"/qualify_engine.py",(line,),env_extra=env)
  checks[line]={name:sha(o/"runs"/line/name) for name in ["INPUT_BOUNDARY_QUALIFICATION.json","ENGINE_QUALIFICATION.json"]}
 lock(stage,recursive=True);deps(stage)
 save(o/"QUALIFICATION.json",{"status":"PASS_isolated_six_input_engine_and_boundaries","tasks":864,"searches_after_coarse":108,"checks":checks,"no_old_fitted_state_import":True,"no_public_scores_read":True})
 if os.environ.get("AKI_PREPARE_ONLY")=="1":print("PASS common6 corrected-source input and engine preparation only");sys.exit(0)
 run(stage+"/supervisor.py")
from common6_checkpoint_reuse import verify
verify()
save(o/"REVIEW_ACCEPTANCE.json",{"status":"PASS_864_coarse_imputation_108_searches_reverified","new_source_sha256":sha(source/"SOURCE_LOCK.json"),"no_foreign_model_or_cache":True,"checkpoint_reuse":{"path":str(o/"CHECKPOINT_REUSE_SEAL.json"),"sha256":sha(o/"CHECKPOINT_REUSE_SEAL.json")}})
if os.environ.get("AKI_PREPARE_ONLY")=="1":print("PASS common6 completed-current-source checkpoint reuse only");sys.exit(0)
stage="common6_strict_continuation_20261024_v1";oo=R/"outputs"/stage;oo.mkdir(parents=True,exist_ok=True)
# Provenance is rebound to matching current-source implementations.
prov=read(P/"outputs"/stage/"ENGINE_PROVENANCE.json")
for name,z in prov.items():
 src=Path(z["source"].replace(str(P)+"/",str(R)+"/",1));assert sha(R/"work"/stage/"engine"/name)==sha(src),name
 z.update(source=str(src),sha256=sha(src))
save(oo/"ENGINE_PROVENANCE.json",prov)
run(stage+"/prepare_scopes.py");lock(stage,"EXECUTION_CODE_LOCK.json",recursive=True);deps(stage,"EXECUTION_DEPENDENCIES.json")
run(stage+"/qualify_execution.py");run(stage+"/supervisor.py")
stage="common6_internal_assembly_20261024_v1";oo=R/"outputs"/stage;oo.mkdir(parents=True,exist_ok=True)
man=read(R/"outputs/common6_strict_continuation_20261024_v1/SCOPE_MANIFEST.json")
for root in man["roots"]:root["base_output"]=str(oo/"root_runs"/root["line"])
save(oo/"SCOPE_MANIFEST.json",man)
for line in ["geography","temporal"]:
 d=oo/"root_runs"/line;d.mkdir(parents=True,exist_ok=True)
 for name in ["search_v2","selection"]:
  target=o/"runs"/line/name
  if not (d/name).exists():(d/name).symlink_to(target,target_is_directory=True)
lock(stage,recursive=True);deps(stage)
run(stage+"/qualify.py");run(stage+"/supervisor.py")
stage="common6_model_lock_20261024_v1";lock(stage);deps(stage)
run(stage+"/qualify.py");run(stage+"/lock_models.py")
stage="common6_public_evaluation_20261024_v1"
public="public_eligibility_refinement_20261022_v1";origin=P/"outputs"/public;target=R/"outputs"/public;target.mkdir(parents=True,exist_ok=True)
c=read(origin/"COMMON6_INPUT_CONTRACT.json")
for f,h in c["files"].items():
 assert sha(origin/f)==h
 if not (target/f).exists():(target/f).symlink_to(origin/f)
save(target/"COMMON6_INPUT_CONTRACT.json",c)
for name in ["ACCEPTANCE.json","PUBLIC_ENDPOINT_RULE_LOCK.json"]:
 if not (target/name).exists():(target/name).symlink_to(origin/name)

lock(stage);deps(stage);run(stage+"/qualify.py");run(stage+"/evaluate_public.py")
