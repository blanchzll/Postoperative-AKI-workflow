from secondary_runtime import *
stage="fixed_flow_sensitivity_20261024_v1";o=R/"outputs"/stage;o.mkdir(parents=True,exist_ok=True)
save(o/"ROUTES.json",read(R/"outputs/conditional_stability_20261024_v1/ROUTES.json"))
contract=read(P/"outputs"/stage/"EXECUTION_CONTRACT.json")
contract.update(high_baseline_record_ids=[],high_baseline_resolution="Source-corrected former three records; exclusion target is empty and cannot establish robustness",reference="New corrected-source main locked choices; all scenario fitted states new",data_storage=str(o.resolve()))
save(o/"EXECUTION_CONTRACT.json",contract);lock(stage);deps(stage)
run(stage+"/qualify.py");run(stage+"/supervisor.py")
stage="fixed_flow_sensitivity_evaluation_20261024_v1";lock(stage);deps(stage)
run(stage+"/qualify.py");run(stage+"/evaluate.py")
