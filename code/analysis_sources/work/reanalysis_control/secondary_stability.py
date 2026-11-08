from secondary_runtime import *
stage="conditional_stability_20261024_v1";o=R/"outputs"/stage;o.mkdir(parents=True,exist_ok=True)
models=read(R/"outputs/model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json")["models"]
routes={"_".join(ref["identity"][k] for k in ["line","cohort","stage","branch"]):dict(ref) for ref in models if ref["primary_route_winner"]}
assert len(routes)==10
save(o/"ROUTES.json",routes)
save(o/"EXECUTION_CONTRACT.json",read(P/"outputs"/stage/"EXECUTION_CONTRACT.json"))
lock(stage);deps(stage)
save(o/"FIT_IMPLEMENTATION_CODE_LOCK.json",{"stability.py":sha(R/"work/conditional_stability_20261024_v1/stability.py"),"model_runtime.py":sha(R/"work/model_lock_20261022_v1/runtime.py")})
run(stage+"/qualify.py");run(stage+"/supervisor.py")
