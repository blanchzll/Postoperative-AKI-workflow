from pathlib import Path
import subprocess,os
R=Path(__file__).resolve().parents[2];O=R/"outputs/tabpfn_benchmark_20261025_v1"
import json
for rel in ['model_lock_20261022_v1/STAGE_MODEL_LOCK_ACCEPTANCE.json', 'formal_evaluation_20261023_v1/STAGE_FINAL_EVALUATION_ACCEPTANCE.json', 'fixed_flow_sensitivity_evaluation_20261024_v1/STAGE_EVALUATION_ACCEPTANCE.json']:
 if not (R/"outputs"/rel).exists():raise RuntimeError("TabPFN deferred by user until accepted main pipeline: "+rel)
if (O/"STATUS.json").exists() and json.loads((O/"STATUS.json").read_text()).get("state")=="predictions_complete_pending_statistics":
 print("Ten verified predictions already complete; perform deferred paired comparison, do not refit.");raise SystemExit(0)
PY="/home/lei/codex-workspaces/shared-python-envs/envs/tabular-foundation-py312-torch260-cu118-tabpfn221-v1/bin/python"
os.environ.update(PYTHONDONTWRITEBYTECODE="1",TABPFN_DISABLE_TELEMETRY="1",HF_HUB_DISABLE_TELEMETRY="1",HF_HUB_OFFLINE="1")
for s in ["smoke.py","supervisor.py"]:
 with (O/(s+".launch.log")).open("a") as f:
  subprocess.run([PY,str(O/s)],cwd=R,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
