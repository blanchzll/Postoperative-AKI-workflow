"""Read-only progress verification, no computation relaunch or final score access."""
from core import *
import datetime
def process(pid,expected):
 try:
  p=Path("/proc")/str(pid);stat=(p/"stat").read_text().split(") ",1)[1].split()
  cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode()
  return {"pid":pid,"alive":stat[0]!="Z" and expected in cmd,"state":stat[0],"start_ticks":stat[19],"cmd":cmd}
 except FileNotFoundError:return {"pid":pid,"alive":False}
def main():
 readings={}
 for n in ["AGENTS.md","outputs/research_plan_revision_20261018_094007/CURRENT_RESEARCH_PLAN_ZH.md","outputs/ensemble_continuation_20261018_v1/NEXT_EXECUTION_HANDOFF_ZH.md"]:
  (ROOT/n).read_text();readings[n]=sha(ROOT/n)
 d=ROOT/"outputs/ensemble_continuation_20261018_v1"
 assert json.loads((d/"STAGE_D_ACCEPTANCE.json").read_text())["status"]=="complete_nested_raw_ensembles_and_base_calibration_only"
 ds=json.loads((d/"status_d.json").read_text());assert ds["task_owned_workers_remaining"]==0
 assert json.loads((d/"CODE_LOCK.json").read_text())=={p.name:sha(p) for p in (ROOT/"work/ensemble_continuation_20261018_v1").glob("*.py")}
 lock=json.loads((OUT/"CODE_LOCK.json").read_text());assert lock=={p.name:sha(p) for p in CODE.glob("*.py")}
 s=json.loads((OUT/"status_e.json").read_text())
 controller=process(s.get("pid",0),str(CODE/"coordinator_e.py"))
 live=[process(x["pid"],str(CODE)) for x in s.get("running",[])]
 errors=[]
 p=OUT/"COORDINATOR_E_FAILURE.json"
 if p.exists():errors.append(json.loads(p.read_text()))
 acceptance=None
 if (OUT/"STAGE_E_ACCEPTANCE.json").exists():acceptance=json.loads((OUT/"STAGE_E_ACCEPTANCE.json").read_text())
 healthy=controller["alive"] and not errors and not s.get("failed") and s["state"] in ["running","waiting_memory","waiting_disk"]
 rec={"checked_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"state":s["state"],"counts":s.get("counts"),"controller":controller,"live_workers":[v for v in live if v["alive"]],
 "code_lock_verified":True,"D_completed_not_restarted":True,"readings":readings,"resources":resource_snapshot(),"data_free_bytes":os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize,
 "errors":errors,"acceptance":acceptance,"needs_action":acceptance is not None or not healthy or s["state"]=="waiting_disk","final_model_locked":False}
 p=OUT/"monitoring"/("heartbeat_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
 atomic(p,rec)
 print(canonical({"receipt":str(p),"state":rec["state"],"counts":rec["counts"],"live_workers":len(rec["live_workers"]),"controller_alive":controller["alive"],"code_lock_verified":True,"needs_action":rec["needs_action"],"errors":errors,"acceptance":acceptance}),flush=True)
if __name__=="__main__":main()
