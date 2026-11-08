"""Resource admission only: preserve live main-E growth reservations and OS headroom."""
from pathlib import Path
import json,os,time
G=1024**3
LAST={}
def host_available():
 return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
def available(root,out):
 global LAST
 host=host_available();p=root/"outputs/ensemble_calibration_20261020_v1/status_e.json"
 try:
  s=json.loads(p.read_text());age=time.time()-s["updated"]
  if s.get("state")=="complete_strict_calibration_internal_evaluation":
   join=root/"outputs/resource_acceleration_20261026_v1/pod1_fixed_advance/MAIN_JOIN_REQUEST.json"
   main_here="aki-postslot-full-reanalysis-20261025.service" in Path("/proc/self/cgroup").read_text()
   main_waiting=False
   if join.exists():
    j=json.loads(join.read_text());proc=Path("/proc")/str(j["pid"])
    try:
     main_waiting=(proc/"stat").read_text().split()[21]==j["start_ticks"] and "secondary_pod1.py" in (proc/"cmdline").read_bytes().decode(errors="replace")
    except FileNotFoundError:pass
   if main_here or main_waiting:
    LAST={"main_POD1_step_joined":True,"host_MemAvailable_bytes":host,"spare_bytes_before_auxiliary_reservations":host,"native_POD1_headroom_still_applies":True}
    return host
   raise ValueError("E completed; preserve later main stages until main POD1 step joins")
  if s.get("state")!="running" or age>65 or s.get("failed"):raise ValueError("Main-E status not fresh/healthy")
  growth=0;seen=0
  for v in s["running"]:
   pid=v["pid"];proc=Path("/proc")/str(pid)
   try:
    cmd=(proc/"cmdline").read_bytes().decode(errors="replace")
    assert str(root/"work/ensemble_calibration_20261020_v1") in cmd,"Unexpected main PID owner"
    rss=int((proc/"statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
   except FileNotFoundError:continue
   jid=v["id"];kind=jid.split("_",1)[0]
   if kind=="search":
    t=json.loads((p.parent/"requests"/(jid+".json")).read_text())
    nominal=({"RF":1.7,"XGB":1.4,"SVM":1.2}.get(t.get("family"),1.0)+(.5 if t.get("branch")=="selected" else 0))*G
    reserve=max(nominal,1.4*(rss+256*1024**2))
   else:reserve=max((.7 if kind=="ensemble" else .9)*G,1.4*rss)
   growth+=max(0,reserve-rss);seen+=1
  head=max(4*G,int(host*.10));spare=max(0,int(host-head-growth))
  LAST={"main_status_age_seconds":age,"main_live_workers_seen":seen,"main_future_growth_protected_bytes":int(growth),
   "shared_OS_headroom_bytes":head,"host_MemAvailable_bytes":host,"spare_bytes_before_auxiliary_reservations":spare,
   "reservation_note":"Conservative admission estimate using original per-kind defaults and live RSS; not a guaranteed peak prediction",
   "main_not_signalled_or_restarted":True,"auxiliary_additional_native_headroom_minimum_bytes":3*G,"main_own_25percent_admission_unchanged":True}
  return spare
 except (OSError,ValueError,KeyError,AssertionError) as error:
  LAST={"spare_bytes_before_auxiliary_reservations":0,"wait_reason":repr(error),"fail_closed":True}
  return 0
