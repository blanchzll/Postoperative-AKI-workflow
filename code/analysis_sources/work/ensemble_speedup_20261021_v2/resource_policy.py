"""Measured resource admission; no model, sampling or score changes."""
from pathlib import Path
import os,time,json,math
M=1024**2;G=1024**3
BASELINE=int(json.loads((Path(__file__).resolve().parents[2]/"outputs/ensemble_speedup_20261021_v2/BASELINE_CPU_SHARE.json").read_text())["ceil_baseline"])
def fair_cpu_budget(effective,other):
 return max(1,min(effective,max(BASELINE,int(math.floor(effective-other)))))
def key(j):
 t=j.get("task",{})
 return (j["kind"],t.get("family",""),t.get("branch",""),t.get("cohort",""))
def historical_envelopes(history):
 bank={}
 for r in history["observed_completed_task_resource_receipts"]:
  bank[(r["kind"],r["family"],r["branch"],r["cohort"])]=r["max_MiB"]*M
 return bank
def reservation(j,peaks):
 k=j["kind"];t=j.get("task",{});branch=t.get("branch","");f=t.get("family","")
 floor={"coarse":1024*M,"fit":512*M,"ensemble":512*M,"search":768*M if branch=="selected" or f=="XGB" else 512*M}[k]
 peak=peaks.get(key(j),0)
 # Max observed original-model RSS plus controller/IPC and future size margin.
 if k=="search":margin=128*M
 else:margin=64*M
 return int(max(floor,peak*1.6+margin))
def slots(available,growth,reserve):
 return max(0,int((available-max(4*G,available*.25)-growth)//reserve))
def read_cpu():
 v={}
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:
   a=(p/"stat").read_text().split(") ",1)[1].split()
   v[int(p.name)]=(int(a[11])+int(a[12]),int(a[1]),a[19])
  except (OSError,ValueError):pass
 return time.monotonic(),v
class CPUShare:
 def __init__(self):self.prev=read_cpu();self.other_cores=0.;self.ready=False
 def sample(self,effective,owned_roots):
  now,rows=read_cpu();old,prev=self.prev;dt=now-old
  owned=set(owned_roots)|{os.getpid()}
  change=True
  while change:
   size=len(owned)
   owned.update(pid for pid,v in rows.items() if v[1] in owned)
   change=len(owned)!=size
  if dt>=1:
   ticks=sum(max(0,v[0]-prev[p][0]) for p,v in rows.items() if p not in owned and p in prev and v[2]==prev[p][2])
   actual=ticks/os.sysconf("SC_CLK_TCK")/dt
   self.other_cores=actual if not self.ready else self.other_cores*.65+actual*.35
   self.ready=True;self.prev=(now,rows)
  # Use actual headroom; no fixed small worker cap and no unrelated task control.
  return fair_cpu_budget(effective,self.other_cores)
