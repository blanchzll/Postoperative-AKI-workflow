"""Current calculator numeric replay and inference-context qualification; no source fitting."""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib,datetime,fcntl,os,time
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;O=R/"outputs/calculator_release_20261025_v1";A=O/"app"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 t=Path(str(p)+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def expected_signature():
 c=read(O/"CURRENT_QUALIFICATION_CONTRACT.json")
 for f,h in c["source_files"].items():assert sha(f)==h
 for f,h in c["app_code_and_catalog"].items():assert sha(f)==h
 for f,h in c["current_work_code"].items():assert sha(W/f)==h
 return sha(O/"CURRENT_QUALIFICATION_CONTRACT.json")
def worker(route):
 sig=expected_signature()
 import native_route_qualification as native
 result=native.execute(route)
 if result["status"]!="PASS_full_route_inference":return 21
 path=O/"route_checks"/(route+".json");result.update(current_qualification_signature=sig,native_numeric_checks_unchanged=True)
 save(path,result);return 0
def resources():
 available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
 reserve=max(4*1024**3,int(.25*available));own=Path("/proc/self/cgroup").read_text().strip().split(":")[-1];cg=Path("/sys/fs/cgroup")/own.lstrip("/")
 maximum=(cg/"memory.max").read_text().strip();assert maximum!="max"
 own_remaining=max(0,int(maximum)-int((cg/"memory.current").read_text()));effective=available;parent=cg.parent;others=[];extra=0
 while parent!=Path("/sys/fs/cgroup"):
  mx=(parent/"memory.max").read_text().strip()
  if mx!="max":effective=min(effective,max(0,int(mx)-int((parent/"memory.current").read_text())))
  parent=parent.parent
 units=subprocess.run(["systemctl","--user","list-units","--state=active","--plain","--no-legend","aki-postslot-*"],capture_output=True,text=True,check=True).stdout.splitlines()
 for line in units:
  unit=line.split()[0]
  if unit in own:continue
  props=subprocess.run(["systemctl","--user","show",unit,"-p","ActiveState","-p","MemoryMax","-p","ControlGroup"],capture_output=True,text=True,check=True).stdout
  p=dict(v.split("=",1) for v in props.splitlines() if "=" in v)
  if p.get("ActiveState")!="active":continue
  assert p["MemoryMax"].isdigit()
  stat=Path("/sys/fs/cgroup")/p["ControlGroup"].lstrip("/")/"memory.stat"
  try:s=dict(v.split() for v in stat.read_text().splitlines())
  except FileNotFoundError:
   if "ActiveState=active" not in subprocess.run(["systemctl","--user","show",unit,"-p","ActiveState"],capture_output=True,text=True).stdout:continue
   raise
  growth=max(0,int(p["MemoryMax"])-sum(int(s.get(k,0)) for k in ["anon","kernel","shmem"]));extra+=growth;others.append({"unit":unit,"growth_reserved_bytes":growth})
 return max(0,min(own_remaining,effective-reserve-extra)),{"host_MemAvailable":available,"OS_headroom":reserve,"other_reservations":others,"own_cgroup_remaining":own_remaining}
def collect(route,sig):
 p=O/"route_checks"/(route+".json");r=read(p);assert r["status"]=="PASS_full_route_inference"
 assert r["current_qualification_signature"]==sig
 assert sha(r["private_replay_path"])==r["private_replay_sha256"]
 assert max(r["errors"].values())<=1e-12
 return r
def controller():
 with (O/"QUALIFICATION.lock").open("a") as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);sig=expected_signature()
  routes=list(read(A/"catalog.json"));result=[]
  pilot="temporal_preop_preop_full";assert pilot in routes
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
  p=O/"route_checks"/(pilot+".json")
  if not p.exists():
   with (O/"largest_route_pilot.log").open("ab") as stream:
    child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--route",pilot],env=env,stdout=stream,stderr=stream)
    save(O/"ROUTE_STATUS.json",{"state":"current_largest_route_real_pilot","time":now(),"owned_worker":child.pid,"route":pilot})
    try:code=child.wait(timeout=1500)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned route qualification pilot budget exceeded")
   assert code==0,"Inspect largest route pilot before new attempts"
  pilot_result=collect(pilot,sig);result.append(pilot_result)
  usable,admission=resources();perworker=max(1024**3,int(1.6*pilot_result["peak_rss_bytes"]))
  assert usable>perworker+1024**3
  workers=max(1,min(len(routes)-1,len(os.sched_getaffinity(0)),int((usable-1024**3)//perworker)))
  save(O/"ROUTE_RESOURCE_BUDGET.json",{"time":now(),"measured_largest_route_RSS":pilot_result["peak_rss_bytes"],"per_worker_peak_budget_bytes":perworker,"workers":workers,"actual_resource_admission":admission,"primary_LLM_not_modified":True})
  pending=[]
  for route in routes:
   if route==pilot:continue
   if (O/"route_checks"/(route+".json")).exists():result.append(collect(route,sig))
   else:pending.append(route)
  active={}
  try:
   while pending or active:
    while pending and len(active)<workers:
     route=pending.pop(0);log=(O/(route+".log")).open("ab")
     child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--route",route],env=env,stdout=log,stderr=log)
     active[route]=(child,log,time.monotonic())
    for route,(child,log,started) in list(active.items()):
     rc=child.poll()
     if rc is None:
      if time.monotonic()-started>1500:child.kill();child.wait();raise TimeoutError("Owned calculator route exceeded finite budget")
      continue
     log.close();del active[route];assert rc==0,"Inspect calculator route "+route
     result.append(collect(route,sig))
    save(O/"ROUTE_STATUS.json",{"state":"running_current_full_reference_and_invariance","time":now(),"completed":len(result),"total":10,"workers":workers,"active":{route:v[0].pid for route,v in active.items()}})
    if pending or active:time.sleep(2)
  finally:
   for child,log,started in active.values():child.wait();log.close()
  assert len(result)==10
  expected_signature()
  sources=read(O/"CURRENT_QUALIFICATION_CONTRACT.json")["source_files"]
  assert all(sha(f)==h for f,h in sources.items())
  receipt={"status":"PASS_ten_primary_routes_full_saved_reference_and_invariance","time":now(),"passed":10,"total":10,"heldout_records_replayed":sum(v["heldout_reference_n"] for v in result),"max_absolute_difference":max(max(v["errors"].values()) for v in result),"strict_absolute_tolerance":1e-12,"current_qualification_signature":sig,"route_receipts":{str(O/"route_checks"/(route+".json")):sha(O/"route_checks"/(route+".json")) for route in routes},"app_code_sha256":{p.name:sha(p) for p in A.iterdir() if p.is_file()},"protected_frozen_states_unchanged":len(sources),"final_scientific_review_performed":False}
  save(O/"ROUTE_ACCEPTANCE.json",receipt);save(O/"ROUTE_STATUS.json",{"state":"complete_current_route_engineering_qualification","time":now(),"passed":10,"workers":0})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--route");a=p.parse_args()
 if a.route:raise SystemExit(worker(a.route))
 controller()
