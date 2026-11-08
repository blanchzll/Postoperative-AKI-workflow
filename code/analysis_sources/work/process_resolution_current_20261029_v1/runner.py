"""Finite, resumable paired support calculations; no source model mutations."""
from pathlib import Path
import os,sys,json,time,pickle,gzip,hashlib,fcntl,resource,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import analysis_core as ac
R=ac.R;O=ac.O;W=ac.W
CH=O/"chunks"
def now():return datetime.now(timezone.utc).isoformat()
def memavailable():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):return int(line.split()[1])*1024
def owned_worker_budget():
    host=memavailable();headroom=max(4*1024**3,int(host*.25))
    own=Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    cg=Path("/sys/fs/cgroup")/own.lstrip("/")
    maximum=(cg/"memory.max").read_text().strip();assert maximum!="max"
    own_remaining=max(0,int(maximum)-int((cg/"memory.current").read_text()))
    effective=host;ancestors=[];parent=cg.parent
    while parent!=Path("/sys/fs/cgroup"):
        mx=(parent/"memory.max").read_text().strip();used=int((parent/"memory.current").read_text())
        if mx!="max":
            remaining=max(0,int(mx)-used);effective=min(effective,remaining);ancestors.append({"cgroup":str(parent),"remaining_bytes":remaining})
        parent=parent.parent
    import subprocess
    extra=0;reservations=[]
    units=subprocess.run(["systemctl","--user","list-units","--state=active","--plain","--no-legend","aki-postslot-*"],capture_output=True,text=True,check=True).stdout.splitlines()
    for row in units:
        name=row.split()[0]
        if name in own:continue
        output=subprocess.run(["systemctl","--user","show",name,"-p","MemoryMax","-p","ControlGroup","-p","ActiveState"],capture_output=True,text=True,check=True).stdout
        props=dict(row.split("=",1) for row in output.splitlines() if "=" in row)
        if props.get("ActiveState")!="active":continue
        assert props["MemoryMax"].isdigit(),"Other active AKI owner requires explicit memory budget"
        stat=Path("/sys/fs/cgroup")/props["ControlGroup"].lstrip("/")/"memory.stat"
        try:mem=dict(row.split() for row in stat.read_text().splitlines())
        except FileNotFoundError:
            state=subprocess.run(["systemctl","--user","show",name,"-p","ActiveState"],capture_output=True,text=True,check=True).stdout.strip()
            if state!="ActiveState=active":continue
            raise
        used=sum(int(mem.get(k,0)) for k in ["anon","kernel","shmem"])
        growth=max(0,int(props["MemoryMax"])-used);extra+=growth
        reservations.append({"unit":name,"growth_reserved_bytes":growth})
    available=max(0,min(own_remaining,effective-headroom-extra))
    return available,{"actual_host_MemAvailable_bytes":host,"own_cgroup_remaining_bytes":own_remaining,"shared_ancestor_limits":ancestors,"OS_headroom_bytes":headroom,"other_AKI_growth_reserved_bytes":extra,"other_AKI_reservations":reservations,"usable_for_new_owned_workers_bytes":available}
def processmem():
    vals={}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        if ":" in line:
            k,v=line.split(":",1)
            if v.strip().endswith("kB"):vals[k]=int(v.split()[0])*1024
    return {"pss":vals.get("Pss",0),"private":vals.get("Private_Clean",0)+vals.get("Private_Dirty",0),"rss":vals.get("Rss",0)}
def key(job):return hashlib.sha256(json.dumps(job).encode()).hexdigest()[:24]
def paths(job):return CH/(key(job)+".pkl.gz"),CH/(key(job)+".json")
def valid(job,signature):
    a,b=paths(job)
    if not a.exists() or not b.exists():return False
    d=json.loads(b.read_text())
    return d["job"]==list(job) and d["signature"]==signature and ac.sha(a)==d["sha256"]
def execute(job,signature):
    t=time.monotonic();before=processmem();value=ac.task(job)
    a,b=paths(job);tmp=a.with_name(a.name+"."+str(os.getpid())+".tmp")
    with gzip.open(tmp,"wb",compresslevel=3) as f:pickle.dump(value,f,protocol=5)
    tmp.replace(a);after=processmem()
    receipt={"job":job,"signature":signature,"sha256":ac.sha(a),"bytes":a.stat().st_size,
             "seconds":time.monotonic()-t,"worker_pid":os.getpid(),"memory_before":before,
             "memory_after":after,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    ac.save(b,receipt);return receipt
def main():
    lock=(O/"RUN.lock").open("a+")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("A live resolution runner owns RUN.lock")
    assert json.loads((O/"QUALIFICATION.json").read_text())["status"].startswith("PASS_")
    CH.mkdir(parents=True,exist_ok=True,mode=0o700)
    link=O/"chunks"
    if not link.exists():link.symlink_to(CH,target_is_directory=True)
    assert link.resolve()==CH.resolve()
    signatures={str(p.relative_to(R)):ac.sha(p) for p in [O/"ANALYSIS_CONTRACT.json",O/"INPUT_ACCEPTANCE.json",W/"PRIVATE_INPUTS.pkl",W/"analysis_core.py",W/"calibration_solver.py",W/"runner.py"]}
    signature=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()
    state={"status":"initializing","pid":os.getpid(),"start_ticks":int(Path("/proc/self/stat").read_text().split()[21]),"started":now(),"signature":signature}
    ac.save(O/"RUN_STATUS.json",state)
    ac.initialize()
    jobs=[]
    chinese=[c for c in ac.DATA if c!="INSPIRE_common6"]
    for c in chinese:
        for a in range(0,2000,50):jobs.append(("alt",c,"",a,a+50))
    for c,d in ac.DATA.items():
        for name in d["cal_routes"]:
            for a in range(0,2000,50):jobs.append(("cal",c,name,a,a+50))
    for c in ["temporal_preop","temporal_paired"]:
        for a in range(0,2000,50):jobs.append(("nonoverlap",c,"",a,a+50))
    for c in chinese:
        for a in range(0,2000,200):jobs.append(("random",c,"",a,a+200))
    ac.save(O/"RUN_MANIFEST.json",{"signature":signature,"files":signatures,"jobs":jobs,"bootstrap_proofs":ac.BOOT_PROOF,"checkpoint_root":str(CH),"created":now()})
    pending=[j for j in jobs if not valid(j,signature)]
    old=len(jobs)-len(pending);state.update(total=len(jobs),completed=old,status="pilot",bootstrap_proofs=ac.BOOT_PROOF)
    ac.save(O/"RUN_STATUS.json",state)
    if not pending:
        state.update(status="completed",ended=now());ac.save(O/"RUN_STATUS.json",state);return
    target=[next((j for j in pending if j[0]=="cal" and j[1]=="INSPIRE_common6"),pending[0]),
            next((j for j in pending if j[0]=="nonoverlap"),pending[-1])]
    pilots=list(dict.fromkeys(target));measured=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(execute,j,signature):j for j in pilots}
        for f in futures:
            measured.append(f.result());pending.remove(futures[f]);old+=1
    perworker=max(max(z["memory_after"]["private"]*1.8,
                         z["memory_after"]["private"]+max(0,z["peak_rss_bytes"]-z["memory_before"]["rss"])*1.5,
                         128*1024**2) for z in measured)
    available,admission=owned_worker_budget();headroom=1024**3
    cpu=len(os.sched_getaffinity(0))
    if available<headroom+perworker:
        state.update(status="WAIT_RESOURCE_SAVED_REAL_PILOTS",completed=old,resource_admission=admission)
        ac.save(O/"RUN_STATUS.json",state);return
    workers=max(1,min(cpu,int((available-headroom)//perworker)))
    ac.save(O/"RESOURCE_BUDGET.json",{"CPU_affinity":cpu,"MemAvailable_bytes":available,"OS_and_other_jobs_headroom_bytes":headroom,"measured_pilots":measured,"per_worker_private_peak_budget_bytes":perworker,"workers":workers,"numerical_threads":1,"not_counting_VRAM_as_RAM":True,"actual_resource_admission":admission})
    state.update(status="running",completed=old,workers=workers);ac.save(O/"RUN_STATUS.json",state)
    errors=[];active={};cursor=0;done=old;start=time.monotonic();last=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while cursor<len(pending) or active:
            while cursor<len(pending) and len(active)<workers and owned_worker_budget()[0]>headroom+perworker:
                j=pending[cursor];cursor+=1;active[pool.submit(execute,j,signature)]=j
            if not active:
                state.update(status="WAIT_RESOURCE_SAVED_VALID_CHECKPOINTS",completed=done,active=0)
                ac.save(O/"RUN_STATUS.json",state);return
            finished,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
            for f in finished:
                j=active.pop(f)
                try:f.result();done+=1
                except Exception as e:
                    errors.append({"job":j,"error":repr(e),"traceback":traceback.format_exc()})
            if errors:raise RuntimeError(json.dumps(errors))
            if time.monotonic()-last>=15 or done==len(jobs):
                state.update(completed=done,active=len(active),unsubmitted=len(pending)-cursor,updated=now(),elapsed_seconds=time.monotonic()-start,MemAvailable_bytes=memavailable())
                ac.save(O/"RUN_STATUS.json",state);last=time.monotonic()
    assert all(valid(j,signature) for j in jobs)
    changed=[p for p,h in ac.PROTECTED.items() if ac.sha(R/p)!=h]
    assert not changed,changed
    state.update(status="completed",completed=len(jobs),active=0,ended=now(),protected_frozen_files_unchanged=True)
    ac.save(O/"RUN_STATUS.json",state)
if __name__=="__main__":
    try:main()
    except Exception:
        ac.save(O/"RUN_FAILURE.json",{"time":now(),"pid":os.getpid(),"error":traceback.format_exc()})
        raise
