from pathlib import Path
import os,sys,json,hashlib,subprocess,time
R=Path(__file__).resolve().parents[2];P=R.parents[1];PY=R/"envs/prediction_gpu/bin/python"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(q,p)
def lock(stage,name="CODE_LOCK.json",recursive=False):
 c=R/"work"/stage;o=R/"outputs"/stage;o.mkdir(parents=True,exist_ok=True)
 v={str(f.relative_to(c)):sha(f) for f in (c.rglob("*.py") if recursive else c.glob("*.py"))}
 save(o/name,v);return v
def deps(stage,name="DEPENDENCIES.json"):
 original=P/"outputs"/stage/name
 mapping={}
 for path in read(original):
  text=path.replace("/data/codex-workspaces/aki-prediction-manuscript/",str(R)+"/")
  if text==path:text=path.replace(str(P)+"/",str(R)+"/",1)
  f=Path(text);assert f.exists(),"Unprepared fresh dependency: "+text
  mapping[text]=sha(f)
 save(R/"outputs"/stage/name,mapping);return mapping
def run(relative,args=(),env_extra=None,gpu=False,launch_record=None):
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0" if gpu else "")
 if env_extra:env.update(env_extra)
 proc=subprocess.Popen([str(PY),"-B",str(R/"work"/relative),*args],cwd=R,env=env)
 if launch_record:save(launch_record,{"pid":proc.pid,"start_ticks":Path("/proc",str(proc.pid),"stat").read_text().split()[21],"finite":True,"at":time.time()})
 rc=proc.wait();assert rc==0,relative+" failed, preserved output and logs"
