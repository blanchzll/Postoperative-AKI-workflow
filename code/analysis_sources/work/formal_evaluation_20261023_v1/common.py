from pathlib import Path
import json,hashlib,os,datetime,importlib.util
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/formal_evaluation_20261023_v1'
SOURCE=ROOT/'outputs/v4_development_inputs_20261018'
MODELS=ROOT/'outputs/model_lock_20261022_v1'
POLICIES=ROOT/'outputs/selective_policy_lock_20261023_v1'
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
EXPECTED={('geography', 'preop'): (1159, 72), ('temporal', 'preop'): (1683, 66), ('geography', 'paired'): (1158, 72), ('temporal', 'paired'): (1011, 36)}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def gate():
 a=read(MODELS/'STAGE_MODEL_LOCK_ACCEPTANCE.json')
 b=read(POLICIES/'STAGE_POLICY_LOCK_ACCEPTANCE.json')
 assert a['status']=='PASS' and a['final_model_locked']
 assert b['status']=='PASS_two_selective_strategies_and_prespecified_variants_locked'
 assert b['final_hospital_time_scoring_permitted'] and b['policy_variant_count']==24
 assert sha(POLICIES/'CLINICAL_INFERENCE_QUALIFICATION.json')==b['clinical_inference_qualification_sha256']
 assert read(POLICIES/'CLINICAL_INFERENCE_QUALIFICATION.json')['status']=='PASS_development_only_exact_source_policy_inference'
 for rec in a['models']+b['policy_manifests']:assert sha(rec['path'])==rec['sha256']
 for d,src in [(MODELS,ROOT/'work/model_lock_20261022_v1'),(POLICIES,ROOT/'work/selective_policy_lock_20261023_v1')]:
  for name,h in read(d/'CODE_LOCK.json').items():assert sha(src/name)==h
 for name,h in read(SOURCE/'SOURCE_LOCK.json')['input_sha256'].items():assert sha(SOURCE/name)==h
 return a,b
def mem():
 n=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
 for line in Path('/proc/self/cgroup').read_text().splitlines():
  _,ctrl,rel=line.split(':',2)
  if not ctrl:
   base=Path('/sys/fs/cgroup')/rel.lstrip('/')
   for p in [base,*base.parents]:
    lim=p/'memory.max';used=p/'memory.current'
    if lim.exists() and used.exists() and lim.read_text().strip()!='max':n=min(n,max(0,int(lim.read_text())-int(used.read_text())))
 return n

# Admission only: measure actual other CPU use and inherited cgroup/affinity headroom.
_cpu_admission=None
def effective_cpu():
 global _cpu_admission
 if _cpu_admission is None:
  _cpu_admission=module("corrected_run_resource_admission",ROOT/"work/formal_development_20261018_v1/core.py")
 return _cpu_admission.resource_snapshot()["effective_cpu"]
