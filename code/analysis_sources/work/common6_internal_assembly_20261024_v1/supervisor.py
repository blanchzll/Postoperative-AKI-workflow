"""Wait for strict bank, then finite root fits/ensembles and calibrated internal evaluation."""
from pathlib import Path
import sys,os,time,fcntl,subprocess,traceback,shutil
from complete_internal import ROOT,CODE,OUT,BANK,AB,base,FAMILIES
PYTHON=ROOT/'envs/prediction_gpu/bin/python';OWNED=[]
def build():
    man=base.read(OUT/'SCOPE_MANIFEST.json');jobs=[]
    for root in man['roots']:
        deps=[]
        if root['context']!='final':
            method=base.read(AB/'runs'/root['line']/'selection'/root['line']/'preop'/root['context']/'METHOD_SELECTION.json')['selected_method']
            for f in FAMILIES:
                task={'line':root['line'],'cohort':'preop','stage':'preop','branch':'full','context':root['context'],'family':f,'method':method}
                key='base_'+root['id']+'_'+f;deps.append(key)
                dest=Path(root['base_output'])/'outer_raw'/root['line']/'preop/preop/full'/root['context']/f/'COMPLETE.json'
                jobs.append({'id':key,'kind':'fit','line':root['line'],'task':task,'dest':str(dest),'depends':[]})
        jobs.append({'id':'ensemble_'+root['id'],'kind':'ensemble','line':root['line'],'task':{'root':root,'stage':'preop','branch':'full'},
          'dest':str(OUT/'ensemble_results'/root['line']/'preop/preop/full'/root['context']/'COMPLETE.json'),'depends':deps})
    assert len(jobs)==102;return jobs
def good(j):
    p=Path(j['dest'])
    if not p.exists():return False
    for f,h in base.read(p)['files'].items():assert base.sha(p.parent/f)==h
    return True
def rss(pid):
    v=0
    try:
        v=int(next(s.split()[1] for s in Path('/proc',str(pid),'status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
        for child in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():v+=rss(int(child))
    except (OSError,StopIteration):pass
    return v
def check():
    for n,h in base.read(OUT/'CODE_LOCK.json').items():assert base.sha(CODE/n)==h
    for p,h in base.read(OUT/'DEPENDENCIES.json').items():assert base.sha(p)==h
def main():
    guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
    assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_common6_internal_assembly_contract'
    base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
    while not (BANK/'NESTED_BANK_ACCEPTANCE.json').exists():
        if (BANK/'FAILURE.json').exists():raise RuntimeError('Upstream nested common6 failed; diagnose before continuation')
        base.atomic(OUT/'status.json',{'state':'waiting_nested_common6_bank','pid':os.getpid(),'time_UTC':base.stamp(),'owned_workers':0})
        time.sleep(30)
    assert base.read(BANK/'NESTED_BANK_ACCEPTANCE.json')['status']=='PASS_common6_nested_base_bank_and_first_layer_raw_ensembles_only'
    prior=(BANK/'SUPERVISOR.lock').open('a')
    while True:
        try:fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(10)
    if (OUT/'STAGE_INTERNAL_ACCEPTANCE.json').exists():return
    jobs=build();base.atomic(OUT/'JOB_MANIFEST.json',{'jobs':jobs,'total':102})
    for n in ['logs','requests']:(OUT/n).mkdir(exist_ok=True)
    done={j['id'] for j in jobs if good(j)};live={};failed=[];peaks={};budget={}
    pilot=next(j['id'] for j in jobs if j['kind']=='fit' and j['task']['family']=='LR' and j['line']=='geography');pilotok=pilot in done
    env0=os.environ.copy();env0.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    while len(done)<len(jobs):
        for key,(p,log,j) in list(live.items()):
            peaks[key]=max(peaks.get(key,0),rss(p.pid));rc=p.poll()
            if rc is None:continue
            log.close();del live[key]
            if rc or not good(j):failed.append({'id':key,'returncode':rc});continue
            done.add(key);rec=base.read(j['dest']);budget[j['kind']]=max(budget.get(j['kind'],0),768*1024**2,2*max(peaks[key],rec.get('peak_rss_bytes',0)))
        if not pilotok and pilot in done:
            pilotok=True;base.atomic(OUT/'PILOT_ACCEPTANCE.json',{'status':'PASS_source_only_outer_raw_full_pipeline','id':pilot,'receipt_sha256':base.sha(next(j['dest'] for j in jobs if j['id']==pilot))})
        if failed and not live:raise RuntimeError('Assembly failed; retain checkpoints '+str(failed))
        check();avail=base.mem();head=max(4*1024**3,int(.25*avail))
        reserve=lambda j:budget.get(j['kind'],2*1024**3)
        growth=sum(max(0,reserve(j)-rss(p.pid)) for p,l,j in live.values())
        for j in jobs:
            if failed or len(live)>=len(os.sched_getaffinity(0)):break
            if j['id'] in done or j['id'] in live or not set(j['depends'])<=done or (not pilotok and j['id']!=pilot):continue
            if avail-head-growth<reserve(j):continue
            env=env0.copy();env['AKI_SCOPE_INPUT']=str(AB/'inputs'/j['line'])
            env['AKI_SCOPE_OUT']=str(OUT/'root_runs'/j['line'] if j['kind']=='fit' else OUT)
            req=OUT/'requests'/(j['id']+'.json')
            if req.exists():assert base.read(req)==j['task']
            else:base.atomic(req,j['task'])
            worker='raw_outer.py' if j['kind']=='fit' else 'root_ensemble.py'
            cmd=[str(PYTHON),'-B',str(CODE/'engine'/worker),str(req)]
            if j['kind']=='fit':cmd+=['--threads','1']
            log=(OUT/'logs'/(j['id']+'.log')).open('ab')
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);OWNED.append((p,log))
            live[j['id']]=(p,log,j);peaks[j['id']]=0;growth+=reserve(j)
        base.atomic(OUT/'status.json',{'state':'running_root_common6_assembly','time_UTC':base.stamp(),'pid':os.getpid(),'completed':len(done),'total':102,
          'running':{k:v[0].pid for k,v in live.items()},'failures':failed,'model_locked':False,'public_scores_accessed':False})
        if len(done)<len(jobs):time.sleep(15)
    base.atomic(OUT/'ASSEMBLY_COMPUTE_ACCEPTANCE.json',{'status':'PASS_90_outer_base_fits_12_root_ensembles','receipts':{j['id']:base.sha(j['dest']) for j in jobs},'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
    with (OUT/'INTERNAL_CALIBRATION.log').open('ab') as log:
        p=subprocess.Popen([str(PYTHON),'-B',str(CODE/'complete_internal.py')],cwd=ROOT,env=env0,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);OWNED.append((p,log))
        base.atomic(OUT/'status.json',{'state':'assembling_strict_common6_calibration','pid':os.getpid(),'worker':p.pid,'time_UTC':base.stamp()})
        if p.wait():raise RuntimeError('Internal common6 calibration failed; preserve outputs')
    check();assert (OUT/'STAGE_INTERNAL_ACCEPTANCE.json').exists()
    base.atomic(OUT/'status.json',{'state':'complete_strict_common6_internal_evaluation','time_UTC':base.stamp(),'owned_workers':0,'model_locked':False,'public_scores_accessed':False,'entire_research_complete':False})
    prior.close();guard.close()
if __name__=='__main__':
    try:main()
    except Exception:
        base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'owned_running_pids':[p.pid for p,l in OWNED if p.poll() is None]})
        for p,l in OWNED:
            p.wait()
            if not l.closed:l.close()
        raise
