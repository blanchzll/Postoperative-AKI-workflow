"""Finite common6 nested base bank and first-layer raw ensembles; no old model reuse."""
from pathlib import Path
import sys,os,json,hashlib,time,fcntl,subprocess,traceback,shutil
ROOT=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
CODE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/common6_strict_continuation_20261024_v1';AB=ROOT/'outputs/common6_development_20261024_v1'
sys.path.insert(0,str(ROOT/'work/formal_evaluation_20261023_v1'))
import common as base
PYTHON=ROOT/'envs/prediction_gpu/bin/python'
SHARED=Path('/home/lei/codex-workspaces/shared-python-envs/envs/clinical-ml-py312-cpu-xgb214-20261016-v1/bin/python')
FAMILIES=['LR','DT','RF','KNN','SVM','NB','XGB','SGBT','NNET'];OWNED=[]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def build(man):
    jobs=[];bank={};by={s['id']:s for s in man['scopes']}
    for s in man['scopes']:
        sid=s['id'];scopefile=OUT/'scopes'/sid/'SCOPE.json';coarse='coarse_'+sid
        jobs.append(dict(id=coarse,kind='coarse',scope_id=sid,dest=str(Path(s['output'])/'COARSE_ACCEPTANCE.json'),depends=[]))
        for f in FAMILIES:
            task={'line':s['line'],'cohort':'preop','stage':'preop','branch':'full','context':'outer_0','family':f,'scope_id':sid,'scope_sha256':base.sha(scopefile)}
            key=digest(task)[:24]
            sd=Path(s['output'])/'search_v2'/s['line']/'preop/preop/full/outer_0'/f
            fd=Path(s['output'])/'outer_raw'/s['line']/'preop/preop/full/outer_0'/f
            jobs.append(dict(id='search_'+key,kind='search',scope_id=sid,task=task,dest=str(sd/'COMPLETE.json'),depends=[coarse]))
            jobs.append(dict(id='fit_'+key,kind='fit',scope_id=sid,task=task,dest=str(fd/'COMPLETE.json'),depends=['search_'+key]))
            bank[(sid,f)]='fit_'+key
    for r in man['ensemble_calibration_roots']:
        second=[s for s in man['scopes'] if s['parent']==r['id']]
        third=[s for s in man['scopes'] if s['parent'] in {t['id'] for t in second}]
        assert len(second)==3 and len(third)==9
        ss=[by[r['id']]]+second+third
        jobs.append(dict(id='ensemble_'+r['id'],kind='ensemble',task={'root':r,'stage':'preop','branch':'full'},
          dest=str(OUT/'ensemble_results'/r['line']/'preop/preop/full'/r['id']/'COMPLETE.json'),
          depends=[bank[(s['id'],f)] for s in ss for f in FAMILIES]))
    assert len(jobs)==8928
    return jobs
def good(j):
    p=Path(j['dest'])
    if not p.exists():return False
    q=base.read(p)
    if j['kind']=='coarse':
        assert q['status']=='PASS' and q['source_scope_sha256']==base.sha(OUT/'scopes'/j['scope_id']/'SCOPE.json')
        return True
    for f,h in q.get('files',{}).items():assert base.sha(p.parent/f)==h
    if j['kind'] in ['search','fit']:
        t=q['signature']['task'];assert t['scope_sha256']==j['task']['scope_sha256']
        assert t['branch']=='full' and t['stage']=='preop'
    return True
def rss(pid):
    total=0
    try:
        for s in Path('/proc',str(pid),'status').read_text().splitlines():
            if s.startswith('VmRSS:'):total=int(s.split()[1])*1024;break
        for child in Path('/proc',str(pid),'task',str(pid),'children').read_text().split():total+=rss(int(child))
    except OSError:pass
    return total
def check_code():
    for n,h in base.read(OUT/'EXECUTION_CODE_LOCK.json').items():assert base.sha(CODE/n)==h
    for p,h in base.read(OUT/'EXECUTION_DEPENDENCIES.json').items():assert base.sha(p)==h
def launch(j,by):
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    req=OUT/'requests'/(j['id']+'.json')
    if j['kind']=='ensemble':
        env.update(AKI_SCOPE_OUT=str(OUT),AKI_SCOPE_INPUT=str(AB/'inputs/geography'))
        task=j['task'];worker='ensemble_e_worker.py';cmd=[str(PYTHON),'-B',str(CODE/'engine'/worker),str(req)]
    else:
        s=by[j['scope_id']];env.update(AKI_SCOPE_OUT=s['output'],AKI_SCOPE_INPUT=s['input'])
        sp=OUT/'scopes'/s['id']/'SCOPE.json'
        if j['kind']=='coarse':
            task=None;cmd=[str(PYTHON),'-B',str(CODE/'engine/scope_worker.py'),'coarse',str(sp),'--threads','1']
        else:
            method=base.read(Path(s['output'])/'selection'/s['line']/'preop/outer_0/METHOD_SELECTION.json')['selected_method']
            task={**j['task'],'method':method}
            cmd=([str(SHARED),'-B',str(CODE/'engine/search_worker.py'),str(req),'--threads','1'] if j['kind']=='search' else
                 [str(PYTHON),'-B',str(CODE/'engine/scope_worker.py'),'fit',str(sp),'--task',str(req),'--threads','1'])
    if task is not None:
        if req.exists():assert base.read(req)==task
        else:base.atomic(req,task)
    log=(OUT/'logs'/(j['id']+'.log')).open('ab')
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);OWNED.append((p,log));return p,log
def main():
    guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_code()
    assert base.read(OUT/'EXECUTION_QUALIFICATION.json')['status']=='PASS_common6_scoped_DAG_and_original_engine_contract'
    assert base.read(AB/'REVIEW_ACCEPTANCE.json')['status']=='PASS_864_coarse_imputation_108_searches_reverified'
    prior=(AB/'SUPERVISOR.lock').open('a');fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'NESTED_BANK_ACCEPTANCE.json').exists():return
    man=base.read(OUT/'SCOPE_MANIFEST.json');by={s['id']:s for s in man['scopes']};jobs=build(man)
    for d in ['logs','requests']:(OUT/d).mkdir(exist_ok=True)
    base.atomic(OUT/'JOB_MANIFEST.json',{'jobs':jobs,'counts':{k:sum(j['kind']==k for j in jobs) for k in ['coarse','search','fit','ensemble']}})
    base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'EXECUTION_CODE_LOCK.json')})
    # Largest training scope pilot traverses actual coarse -> full LR search -> raw fit.
    pilot_scope=max(man['scopes'],key=lambda s:(s['train_n'],s['id']))['id']
    pilots={j['id'] for j in jobs if j.get('scope_id')==pilot_scope and (j['kind']=='coarse' or j.get('task',{}).get('family')=='LR')}
    assert len(pilots)==3
    done={j['id'] for j in jobs if good(j)};live={};fail=[];peak={};budget={};pilot_ok=pilots<=done
    while len(done)<len(jobs):
        for key,(p,log,j) in list(live.items()):
            peak[key]=max(peak.get(key,0),rss(p.pid));rc=p.poll()
            if rc is None:continue
            log.close();del live[key]
            if rc or not good(j):fail.append({'id':key,'returncode':rc});continue
            done.add(key);rec=base.read(j['dest']);kind=(j['kind'],j.get('task',{}).get('family',''))
            observed=max(peak[key],rec.get('peak_rss_bytes',0),rec.get('peak_model_rss_bytes',0))
            budget[kind]=max(budget.get(kind,0),768*1024**2,2*observed)
        if not pilot_ok and pilots<=done:
            pilot_ok=True;base.atomic(OUT/'PILOT_ACCEPTANCE.json',{'status':'PASS_largest_scope_coarse_full_search_raw_fit','pilot_scope':pilot_scope,
                'receipts':{j['id']:base.sha(j['dest']) for j in jobs if j['id'] in pilots},'time_UTC':base.stamp()})
        if fail and not live:raise RuntimeError('Nested common6 failure; diagnose before retry '+str(fail))
        check_code();available=base.mem();head=max(4*1024**3,int(.25*available))
        def reserve(j):return budget.get((j['kind'],j.get('task',{}).get('family','')),2*1024**3)
        growth=sum(max(0,reserve(j)-rss(p.pid)) for p,log,j in live.values())
        pending=[j for j in jobs if j['id'] not in done and j['id'] not in live and set(j['depends'])<=done and (pilot_ok or j['id'] in pilots)]
        pending.sort(key=lambda j:({'ensemble':0,'fit':1,'search':2,'coarse':3}[j['kind']],j['id']))
        for j in pending:
            if fail or len(live)>=base.effective_cpu():break
            if available-head-growth<reserve(j) or shutil.disk_usage(OUT).free<100*1024**3:continue
            p,log=launch(j,by);live[j['id']]=(p,log,j);peak[j['id']]=0;growth+=reserve(j)
        base.atomic(OUT/'status.json',{'state':'running_nested_common6' if live else 'waiting_resources','pid':os.getpid(),'time_UTC':base.stamp(),
            'completed':len(done),'total':len(jobs),'counts':{k:{'completed':sum(j['kind']==k and j['id'] in done for j in jobs),'total':sum(j['kind']==k for j in jobs)} for k in ['coarse','search','fit','ensemble']},
            'running':{k:v[0].pid for k,v in live.items()},'failures':fail,'pilot_accepted':pilot_ok,'available_bytes':available,'growth_reserved_bytes':growth,
            'model_locked':False,'public_performance_accessed':False,'entire_research_complete':False})
        if len(done)<len(jobs):time.sleep(15)
    base.atomic(OUT/'NESTED_BANK_ACCEPTANCE.json',{'status':'PASS_common6_nested_base_bank_and_first_layer_raw_ensembles_only',
      'task_count':len(jobs),'receipts':{j['id']:base.sha(j['dest']) for j in jobs},'code_lock_sha256':base.sha(OUT/'EXECUTION_CODE_LOCK.json'),
      'owned_workers_remaining':0,'not_complete_internal_evaluation_or_model_lock':True,'public_performance_accessed':False})
    base.atomic(OUT/'status.json',{'state':'complete_nested_base_bank_and_first_layer_raw_ensembles','completed':len(jobs),'owned_workers_remaining':0,
       'next_required':'Root ensemble selection, strict calibration and internal report, then separate common6 model locks and INSPIRE scoring','model_locked':False,'entire_research_complete':False})
    prior.close();guard.close()
if __name__=='__main__':
    try:main()
    except Exception:
        base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'owned_running_pids':[p.pid for p,l in OWNED if p.poll() is None]})
        for p,log in OWNED:
            p.wait()
            if not log.closed:log.close()
        raise
