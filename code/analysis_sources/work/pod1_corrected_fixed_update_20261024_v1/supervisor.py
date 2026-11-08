"""Finite memory/CPU/VRAM-admitted current-input secondary update."""
from pathlib import Path
import os,sys,time,subprocess,fcntl,json,traceback
import engine as e
import spare_resource_admission as spare
OUT=e.OUT;ROOT=e.ROOT;CODE=e.CODE;base=e.base
def gpu_free():
 r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 assert r.returncode==0,r.stderr
 return int(r.stdout.strip().splitlines()[0])*1024**2
def rss(pid):
 try:return int(Path('/proc',str(pid),'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
 except FileNotFoundError:return 0
def complete(i):
 p=OUT/'tasks'/str(i).zfill(3)/'COMPLETE.json'
 if not p.exists():return False
 r=base.read(p);assert r['status']=='PASS_fixed_choice_POD1_context_fitted'
 for f,h in r['files'].items():assert base.sha(p.parent/f)==h
 assert r['signature']['code']==base.sha(OUT/'CODE_LOCK.json')
 return True
def main():
 guard=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);e.check()
 assert base.read(OUT/'QUALIFICATION.json')['status']=='PASS_POD1_current_inputs_fixed_choices_nested_group_boundaries_and_CUDA'
 base.atomic(OUT/'LAUNCH.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'time_UTC':base.stamp(),'code_lock_sha256':base.sha(OUT/'CODE_LOCK.json')})
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 tasks=base.read(OUT/'TASKS.json');affinity=base.effective_cpu();active={};logs=OUT/'logs';logs.mkdir(exist_ok=True)
 budget=1024**3;head=max(3*1024**3,int(min(base.mem(),spare.available(ROOT,OUT))*.2));gpu_head=2*1024**3;vram_budget=1024**3
 for phase,indices in [('base',[i for i,t in enumerate(tasks) if t['family']!='Stacking']),('strict_stacking',[i for i,t in enumerate(tasks) if t['family']=='Stacking'])]:
  pending=[i for i in indices if not complete(i)]
  try:
   while pending or active:
    e.check()
    for i,(p,h,cost,gg) in list(active.items()):
     rc=p.poll()
     if rc is None:continue
     h.close();del active[i]
     assert rc==0,'Worker failed; diagnose before restart: '+str(i)
     assert complete(i)
    while pending:
     available=min(base.mem(),spare.available(ROOT,OUT));reserved=sum(max(0,budget-rss(p.pid)) for p,_,_,_ in active.values())
     cpu_free=affinity-sum(cost for _,_,cost,_ in active.values())
     if available<head+reserved+budget or cpu_free<1:break
     free=gpu_free();gactive=sum(1 for _,_,_,gg in active.values() if gg)
     candidate=None
     for i in pending:
      uses=tasks[i]['family'] in ['XGB','Stacking']
      if uses and free<gpu_head+vram_budget*(1+gactive):continue
      candidate=i;break
     if candidate is None:break
     i=candidate;pending.remove(i);family=tasks[i]['family'];uses=family in ['XGB','Stacking']
     if family=='LR':threads=1
     else:
      future=max(1,min(len(pending)+1,int(max(1,(available-head)/budget))))
      threads=max(1,cpu_free//future)
     h=(logs/(str(i).zfill(3)+'.log')).open('ab')
     p=subprocess.Popen([str(base.PYTHON),'-B',str(CODE/'engine.py'),str(i),str(threads)],cwd=ROOT,env=env,
      stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
     active[i]=(p,h,threads,uses)
    done=sum(complete(i) for i in range(len(tasks)))
    base.atomic(OUT/'status.json',{'state':'running_current_input_POD1_fixed_choice_update','phase':phase,'completed':done,'total':len(tasks),
     'owned_workers':{str(i):{'pid':p.pid,'threads':c,'GPU_eligible':gg} for i,(p,h,c,gg) in active.items()},
     'pending_phase':len(pending),'effective_cpu_affinity':affinity,'memory_available_bytes':base.mem(),'memory_headroom_bytes':head,
     'GPU_free_bytes':gpu_free(),'GPU_headroom_bytes':gpu_head,'main_spare_admission':spare.LAST,'time_UTC':base.stamp()})
    if pending or active:time.sleep(5)
  finally:
   for p,h,c,gg in active.values():p.wait();h.close()
   active.clear()
 refs={str(OUT/'tasks'/str(i).zfill(3)/'COMPLETE.json'):base.sha(OUT/'tasks'/str(i).zfill(3)/'COMPLETE.json') for i in range(len(tasks))}
 locks={str(i):{'task':tasks[i],'receipt':str(OUT/'tasks'/str(i).zfill(3)/'COMPLETE.json'),'sha256':refs[str(OUT/'tasks'/str(i).zfill(3)/'COMPLETE.json')]} for i in range(len(tasks)) if tasks[i]['role']=='final'}
 base.atomic(OUT/'MODEL_LOCKS.json',{'status':'PASS_22_current_input_POD1_fixed_choice_model_packages_locked','models':locks,'no_holdout_winner_selection':True})
 base.atomic(OUT/'STAGE_POD1_FIT_ACCEPTANCE.json',{'status':'PASS_132_fixed_choice_POD1_fitting_contexts','receipts':refs,'final_models':22,
  'five_fold_internal_conditional_on_prior_fixed_choices':True,'strict_stack_upstream_reconstructed':True,'evaluation_pending':True,'whole_project_complete':False})
 base.atomic(OUT/'status.json',{'state':'complete_POD1_fits_evaluation_pending','completed':132,'total':132,'owned_workers':0,'time_UTC':base.stamp()})
 guard.close()
if __name__=='__main__':
 try:main()
 except Exception:
  base.atomic(OUT/'FAILURE.json',{'time_UTC':base.stamp(),'traceback':traceback.format_exc(),'no_unchanged_automatic_retry':True});raise
