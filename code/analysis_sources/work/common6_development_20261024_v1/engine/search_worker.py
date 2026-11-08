"""Shared Optuna controller; original runtime owns every scientific fit."""
from core import *
import optuna,subprocess,pickle,itertools,argparse,traceback
optuna.logging.set_verbosity(optuna.logging.WARNING)
SPEC=json.loads((PLAN/"PARAMETER_SEARCH_SPACE.json").read_text())
def study_dir(t):return OUT/"search_v2"/t["line"]/t["cohort"]/t["stage"]/t["branch"]/t["context"]/t["family"]
def seed_for(t):return int(digest({k:v for k,v in t.items() if k not in ["target_trials"]})[:8],16)
def simple(trial,key,value):
 if isinstance(value,list):return trial.suggest_categorical(key,value)
 if isinstance(value,dict) and "low" in value:return trial.suggest_float(key,value["low"],value["high"],log=value.get("distribution")=="log")
 raise ValueError((key,value))
def sample(trial,t):
 f=t["family"];cfg=SPEC["families"][f];params={}
 if t["branch"]=="selected":params["selector_k"]=trial.suggest_categorical("selector_k",[3,5,7,9,12,15,18,21,24,27])
 if f=="LR":
  params["variant"]=trial.suggest_categorical("variant",["linear","spline"]);params["C"]=simple(trial,"C",cfg["C"])
  if params["variant"]=="spline":params["n_knots"]=trial.suggest_categorical("n_knots",cfg["spline"]["n_knots"])
 elif f=="SVM":
  params["C"]=simple(trial,"C",cfg["C"])
  kind=trial.suggest_categorical("gamma_kind",["scale","numeric"])
  params["gamma"]="scale" if kind=="scale" else trial.suggest_float("gamma",*cfg["gamma"]["additional_log_range"],log=True)
 elif f=="NB":params["var_smoothing"]=10.0**trial.suggest_categorical("var_smoothing_exponent",cfg["var_smoothing"]["grid_log10_exponents"])
 elif f=="NNET":
  key=trial.suggest_categorical("hidden_layer_key",[str(i) for i in range(len(cfg["hidden_layer_sizes"]))])
  params["hidden_layer_sizes"]=cfg["hidden_layer_sizes"][int(key)]
  params["alpha"]=simple(trial,"alpha",cfg["alpha"]);params["learning_rate_init"]=simple(trial,"learning_rate_init",cfg["learning_rate_init"])
 else:
  names={
   "DT":["max_depth","min_samples_leaf","min_samples_split","ccp_alpha"],
   "RF":["n_estimators","max_depth","min_samples_leaf","max_features"],
   "KNN":["n_neighbors","weights","p"],
   "XGB":["n_estimators","max_depth","learning_rate","min_child_weight","subsample","colsample_bytree","gamma","reg_alpha","reg_lambda"],
   "SGBT":["n_estimators","max_depth","learning_rate","min_samples_leaf","subsample"]
  }[f]
  for key in names:params[key]=simple(trial,key,cfg[key])
 return params
def finite_points(t):
 if t["branch"]=="selected":return None
 if t["family"]=="NB":return [{"var_smoothing_exponent":e} for e in SPEC["families"]["NB"]["var_smoothing"]["grid_log10_exponents"]]
 if t["family"]=="KNN":
  c=SPEC["families"]["KNN"]
  return [dict(zip(["n_neighbors","weights","p"],v)) for v in itertools.product(c["n_neighbors"],c["weights"],c["p"])]
 return None
def choose_best(ledger):
 bestAP=max(x["mean_AP"] for x in ledger)
 eligible=[x for x in ledger if x["mean_AP"]>=bestAP-1e-12]
 bestBrier=min(x["mean_Brier"] for x in eligible)
 return min([x for x in eligible if x["mean_Brier"]<=bestBrier+1e-12],key=lambda x:x["candidate_id"])
def save_checkpoint(path,study,ledger,phase):
 q=Path(str(path)+".tmp")
 with q.open("wb") as f:pickle.dump({"study":study,"ledger":ledger,"phase":phase},f);f.flush();os.fsync(f.fileno())
 os.replace(q,path)
def run(t,threads,target=None):
 dest=study_dir(t);dest.mkdir(parents=True,exist_ok=True)
 lock=(dest/"study.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signature={"task":{k:v for k,v in t.items() if k!="target_trials"},"plan_sha256":sha(PLAN/"PROTOCOL.json"),"search_space_sha256":sha(PLAN/"PARAMETER_SEARCH_SPACE.json"),"controller_sha256":sha(__file__),"engine_sha256":sha(CODE/"model_engine.py"),"source_input_lock_sha256":sha(INPUT/"SOURCE_LOCK.json"),"core_sha256":sha(CODE/"core.py"),"optuna_version":optuna.__version__,"original_model_runtime":str(PYTHON)}
 if ready(dest,signature):return
 if (dest/"SIGNATURE.json").exists():assert json.loads((dest/"SIGNATURE.json").read_text())==signature
 else:atomic(dest/"SIGNATURE.json",signature)
 fulltarget=96;finite=finite_points(t)
 if finite is not None:fulltarget=len(finite)
 goal=min(fulltarget,target) if target is not None else fulltarget
 ckpt=dest/"checkpoint.pkl"
 if ckpt.exists():
  with ckpt.open("rb") as f:obj=pickle.load(f)
  study=obj["study"];ledger=obj["ledger"];phase=obj["phase"]
 else:
  sampler=optuna.samplers.RandomSampler(seed=seed_for(t))
  study=optuna.create_study(direction="maximize",sampler=sampler);ledger=[];phase="random"
  if finite is not None:
   for point in finite:study.enqueue_trial(point)
 request=dest/"ENGINE_REQUEST.json";atomic(request,t)
 env=os.environ.copy();env.update({k:"1" for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]});env["PYTHONDONTWRITEBYTECODE"]="1"
 log=(dest/"model_runtime.log").open("a")
 worker=subprocess.Popen([str(PYTHON),"-B",str(CODE/"model_engine.py"),str(request),str(threads),str(dest)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,cwd=ROOT,env=env)
 try:
  hello=worker.stdout.readline()
  assert hello and json.loads(hello)["sklearn"]=="1.5.2",hello
  seen={canonical(r["params"]) for r in ledger};proposals=0
  while len(ledger)<goal:
   if len(ledger)>=48 and phase=="random" and fulltarget>48:
    refined=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(n_startup_trials=0,seed=seed_for(t),constant_liar=False))
    refined.add_trials([v for v in study.trials if v.state==optuna.trial.TrialState.COMPLETE])
    study=refined;phase="tpe"
   trial=study.ask();params=sample(trial,t);key=canonical(params)
   if key in seen:
    study.tell(trial,state=optuna.trial.TrialState.PRUNED);proposals+=1
    if proposals>10*fulltarget:raise RuntimeError("Cannot obtain unique candidates under frozen search space")
    continue
   start=time.time()
   worker.stdin.write(canonical({"action":"trial","params":params,"trial":len(ledger)})+"\n");worker.stdin.flush()
   line=worker.stdout.readline()
   if not line:raise RuntimeError("Model runtime exited; see log:"+str(dest/"model_runtime.log"))
   reply=json.loads(line)
   if not reply["ok"]:
    study.tell(trial,state=optuna.trial.TrialState.FAIL)
    atomic(dest/"FAILED_TRIAL.json",{"trial":trial.number,"params":params,"reply":reply})
    save_checkpoint(ckpt,study,ledger,phase)
    raise RuntimeError(reply["error"])
   study.tell(trial,reply["mean_AP"])
   rec={"trial":len(ledger),"optuna_trial_number":trial.number,"phase":phase,"params":params,"candidate_id":digest(params),"seconds":time.time()-start,**{k:v for k,v in reply.items() if k not in ["ok","trial"]}}
   ledger.append(rec);seen.add(key);save_checkpoint(ckpt,study,ledger,phase)
   best=choose_best(ledger)
   atomic(dest/"PROGRESS.json",{"complete_unique":len(ledger),"target":fulltarget,"state":"running","best":best,"updated":time.time(),"fits":len(ledger)*3,"scoring_scope":"development_inner_selection_not_unbiased_validation"})
  best=choose_best(ledger)
  atomic(dest/"TRIALS.json",{"trials":ledger,"selected":best,"development_selection_only":True})
  if len(ledger)==fulltarget:
   atomic(dest/"COMPLETE.json",{"signature":signature,"files":{n:sha(dest/n) for n in ["TRIALS.json","checkpoint.pkl"]},"selected":best,"unique_trials":len(ledger),"fits":len(ledger)*3,"peak_model_rss_bytes":max(x["peak_rss_bytes"] for x in ledger),"no_final_evaluation_used":True})
  else:
   atomic(dest/"PARTIAL_QUALIFICATION.json",{"complete_unique":len(ledger),"target":fulltarget,"fully_resumable":True,"not_completed_study":True})
 finally:
  if worker.poll() is None:
   try:worker.stdin.write('{"action":"close"}\n');worker.stdin.flush()
   except BrokenPipeError:pass
  worker.wait();log.close();lock.close()
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("task");p.add_argument("--threads",type=int,default=1);p.add_argument("--target",type=int);a=p.parse_args()
 t=json.loads(Path(a.task).read_text())
 try:run(t,a.threads,a.target)
 except Exception:
  atomic(study_dir(t)/"SEARCH_FAILURE.json",{"traceback":traceback.format_exc(),"time":time.time()})
  raise
