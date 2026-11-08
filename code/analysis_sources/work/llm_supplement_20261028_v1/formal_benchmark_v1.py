"""Finite source-locked LLM benchmark fits and prediction banks.
Evaluation labels never enter training, prompts, budget selection, or stopping.
"""
from pathlib import Path
import argparse,csv,datetime,fcntl,gc,hashlib,importlib.util,json,os,random,resource,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[2];O=R/"outputs/llm_supplement_20261028_v1";D=O/"data_preparation_v1";F=O/"formal_v1";GIB=1024**3
MODELS=["biomedlm_2_7b","medgemma_4b_it","llama3_8b_instruct_ngc_v1"]
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700);t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def helper(model):
 fn="gpu_engineering_probe_llama3_ngc_v1.py" if model.startswith("llama") else "gpu_engineering_probe.py";p=Path(__file__).parent/fn
 expected="0cc34f9c74b277dcf2d67e17724a22a12fb51af6925997280da4f429a715ec12" if model.startswith("llama") else "69bd40f9aa34a38667be10e886e2f91b563d48b361112a8db45b936390cbc86c"
 assert sha(p)==expected
 spec=importlib.util.spec_from_file_location("qualified_source",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def admission():
 mem=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
 own=Path("/proc/self/cgroup").read_text().strip().split(":")[-1];cg=Path("/sys/fs/cgroup")/own.lstrip("/")
 effective=mem;parent=cg.parent
 while parent!=Path("/sys/fs/cgroup"):
  maximum=(parent/"memory.max").read_text().strip()
  if maximum!="max":effective=min(effective,max(0,int(maximum)-int((parent/"memory.current").read_text())))
  parent=parent.parent
 reserve=max(4*GIB,int(.25*mem));extra=0;others=[]
 units=subprocess.run(["systemctl","--user","list-units","--state=active","--plain","--no-legend","aki-postslot-*"],capture_output=True,text=True,check=True).stdout.splitlines()
 for row in units:
  name=row.split()[0]
  if name in own:continue
  props=subprocess.run(["systemctl","--user","show",name,"-p","MemoryMax","-p","ControlGroup"],capture_output=True,text=True,check=True).stdout
  p=dict(v.split("=",1) for v in props.splitlines() if "=" in v);assert p["MemoryMax"].isdigit(),"Unbounded concurrent AKI unit"
  stat=dict(v.split() for v in (Path("/sys/fs/cgroup")/p["ControlGroup"].lstrip("/")/"memory.stat").read_text().splitlines())
  growth=max(0,int(p["MemoryMax"])-sum(int(stat.get(k,0)) for k in ["anon","kernel","shmem"]));extra+=growth;others.append({"unit":name,"reserved_growth_bytes":growth})
 gpu=subprocess.run(["nvidia-smi","--id=0","--query-gpu=memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
 apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.strip()
 margin=effective-reserve-extra
 return {"allowed":margin>=16*GIB and int(gpu.stdout.strip())*1024**2>=20*GIB and not apps,"time":now(),"MemAvailable_bytes":mem,"shared_effective_available_bytes":effective,"OS_reserve_bytes":reserve,"other_AKI_reservations":others,"remaining_bytes":margin,"required_host_loading_bytes":16*GIB,"free_GPU_MiB":int(gpu.stdout.strip()),"existing_GPU_compute_PIDs":apps}
def bundle(model,stage,line,fit,kind):
 recipe=read(F/"FORMAL_TRAINING_CONTRACT.json");assert recipe["worker_code_sha256"]==sha(__file__)
 for p,h in recipe["locked_receipts"].items():assert sha(p)==h
 m=helper(model);b=m.bundle(model);meta={}
 with (D/"labels_and_partitions.csv").open() as stream:meta={v["record_key"]:v for v in csv.DictReader(stream)}
 bodies={v["record_key"]:v["clinical_text"] for v in map(json.loads,(D/(stage+"_clinical_text.jsonl")).read_text().splitlines())}
 assert set(meta)==set(bodies) and len(meta)==2428
 folds=read(D/"ORIGINAL_OOF_FOLDS.json")
 if kind=="before":train=[];pred=list(meta);dest=F/"before"/model/stage
 elif kind=="pilot":train=b["record_keys"];pred=b["record_keys"];dest=F/"pilots"/model
 else:
  part=line+"_partition_v2";dev=[k for k,v in meta.items() if v[part]=="development"];val=[k for k,v in meta.items() if v[part]=="validation"]
  assert (len(dev),len(val))==({"geography":(1270,1158),"temporal":(1414,1011)}[line])
  if fit=="final":train=dev;pred=val
  else:
   f=next(v for v in folds if v["line"]==line and v["fold"]==int(fit));train=f["train_record_keys"];pred=f["assessment_record_keys"]
   assert set(train)|set(pred)==set(dev) and not(set(train)&set(pred))
  dest=F/"fits"/model/(line+"_"+stage)/str(fit)
 if kind=="fit":
  assert not ({meta[k]["group_hash"] for k in train}&{meta[k]["group_hash"] for k in pred})
  assert all(meta[k][line+"_partition_v2"]=="development" for k in train)
 assert len(train)==len(set(train)) and len(pred)==len(set(pred))
 fields={"model":model,"stage":stage,"line":line,"fit":str(fit),"kind":kind,"contract_sha256":sha(F/"FORMAL_TRAINING_CONTRACT.json"),"source_signature":b["signature"],"train_record_keys":train,"prediction_record_keys":pred,"code_sha256":sha(__file__)}
 sig=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
 return m,b,meta,bodies,train,pred,dest,fields,sig,recipe
def worker(a):
 assert os.environ.get("AKI_LLM_FORMAL_LOCK_FD");os.fstat(int(os.environ["AKI_LLM_FORMAL_LOCK_FD"]))
 m,b,meta,bodies,train,pred,dest,fields,sig,recipe=bundle(a.model,a.stage,a.line,a.fit,a.kind)
 dest.mkdir(parents=True,exist_ok=True,mode=0o700)
 if (dest/"COMPLETE.json").exists():
  done=read(dest/"COMPLETE.json");assert done["signature"]==sig
  for p,h in done["files"].items():assert sha(p)==h
  return 0
 gate=admission();save(dest/"RESOURCE_ADMISSION.json",gate)
 if not gate["allowed"]:return 20
 save(dest/"SIGNATURE.json",{"signature":sig,"fields":fields})
 import importlib.metadata as md
 assert {p:md.version(p) for p in m.EXPECTED_VERSIONS}==m.EXPECTED_VERSIONS
 import numpy as np,torch
 from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer,BitsAndBytesConfig
 from peft import prepare_model_for_kbit_training,LoraConfig,TaskType,get_peft_model
 torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False
 props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction(12*GIB/props.total_memory,0)
 local=Path(b["asset"]["local_path"])
 for p,h in b["asset"]["files"].items():assert sha(p)==h
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)
 cls=AutoModelForImageTextToText if a.model=="medgemma_4b_it" else AutoModelForCausalLM
 begin=time.monotonic()
 model,info=cls.from_pretrained(local,local_files_only=True,trust_remote_code=False,weights_only=True,use_safetensors=a.model!="biomedlm_2_7b",quantization_config=quant,device_map={"":0},dtype=torch.float32,attn_implementation="eager",low_cpu_mem_usage=True,output_loading_info=True)
 assert not any(info.get(k) for k in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"]);model.config.use_cache=False
 save(dest/"CHECKPOINT_LOAD_KEYS.json",{"status":"PASS_ALL_BASE_MODEL_KEYS_LOADED","information":info,"source_signature":b["signature"]})
 tok=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False,use_fast=True,fix_mistral_regex=False);tok.padding_side="left"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 label_ids=b["tokenizer_receipt"]["label_token_ids"];class_ids=torch.tensor(label_ids,device="cuda")
 allkeys=list(dict.fromkeys(train+pred))
 prompts={k:bodies[k]+"\nAnswer:" if a.model=="biomedlm_2_7b" else tok.apply_chat_template([{"role":"user","content":bodies[k]}],tokenize=False,add_generation_prompt=True) for k in allkeys}
 tokens={k:tok.encode(prompts[k],add_special_tokens=False) for k in allkeys}
 maxlen=b["tokenizer_receipt"]["stages"][a.stage]["prefix_tokens_max"]
 assert max(map(len,tokens.values()))<=maxlen
 assert tok.encode("0",add_special_tokens=False)==[label_ids[0]] and tok.encode("1",add_special_tokens=False)==[label_ids[1]]
 def packed(keys):
  lengths=[len(tokens[k]) for k in keys];width=max(lengths)
  ids=torch.full((len(keys),width),tok.pad_token_id,dtype=torch.long,device="cuda");mask=torch.zeros_like(ids)
  for row,k in enumerate(keys):ids[row,-lengths[row]:]=torch.tensor(tokens[k],device="cuda");mask[row,-lengths[row]:]=1
  pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
  return {"input_ids":ids,"attention_mask":mask,"position_ids":pos}
 def logits(keys):return model(**packed(keys),use_cache=False,return_dict=True,logits_to_keep=1).logits[:,-1,:].float()
 config=recipe["model_hardware"][a.model];completed=0;training_trace=[];nsteps=0
 if train:
  targets=[n for n,v in model.named_modules() if (n.endswith(".c_attn") if a.model=="biomedlm_2_7b" else (n.endswith((".q_proj",".v_proj")) and "vision" not in n and (a.model!="medgemma_4b_it" or "language_model" in n)))]
  expected=int(model.config.n_layer) if a.model=="biomedlm_2_7b" else 2*int(model.config.text_config.num_hidden_layers if a.model=="medgemma_4b_it" else model.config.num_hidden_layers)
  assert len(targets)==expected
  model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=0.,target_modules=targets,bias="none"))
  parameters={n:p for n,p in model.named_parameters() if p.requires_grad};assert parameters and all("lora_" in n and "vision" not in n for n in parameters)
  initial_adapter={n:p.detach().cpu().clone() for n,p in parameters.items()}
  optimizer=torch.optim.AdamW(parameters.values(),lr=1e-4,weight_decay=.01)
  order=np.random.default_rng(42).permutation(len(train)).tolist();ordered=[train[i] for i in order];nsteps=(len(ordered)+3)//4
  checkpoint=dest/"CHECKPOINT_LATEST.json"
  if checkpoint.exists():
   receipt=read(checkpoint);assert receipt["signature"]==sig and sha(receipt["path"])==receipt["sha256"]
   state=torch.load(receipt["path"],map_location="cpu",weights_only=True)
   assert state["signature"]==sig and state["training_order"]==ordered
   with torch.no_grad():
    for n,p in parameters.items():p.copy_(state["adapter_parameters"][n].to(p.device))
   optimizer.load_state_dict(state["optimizer"]);completed=state["completed_updates"]
   torch.set_rng_state(state["torch_RNG"]);torch.cuda.set_rng_state_all(state["CUDA_RNG"]);random.setstate(state["python_RNG"]);training_trace=state["training_trace"]
   del state
  def checkpoint_save(step):
   path=dest/f"checkpoint_{step:06d}.pt";tmp=path.with_suffix(".tmp")
   state={"signature":sig,"training_order":ordered,"completed_updates":step,"adapter_parameters":{n:p.detach().cpu() for n,p in parameters.items()},"optimizer":optimizer.state_dict(),"torch_RNG":torch.get_rng_state(),"CUDA_RNG":torch.cuda.get_rng_state_all(),"python_RNG":random.getstate(),"training_trace":training_trace}
   torch.save(state,tmp);os.replace(tmp,path);digest=sha(path)
   save(checkpoint,{"signature":sig,"completed_updates":step,"path":str(path),"sha256":digest})
   del state
  training_start=time.monotonic()
  for step in range(completed,nsteps):
   batch=ordered[step*4:(step+1)*4];micro=config["training_microbatch"]
   model.train();optimizer.zero_grad(set_to_none=True);loss_value=0.;step_start=time.monotonic()
   for start in range(0,len(batch),micro):
    keys=batch[start:start+micro];score=logits(keys);target=torch.tensor([label_ids[int(meta[k]["y"])] for k in keys],device="cuda")
    loss=torch.nn.functional.cross_entropy(score,target)*(len(keys)/len(batch));assert torch.isfinite(loss)
    loss_value+=float(loss.detach().cpu());loss.backward();del score,target,loss
   gradients=[p.grad for p in parameters.values() if p.grad is not None]
   assert gradients and all(torch.isfinite(g).all() for g in gradients)
   norm=max(float(g.detach().abs().max().cpu()) for g in gradients);assert norm>=0
   optimizer.step();assert all(torch.isfinite(p).all() for p in parameters.values());del gradients
   completed=step+1;training_trace.append({"update":completed,"examples":len(batch),"loss_operator_diagnostic":loss_value,"gradient_max_abs":norm,"seconds":time.monotonic()-step_start})
   if completed%25==0 or completed==nsteps:checkpoint_save(completed)
   if completed%10==0 or completed==nsteps:save(dest/"PROGRESS.json",{"status":"TRAINING_FIXED_ONE_PASS","time":now(),"signature":sig,"completed_updates":completed,"total_updates":nsteps,"completed_training_examples":min(4*completed,len(train)),"formal_fit":a.kind=="fit"})
  assert completed==nsteps and any(v["gradient_max_abs"]>0 for v in training_trace)
  maximum_parameter_change=max(float((p.detach().cpu()-initial_adapter[n]).abs().max()) for n,p in parameters.items());assert maximum_parameter_change>0
  adapter=dest/("PILOT_ADAPTER_NEVER_FOR_FORMAL" if a.kind=="pilot" else "adapter")
  model.save_pretrained(adapter,safe_serialization=True,save_embedding_layers=False)
  save(dest/"TRAINING_ACCEPTANCE.json",{"status":"PASS_FIXED_ONE_PASS_DEVELOPMENT_ONLY_PEFT","signature":sig,"model":a.model,"line":a.line,"stage":a.stage,"fit":a.fit,"n_training_records":len(train),"n_training_groups":len({meta[k]["group_hash"] for k in train}),"completed_updates":completed,"maximum_parameter_change":maximum_parameter_change,"training_passes":1,"training_order_sha256":hashlib.sha256("\n".join(ordered).encode()).hexdigest(),"training_seconds_this_process":time.monotonic()-training_start,"all_training_keys_are_original_development":True,"no_evaluation_early_stopping_or_search":True,"engineering_adapter_loaded":False,"formal_fit":a.kind=="fit","adapter_files":{str(p):sha(p) for p in adapter.rglob("*") if p.is_file()}})
  optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
 # Score without outcome labels, fixed class ordering. Durable chunks resume without recomputing valid predictions.
 model.eval();batch_size=config["inference_batch"];chunks=[]
 for start in range(0,len(pred),64):
  keys=pred[start:start+64];path=dest/f"PREDICTION_CHUNK_{start:05d}.npz";receipt=path.with_suffix(".json")
  if receipt.exists():
   rec=read(receipt);assert rec["signature"]==sig and sha(path)==rec["sha256"]
   with np.load(path,allow_pickle=False) as z:assert z["record_key"].tolist()==keys
   chunks.append(path);continue
  probabilities=[];scores=[]
  for offset in range(0,len(keys),batch_size):
   chosen=keys[offset:offset+batch_size]
   with torch.no_grad():
    values=logits(chosen);pair=values.index_select(-1,class_ids);prob=torch.softmax(pair,-1)[:,1];raw=pair[:,1]-pair[:,0]
    assert torch.isfinite(values).all() and torch.isfinite(prob).all() and torch.isfinite(raw).all()
    probabilities.extend(prob.cpu().tolist());scores.extend(raw.cpu().tolist())
   del values,pair,prob,raw
  tmp=path.with_suffix(".tmp")
  with tmp.open("wb") as stream:np.savez_compressed(stream,record_key=np.asarray(keys,dtype=str),prob_raw=np.asarray(probabilities,float),binary_log_odds=np.asarray(scores,float))
  os.replace(tmp,path);save(receipt,{"signature":sig,"sha256":sha(path),"n_records":len(keys),"no_outcome_labels_used_for_scoring":True});chunks.append(path)
  save(dest/"PREDICTION_PROGRESS.json",{"time":now(),"signature":sig,"completed":min(start+64,len(pred)),"total":len(pred)})
 raw=[];odds=[];ordered_keys=[]
 for path in chunks:
  with np.load(path,allow_pickle=False) as z:ordered_keys.extend(z["record_key"].tolist());raw.extend(z["prob_raw"].tolist());odds.extend(z["binary_log_odds"].tolist())
 assert ordered_keys==pred
 result=dest/"RAW_PREDICTIONS.npz"
 with result.open("wb") as stream:np.savez_compressed(stream,record_key=np.asarray(pred,dtype=str),prob_raw=np.asarray(raw,float),binary_log_odds=np.asarray(odds,float))
 files={str(result):sha(result),str(dest/"SIGNATURE.json"):sha(dest/"SIGNATURE.json"),str(dest/"CHECKPOINT_LOAD_KEYS.json"):sha(dest/"CHECKPOINT_LOAD_KEYS.json")}
 if train:
  files[str(dest/"TRAINING_ACCEPTANCE.json")]=sha(dest/"TRAINING_ACCEPTANCE.json")
  files.update(read(dest/"TRAINING_ACCEPTANCE.json")["adapter_files"])
 save(dest/"COMPLETE.json",{"status":"PASS_FIXED_CHECKPOINT_PROBABILITY_BANK_AND_ONE_PASS_PEFT" if train else "PASS_ORIGINAL_CHECKPOINT_PRE_TASK_PEFT_PROBABILITY_BANK","time":now(),"signature":sig,"model":a.model,"stage":a.stage,"line":a.line,"fit":a.fit,"kind":a.kind,"files":files,"n_training_records":len(train),"n_prediction_records":len(pred),"clinical_performance_metrics_computed":False,"formal_fit":a.kind=="fit","peak_GPU_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_GPU_reserved_bytes":torch.cuda.max_memory_reserved(),"peak_host_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"elapsed_seconds":time.monotonic()-begin})
 return 0
def verify_pilot_checkpoint(model):
 import torch
 p=F/"pilots"/model/"CHECKPOINT_LATEST.json";r=read(p);assert sha(r["path"])==r["sha256"]
 state=torch.load(r["path"],map_location="cpu",weights_only=True)
 assert state["signature"]==r["signature"] and state["completed_updates"]==1 and len(state["training_order"])==4
 assert state["adapter_parameters"] and all(torch.isfinite(t).all() for t in state["adapter_parameters"].values())
 assert state["optimizer"]["state"] and all(t.dtype==torch.uint8 for t in state["CUDA_RNG"])
 save(F/"pilots"/model/"CHECKPOINT_RELOAD_ACCEPTANCE.json",{"status":"PASS_NEW_PROCESS_OWN_CHECKPOINT_WEIGHTS_ONLY_LOAD_STATE_AND_RNG","signature":r["signature"],"checkpoint_sha256":r["sha256"],"completed_updates":1,"formal_fit":False})
def controller():
 F.mkdir(parents=True,exist_ok=True,mode=0o700)
 with (O/"engineering/GPU_ENGINEERING_OWNER.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  start=time.monotonic();recipe=read(F/"FORMAL_TRAINING_CONTRACT.json");assert recipe["worker_code_sha256"]==sha(__file__)
  jobs=[]
  for model in MODELS:jobs.append((model,"endop","geography","pilot","pilot"))
  for model in MODELS:
   for stage in ["preop","endop"]:jobs.append((model,stage,"all","before","before"))
   for line in ["geography","temporal"]:
    for stage in ["preop","endop"]:
     for fit in ["0","1","2","final"]:jobs.append((model,stage,line,fit,"fit"))
  accepted=[]
  for model,stage,line,fit,kind in jobs:
   m,b,meta,bodies,train,pred,dest,fields,sig,c=bundle(model,stage,line,fit,kind)
   if (dest/"COMPLETE.json").exists():
    done=read(dest/"COMPLETE.json");assert done["signature"]==sig
    for p,h in done["files"].items():assert sha(p)==h
   else:
    gate=admission();save(F/"LATEST_RESOURCE_ADMISSION.json",gate)
    if not gate["allowed"]:
     save(F/"STATUS.json",{"status":"WAIT_RESOURCE_SAVED_VALID_RESULTS_PRESERVED","time":now(),"next_task":[model,stage,line,fit,kind],"gate":gate});return
    env=os.environ.copy();env.update(AKI_LLM_FORMAL_LOCK_FD=str(lock.fileno()),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",USE_TORCH="1",USE_TF="0",USE_FLAX="0",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    dest.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (dest/"worker.log").open("ab") as log:
     child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--worker","--model",model,"--stage",stage,"--line",line,"--fit",fit,"--kind",kind],env=env,stdout=log,stderr=log,pass_fds=(lock.fileno(),))
     save(F/"STATUS.json",{"status":"RUNNING_PILOT" if kind=="pilot" else "RUNNING_FIXED_FORMAL_BENCHMARK","time":now(),"task":[model,stage,line,fit,kind],"worker_PID":child.pid,"accepted_formal_fits":sum(1 for v in accepted if v["kind"]=="fit"),"accepted_before_banks":sum(1 for v in accepted if v["kind"]=="before"),"controller_PID":os.getpid(),"contract_sha256":sha(F/"FORMAL_TRAINING_CONTRACT.json")})
     try:rc=child.wait(timeout=recipe["maximum_per_worker_seconds"])
     except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned finite formal worker exceeded fixed wallclock; checkpoint preserved")
    if rc==20:return
    assert rc==0 and (dest/"COMPLETE.json").exists(),"Inspect model/task worker failure; do not duplicate"
   if kind=="pilot" and not (dest/"CHECKPOINT_RELOAD_ACCEPTANCE.json").exists():
    v=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--verify-checkpoint",model],capture_output=True,text=True,timeout=180);assert v.returncode==0
   accepted.append({"model":model,"stage":stage,"line":line,"fit":fit,"kind":kind,"receipt":str(dest/"COMPLETE.json"),"sha256":sha(dest/"COMPLETE.json")})
   save(F/"ACCEPTED_TASKS.json",{"time":now(),"tasks":accepted})
   assert time.monotonic()-start<recipe["maximum_controller_seconds"]
  save(F/"TRAINING_AND_PREDICTION_ACCEPTANCE.json",{"status":"PASS_48_FORMAL_ONE_PASS_FITS_6_ORIGINAL_CHECKPOINT_BANKS_ENGINEERING_PILOTS_SEPARATE","time":now(),"formal_fits":sum(v["kind"]=="fit" for v in accepted),"original_checkpoint_banks":sum(v["kind"]=="before" for v in accepted),"engineering_pilots":sum(v["kind"]=="pilot" for v in accepted),"contract_sha256":sha(F/"FORMAL_TRAINING_CONTRACT.json"),"tasks":accepted,"calibration_thresholds_and_statistical_comparison_complete":False})
  save(F/"STATUS.json",{"status":"COMPLETE_TRAINING_AND_PREDICTION_BANKS_STATISTICAL_COMPARISON_REQUIRED","time":now(),"formal_fits":48,"before_banks":6,"owned_GPU_workers":0,"full_supplement_complete":False})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--worker",action="store_true");p.add_argument("--model",choices=MODELS);p.add_argument("--stage");p.add_argument("--line");p.add_argument("--fit");p.add_argument("--kind",choices=["pilot","before","fit"]);p.add_argument("--verify-checkpoint",choices=MODELS);a=p.parse_args()
 try:
  if a.verify_checkpoint:verify_pilot_checkpoint(a.verify_checkpoint)
  elif a.worker:raise SystemExit(worker(a))
  else:controller()
 except SystemExit:raise
 except BaseException as e:
  dest=F/"FAILURE.json"
  if a.worker:
   try:dest=bundle(a.model,a.stage,a.line,a.fit,a.kind)[6]/"FAILURE.json"
   except Exception:pass
  save(dest,{"time":now(),"error_class":type(e).__name__,"stack":[{"file":str(Path(v.filename)),"line":v.lineno,"function":v.name} for v in traceback.extract_tb(e.__traceback__)[-10:]],"existing_valid_results_and_checkpoints_preserved":True})
  raise SystemExit("Finite LLM benchmark stopped: "+type(e).__name__+"; inspect saved diagnostics")
