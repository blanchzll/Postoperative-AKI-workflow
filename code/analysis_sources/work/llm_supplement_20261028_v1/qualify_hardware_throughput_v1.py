"""Finite hardware batch/accumulation throughput qualification on four development patients.
No formal fit, no validation performance, no saved adapter for subsequent science.
"""
from pathlib import Path
import argparse,datetime,fcntl,gc,hashlib,importlib.util,json,os,resource,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[2];O=R/"outputs/llm_supplement_20261028_v1";E=O/"engineering";T=E/"throughput_v1"
MODELS=["biomedlm_2_7b","medgemma_4b_it","llama3_8b_instruct_ngc_v1"]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700);t=p.with_name(p.name+".tmp")
 t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def helper(model):
 fn="gpu_engineering_probe_llama3_ngc_v1.py" if model.startswith("llama") else "gpu_engineering_probe.py"
 p=Path(__file__).parent/fn
 expected="0cc34f9c74b277dcf2d67e17724a22a12fb51af6925997280da4f429a715ec12" if model.startswith("llama") else "69bd40f9aa34a38667be10e886e2f91b563d48b361112a8db45b936390cbc86c"
 assert sha(p)==expected
 spec=importlib.util.spec_from_file_location("qualified_model_helper",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def worker(model_id):
 m=helper(model_id);b=m.bundle(model_id);out=T/model_id;out.mkdir(parents=True,exist_ok=True,mode=0o700)
 assert os.environ.get("AKI_THROUGHPUT_LOCK_FD");os.fstat(int(os.environ["AKI_THROUGHPUT_LOCK_FD"]))
 ac=E/model_id/"GPU_ENGINEERING_ACCEPTANCE.json";a=read(ac)
 assert a["signature"]==b["signature"] and not a["formal_training_started"]
 gate=m.admission();save(out/"RESOURCE_ADMISSION.json",gate)
 if not gate["allowed"]:return 20
 import importlib.metadata as md
 assert {p:md.version(p) for p in m.EXPECTED_VERSIONS}==m.EXPECTED_VERSIONS
 import random,numpy as np,torch
 from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer,BitsAndBytesConfig
 from peft import prepare_model_for_kbit_training,LoraConfig,TaskType,get_peft_model
 torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.backends.cuda.matmul.allow_tf32=False
 props=torch.cuda.get_device_properties(0)
 torch.cuda.set_per_process_memory_fraction(12*1024**3/props.total_memory,0)
 local=Path(b["asset"]["local_path"])
 for f,h in b["asset"]["files"].items():assert sha(f)==h
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)
 cls=AutoModelForImageTextToText if model_id=="medgemma_4b_it" else AutoModelForCausalLM
 started=time.monotonic()
 model,info=cls.from_pretrained(local,local_files_only=True,trust_remote_code=False,weights_only=True,use_safetensors=model_id!="biomedlm_2_7b",quantization_config=quant,device_map={"":0},dtype=torch.float32,attn_implementation="eager",low_cpu_mem_usage=True,output_loading_info=True)
 assert not any(info.get(k) for k in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"])
 model.config.use_cache=False
 # 4.57.6's local-config heuristic misidentifies Llama as Mistral.
 # Explicit False preserves the independently qualified original Meta encoding.
 tok=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False,use_fast=True,fix_mistral_regex=False)
 tok.padding_side="left"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 labels=b["tokenizer_receipt"]["label_token_ids"]
 bodies=b["bodies"]["endop"]
 prompts=[s+"\nAnswer:" if model_id=="biomedlm_2_7b" else tok.apply_chat_template([{"role":"user","content":s}],tokenize=False,add_generation_prompt=True) for s in bodies]
 maxlen=b["tokenizer_receipt"]["stages"]["endop"]["prefix_tokens_max"]
 encoded=tok(prompts,padding="max_length",max_length=maxlen,truncation=False,add_special_tokens=False,return_tensors="pt")
 assert encoded["input_ids"].shape==(4,maxlen)
 for j,prompt in enumerate(prompts):
  assert encoded["input_ids"][j][encoded["attention_mask"][j].bool()].tolist()==tok.encode(prompt,add_special_tokens=False)
 input_ids=encoded["input_ids"].to("cuda");masks=encoded["attention_mask"].to("cuda")
 positions=masks.long().cumsum(-1)-1;positions.masked_fill_(masks==0,0)
 class_ids=torch.tensor(labels,device="cuda")
 y=torch.tensor([labels[v] for v in b["labels"]],device="cuda")
 def logits(indices):
  return model(input_ids=input_ids[indices],attention_mask=masks[indices],position_ids=positions[indices],use_cache=False,return_dict=True,logits_to_keep=1).logits[:,-1,:].float()
 def probs(indices):
  with torch.no_grad():return torch.softmax(logits(indices).index_select(-1,class_ids),-1)[:,1].cpu()
 targets=[n for n,v in model.named_modules() if (n.endswith(".c_attn") if model_id=="biomedlm_2_7b" else (n.endswith((".q_proj",".v_proj")) and "vision" not in n and (model_id!="medgemma_4b_it" or "language_model" in n)))]
 expected=int(model.config.n_layer) if model_id=="biomedlm_2_7b" else 2*int(model.config.text_config.num_hidden_layers if model_id=="medgemma_4b_it" else model.config.num_hidden_layers)
 assert len(targets)==expected and targets
 model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
 model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=0.0,target_modules=targets,bias="none"))
 params={n:p for n,p in model.named_parameters() if p.requires_grad}
 assert params and all("lora_" in n and "vision" not in n for n in params)
 initial={n:p.detach().cpu().clone() for n,p in params.items()}
 results=[];blocked=False
 for micro in [1,2,4]:
  if blocked:
   results.append({"microbatch":micro,"status":"SKIPPED_AFTER_SMALLER_BATCH_EXCEEDED_ALLOCATOR_BUDGET"});continue
  with torch.no_grad():
   for n,p in params.items():p.copy_(initial[n].to(p.device))
  model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=.01);times=[];losses=[]
  try:
   for update in range(3):
    model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.monotonic();loss_sum=0.
    for start in range(0,4,micro):
     idx=list(range(start,min(start+micro,4)));score=logits(idx)
     loss=torch.nn.functional.cross_entropy(score,y[idx])*(len(idx)/4)
     assert torch.isfinite(loss);loss_sum+=float(loss.detach().cpu());loss.backward()
     del score,loss
    gradients=[p.grad for p in params.values() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert max(float(g.detach().abs().max().cpu()) for g in gradients)>0
    optimizer.step();assert all(torch.isfinite(p).all() for p in params.values())
    torch.cuda.synchronize();times.append(time.monotonic()-t0);losses.append(loss_sum);del gradients
   results.append({"microbatch":micro,"gradient_accumulation":4//micro,"effective_batch":4,"status":"PASS_FINITE_GRADIENTS_UPDATES","warmup_update_seconds":times[0],"timed_update_seconds":times[1:],"timed_examples_per_second":8/sum(times[1:]),"training_loss_operator_diagnostic_only":losses,"peak_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_VRAM_bytes":torch.cuda.max_memory_reserved()})
  except torch.cuda.OutOfMemoryError:
   blocked=True;results.append({"microbatch":micro,"status":"CUDA_ALLOCATOR_BUDGET_EXCEEDED_BOUNDED_TRIAL","peak_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_VRAM_bytes":torch.cuda.max_memory_reserved()})
  finally:
   model.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
  save(out/"PROGRESS.json",{"time":now(),"model":model_id,"training_candidates":results,"formal_training_started":False})
 valid=[v for v in results if v["status"]=="PASS_FINITE_GRADIENTS_UPDATES"];assert valid
 # A single fixed final engineering state is used for every inference batch, never clinical metrics.
 model.eval();reference=probs([0,1,2,3]);inference=[]
 for batch in [1,2,4,8]:
  torch.cuda.reset_peak_memory_stats()
  try:
   idx=[i%4 for i in range(batch)]
   for trial in range(2):
    torch.cuda.synchronize();t0=time.monotonic();p=probs(idx);torch.cuda.synchronize();elapsed=time.monotonic()-t0
   delta=float((p-reference[idx]).abs().max());assert delta<=1e-4
   inference.append({"batch":batch,"status":"PASS_BATCH_PARITY","timed_examples_per_second":batch/elapsed,"max_singleton_probability_difference":delta,"peak_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_VRAM_bytes":torch.cuda.max_memory_reserved()})
   del p
  except torch.cuda.OutOfMemoryError:
   inference.append({"batch":batch,"status":"CUDA_ALLOCATOR_BUDGET_EXCEEDED_BOUNDED_TRIAL"});break
  finally:gc.collect();torch.cuda.empty_cache()
 winner=max(valid,key=lambda z:z["timed_examples_per_second"])
 infer=max([v for v in inference if v["status"]=="PASS_BATCH_PARITY"],key=lambda z:z["timed_examples_per_second"])
 save(out/"THROUGHPUT_ACCEPTANCE.json",{"status":"PASS_FINITE_HARDWARE_THROUGHPUT_AND_BATCH_ACCUMULATION_ONLY","time":now(),"model":model_id,"source_signature":b["signature"],"GPU_engineering_acceptance_sha256":sha(ac),"code_sha256":sha(__file__),"padded_longest_current_endop_prefix":maxlen,"existing_development_patient_groups":4,"training_candidates":results,"inference_candidates":inference,"selected_training_microbatch":winner["microbatch"],"selected_gradient_accumulation":4//winner["microbatch"],"fixed_effective_batch":4,"selected_inference_batch":infer["batch"],"host_peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"elapsed_seconds":time.monotonic()-started,"selection_uses_clinical_validation_metrics":False,"formal_training_started":False,"saved_adapter":False})
 return 0
def controller():
 T.mkdir(parents=True,exist_ok=True,mode=0o700)
 with (E/"GPU_ENGINEERING_OWNER.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for model in MODELS:
   p=T/model/"THROUGHPUT_ACCEPTANCE.json"
   if p.exists():assert read(p)["code_sha256"]==sha(__file__);continue
   env=os.environ.copy();env.update({"AKI_THROUGHPUT_LOCK_FD":str(lock.fileno()),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","USE_TORCH":"1","USE_TF":"0","USE_FLAX":"0","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"})
   (T/model).mkdir(parents=True,exist_ok=True,mode=0o700)
   with (T/model/"worker.log").open("ab") as log:
    child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",model],env=env,stdout=log,stderr=log,pass_fds=(lock.fileno(),))
    save(T/"STATUS.json",{"status":"RUNNING_FINITE_HARDWARE_QUALIFICATION","time":now(),"model":model,"worker_PID":child.pid,"formal_training_started":False})
    try:code=child.wait(timeout=2400)
    except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError("Owned finite throughput worker budget exceeded")
   if code==20:
    save(T/"STATUS.json",{"status":"WAIT_RESOURCE_NO_RETRY_OF_IDENTICAL_COMMAND","time":now(),"model":model});return
   assert code==0 and p.exists(),"Inspect model-specific finite qualification failure"
  save(T/"STATUS.json",{"status":"COMPLETE_THREE_HARDWARE_THROUGHPUT_QUALIFICATIONS_FORMAL_BUDGET_NOT_YET_LOCKED","time":now(),"receipts":{str(T/model/"THROUGHPUT_ACCEPTANCE.json"):sha(T/model/"THROUGHPUT_ACCEPTANCE.json") for model in MODELS},"formal_training_started":False})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--worker",choices=MODELS);a=p.parse_args()
 try:
  if a.worker:raise SystemExit(worker(a.worker))
  controller()
 except SystemExit:raise
 except BaseException as e:
  dest=T/(a.worker or "controller")/"FAILURE.json"
  save(dest,{"time":now(),"error_class":type(e).__name__,"stack":[{"file":str(Path(v.filename)),"line":v.lineno,"function":v.name} for v in traceback.extract_tb(e.__traceback__)[-10:]],"formal_training_started":False})
  raise SystemExit("Finite throughput qualification stopped: "+type(e).__name__)
