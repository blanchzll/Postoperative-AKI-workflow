"""Finite real-data CUDA/LoRA training and fresh-process reload qualification.
Engineering only: four existing development patients; never a formal fit or clinical evaluation.
"""
from pathlib import Path
import argparse,csv,datetime,fcntl,hashlib,json,os,random,resource,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[2]
O=R/"outputs/llm_supplement_20261028_v1"
D=O/"data_preparation_v1"
E=O/"engineering"
GIB=1024**3
CONTRACT={"version":1,"seed":42,"engineering_steps":6,"training_stage":"endop","lora_rank":8,"lora_alpha":16,"lora_dropout":0.0,"optimizer":"AdamW","learning_rate":0.0001,"weight_decay":0.01,"loss":"full_vocabulary_cross_entropy_for_one_first_answer_token","quantization":"NF4_double_quant","compute_dtype":"float32","host_loading_budget_bytes":16*GIB,"new_GPU_allocator_budget_bytes":12*GIB,"primary_GPU_headroom_bytes":8*GIB,"batch_probability_tolerance":0.0001,"reload_probability_tolerance":0.00001,"maximum_worker_seconds":1200,"engineering_adapters_must_not_initialize_formal_fits":True,"formal_fits_started":False}
EXPECTED_VERSIONS={"torch":"2.6.0+cu118","transformers":"4.57.6","peft":"0.18.0","bitsandbytes":"0.48.1"}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");os.replace(t,p)
def read(p):return json.loads(Path(p).read_text())
def unit_properties(unit):
    r=subprocess.run(["systemctl","--user","show",unit,"--property=ActiveState,MainPID,MemoryCurrent,MemoryMax,MemorySwapMax,ControlGroup"],capture_output=True,text=True)
    if r.returncode:raise RuntimeError("Cannot verify task-scoped systemd resource properties")
    return dict(l.split("=",1) for l in r.stdout.splitlines() if "=" in l)
def ancestor_headroom(main):
    # Budget shared ancestors; a probe's own 16GiB cap is independently enforced at launch.
    root=Path("/sys/fs/cgroup")
    current=Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    branches=[current,main.get("ControlGroup","")]
    paths=set()
    for branch in branches:
        if not branch:continue
        path=root/branch.lstrip("/")
        path=path.parent
        while path!=root:
            assert root in path.parents
            paths.add(path);path=path.parent
    limits=[]
    for path in sorted(paths):
        maxfile=path/"memory.max";curfile=path/"memory.current"
        if not maxfile.exists() or not curfile.exists():
            raise RuntimeError("Cannot verify shared cgroup memory bound")
        maximum=maxfile.read_text().strip();used=int(curfile.read_text().strip())
        if maximum!="max":
            limits.append({"cgroup":str(path.relative_to(root)),"maximum_bytes":int(maximum),"current_bytes":used,"remaining_bytes":max(0,int(maximum)-used)})
    return limits
def admission():
    reason=[];available=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))*1024
    main=unit_properties("aki-postslot-full-reanalysis-20261025.service")
    growth=0
    if main.get("ActiveState")=="active":
        p=R/"outputs/common6_strict_continuation_20261024_v1/status.json"
        if time.time()-p.stat().st_mtime>180:reason.append("main_resource_status_stale")
        else:growth=int(read(p)["growth_reserved_bytes"])
    own=(Path("/proc/self/cgroup").read_text().strip().split(":")[-1])
    own_service=next((v for v in own.split("/") if v.endswith(".service") and v.startswith("aki-postslot-llm-gpu-qual")),"")
    listed=subprocess.run(["systemctl","--user","list-units","--state=active","--plain","--no-legend","aki-postslot-*"],capture_output=True,text=True).stdout.splitlines()
    aux=[];extra=0
    for row in listed:
        unit=row.split()[0]
        if unit in ["aki-postslot-full-reanalysis-20261025.service",own_service]:continue
        p=unit_properties(unit)
        if p.get("ActiveState")!="active":continue
        maximum=p.get("MemoryMax","infinity");current=p.get("MemoryCurrent","0")
        if not maximum.isdigit() or not current.isdigit():
            reason.append("unbounded_active_AKI_auxiliary:"+unit);continue
        # MemAvailable includes reclaimable file cache. Subtracting full MemoryCurrent
        # here would incorrectly treat charged download cache as future anonymous-RAM reserve.
        stat_path=Path("/sys/fs/cgroup")/p.get("ControlGroup","").lstrip("/")/"memory.stat"
        if not p.get("ControlGroup") or not stat_path.exists():
            reason.append("auxiliary_anonymous_memory_telemetry_unavailable:"+unit);continue
        memory_stat=dict(row.split() for row in stat_path.read_text().splitlines())
        nonreclaimable=sum(int(memory_stat.get(key,0)) for key in ["anon","kernel","shmem"])
        reserve=max(0,int(maximum)-nonreclaimable);extra+=reserve
        aux.append({"unit":unit,"growth_reserved_bytes":reserve,"current_charged_bytes":int(current),"current_nonreclaimable_budget_bytes":nonreclaimable})
    head=max(4*GIB,int(.25*available))
    ancestors=ancestor_headroom(main)
    effective=min([available]+[v["remaining_bytes"] for v in ancestors])
    margin=effective-growth-extra-head
    if margin<CONTRACT["host_loading_budget_bytes"]:reason.append("insufficient_host_RAM_after_main_auxiliary_OS_reservations")
    gpu=subprocess.run(["nvidia-smi","--id=0","--query-gpu=name,memory.free,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True)
    if gpu.returncode:reason.append("GPU_telemetry_unavailable")
    fields=[v.strip() for v in gpu.stdout.strip().split(",")]
    free=int(fields[1])*1024**2 if len(fields)>=3 else 0
    if free<CONTRACT["new_GPU_allocator_budget_bytes"]+CONTRACT["primary_GPU_headroom_bytes"]:reason.append("insufficient_free_VRAM_including_primary_reserve")
    appquery=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],capture_output=True,text=True)
    if appquery.returncode:reason.append("GPU_owner_telemetry_unavailable")
    apps=appquery.stdout.strip()
    if apps:reason.append("existing_CUDA_compute_owner_do_not_interrupt")
    return {"allowed":not reason,"reasons":reason,"time":now(),"host_available_bytes":available,"effective_shared_available_bytes":effective,"shared_cgroup_bounds":ancestors,"main_growth_reserved_bytes":growth,"other_owned_auxiliary_growth_reserved_bytes":extra,"auxiliary_reservations":aux,"OS_headroom_bytes":head,"residual_bytes":margin,"required_loading_bytes":CONTRACT["host_loading_budget_bytes"],"GPU":gpu.stdout.strip(),"GPU_existing_compute_processes":apps,"own_probe_service":own_service}
def bundle(model_id):
    prep=read(D/"DATA_PREPARATION_CHECK.json");tok=read(O/"CPU_TOKENIZATION_QUALIFICATION.json")
    assert sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")==prep["current_input_lock_sha256"]=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
    assert tok["data_preparation_check_sha256"]==sha(D/"DATA_PREPARATION_CHECK.json")
    tr=next(m for m in tok["models"] if m["model"]==model_id)
    assert tr["class_order"]==["no_AKI","AKI"] and tr["label_text"]==["0","1"]
    asset_path=O/"models"/model_id/"MODEL_DOWNLOAD_ACCEPTANCE.json";asset=read(asset_path)
    assert sha(asset_path)==tr["download_acceptance_sha256"] and asset["revision"]==tr["revision"]
    for name in ["ENGINEERING_RECORDS.json","ORIGINAL_OOF_FOLDS.json","labels_and_partitions.csv","preop_clinical_text.jsonl","endop_clinical_text.jsonl"]:
        p=D/name;assert sha(p)==prep["files"][str(p)]
    ids=read(D/"ENGINEERING_RECORDS.json")["record_keys"];assert len(ids)==len(set(ids))==4
    fold=next(f for f in read(D/"ORIGINAL_OOF_FOLDS.json") if f["line"]=="geography" and f["fold"]==0)
    assert set(ids)<=set(fold["train_record_keys"])
    with (D/"labels_and_partitions.csv").open() as f:metadata={r["record_key"]:r for r in csv.DictReader(f)}
    assert len({metadata[i]["group_hash"] for i in ids})==4
    y=[int(metadata[i]["y"]) for i in ids];assert sorted(y)==[0,0,1,1]
    bodies={}
    for stage in ["preop","endop"]:
        rows=[json.loads(l) for l in (D/(stage+"_clinical_text.jsonl")).open()]
        look={r["record_key"]:r["clinical_text"] for r in rows}
        bodies[stage]=[look[i] for i in ids]
    sig={"source_lock":prep["current_input_lock_sha256"],"data_receipt_sha256":sha(D/"DATA_PREPARATION_CHECK.json"),"CPU_tokenization_receipt_sha256":sha(O/"CPU_TOKENIZATION_QUALIFICATION.json"),"asset_receipt_sha256":sha(asset_path),"model_id":model_id,"revision":asset["revision"],"record_keys":ids,"labels":y,"contract":CONTRACT,"expected_versions":EXPECTED_VERSIONS,"code_sha256":sha(__file__)}
    signature=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()
    return {"asset":asset,"tokenizer_receipt":tr,"record_keys":ids,"labels":y,"bodies":bodies,"signature":signature,"signature_fields":sig}
def safe_error(e):
    return {"error_class":type(e).__name__,"stack":[{"file":str(Path(v.filename)),"line":v.lineno,"function":v.name} for v in traceback.extract_tb(e.__traceback__)[-8:]],"raw_messages_clinical_texts_or_credentials_saved":False}
def worker(model_id,phase):
    assert os.environ.get("AKI_GPU_PROBE_LOCK_FD"),"Workers require their finite controller's inherited lock"
    os.fstat(int(os.environ["AKI_GPU_PROBE_LOCK_FD"]))
    b=bundle(model_id);out=E/model_id
    gate=admission();save(out/(phase+"_RESOURCE_ADMISSION.json"),gate)
    if not gate["allowed"]:
        save(out/"CONTROL_STATUS.json",{"status":"WAIT_RESOURCE_BEFORE_"+phase.upper(),"signature":b["signature"],"gate":gate,"GPU_training_qualified":False});return 20
    import importlib.metadata as md
    assert {p:md.version(p) for p in EXPECTED_VERSIONS}==EXPECTED_VERSIONS
    import numpy as np
    import torch,transformers,peft,bitsandbytes
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training,PeftModel
    assert torch.__version__=="2.6.0+cu118" and torch.cuda.is_available()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    props=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0,CONTRACT["new_GPU_allocator_budget_bytes"]/props.total_memory),0)
    torch.cuda.reset_peak_memory_stats()
    local=Path(b["asset"]["local_path"])
    # Verify exactly the files about to be loaded. Never fetch a remote model or execute remote code.
    for p,digest in b["asset"]["files"].items():assert sha(p)==digest
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)
    cls=AutoModelForCausalLM if model_id=="biomedlm_2_7b" else AutoModelForImageTextToText
    started=time.monotonic()
    model=cls.from_pretrained(local,local_files_only=True,trust_remote_code=False,weights_only=True,use_safetensors=model_id!="biomedlm_2_7b",quantization_config=quant,device_map={"":0},dtype=torch.float32,attn_implementation="eager",low_cpu_mem_usage=True)
    model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False,use_fast=True)
    tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    assert tokenizer.pad_token_id is not None
    label_ids=b["tokenizer_receipt"]["label_token_ids"];label_text=b["tokenizer_receipt"]["label_text"]
    prompts={}
    for stage,bodies in b["bodies"].items():
        prompts[stage]=[body+"\nAnswer:" if model_id=="biomedlm_2_7b" else tokenizer.apply_chat_template([{"role":"user","content":body}],tokenize=False,add_generation_prompt=True) for body in bodies]
    device=torch.device("cuda:0");class_ids=torch.tensor(label_ids,device=device)
    def packed(texts):
        data=tokenizer(texts,return_tensors="pt",padding=True,truncation=False,add_special_tokens=False)
        ids=data["input_ids"].to(device);mask=data["attention_mask"].to(device)
        pos=mask.long().cumsum(-1)-1;pos.masked_fill_(mask==0,0)
        assert ids.shape[1]<=max(b["tokenizer_receipt"]["stages"][s]["prefix_tokens_max"] for s in ["preop","endop"])
        return {"input_ids":ids,"attention_mask":mask,"position_ids":pos}
    def last_logits(texts):
        result=model(**packed(texts),use_cache=False,return_dict=True,logits_to_keep=1)
        result=result.logits[:,-1,:].float();assert torch.isfinite(result).all()
        return result
    def probabilities(texts):
        model.eval()
        with torch.no_grad():
            logits=last_logits(texts);pair=logits.index_select(-1,class_ids)
            prob=torch.softmax(pair,-1)[:,1]
            assert torch.isfinite(prob).all()
            return prob.cpu().tolist()
    def parity():
        result={};maximum=0.0
        for stage in ["preop","endop"]:
            batched=probabilities(prompts[stage]);single=[probabilities([p])[0] for p in prompts[stage]]
            delta=max(abs(a-z) for a,z in zip(batched,single));maximum=max(maximum,delta)
            result[stage]={"batched":batched,"single":single,"max_absolute_difference":delta}
        assert maximum<=CONTRACT["batch_probability_tolerance"]
        return result,maximum
    adapter=out/"engineering_adapter_NOT_FOR_FORMAL_FITS"
    training=[]
    if phase=="train":
        before,before_delta=parity()
        targets=[name for name,module in model.named_modules() if name.endswith(".c_attn")] if model_id=="biomedlm_2_7b" else [name for name,module in model.named_modules() if "language_model" in name and name.endswith((".q_proj",".v_proj")) and "vision" not in name]
        expected=int(model.config.n_layer) if model_id=="biomedlm_2_7b" else 2*int(model.config.text_config.num_hidden_layers)
        assert len(targets)==expected and targets
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=0.0,target_modules=targets,bias="none")
        model=get_peft_model(model,config)
        parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert parameters and all("lora_" in n and "vision" not in n for n in parameters)
        if model_id=="medgemma_4b_it":assert all("language_model" in n for n in parameters)
        initial={n:p.detach().cpu().clone() for n,p in parameters.items()}
        optimizer=torch.optim.AdamW(parameters.values(),lr=0.0001,weight_decay=0.01)
        torch.cuda.synchronize();training_start=time.monotonic()
        for step in range(6):
            idx=step%4;model.train();optimizer.zero_grad(set_to_none=True)
            logits=last_logits([prompts["endop"][idx]])
            target=torch.tensor([label_ids[b["labels"][idx]]],device=device)
            # The one predicted answer token is scored against the full vocabulary.
            # Passing labels into a one-logit CausalLM output would incorrectly shift to an empty target.
            loss=torch.nn.functional.cross_entropy(logits,target)
            assert torch.isfinite(loss)
            loss.backward()
            grads=[p.grad for p in parameters.values() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            maximum=max(float(g.detach().abs().max().cpu()) for g in grads)
            assert maximum>0
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in parameters.values())
            training.append({"step":step+1,"loss":float(loss.detach().cpu()),"gradient_max_abs":maximum})
        torch.cuda.synchronize();training_seconds=time.monotonic()-training_start
        delta=max(float((p.detach().cpu()-initial[n]).abs().max()) for n,p in parameters.items())
        assert delta>0
        after,after_delta=parity()
        model.save_pretrained(adapter,safe_serialization=True,save_embedding_layers=False)
        files={str(p):sha(p) for p in adapter.rglob("*") if p.is_file()}
        assert any(p.endswith("adapter_model.safetensors") for p in files)
        result={"signature":b["signature"],"phase":"real_GPU_training_and_adapter_save","worker_PID":os.getpid(),"before":before,"after":after,"batch_parity_max_before":before_delta,"batch_parity_max_after":after_delta,"real_optimizer_steps":6,"loss_records":training,"parameter_delta_max_abs":delta,"trainable_parameter_count":sum(p.numel() for p in parameters.values()),"LoRA_target_modules":targets,"training_seconds":training_seconds,"files":files,"engineering_only":True,"formal_training_started":False,"performance_metrics_computed":False}
        name="TRAIN_RESULT.json"
    else:
        saved=read(out/"TRAIN_RESULT.json");assert saved["signature"]==b["signature"]
        assert all(Path(p).is_file() and sha(p)==h for p,h in saved["files"].items())
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
        after,batch_delta=parity()
        delta=max(abs(a-z) for stage in ["preop","endop"] for a,z in zip(after[stage]["batched"],saved["after"][stage]["batched"]))
        assert delta<=CONTRACT["reload_probability_tolerance"]
        result={"signature":b["signature"],"phase":"fresh_process_real_model_and_adapter_reload","worker_PID":os.getpid(),"after":after,"batch_parity_max":batch_delta,"reload_probability_max_abs_difference":delta,"engineering_only":True,"formal_training_started":False}
        name="RELOAD_RESULT.json"
    torch.cuda.synchronize()
    result.update(time=now(),versions={p:md.version(p) for p in EXPECTED_VERSIONS},GPU_name=props.name,GPU_capability=list(torch.cuda.get_device_capability()),compute_dtype="float32",quantization="NF4_double_quant",elapsed_seconds=time.monotonic()-started,peak_allocated_VRAM_bytes=torch.cuda.max_memory_allocated(),peak_reserved_VRAM_bytes=torch.cuda.max_memory_reserved(),host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save(out/name,result)
    return 0
def controller(model_id,run):
    E.mkdir(parents=True,exist_ok=True,mode=0o700)
    guard=(E/"GPU_ENGINEERING_OWNER.lock").open("a")
    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status":"EXISTING_ENGINEERING_OWNER_DO_NOT_DUPLICATE"}));return
    b=bundle(model_id);out=E/model_id;out.mkdir(parents=True,exist_ok=True,mode=0o700)
    signature_path=out/"SIGNATURE.json"
    if signature_path.exists():assert read(signature_path)["signature"]==b["signature"],"Existing engineering version needs inspection, not relabelling"
    else:save(signature_path,{"signature":b["signature"],"fields":b["signature_fields"]})
    accepted_path=out/"GPU_ENGINEERING_ACCEPTANCE.json"
    if accepted_path.exists():
        accepted=read(accepted_path)
        assert accepted["signature"]==b["signature"]
        assert sha(out/"TRAIN_RESULT.json")==accepted["train_result_sha256"]
        assert sha(out/"RELOAD_RESULT.json")==accepted["reload_result_sha256"]
        trained=read(out/"TRAIN_RESULT.json")
        assert all(Path(p).is_file() and sha(p)==h for p,h in trained["files"].items())
        print(json.dumps({"status":"ALREADY_ACCEPTED_VALID_ENGINEERING_ONLY_NO_RERUN","model":model_id,"acceptance_sha256":sha(accepted_path),"formal_training_started":False}));return
    gate=admission()
    status={"status":"RESOURCE_ELIGIBLE_NOT_RUN" if gate["allowed"] else "WAIT_RESOURCE_NO_MODEL_OR_CUDA_LOADED","time":now(),"model":model_id,"signature":b["signature"],"gate":gate,"GPU_training_qualified":False,"formal_training_started":False}
    save(out/"CONTROL_STATUS.json",status)
    if not run or not gate["allowed"]:
        print(json.dumps(status,ensure_ascii=False));return
    assert gate["own_probe_service"],"Use a task-scoped managed unit for heavy qualification"
    own=unit_properties(gate["own_probe_service"])
    assert own.get("MemorySwapMax")=="0" and own.get("MemoryMax","").isdigit() and int(own["MemoryMax"])==16*GIB
    exited=[]
    for phase,name in [("train","TRAIN_RESULT.json"),("reload","RELOAD_RESULT.json")]:
        result_path=out/name
        if result_path.exists():
            result=read(result_path);assert result["signature"]==b["signature"]
            assert all(Path(p).is_file() and sha(p)==h for p,h in result.get("files",{}).items())
            continue
        fresh=admission()
        if not fresh["allowed"]:
            save(out/"CONTROL_STATUS.json",{"status":"WAIT_RESOURCE_SAVED_PHASE_REUSED","time":now(),"next_phase":phase,"gate":fresh,"GPU_training_qualified":False});return
        env=os.environ.copy()
        env.update({"AKI_GPU_PROBE_LOCK_FD":str(guard.fileno()),"USE_TORCH":"1","USE_TF":"0","USE_FLAX":"0","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8","CUDA_VISIBLE_DEVICES":"0"})
        log=out/(phase+"_worker.log")
        with log.open("ab") as stream:
            child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--model",model_id,"--worker",phase],env=env,stdout=stream,stderr=stream,pass_fds=(guard.fileno(),))
            try:code=child.wait(timeout=CONTRACT["maximum_worker_seconds"])
            except subprocess.TimeoutExpired:
                child.kill();child.wait();raise TimeoutError("Owned finite GPU probe exceeded its budget")
        exited.append({"PID":child.pid,"exit_code":code})
        if code==20:return
        assert code==0 and result_path.exists(),"Inspect sanitized worker failure without retraining a saved valid phase"
        assert not (Path("/proc")/str(child.pid)).exists()
    trained=read(out/"TRAIN_RESULT.json");reloaded=read(out/"RELOAD_RESULT.json")
    assert trained["signature"]==reloaded["signature"]==b["signature"]
    assert trained["real_optimizer_steps"]==6 and trained["parameter_delta_max_abs"]>0
    assert reloaded["reload_probability_max_abs_difference"]<=CONTRACT["reload_probability_tolerance"]
    record={"status":"PASS_REAL_GPU_TRAIN_SAVE_FRESH_PROCESS_RELOAD_AND_BATCH_PARITY_ENGINEERING_ONLY","time":now(),"signature":b["signature"],"model":model_id,"revision":b["asset"]["revision"],"source_training_patients":4,"real_engineering_steps":6,"train_result_sha256":sha(out/"TRAIN_RESULT.json"),"reload_result_sha256":sha(out/"RELOAD_RESULT.json"),"worker_exits":exited,"formal_training_started":False,"clinical_performance_evaluation_performed":False,"engineering_adapter_reuse_for_formal_training_prohibited":True}
    save(out/"GPU_ENGINEERING_ACCEPTANCE.json",record)
    save(out/"CONTROL_STATUS.json",{"status":record["status"],"time":now(),"GPU_training_qualified":True,"formal_training_started":False})
    print(json.dumps(record,ensure_ascii=False))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--model",choices=["biomedlm_2_7b","medgemma_4b_it"],required=True);parser.add_argument("--run",action="store_true");parser.add_argument("--worker",choices=["train","reload"])
    args=parser.parse_args()
    try:
        if args.worker:raise SystemExit(worker(args.model,args.worker))
        controller(args.model,args.run)
    except SystemExit:raise
    except BaseException as e:
        out=E/args.model;out.mkdir(parents=True,exist_ok=True,mode=0o700)
        error={"time":now(),"phase":args.worker or "controller",**safe_error(e),"formal_training_started":False}
        save(out/((args.worker or "CONTROLLER")+"_FAILURE.json"),error)
        raise SystemExit("GPU engineering stopped: "+type(e).__name__+"; inspect sanitized failure receipt")
