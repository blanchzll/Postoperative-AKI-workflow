"""CPU-only local tokenizer and class-continuation checks. No model weights, CUDA or performance evaluation."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys
from transformers import AutoTokenizer,AutoConfig
R=Path(__file__).resolve().parents[2]
O=R/"outputs/llm_supplement_20261028_v1"
B=O/"data_preparation_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
def main():
    assert "torch" not in sys.modules,"CPU tokenizer process must not initialize the model backend"
    data=json.loads((B/"DATA_PREPARATION_CHECK.json").read_text())
    for stage in ["preop","endop"]:
        p=B/(stage+"_clinical_text.jsonl")
        assert sha(p)==data["files"][str(p)]
    results=[]
    for model_id in ["biomedlm_2_7b","medgemma_4b_it"]:
        receipt=O/"models"/model_id/"MODEL_DOWNLOAD_ACCEPTANCE.json"
        downloaded=json.loads(receipt.read_text())
        assert downloaded["status"].startswith("PASS_official_pinned_")
        local=Path(downloaded["local_path"])
        for name in ["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","tokenizer.model","vocab.json","merges.txt","added_tokens.json","chat_template.jinja"]:
            p=local/name
            if p.exists():assert sha(p)==downloaded["files"][str(p)]
        config=AutoConfig.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        c=getattr(config,"text_config",config)
        context=int(getattr(c,"max_position_embeddings",getattr(c,"n_positions",0)))
        assert context>0
        tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False,use_fast=True)
        stages={};label_ids=None;label_text=None;wrapping="causal_completion" if model_id=="biomedlm_2_7b" else "official_text_only_chat_template"
        for stage in ["preop","endop"]:
            counts=[];cases=0
            for line in (B/(stage+"_clinical_text.jsonl")).open():
                row=json.loads(line);assert set(row)=={"record_key","clinical_text"}
                body=row["clinical_text"]
                if model_id=="biomedlm_2_7b":prompt=body+"\nAnswer:"
                else:
                    prompt=tokenizer.apply_chat_template([{"role":"user","content":body}],tokenize=False,add_generation_prompt=True)
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                if label_text is None:
                    for pair in [("0","1"),(" 0"," 1")]:
                        continuation=[tokenizer.encode(prompt+s,add_special_tokens=False) for s in pair]
                        if all(v[:-1]==ids and len(v)==len(ids)+1 for v in continuation):
                            label_text=list(pair);label_ids=[v[-1] for v in continuation];break
                    assert label_text is not None,"A qualified binary single-token continuation is required"
                    assert label_ids[0]!=label_ids[1]
                for text,target in zip(label_text,label_ids):
                    full=tokenizer.encode(prompt+text,add_special_tokens=False)
                    assert full[:-1]==ids and full[-1]==target and len(full)==len(ids)+1
                assert len(ids)+1<=context,"Full clinical field retention exceeds actual checkpoint context"
                counts.append(len(ids));cases+=1
            assert cases==2428
            counts.sort()
            stages[stage]={"n_records":cases,"prefix_tokens_min":counts[0],"prefix_tokens_median":statistics.median(counts),"prefix_tokens_p95":counts[int(.95*(cases-1))],"prefix_tokens_max":counts[-1],"actual_context_limit":context,"full_field_retention_without_truncation":True,"all_class_continuations_preserve_prefix_and_add_one_token":True}
        results.append({"model":model_id,"revision":downloaded["revision"],"download_acceptance_sha256":sha(receipt),"wrapper":wrapping,"label_text":label_text,"label_token_ids":label_ids,"class_order":["no_AKI","AKI"],"stages":stages,"GPU_training_qualified":False})
        del tokenizer,config
    assert "torch" not in sys.modules
    rec={"status":"PASS_TWO_LOCAL_TOKENIZERS_ALL_CURRENT_TEXTS_CONTEXT_AND_SINGLE_TOKEN_BINARY_CONTINUATIONS_ONLY","time":now(),"data_preparation_check_sha256":sha(B/"DATA_PREPARATION_CHECK.json"),"models":results,"llama3_tokenizer_still_pending_source_and_conversion":True,"CUDA_initialized":False,"model_weights_loaded":False,"formal_training_started":False,"scientific_performance_evaluated":False,"script_sha256":sha(Path(__file__))}
    save(O/"CPU_TOKENIZATION_QUALIFICATION.json",rec)
    print(json.dumps(rec,ensure_ascii=False))
if __name__=="__main__":
    try:main()
    except BaseException as e:
        save(O/"CPU_TOKENIZATION_FAILURE.json",{"time":now(),"error_class":type(e).__name__,"model_weights_loaded":False,"formal_training_started":False})
        raise SystemExit("CPU tokenizer qualification stopped: "+type(e).__name__+"; inspect without printing clinical texts")
