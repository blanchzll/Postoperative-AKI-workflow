"""Construct/qualify local HF-format NGC candidate without loading weights or changing selection."""
from pathlib import Path
import datetime,hashlib,json,os,sys,tarfile
R=Path(__file__).resolve().parents[2];O=R/"outputs/llm_supplement_20261028_v1";A=O/"models/llama3_8b_instruct_ngc_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def exact(p,d):
 p=Path(p)
 if p.exists():assert read(p)==d,str(p)
 else:p.write_text(json.dumps(d,indent=2)+"\n")
def main():
 import yaml
 from transformers import AutoConfig,AutoTokenizer
 tokenrec=read(A/"LLAMA3_NGC_TOKENIZATION_QUALIFICATION.json")
 assert tokenrec["all_256_special_tokens_match_original_Meta_ids"] and tokenrec["original_Meta_regex_matches"]
 fmt=read(A/"ORIGINAL_META_LLAMA3_CHAT_FORMAT.json");assert sha(A/"ORIGINAL_META_LLAMA3_CHAT_FORMAT.json")==tokenrec["chat_format_file_sha256"]
 archive_rec=read(A/"CANDIDATE_DOWNLOAD_ACCEPTANCE.json")
 export=A/"tensor_export_v1";converted=read(export/"TENSOR_EXPORT_ACCEPTANCE.json")
 assert converted["tensor_count"]==291 and converted["tensor_bytes"]==16060522496
 assert converted["selected_HF_checkpoint_changed"] is False
 with tarfile.open(archive_rec["archive_path"],"r:") as tf:cfg=yaml.safe_load(tf.extractfile("./model_config.yaml").read())
 assert cfg["precision"]=="bf16" and cfg["activation"]=="fast-swiglu" and cfg["normalization"]=="rmsnorm"
 assert cfg["hidden_dropout"]==cfg["attention_dropout"]==0.0 and not cfg["bias"]
 assert cfg["rotary_percentage"]==1.0 and not cfg["share_embeddings_and_output_weights"]
 asset=read(A/"NVIDIA_PUBLIC_TOKENIZER_ASSET_ACCEPTANCE.json")
 tokenfiles={Path(f["file"]).name:Path(f["file"]) for f in asset["files"]}
 for f in asset["files"]:assert sha(f["file"])==f["local_SHA256"]
 dest=R/"model_assets/llama3_8b_instruct_ngc_v1/1.0/hf_candidate_ngc_v1"
 dest.mkdir(parents=True,exist_ok=True,mode=0o700)
 weights=converted["shards"]
 files={}
 for src,digest in weights.items():
  src=Path(src);target=dest/src.name
  if target.exists():assert target.resolve()==src.resolve()
  else:target.symlink_to(src)
  files[str(target)]=digest
 origin=Path(next(iter(weights))).parent
 for name in ["model.safetensors.index.json"]:
  target=dest/name
  if not target.exists():target.symlink_to(origin/name)
  assert sha(target)==converted["index_sha256"];files[str(target)]=sha(target)
 for name in ["tokenizer.json","special_tokens_map.json"]:
  target=dest/name
  if not target.exists():target.symlink_to(tokenfiles[name])
  assert target.resolve()==tokenfiles[name].resolve();files[str(target)]=sha(target)
 config={"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":128256,"hidden_size":cfg["hidden_size"],"intermediate_size":cfg["ffn_hidden_size"],"num_hidden_layers":cfg["num_layers"],"num_attention_heads":cfg["num_attention_heads"],"num_key_value_heads":cfg["num_query_groups"],"hidden_act":"silu","max_position_embeddings":cfg["max_position_embeddings"],"initializer_range":cfg["init_method_std"],"rms_norm_eps":cfg["layernorm_epsilon"],"rope_theta":cfg["rotary_base"],"rope_scaling":None,"attention_bias":False,"attention_dropout":0.0,"mlp_bias":False,"tie_word_embeddings":False,"bos_token_id":128000,"eos_token_id":[128001,128009],"pad_token_id":128001,"pretraining_tp":1,"use_cache":False,"torch_dtype":"bfloat16"}
 exact(dest/"config.json",config)
 tokcfg=read(tokenfiles["tokenizer_config.json"])
 tokcfg.update(chat_template=fmt["chat_template"],model_max_length=8192,add_bos_token=False,add_eos_token=False)
 # Keep source's original token mapping; replace only its ChatQA formatting and unbounded context claim.
 exact(dest/"tokenizer_config.json",tokcfg)
 for name in ["config.json","tokenizer_config.json"]:files[str(dest/name)]=sha(dest/name)
 check=AutoConfig.from_pretrained(dest,local_files_only=True,trust_remote_code=False)
 for k in ["vocab_size","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","num_key_value_heads","max_position_embeddings","rms_norm_eps","rope_theta","tie_word_embeddings"]:
  assert getattr(check,k)==config[k],k
 assert check.head_dim==128 and check.pad_token_id==128001
 tokenizer=AutoTokenizer.from_pretrained(dest,local_files_only=True,trust_remote_code=False,use_fast=True)
 assert [tokenizer.encode(z,add_special_tokens=False) for z in ["0","1"]]==[[15],[16]]
 digest=hashlib.sha256();lengths={}
 for stage in ["preop","endop"]:
  rows=[json.loads(l) for l in (O/"data_preparation_v1"/(stage+"_clinical_text.jsonl")).open()]
  sizes=[]
  for row in rows:
   text=tokenizer.apply_chat_template([{"role":"user","content":row["clinical_text"]}],tokenize=False,add_generation_prompt=True)
   ids=tokenizer.encode(text,add_special_tokens=False)
   for label,token in zip(["0","1"],[15,16]):assert tokenizer.encode(text+label,add_special_tokens=False)==ids+[token]
   digest.update(json.dumps(ids,separators=(",",":")).encode());sizes.append(len(ids))
  assert max(sizes)==tokenrec["stages"][stage]["prefix_tokens_max"]
  lengths[stage]={"n":len(rows),"prefix_tokens_max":max(sizes)}
 assert digest.hexdigest()==tokenrec["all_prefixes_sha256"]
 assert "torch" not in sys.modules
 derivation={"NGC_archive_local_SHA256":archive_rec["archive_sha256"],"NGC_model":"nvidia/nemo/llama-3-8b-instruct-nemo:1.0","source_model_config":"original immutable model_config.yaml within NGC archive; no ChatQA model configuration","tensor_mapping_source":"Saved NVIDIA v2.0.0rc0 official Llama exporter; all291 BF16 tensors bit-preserved and QKV source-roundtrip verified","tokenizer_repository":asset["source_repository"],"tokenizer_commit":asset["source_commit"],"tokenizer_format":"Original Meta Llama3 ChatFormat and all256 IDs; all4856 current prefixes independently byte-BPE verified","actual_context_limit":8192,"EOS_and_padding":"Original Meta end-of-text128001 and end-of-turn128009 stop semantics; existing end-of-text aliases masked left-padding; no free generation used","selected_HF_bit_identity":"UNVERIFIED; no selected-HF contents accessed","ChatQA_model_weights_used":False,"generation_or_prompt_tuned_against_outcomes":False}
 exact(dest/"CANDIDATE_DERIVATION.json",derivation);files[str(dest/"CANDIDATE_DERIVATION.json")]=sha(dest/"CANDIDATE_DERIVATION.json")
 record={"status":"PASS_NGC_HF_FORMAT_CANDIDATE_CURRENT_TENSOR_CONFIG_AND_TOKENIZER_FILES_ONLY","time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"model_id":"llama3_8b_instruct_ngc_v1","revision":"NGC1.0_localSHA256_"+archive_rec["archive_sha256"],"local_path":str(dest),"files":files,"weight_file_hashes_source":"Current accepted bit-preserving export, actual weight hashes rechecked again before any GPU loading","tensor_export_receipt_sha256":sha(export/"TENSOR_EXPORT_ACCEPTANCE.json"),"tokenizer_qualification_sha256":sha(A/"LLAMA3_NGC_TOKENIZATION_QUALIFICATION.json"),"original_Meta_chat_format_sha256":sha(A/"ORIGINAL_META_LLAMA3_CHAT_FORMAT.json"),"all_AutoTokenizer_prefixes_match_independent_qualification":True,"stage_lengths":lengths,"config_derived_from_NGC_and_checked_with_AutoConfig":True,"selected_HF_checkpoint_changed":False,"scientific_source_adoption_pending_GPU_qualification":True,"GPU_training_qualified":False,"formal_training_started":False,"torch_imported":False,"script_sha256":sha(__file__)}
 target=A/"NGC_HF_CANDIDATE_ASSET_ACCEPTANCE.json"
 assert not target.exists();target.write_text(json.dumps(record,indent=2)+"\n")
 print(json.dumps({k:v for k,v in record.items() if k!="files"}))
if __name__=="__main__":
 try:main()
 except BaseException as e:
  import traceback
  p=A/"HF_CANDIDATE_PREPARATION_FAILURE.json";p.write_text(json.dumps({"error_class":type(e).__name__,"stack":[{"line":x.lineno,"function":x.name} for x in traceback.extract_tb(e.__traceback__)[-5:]],"raw_clinical_text_or_credentials_saved":False},indent=2)+"\n");raise
