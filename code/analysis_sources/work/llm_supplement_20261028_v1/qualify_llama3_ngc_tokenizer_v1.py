"""Independent CPU validation of public NVIDIA Llama3 byte-BPE and original Meta chat format.
No Torch/CUDA/model weights, clinical outcomes, ChatQA weights or remote inference.
"""
from pathlib import Path
import ast,datetime,functools,hashlib,json,os,statistics,sys
R=Path(__file__).resolve().parents[2]
O=R/"outputs/llm_supplement_20261028_v1"
D=O/"data_preparation_v1"
A=O/"models/llama3_8b_instruct_ngc_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,indent=2)+"\n");os.replace(t,p)
def main():
 import regex
 from tokenizers import Tokenizer
 receipt=read(A/"NVIDIA_PUBLIC_TOKENIZER_ASSET_ACCEPTANCE.json")
 paths={Path(x["file"]).name:Path(x["file"]) for x in receipt["files"]}
 for item in receipt["files"]:assert sha(item["file"])==item["local_SHA256"]
 assert receipt["source_commit"]=="3b98162e3f97550d62aeeb19ea50208f968c678a"
 assert receipt["no_ChatQA_model_weights_downloaded_or_used"] and not receipt["auth_token_used"]
 spec=O/"references/nemo_conversion_review_20261028_v1/meta_llama3_tokenizer.py"
 assert sha(spec)=="e29045cbbae33e07bb1ad92bd8249032c4a802f7d9676f115c19aeebd114f9cd"
 tree=ast.parse(spec.read_text())
 pattern=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="pat_str" for t in n.targets))
 first=next(ast.literal_eval(n.value.left) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="special_tokens" for t in n.targets) and isinstance(n.value,ast.BinOp))
 special=first+[f"<|reserved_special_token_{i}|>" for i in range(5,251)]
 assert len(special)==256
 raw=read(paths["tokenizer.json"]);vocab=raw["model"]["vocab"]
 assert len(vocab)==128000 and raw["model"]["type"]=="BPE" and raw["normalizer"] is None
 actual={x["content"]:x["id"] for x in raw["added_tokens"]}
 assert actual=={word:128000+i for i,word in enumerate(special)}
 pre=raw["pre_tokenizer"]["pretokenizers"]
 assert pre[0]["pattern"]["Regex"]==pattern and pre[1]["type"]=="ByteLevel" and not pre[1]["add_prefix_space"]
 assert pre[1]["use_regex"] is False
 # Standard reversible GPT2/ByteLevel alphabet; then recover the published byte->rank codebook.
 initial=list(range(33,127))+list(range(161,173))+list(range(174,256))
 bytevalues=initial[:];codepoints=initial[:];n=0
 for b in range(256):
  if b not in initial:bytevalues.append(b);codepoints.append(256+n);n+=1
 inverse={chr(u):b for b,u in zip(bytevalues,codepoints)}
 ranks={bytes(inverse[c] for c in token):rank for token,rank in vocab.items()}
 assert len(ranks)==128000 and all(bytes([b]) in ranks for b in range(256))
 pat=regex.compile(pattern)
 @functools.lru_cache(maxsize=131072)
 def byte_bpe(data):
  # CoreBPE's whole-piece lookup plus the merge loop in the saved OpenAI educational source.
  if data in ranks:return (ranks[data],)
  parts=[bytes([b]) for b in data]
  while True:
   candidate=None
   for i in range(len(parts)-1):
    rank=ranks.get(parts[i]+parts[i+1])
    if rank is not None and (candidate is None or rank<candidate[0]):candidate=(rank,i)
   if candidate is None:break
   i=candidate[1];parts=parts[:i]+[parts[i]+parts[i+1]]+parts[i+2:]
  return tuple(ranks[part] for part in parts)
 def independent(text):
  pieces=pat.findall(text);assert "".join(pieces)==text
  return [token for piece in pieces for token in byte_bpe(piece.encode())]
 tokenizer=Tokenizer.from_file(str(paths["tokenizer.json"]))
 def fast(text):return tokenizer.encode(text,add_special_tokens=False).ids
 def header(role):return [actual["<|start_header_id|>"]]+independent(role)+[actual["<|end_header_id|>"]]+independent("\n\n")
 def prompt(body):return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"+body.strip()+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
 label_ids=[fast("0"),fast("1")];assert all(len(x)==1 for x in label_ids);label_ids=[x[0] for x in label_ids]
 assert label_ids==[independent("0")[0],independent("1")[0]]
 template="{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'].strip() + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
 # Jinja renders the same literal original Meta headers, not NVIDIA's ChatQA conversation format.
 from jinja2 import Environment,StrictUndefined
 render=Environment(undefined=StrictUndefined).from_string(template)
 assert render.render(bos_token="<|begin_of_text|>",messages=[{"role":"user","content":"test"}],add_generation_prompt=True)==prompt("test")
 prep=read(D/"DATA_PREPARATION_CHECK.json")
 assert sha(R/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")==prep["current_input_lock_sha256"]=="afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3"
 reports={};digest=hashlib.sha256()
 for stage in ["preop","endop"]:
  path=D/(stage+"_clinical_text.jsonl");assert sha(path)==prep["files"][str(path)]
  rows=[json.loads(line) for line in path.open()];assert len(rows)==2428 and len({x["record_key"] for x in rows})==2428
  lengths=[]
  for row in rows:
   body=row["clinical_text"];assert body==body.strip() and not any(k in body for k in ["<|begin_of_text|>","<|eot_id|>","<|start_header_id|>"])
   pure=independent(body);assert pure==fast(body)
   assert tokenizer.decode(pure,skip_special_tokens=False)==body
   expected=[actual["<|begin_of_text|>"]]+header("user")+pure+[actual["<|eot_id|>"]]+header("assistant")
   text=prompt(body);encoded=fast(text);assert encoded==expected
   assert render.render(bos_token="<|begin_of_text|>",messages=[{"role":"user","content":body}],add_generation_prompt=True)==text
   for label,token in zip(["0","1"],label_ids):assert fast(text+label)==expected+[token]
   assert len(encoded)+1<=8192
   digest.update(json.dumps(encoded,separators=(",",":")).encode());lengths.append(len(encoded))
  reports[stage]={"n_records":len(rows),"prefix_tokens_min":min(lengths),"prefix_tokens_median":statistics.median(lengths),"prefix_tokens_max":max(lengths),"context_limit":8192,"all_fields_kept_without_truncation":True,"independent_Meta_regex_byte_BPE_matches_fast_body_and_full_format":True,"all_label_continuations_one_token_preserving_prefix":True}
  save(A/"TOKENIZER_QUALIFICATION_PROGRESS.json",{"stage_complete":stage,"time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"model_weights_loaded":False})
 assert "torch" not in sys.modules
 format_path=A/"ORIGINAL_META_LLAMA3_CHAT_FORMAT.json"
 save(format_path,{"format":"Meta original Llama3 ChatFormat one user message and assistant header","chat_template":template,"source_Meta_code_sha256":sha(spec),"body_stripping":"Only original Meta .strip(); all current source texts already stripped","no_system_or_ChatQA_prompt_inserted":True,"label_text":["0","1"],"label_token_ids":label_ids,"class_order":["no_AKI","AKI"]})
 result={"status":"PASS_PUBLIC_NVIDIA_LLAMA3_TOKENIZER_ORIGINAL_META_FORMAT_AND_INDEPENDENT_BYTE_BPE_FOR_ALL_CURRENT_TEXTS_ONLY","time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"source_asset_receipt_sha256":sha(A/"NVIDIA_PUBLIC_TOKENIZER_ASSET_ACCEPTANCE.json"),"source_Data_preparation_sha256":sha(D/"DATA_PREPARATION_CHECK.json"),"original_Meta_tokenizer_spec_sha256":sha(spec),"original_OpenAI_BPE_source_sha256":sha(O/"references/tiktoken_educational_primary_source.py"),"script_sha256":sha(__file__),"public_NVIDIA_tokenizer_revision":receipt["source_commit"],"all_256_special_tokens_match_original_Meta_ids":True,"original_Meta_regex_matches":True,"model_vocabulary_size":128000,"label_token_ids":label_ids,"class_order":["no_AKI","AKI"],"chat_format_file_sha256":sha(format_path),"stages":reports,"all_prefixes_sha256":digest.hexdigest(),"exact_file_identity_with_selected_HF_tokenizer":False,"selected_HF_contents_not_accessed_or_compared":True,"scope":"Frozen NGC-source candidate with openly distributed NVIDIA original-Llama3 vocabulary and original Meta format; not a claim of byte identity with selected HF commit","ChatQA_model_weights_or_prompt_used":False,"model_config_qualified":False,"GPU_training_qualified":False,"formal_training_started":False,"Torch_or_CUDA_initialized":False}
 save(A/"LLAMA3_NGC_TOKENIZATION_QUALIFICATION.json",result)
 print(json.dumps(result))
if __name__=="__main__":
 try:main()
 except BaseException as e:
  import traceback
  save(A/"TOKENIZER_QUALIFICATION_FAILURE.json",{"time":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"error_class":type(e).__name__,"stack":[{"line":x.lineno,"function":x.name} for x in traceback.extract_tb(e.__traceback__)[-5:]],"raw_clinical_text_or_credentials_saved":False});raise
