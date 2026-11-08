"""Stream a verified public NeMo Zarr/BF16 candidate to safetensors.
No torch, no pickle, no clinical data. This is a tensor-format qualification only.
Mapping follows the saved NVIDIA v2.0.0rc0 Llama exporter, lines143-215.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,math,os,struct,subprocess,sys,tarfile
R=Path(__file__).resolve().parents[2]
O=R/"outputs/llm_supplement_20261028_v1"
A=O/"models/llama3_8b_instruct_ngc_v1"
E=A/"tensor_export_v1"
DEST=R/"model_assets/llama3_8b_instruct_ngc_v1/1.0/tensor_export_candidate_v1"
GIB=1024**3
BLOCK=4*1024**2
EXPECTED_DATA_BYTES=16060522496
EXPECTED_ARCHIVE_SHA="bb9fac0d2a168a04f0a1e764e0f7741759fb5ba02553e2396cd6bdfcbb8f2776"
PFX="./model_weights/"
MODEL_CONFIG_EXPECTED={"num_layers":32,"hidden_size":4096,"ffn_hidden_size":14336,"num_attention_heads":32,"num_query_groups":8,"rotary_base":500000.0,"precision":"bf16","share_embeddings_and_output_weights":False}
ARRAYS={
"model.embedding.word_embeddings.weight":([128256,4096],[128256,4096]),
"model.output_layer.weight":([128256,4096],[128256,4096]),
"model.decoder.final_layernorm.weight":([4096],[4096]),
"model.decoder.layers.self_attention.linear_qkv.weight":([32,6144,4096],[1,6144,4096]),
"model.decoder.layers.self_attention.linear_proj.weight":([32,4096,4096],[1,4096,4096]),
"model.decoder.layers.self_attention.linear_qkv.layer_norm_weight":([32,4096],[1,4096]),
"model.decoder.layers.mlp.linear_fc1.weight":([32,28672,4096],[1,14336,4096]),
"model.decoder.layers.mlp.linear_fc2.weight":([32,4096,14336],[1,4096,14336]),
"model.decoder.layers.mlp.linear_fc1.layer_norm_weight":([32,4096],[1,4096])}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(BLOCK),b""):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n");os.replace(t,p)
def prop(unit):
 x=subprocess.run(["systemctl","--user","show",unit,"--property=ActiveState,MainPID,MemoryCurrent,MemoryMax,MemorySwapMax"],capture_output=True,text=True,check=True)
 return dict(v.split("=",1) for v in x.stdout.splitlines() if "=" in v)
def admission():
 available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
 main=prop("aki-postslot-full-reanalysis-20261025.service")
 main_reserve=32*GIB if main.get("ActiveState")=="active" else 0
 rows=subprocess.run(["systemctl","--user","list-units","--state=active","--plain","--no-legend","aki-postslot-llm-gpu-qual*","aki-postslot-review-status-*"],capture_output=True,text=True,check=True).stdout.splitlines()
 extra=0
 for row in rows:
  d=prop(row.split()[0]);assert d["MemoryMax"].isdigit()
  extra+=int(d["MemoryMax"])
 os_head=max(4*GIB,int(.25*available))
 return {"time":now(),"allowed":available-main_reserve-extra-os_head>=2*GIB,"host_MemAvailable":available,"main_conservative_growth_reserve":main_reserve,"other_finite_owner_maximum_reserves":extra,"OS_headroom":os_head,"new_task_maximum":2*GIB,"no_CUDA_requested":True}
def inventory():
 receipt=read(A/"CANDIDATE_DOWNLOAD_ACCEPTANCE.json")
 assert receipt["archive_sha256"]==EXPECTED_ARCHIVE_SHA and receipt["archive_bytes"]==16061388800
 assert sha(A/"ARCHIVE_FILE_INDEX.json")==receipt["archive_index_sha256"]
 archive=Path(receipt["archive_path"]);assert archive.stat().st_size==receipt["archive_bytes"]
 ref=O/"references/nemo_conversion_review_20261028_v1/nemo_rc0_llama_export.py"
 assert sha(ref)=="ba2cc9c3b8fa9829e08993e2913f0329f47397477bc18458553833704b376872"
 import yaml
 with tarfile.open(archive,"r:") as tf:
  entries={v.name:v for v in tf.getmembers()}
  cfg=yaml.safe_load(tf.extractfile("./model_config.yaml").read())
  assert all(cfg[k]==v for k,v in MODEL_CONFIG_EXPECTED.items())
  assert cfg["tokenizer"]["type"]=="meta-llama/Meta-Llama-3-8B"
  expected={PFX+k+"/.zarray" for k in ARRAYS}
  assert {n for n in entries if n.endswith("/.zarray")}==expected
  metadata={}
  for name,(shape,chunks) in ARRAYS.items():
   m=json.loads(tf.extractfile(PFX+name+"/.zarray").read())
   assert m["shape"]==shape and m["chunks"]==chunks and m["dtype"]=="bfloat16" and m["order"]=="C"
   assert m["compressor"] is None and m["filters"] is None and m["zarr_format"]==2
   metadata[name]=m
  assert not any("tokenizer" in n.lower() or "vocab" in n.lower() for n in entries)
  assert all(not n.startswith("/") and ".." not in Path(n).parts for n in entries)
 return archive,entries,metadata,receipt,ref
def chunk(entries,array,key):
 m=entries[PFX+array+"/"+key];assert m.isfile()
 expected=math.prod(ARRAYS[array][1])*2;assert m.size==expected
 return (m.offset_data,m.size)
def specs(entries):
 result=[]
 def tensor(name,shape,ranges):return {"name":name,"shape":shape,"ranges":ranges,"bytes":math.prod(shape)*2}
 result.append([tensor("model.embed_tokens.weight",[128256,4096],[chunk(entries,"model.embedding.word_embeddings.weight","0.0")])])
 for layer in range(32):
  base=f"model.layers.{layer}."
  qkv=chunk(entries,"model.decoder.layers.self_attention.linear_qkv.weight",f"{layer}.0.0")
  headbytes=128*4096*2
  qs=[6*g+j for g in range(8) for j in range(4)]
  ks=[6*g+4 for g in range(8)];vs=[6*g+5 for g in range(8)]
  def selected(indices):return [(qkv[0]+h*headbytes,headbytes) for h in indices]
  rows=[
   tensor(base+"self_attn.q_proj.weight",[4096,4096],selected(qs)),
   tensor(base+"self_attn.k_proj.weight",[1024,4096],selected(ks)),
   tensor(base+"self_attn.v_proj.weight",[1024,4096],selected(vs)),
   tensor(base+"self_attn.o_proj.weight",[4096,4096],[chunk(entries,"model.decoder.layers.self_attention.linear_proj.weight",f"{layer}.0.0")]),
   tensor(base+"mlp.gate_proj.weight",[14336,4096],[chunk(entries,"model.decoder.layers.mlp.linear_fc1.weight",f"{layer}.0.0")]),
   tensor(base+"mlp.up_proj.weight",[14336,4096],[chunk(entries,"model.decoder.layers.mlp.linear_fc1.weight",f"{layer}.1.0")]),
   tensor(base+"mlp.down_proj.weight",[4096,14336],[chunk(entries,"model.decoder.layers.mlp.linear_fc2.weight",f"{layer}.0.0")]),
   tensor(base+"input_layernorm.weight",[4096],[chunk(entries,"model.decoder.layers.self_attention.linear_qkv.layer_norm_weight",f"{layer}.0")]),
   tensor(base+"post_attention_layernorm.weight",[4096],[chunk(entries,"model.decoder.layers.mlp.linear_fc1.layer_norm_weight",f"{layer}.0")])]
  result.append(rows)
 result.append([tensor("lm_head.weight",[128256,4096],[chunk(entries,"model.output_layer.weight","0.0")]),tensor("model.norm.weight",[4096],[chunk(entries,"model.decoder.final_layernorm.weight","0")])])
 assert len(result)==34 and sum(len(r) for r in result)==291
 assert sum(t["bytes"] for r in result for t in r)==EXPECTED_DATA_BYTES
 for rows in result:
  for t in rows:assert sum(n for _,n in t["ranges"])==t["bytes"]
 return result
def range_chunks(file,ranges):
 for offset,count in ranges:
  file.seek(offset);left=count
  while left:
   block=file.read(min(BLOCK,left));assert block
   left-=len(block);yield block
def ranges_hash(file,ranges):
 h=hashlib.sha256()
 for b in range_chunks(file,ranges):h.update(b)
 return h.hexdigest()
def header(file):
 file.seek(0);raw=file.read(8);assert len(raw)==8
 size=struct.unpack("<Q",raw)[0];assert 0<size<1024**2
 d=json.loads(file.read(size));return d,8+size
def verify_shard(archive,dest,rows,index,entries):
 with archive.open("rb") as src,dest.open("rb") as out:
  meta,start=header(out)
  assert set(meta)-{"__metadata__"}=={t["name"] for t in rows}
  ranges={}
  for t in rows:
   z=meta[t["name"]];assert z["dtype"]=="BF16" and z["shape"]==t["shape"]
   left,right=z["data_offsets"];assert right-left==t["bytes"]
   ranges[t["name"]]=[(start+left,right-left)]
   assert ranges_hash(out,ranges[t["name"]])==ranges_hash(src,t["ranges"])
  # Independent reverse check reconstructs each original interleaved six-head GQA group
  # from the three exported HF matrices and compares with the original Zarr chunk.
  qkv_hash=None
  if 2<=index<=33:
   layer=index-2;base=f"model.layers.{layer}.self_attn."
   q,k,v=(ranges[base+name+"_proj.weight"][0][0] for name in ["q","k","v"])
   head=128*4096*2;reverse=[]
   for group in range(8):
    reverse.extend([(q+group*4*head,4*head),(k+group*head,head),(v+group*head,head)])
   original=chunk(entries,"model.decoder.layers.self_attention.linear_qkv.weight",f"{layer}.0.0")
   qkv_hash=ranges_hash(out,reverse);assert qkv_hash==ranges_hash(src,[original])
  assert dest.stat().st_size==start+sum(t["bytes"] for t in rows)
 from safetensors import safe_open
 with safe_open(str(dest),framework="np") as f:
  assert set(f.keys())=={t["name"] for t in rows}
  for t in rows:
   s=f.get_slice(t["name"]);assert list(s.get_shape())==t["shape"] and s.get_dtype()=="BF16"
 assert "torch" not in sys.modules
 return {"all_tensor_bytes_equal_mapped_source":True,"source_QKV_reverse_SHA256":qkv_hash,"safetensors_header_reader_verified":True,"torch_imported":False,"tensor_count":len(rows)}
def write_shard(archive,rows,index,entries,signature):
 name=f"model-{index:05d}-of-00034.safetensors";dest=DEST/name;rec=E/(name+".json")
 if rec.exists():
  old=read(rec);assert old["signature"]==signature and sha(dest)==old["output_sha256"]
  return old
 assert not dest.exists(),"Unreceipted export requires inspection"
 data={"__metadata__":{"format":"pt"}};offset=0
 for t in rows:
  data[t["name"]]={"dtype":"BF16","shape":t["shape"],"data_offsets":[offset,offset+t["bytes"]]};offset+=t["bytes"]
 encoded=json.dumps(data,separators=(",",":")).encode();encoded+=b" "*((-len(encoded))%8)
 tmp=dest.with_suffix(dest.suffix+".partial")
 assert not tmp.exists(),"Keep partial for diagnosis; do not silently truncate"
 with archive.open("rb") as src,tmp.open("xb") as out:
  out.write(struct.pack("<Q",len(encoded)));out.write(encoded)
  for t in rows:
   for b in range_chunks(src,t["ranges"]):out.write(b)
  out.flush();os.fsync(out.fileno())
 proof=verify_shard(archive,tmp,rows,index,entries)
 os.replace(tmp,dest)
 result={"signature":signature,"time":now(),"file":str(dest),"output_sha256":sha(dest),"size":dest.stat().st_size,"proof":proof,"tensor_names":[t["name"] for t in rows]}
 save(rec,result);return result
def main(pilot):
 E.mkdir(parents=True,exist_ok=True,mode=0o700)
 lock=(E/"OWNER.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 own=Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
 service=next((n for n in own.split("/") if n.startswith("aki-postslot-llama3-ngc-tensor-export") and n.endswith(".service")),"")
 assert service,"Finite managed unit required"
 properties=prop(service);assert properties["MemoryMax"]==str(2*GIB) and properties["MemorySwapMax"]=="0"
 assert prop("aki-postslot-llama3-ngc-short-range-assets-20261028-v2.service")["ActiveState"]!="active"
 gate=admission();save(E/"RESOURCE_ADMISSION.json",gate)
 if not gate["allowed"]:save(E/"STATUS.json",{"status":"WAIT_RESOURCE_NO_TENSOR_EXPORT","gate":gate});return
 archive,entries,metadata,receipt,ref=inventory();plan=specs(entries)
 assert sys.byteorder=="little"
 DEST.mkdir(parents=True,exist_ok=True,mode=0o700)
 mount=subprocess.run(["findmnt","-T",str(DEST),"-n","-o","TARGET,SOURCE,FSTYPE"],capture_output=True,text=True,check=True).stdout.strip()
 assert mount.split()[0]=="/data" and str(DEST.resolve()).startswith("/data/codex-workspaces/")
 st=os.statvfs(DEST);free=st.f_bavail*st.f_frsize
 assert free>=EXPECTED_DATA_BYTES+4*GIB
 fields={"source_receipt_sha256":sha(A/"CANDIDATE_DOWNLOAD_ACCEPTANCE.json"),"archive_local_SHA256":receipt["archive_sha256"],"archive_bytes":receipt["archive_bytes"],"archive_index_sha256":receipt["archive_index_sha256"],"official_mapping_source_sha256":sha(ref),"script_sha256":sha(__file__),"data_precision":"BF16_bit_patterns_unchanged","source_weight_identity_with_selected_HF_commit":"UNVERIFIED","selected_HF_checkpoint_changed":False,"expected_tensor_bytes":EXPECTED_DATA_BYTES,"expected_tensors":291,"no_pickle_executed":True,"no_Torch_CUDA_or_clinical_data":True}
 signature=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
 contract={"signature":signature,"fields":fields}
 if (E/"CONVERSION_CONTRACT.json").exists():assert read(E/"CONVERSION_CONTRACT.json")==contract
 else:save(E/"CONVERSION_CONTRACT.json",contract)
 if not pilot:
  p=read(E/"PILOT_ACCEPTANCE.json");assert p["signature"]==signature
 indices=[2] if pilot else list(range(1,35));results=[]
 for index in indices:
  z=write_shard(archive,plan[index-1],index,entries,signature);results.append(z)
  save(E/"STATUS.json",{"status":"CONVERTING_TENSOR_FORMAT_ONLY","time":now(),"last_shard":index,"mode":"pilot" if pilot else "full","formal_training_started":False})
 if pilot:
  save(E/"PILOT_ACCEPTANCE.json",{"status":"PASS_REAL_LAYER0_BF16_STREAM_EXPORT_SOURCE_ROUNDTRIP_AND_SAFE_HEADER_ONLY","signature":signature,"shard_receipt_sha256":sha(E/"model-00002-of-00034.safetensors.json"),"tensor_count":9,"GPU_training_qualified":False,"tokenizer_available":False})
 else:
  mapping={t["name"]:f"model-{index:05d}-of-00034.safetensors" for index,rows in enumerate(plan,1) for t in rows}
  save(DEST/"model.safetensors.index.json",{"metadata":{"total_size":EXPECTED_DATA_BYTES},"weight_map":mapping})
  save(E/"TENSOR_EXPORT_ACCEPTANCE.json",{"status":"PASS_ALL_291_BF16_TENSORS_BIT_PRESERVED_SOURCE_ROUNDTRIP_FORMAT_ONLY","time":now(),"signature":signature,"shards":{z["file"]:z["output_sha256"] for z in results},"index_sha256":sha(DEST/"model.safetensors.index.json"),"tensor_count":291,"tensor_bytes":EXPECTED_DATA_BYTES,"tokenizer_available":False,"HF_model_config_qualified":False,"GPU_training_qualified":False,"selected_HF_checkpoint_changed":False,"formal_training_started":False,"clinical_data_read":False})
 save(E/"STATUS.json",{"status":"PILOT_COMPLETE_FULL_EXPORT_PENDING" if pilot else "TENSOR_EXPORT_COMPLETE_EXACT_TOKENIZER_MODEL_CONFIG_AND_GPU_QUALIFICATION_PENDING","time":now(),"owned_workers_remaining":0,"formal_training_started":False})
 print("PASS_TENSOR_FORMAT_PILOT" if pilot else "PASS_TENSOR_FORMAT_ONLY")
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--pilot",action="store_true");args=parser.parse_args()
 try:main(args.pilot)
 except BaseException as e:
  save(E/"FAILURE.json",{"time":now(),"error_class":type(e).__name__,"diagnostic_message":str(e)[:800],"phase":"pilot" if args.pilot else "full","formal_training_started":False});raise
