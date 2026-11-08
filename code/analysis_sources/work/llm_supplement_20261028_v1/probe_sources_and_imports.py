"""Check current local APIs and existing authorized official repository access without exposing credentials."""
from pathlib import Path
import datetime,hashlib,importlib.metadata as md,json,os
import torch,transformers,peft,bitsandbytes
from huggingface_hub import HfApi,hf_hub_url,get_hf_file_metadata,hf_hub_download
R=Path(__file__).resolve().parents[2];O=R/'outputs/llm_supplement_20261028_v1'
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
assert torch.__version__=='2.6.0+cu118'
versions={n:md.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','huggingface-hub','numpy']}
save(O/'IMPORT_QUALIFICATION.json',{'status':'PASS_pinned_extra_imports_only_no_GPU_training_claim','versions':versions,'cuda_build':torch.version.cuda,'compile_arches_include_Pascal': 'sm_60' in torch.cuda.get_arch_list(),'GPU_training_qualified':False,'time':stamp()})
selection=json.loads((O/'MODEL_SELECTION.json').read_text());api=HfApi(endpoint='https://huggingface.co');access=[]
for model in selection['models']:
 repo=model['repository'];item={'id':model['id'],'repository':repo,'time':stamp()}
 try:
  prior=model.get('weight_revision','');revision=prior if len(prior)==40 and all(c in '0123456789abcdef' for c in prior) else None
  info=api.model_info(repo,revision=revision,files_metadata=True,timeout=20)
  model['weight_revision']=info.sha
  item.update(revision=info.sha,official_gate=getattr(info,'gated',None))
  assert len(info.sha)==40
  files={}
  for s in info.siblings:
   if s.rfilename.endswith(('.safetensors','.bin')):
    lfs=getattr(s,'lfs',None);files[s.rfilename]={'bytes':s.size,'provider_SHA256':(lfs.sha256 if hasattr(lfs,'sha256') else lfs.get('sha256')) if lfs else None}
  item['official_weight_files']=files
  metadata=get_hf_file_metadata(hf_hub_url(repo,'config.json',revision=info.sha),timeout=15)
  item['config_authorized_access']='PASS'
  target=O/'source_metadata'/model['id'];target.mkdir(parents=True,exist_ok=True)
  config=Path(hf_hub_download(repo_id=repo,filename='config.json',revision=info.sha,local_dir=str(target)))
  cfg=json.loads(config.read_text());item['architecture']=cfg.get('architectures');item['model_type']=cfg.get('model_type');item['config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
  item['status']='PASS_official_metadata_and_existing_authorized_config_access'
  model.update(weight_revision=info.sha,access=item['status'])
 except Exception as e:
  response=getattr(e,'response',None)
  item.update(status='ACCESS_OR_TRANSPORT_BLOCKED_REQUIRES_DIAGNOSIS',error_class=type(e).__name__,http_status=getattr(response,'status_code',None))
  if type(e).__name__ in ['GatedRepoError','RepositoryNotFoundError'] or getattr(response,'status_code',None) in [401,403]:
   item['status']='OFFICIAL_REPOSITORY_ACCESS_NOT_GRANTED_WITH_CURRENT_SERVER_AUTH'
   item['required_action']='Use own Hugging Face account to accept the official repository access conditions and configure standard server authentication outside project files; do not send tokens in chat.'
  model.update(access=item['status'])
 access.append(item);save(O/'SOURCE_ACCESS_STATUS.json',{'status':'SOURCE_ACCESS_PROBE_IN_PROGRESS' if len(access)<3 else 'COMPLETE_OFFICIAL_SOURCE_ACCESS_PROBE','models':access,'patient_data_sent':False,'credentials_stored_in_project':False})
 print('SOURCE_PROBE',repo,item['status'],item.get('http_status'),flush=True)
selection['status']='THREE_MODELS_SELECTED_EXACT_REVISIONS_IF_ACCESSIBLE_GPU_QUALIFICATION_PENDING';selection['source_probe_time']=stamp()
save(O/'MODEL_SELECTION.json',selection)
