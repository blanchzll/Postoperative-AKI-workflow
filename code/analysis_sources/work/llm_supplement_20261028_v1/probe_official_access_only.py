from pathlib import Path
import datetime,json
from huggingface_hub import HfApi,hf_hub_url,get_hf_file_metadata
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
O=R/'outputs/llm_supplement_20261028_v1';selection=json.loads((O/'MODEL_SELECTION.json').read_text());api=HfApi(endpoint='https://huggingface.co')
out=[]
for model in selection['models']:
 repo=model['repository'];d={'id':model['id'],'repository':repo}
 try:
  info=api.model_info(repo,files_metadata=True,timeout=8);d.update(revision=info.sha,official_gate=getattr(info,'gated',None))
  config=get_hf_file_metadata(hf_hub_url(repo,'config.json',revision=info.sha),timeout=8)
  d.update(status='PASS_existing_authorized_official_config_access',config_commit=config.commit_hash,config_bytes=config.size)
  model.update(weight_revision=info.sha,access=d['status'])
 except Exception as e:
  response=getattr(e,'response',None);d.update(status='TRANSPORT_OR_ACCESS_UNAVAILABLE',error_class=type(e).__name__,http_status=getattr(response,'status_code',None))
  if type(e).__name__=='GatedRepoError':d['status']='OFFICIAL_REPOSITORY_PERMISSION_REQUIRED'
  model['access']=d['status']
 out.append(d)
 (O/'SOURCE_ACCESS_STATUS.json').write_text(json.dumps({'status':'COMPLETE_OFFICIAL_SOURCE_ACCESS_PROBE' if len(out)==3 else 'SOURCE_ACCESS_PROBE_IN_PROGRESS','models':out,'existing_server_credentials_only':True,'credentials_stored_in_project':False,'patient_data_sent':False,'time_Asia_Shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(d,ensure_ascii=False),flush=True)
selection['source_probe_complete']=True
(O/'MODEL_SELECTION.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
