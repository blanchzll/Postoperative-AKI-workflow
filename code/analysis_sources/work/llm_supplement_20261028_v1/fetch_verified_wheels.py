"""Resolve full official PyPI wheels and verify whole-file provider hashes; do not waive hash checks."""
from pathlib import Path
import hashlib,json,os,urllib.request
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename
R=Path(__file__).resolve().parents[2];O=R/'outputs/llm_supplement_20261028_v1';D=R/'model_assets/wheel_cache/provider_verified';D.mkdir(parents=True,exist_ok=True)
order={tag:i for i,tag in enumerate(sys_tags())};records=[];requirements=[]
for package,version in [('peft','0.18.0'),('bitsandbytes','0.48.1')]:
 url=f'https://pypi.org/pypi/{package}/{version}/json'
 with urllib.request.urlopen(url,timeout=30) as response:data=json.load(response)
 candidates=[]
 for item in data['urls']:
  if item['packagetype']!='bdist_wheel':continue
  name,v,build,tags=parse_wheel_filename(item['filename']);ranks=[order[tag] for tag in tags if tag in order]
  if ranks:candidates.append((min(ranks),item))
 assert candidates,(package,'no compatible official wheel')
 _,item=min(candidates,key=lambda pair:pair[0]);expected=item['digests']['sha256'];dest=D/item['filename']
 assert item['url'].startswith('https://files.pythonhosted.org/')
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
  return h.hexdigest()
 if not dest.exists():
  part=dest.with_name(dest.name+'.part')
  with urllib.request.urlopen(item['url'],timeout=60) as response,part.open('wb') as f:
   for block in iter(lambda:response.read(4*1024*1024),b''):f.write(block)
  assert part.stat().st_size==item['size'] and digest(part)==expected,(package,'official full-wheel size/hash mismatch; no installation')
  os.replace(part,dest)
 assert dest.stat().st_size==item['size'] and digest(dest)==expected
 records.append({'package':package,'version':version,'official_version_metadata':url,'official_file_url':item['url'],'path':str(dest),'bytes':item['size'],'sha256':expected})
 requirements.append(str(dest)+' --hash=sha256:'+expected)
 print('VERIFIED_OFFICIAL_WHEEL',package,version,item['size'],flush=True)
(O/'VERIFIED_WHEELS_REQUIREMENTS.txt').write_text('\n'.join(requirements)+'\n')
(O/'WHEEL_INTEGRITY_ACCEPTANCE.json').write_text(json.dumps({'status':'PASS_official_whole_wheel_provider_SHA256_before_offline_hash_required_install','files':records,'hash_verification_disabled':False,'reason':'Original network index detached wheel-metadata hash mismatch. Use independently provider-verified full wheels and hash-required local installation; no package-version or shared-backend change.'},ensure_ascii=False,indent=2)+'\n')
