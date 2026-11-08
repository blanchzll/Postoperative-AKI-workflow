from pathlib import Path
import json,subprocess,shutil,os,time
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/probabilistic_integration_20261105_v1')
P=W/'package'
docs=json.loads((W/'review/RENDER_DOCUMENTS.json').read_text())
out=W/'review/rendered';out.mkdir(exist_ok=True)
cmd=['docker','run','--rm','--name','aki-probabilistic-writing-render-20261105-v1','--user',f'{os.getuid()}:{os.getgid()}','--memory','8g','-v',str(W)+':/work','--entrypoint','/usr/bin/libreoffice','codex-docx-render:20260823','-env:UserInstallation=file:///tmp/aki-probabilistic-render-profile','--headless','--convert-to','pdf','--outdir','/work/review/rendered']
cmd += ['/work/package/'+rel for rel in docs]
started=time.monotonic()
with (W/'review/RENDER_LOG.txt').open('w') as log:
 proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
assert proc.returncode==0,proc.returncode
for rel in docs:
 source=out/Path(rel).with_suffix('.pdf').name
 assert source.is_file() and source.stat().st_size>500,rel
 shutil.copy2(source,(P/rel).with_suffix('.pdf'))
(W/'review/RENDER_ACCEPTANCE.json').write_text(json.dumps({'exit_code':proc.returncode,'elapsed_seconds':time.monotonic()-started,'files':docs,'owned_container_exited':True,'renderer':'existing codex-docx-render:20260823','other_jobs_changed':False},indent=2))
print(json.dumps({'exit_code':proc.returncode,'rendered':len(docs),'elapsed_seconds':round(time.monotonic()-started,2)}))
