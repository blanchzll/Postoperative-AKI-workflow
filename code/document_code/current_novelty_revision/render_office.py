from pathlib import Path
import subprocess,json,os,time
W=Path(__file__).resolve().parents[1]
name=f'aki-final-render-20261101-{os.getpid()}'
files=[str(p).replace(str(W),'/work') for p in (W/'package').rglob('*.docx')]
cmd=['docker','run','--rm','--name',name,'--network','none','--memory','4g','--user',f'{os.getuid()}:{os.getgid()}','--mount',f'type=bind,source={W},target=/work','--entrypoint','/usr/bin/libreoffice','codex-docx-render:20260823','-env:UserInstallation=file:///tmp/aki-final-office','--headless','--convert-to','pdf','--outdir','/work/review/rendered',*files]
start=time.time()
try:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
 (W/'review/OFFICE_RENDER.log').write_text(r.stdout+'\n'+r.stderr)
 print(r.stdout[-1800:]);print('exit',r.returncode)
 assert r.returncode==0
 for p in files:assert (W/'review/rendered'/(Path(p).stem+'.pdf')).exists(),p
finally:
 # Remove only this task's still-running container if the renderer failed or timed out.
 r2=subprocess.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],capture_output=True,text=True)
 if r2.stdout.strip():subprocess.run(['docker','stop','--timeout','5',name],capture_output=True,text=True)
(W/'review/OFFICE_RENDER_RECEIPT.json').write_text(json.dumps({'status':'PASS_LIBREOFFICE_RENDER','files':files,'seconds':time.time()-start,'container_name':name,'remaining_owned_container':False,'renderer_image':'codex-docx-render:20260823','network':'none'},indent=2))
