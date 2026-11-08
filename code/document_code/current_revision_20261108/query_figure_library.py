import json,subprocess,os,select,time,tomllib
from pathlib import Path
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/workbuddy_final_revision_20261108_v1')
cfg=tomllib.loads(Path('/home/lei/codex-workspaces/scientific-figure-library/config/codex-server-entry.toml').read_text())['mcp_servers']['figure-library']
env=os.environ.copy();env.update(cfg['env'])
err=(W/'review/SFL_STDERR.log').open('w')
p=subprocess.Popen([cfg['command']]+cfg['args'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=env)
buf=b''
def send(x):
 p.stdin.write((json.dumps(x)+'\n').encode());p.stdin.flush()
def response(i,timeout=55):
 global buf
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  while b'\n' in buf:
   ln,buf=buf.split(b'\n',1)
   try:r=json.loads(ln)
   except Exception:continue
   if r.get('id')==i:return r
  ready,_,_=select.select([p.stdout],[],[],min(1,max(0,end-time.monotonic())))
  if ready:
   chunk=os.read(p.stdout.fileno(),262144)
   if not chunk:raise RuntimeError('SFL process exited '+str(p.poll()))
   buf+=chunk
 raise TimeoutError('SFL response '+str(i))
try:
 send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'aki-figure-review','version':'1'}}})
 init=response(1)
 send({'jsonrpc':'2.0','method':'notifications/initialized','params':{}})
 send({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})
 listing=response(2);(W/'review/SFL_TOOL_SCHEMAS.json').write_text(json.dumps(listing,indent=2))
 wanted=[t for t in listing.get('result',{}).get('tools',[]) if t['name'] in ['figure_library_source_status','figure_library_search']]
 send({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'figure_library_source_status','arguments':{}}})
 status=response(3)
 (W/'review/SFL_SOURCE_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
 send({'jsonrpc':'2.0','id':4,'method':'tools/call','params':{'name':'figure_library_search','arguments':{'query':'clinical prediction cohort comparison forest plot calibration and workload distribution','dataProfile':'Patient-level cohort evaluations with paired bootstrap intervals, center calibration, daily counts of kidney-risk review demand indexed to actual surgery date. Existing frozen figures must retain their results.','visualProfile':'Publication-size clinical cohort panels, aligned point estimates and confidence intervals; daily-count distributions when scientifically justified; editable Python vector panels.','limit':3}}})
 result=response(4)
 (W/'review/SFL_REFERENCE_SEARCH.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({'source_status':status.get('result',{}).get('structuredContent',{}),'search':result.get('result',{}).get('structuredContent',{})},ensure_ascii=False)[:15000])
finally:
 p.stdin.close()
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
 err.close()
