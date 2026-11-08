from pathlib import Path
import subprocess,json,os,select,time
r=Path(__file__).resolve().parents[1];root=Path('/home/lei/codex-workspaces/scientific-figure-library');env=os.environ.copy()
for k,sub in [('XDG_CONFIG_HOME','config'),('XDG_CACHE_HOME','cache'),('XDG_DATA_HOME','data'),('XDG_STATE_HOME','state'),('TMPDIR','tmp'),('SFL_DIAGNOSTICS_DIR','diagnostics')]:env[k]=str(root/sub)
env['SFL_MCP_UPDATE_NOTICES']='false';err=(r/'logs/FIGURE_REFERENCE_STDERR.log').open('w');p=subprocess.Popen(['/home/lei/.nvm/versions/node/v22.22.0/bin/node',str(root/'scripts/mcp-launch.mjs')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env,cwd=root,bufsize=1);seq=0;events=[]
def send(method,params,notification=False):
 global seq
 seq+=1;key=seq;req={'jsonrpc':'2.0','method':method,'params':params}
 if not notification:req['id']=key
 p.stdin.write(json.dumps(req)+'\n');p.stdin.flush()
 if notification:return
 until=time.monotonic()+45
 while time.monotonic()<until:
  readable,_,_=select.select([p.stdout],[],[],max(0,until-time.monotonic()))
  if not readable:break
  line=p.stdout.readline()
  if not line:raise RuntimeError('owned library stdio closed')
  try:d=json.loads(line)
  except ValueError:continue
  if d.get('id')==key:
   if 'error' in d:raise RuntimeError(str(d['error']))
   events.append({'method':method,'params':params,'result':d['result']});return d['result']
 raise TimeoutError(method)
try:
 send('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'aki-jno-supplement-format-reference','version':'1'}});send('notifications/initialized',{},True)
 schema=send('tools/list',{});schemas={x['name']:x['inputSchema'] for x in schema['tools']}
 status=send('tools/call',{'name':'figure_library_source_status','arguments':{}})
 result=send('tools/call',{'name':'figure_library_search','arguments':{'query':'clinical forest plot confidence intervals multi-panel calibration workload','limit':3}})
 (r/'audit/FIGURE_LIBRARY_REFERENCE.json').write_text(json.dumps({'events':events,'search_schema':schemas.get('figure_library_search'),'source_status':status,'search_result':result,'reference_only_no_template_execution':True},indent=2))
 print(json.dumps({'status':status.get('structuredContent',{}),'search':result.get('structuredContent',{})},ensure_ascii=False)[:4500])
finally:
 p.stdin.close()
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=3)
 err.close()
