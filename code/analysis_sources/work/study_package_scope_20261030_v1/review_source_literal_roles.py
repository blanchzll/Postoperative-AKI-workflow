"""Bounded literal screen plus manually reviewed source roles; never runs candidate code."""
import resource,time,signal,os
resource.setrlimit(resource.RLIMIT_AS,(128*1024*1024,128*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(90,90))
signal.alarm(120)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
started=time.monotonic()
from pathlib import Path
import json,hashlib,re,ast,collections
r=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
b=r/'outputs/study_package_scope_20261030_v1'
rows=json.loads((b/'dependency_disposition_v3/CURRENT_SOURCE_INVENTORY_INTERNAL.json').read_text())
refs=collections.defaultdict(set)
for v in rows:refs[v['path']].add(v['sha256'])
def crawl(x):
 if isinstance(x,dict):
  path=x.get('path');h=x.get('sha256')
  if isinstance(path,str) and isinstance(h,str) and re.fullmatch(r'[0-9a-f]{64}',h):
   if path.startswith(str(r)+'/'):path=str(Path(path).relative_to(r))
   if path.startswith(('work/','outputs/')):refs[path].add(h)
  for k,v in x.items():
   if isinstance(k,str) and isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v):
    z=k
    if z.startswith(str(r)+'/'):z=str(Path(z).relative_to(r))
    if z.startswith(('work/','outputs/')):refs[z].add(v)
   else:crawl(v)
 elif isinstance(x,list):
  for v in x:crawl(v)
for p in sorted(b.glob('SOURCE_INVENTORY_*ADDENDUM*.json')):crawl(json.loads(p.read_text()))
report=[];counts=collections.Counter()
for rel,hashes in sorted(refs.items()):
 p=r/rel
 if p.suffix not in ('.py','.js','.sh','.html','.css'):continue
 raw=p.read_bytes();s=raw.decode();h=hashlib.sha256(raw).hexdigest()
 assert h in hashes, 'Source hash unregistered: '+rel
 counts['source_files']+=1;counts['source_bytes']+=len(raw)
 findings=[]
 for lineno,line in enumerate(s.splitlines(),1):
  if re.search(r'(?<![0-9])(?:171|122|739)(?![0-9])',line) and any(v in line.lower() for v in ['event','aki','preop','paired','assert','n=','clinical','739','count']):
   findings.append({'line':lineno,'type':'old_count_literal_requires_role_check','text':line[:250]})
  if re.search(r'hf_[A-Za-z0-9]{20,}|(?:sk-[A-Za-z0-9]{20,})',line):
   findings.append({'line':lineno,'type':'credential_like_literal_DO_NOT_PRINT','text_sha256':hashlib.sha256(line.encode()).hexdigest()})
 if p.suffix=='.py':
  tree=ast.parse(s)
  for node in ast.walk(tree):
   if not isinstance(node,(ast.Assign,ast.AnnAssign)):continue
   names=[t.id for t in ast.walk(node) if isinstance(t,ast.Name) and isinstance(t.ctx,ast.Store)]
   if not any(re.search(r'(^HIGH$|patient.*id|group.*id|excluded.*rows|exclude.*ids|bad.*rows|row.*ids)',n,re.I) for n in names):continue
   value=getattr(node,'value',None)
   try:v=ast.literal_eval(value)
   except (ValueError,TypeError,SyntaxError):continue
   if isinstance(v,(list,set,tuple,dict)) and len(v)>0:
    findings.append({'line':node.lineno,'type':'literal_row_or_identifier_collection','names':names,'items':len(v),'text':s.splitlines()[node.lineno-1][:250]})
 if findings:report.append({'path':rel,'sha256':h,'findings':findings})
screen_result={'source_counts':dict(counts),'flagged_files':len(report),'findings':report}


from datetime import datetime
from zoneinfo import ZoneInfo
out=b/'source_literal_role_review_v1'
assert not out.exists(), 'Do not overwrite an existing source-role review; inspect its receipt first'
# Keep the registry and each selected original byte-identical.
registry=b/'dependency_disposition_v3/CURRENT_SOURCE_INVENTORY_INTERNAL.json'
registry_sha=hashlib.sha256(registry.read_bytes()).hexdigest()
registered_original={rel:hashlib.sha256((r/rel).read_bytes()).hexdigest() for rel in refs if (r/rel).suffix in ('.py','.js','.sh','.html','.css')}
assert len(registered_original)==307 and len(report)==5
expected_flagged={
'work/fixed_flow_sensitivity_evaluation_20261024_v1/evaluate.py',
'work/llm_delivery_current_20261029_v1/independent_metrics.py',
'work/next_stage_preparation_20261022_v1/prepare.py',
'work/published_benchmark_comparison_20261024_v1/run.py',
'work/study_package_scope_20261030_v1/audit_scope.py'}
assert {x['path'] for x in report}==expected_flagged
guard=json.loads((r/'outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json').read_text())
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(1048576),b''):h.update(q)
 return h.hexdigest()
for path,v in guard['files'].items():
 assert file_sha(path)==(v if isinstance(v,str) else v['sha256'])
assert len(guard['files'])==92
legacy=r/'work/next_stage_preparation_20261022_v1/prepare.py'
t=ast.parse(legacy.read_text())
assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.Expr,ast.If)) for n in t.body)
main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
guarded=[n for n in t.body if isinstance(n,ast.If) and '__main__' in ast.unparse(n.test)]
assert len(guarded)==1 and len(guarded[0].body)==1
assert ast.unparse(guarded[0].body[0])=='main()'
closeout=r/'work/final_evaluation_engine_20261022_v1/closeout.py'
ct=ast.parse(closeout.read_text())
imports=[n for n in ct.body if isinstance(n,ast.ImportFrom) and n.module=='prepare']
assert len(imports)==1 and [a.name for a in imports[0].names]==['sha','read','atomic']
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('main','boundary','children_cover') for n in ast.walk(ct))
helper_ast={name:hashlib.sha256(ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False).encode()).hexdigest() for name in ['sha','read','atomic']}
current_input=r/'outputs/next_stage_preparation_20261022_v1/COMMON6_SOURCE_INPUT_CONTRACT.json'
contract=json.loads(current_input.read_text())
assert contract['source_lock_sha256']=='afa0e1224717aee6b86a8ba3598c9214b24fc02028ebc8e7b4869dbcaf0874d3'
assert contract['expected_development']=={'geography':[2551,133],'temporal':[2024,139]}
for line,counts2 in contract['expected_development'].items():
 d=json.loads((r/('outputs/common6_development_20261024_v1/inputs/'+line+'/INPUT_ACCEPTANCE.json')).read_text())
 assert [d['n'],d['events']]==counts2
assert not (r/'outputs/next_stage_preparation_20261022_v1/ACCEPTANCE.json').exists()
decisions=[
 {'path':'work/fixed_flow_sensitivity_evaluation_20261024_v1/evaluate.py',
  'classification':'CONFIRMED_STALE_EVALUATION_MASK_ALREADY_TRACKED',
  'evidence_lines':[11],'linked_issue':'FF-MASK-STALE-20261030',
  'disposition':'Original preserved as private historical provenance with embedded record indices; do not use its main entry for corrected evaluation or shared export.',
  'current_replacement':'work/fixed_flow_sensitivity_evaluation_current_20261030_r1/evaluate.py',
  'numeric_correction_complete':False},
 {'path':'work/next_stage_preparation_20261022_v1/prepare.py',
  'classification':'LEGACY_MAIN_ASSERTIONS_NOT_CURRENT_SOURCE_PREPARATION',
  'evidence_lines':[60,79,80,91,183],
  'disposition':'Do not run main on corrected inputs. Preserve exact original source for explicit hash/read/write helper imports and historical provenance; not an approved standalone reproduction entry.',
  'helper_consumer':'work/final_evaluation_engine_20261022_v1/closeout.py',
  'helpers_actually_imported':['sha','read','atomic'],
  'helper_AST_sha256':helper_ast,
  'legacy_preparation_acceptance_present':False,
  'current_source_construction':'work/reanalysis_control/secondary_common6.py',
  'current_construction_context':'Executed by the existing master with its control globals; not a new standalone CLI',
  'current_development':contract['expected_development'],
  'no_evidence_of_current_common6_numeric_contamination_from_legacy_main':True},
 {'path':'work/published_benchmark_comparison_20261024_v1/run.py',
  'classification':'HISTORICAL_ENTRYPOINT_PREVIOUSLY_DISPOSED',
  'evidence_lines':[64,65,75,76],
  'disposition':'Reuse prior ENTRYPOINT_DISPOSITIONS; do not run obsolete count/winner assertions.',
  'current_replacement':'work/post_master_delivery_20261028_v1/published_benchmark_current_v1.py'},
 {'path':'work/llm_delivery_current_20261029_v1/independent_metrics.py',
  'classification':'SCREEN_FALSE_POSITIVE_NUMERICAL_BRACKET',
  'evidence_lines':[16,19,20,21,22,23,24],
  'disposition':'low/high are bisection bounds for the calibration-offset intercept, not record identifiers. Preserve numeric source; no correction.'},
 {'path':'work/study_package_scope_20261030_v1/audit_scope.py',
  'classification':'SCREEN_FALSE_POSITIVE_SEARCH_PATTERN',
  'evidence_lines':[114,115,116,117,118,119,125,126],
  'disposition':'Old count strings are patterns used to flag the old builder; current source counts come from the gate. Preserve audit source; no numerical correction.'}]
for row in decisions:row['source_sha256']=registered_original[row['path']]
related_closeout={
 'path':str(closeout.relative_to(r)),'source_sha256':file_sha(closeout),
 'classification':'HISTORICAL_ENGINEERING_CLOSEOUT_NOT_GENERAL_REPLAY_ENTRYPOINT',
 'evidence_lines':[8,17,20,21],
 'reason':'Imports only three I/O helpers and requires the then-running E state; do not advertise direct execution on the completed current analysis.',
 'no_original_source_edit':True}
evidence_paths=[
 'work/reanalysis_control/secondary_common6.py',
 'work/final_evaluation_engine_20261022_v1/closeout.py',
 'outputs/next_stage_preparation_20261022_v1/COMMON6_SOURCE_INPUT_CONTRACT.json',
 'outputs/common6_development_20261024_v1/inputs/geography/INPUT_ACCEPTANCE.json',
 'outputs/common6_development_20261024_v1/inputs/temporal/INPUT_ACCEPTANCE.json',
 'outputs/study_package_scope_20261030_v1/dependency_disposition_v3/ENTRYPOINT_DISPOSITIONS.json',
 'outputs/reanalysis_control/fixed_flow_mask_correction_20261030_r1/DIAGNOSIS.json']
out.mkdir(mode=0o700)
def save(name,x):
 p=out/name
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
 return file_sha(p)
screen_sha=save('TARGETED_LITERAL_SCREEN_INTERNAL.json',{**screen_result,'all_current_source_SHA_verified':True,'automatic_full_privacy_or_runnability_approval':False,'screen_limitations':'Focused literals and assigned-identifier patterns only; no proof of absence of all private content, stale logic, dynamic imports or semantic errors. Do not export this raw screen containing historical record indices.'})
decisions_sha=save('MANUAL_ROLE_DECISIONS.json',{'status':'FIVE_FLAGGED_SOURCE_CONTEXTS_ACTUALLY_READ_AND_DISPOSED','decisions':decisions,'related_closeout':related_closeout,'evidence_sha256':{p:file_sha(r/p) for p in evidence_paths},'scientific_review':False,'full_shared_source_whitelist':False})
inventory_sha=save('SOURCE_SCREENED_HASH_INDEX_INTERNAL.json',[{'path':p,'sha256':h,'content_approval':'NOT_AUTOMATICALLY_SHARED_APPROVED'} for p,h in sorted(registered_original.items())])
assert registry_sha==file_sha(registry)
assert all(file_sha(r/p)==h for p,h in registered_original.items())
for path,v in guard['files'].items():assert file_sha(path)==(v if isinstance(v,str) else v['sha256'])
acceptance={'status':'PASS_BOUNDED_LITERAL_SCREEN_AND_SPECIFIC_MANUAL_ROLE_REVIEW','time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'source_files':len(registered_original),'source_bytes':screen_result['source_counts']['source_bytes'],'flagged_files':5,'manually_read_flagged_contexts':5,'additional_closeout_context_read':True,'new_data_or_fits':0,'original_source_files_changed':0,'original_inventory_changed':False,'protected_files_checked':92,'code_sha256':file_sha(__file__),'outputs_sha256':{'TARGETED_LITERAL_SCREEN_INTERNAL.json':screen_sha,'MANUAL_ROLE_DECISIONS.json':decisions_sha,'SOURCE_SCREENED_HASH_INDEX_INTERNAL.json':inventory_sha},'whole_source_privacy_review':False,'complete_package_acceptance':False,'scientific_review':False,'seconds':time.monotonic()-started,'max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
save('SOURCE_ROLE_REVIEW_RECEIPT.json',acceptance)
print(json.dumps({k:v for k,v in acceptance.items() if k!='outputs_sha256'},ensure_ascii=False))
