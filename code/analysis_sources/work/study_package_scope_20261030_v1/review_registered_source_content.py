#!/usr/bin/env python3
"""Record a fixed, reviewed code-content eligibility snapshot; never run candidate code."""
from pathlib import Path
import ast,csv,datetime,hashlib,json,os,re,resource,signal,time
R=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot")
B=R/"outputs/study_package_scope_20261030_v1"
OUT=B/"registered_source_content_eligibility_v1"
INDEX=B/"source_literal_role_review_v1/SOURCE_SCREENED_HASH_INDEX_INTERNAL.json"
PRIVATE=B/"source_content_disposition_v1/PRIVATE_SOURCE_CONTENT_DISPOSITIONS.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for q in iter(lambda:f.read(1048576),b""):h.update(q)
 return h.hexdigest()
def save(p,x):
 with Path(p).open("x",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write("\n")
 os.chmod(p,0o600)
def main():
 began=time.monotonic()
 resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
 if hasattr(os,"sched_getaffinity"):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 assert sha(INDEX)=="10153225edcb5c1a0ca75a22d0aa44ca8b731062fd53c977dc5f4a4214fafa64"
 assert sha(PRIVATE)=="642758de29886ac790b7ef2d9f0192e30088fe382c06fa34642c1d961a35382b"
 source_rows=json.loads(INDEX.read_text());refs={x["path"]:x["sha256"] for x in source_rows}
 origins={p:["307-file exact-hash source index"] for p in refs}
 addenda={}
 def add(p,h,origin):
  if not isinstance(p,str) or not isinstance(h,str) or not re.fullmatch("[0-9a-f]{64}",h):return
  q=Path(p)
  if q.is_absolute():
   try:q=q.relative_to(R)
   except ValueError:return
  p=q.as_posix()
  if q.suffix not in (".py",".js",".html") or not p.startswith(("work/","outputs/")):return
  assert (R/p).is_file() and sha(R/p)==h,p
  if p in refs:assert refs[p]==h,p
  refs[p]=h;origins.setdefault(p,[]).append(origin)
 def crawl(x,origin):
  if isinstance(x,dict):
   if "path" in x:
    for key in ("sha256","code_sha256","source_sha256"):
     if key in x:add(x["path"],x[key],origin)
   for k,v in x.items():
    if isinstance(v,str):add(k,v,origin)
    else:crawl(v,origin)
  elif isinstance(x,list):
   for v in x:crawl(v,origin)
 for p in sorted(B.glob("SOURCE_INVENTORY_*ADDENDUM*.json")):
  addenda[p.name]=sha(p);crawl(json.loads(p.read_text()),p.name)
 assert len(refs)==310
 this=str(Path(__file__).relative_to(R));refs[this]=sha(__file__);origins[this]=["Current source written and fully reviewed in this task; contains only code, source paths/hashes, classification and screening rules"]
 assert len(refs)==311
 private={x["path"]:x for x in json.loads(PRIVATE.read_text())["private_sources"]};assert len(private)==4
 for p,x in private.items():assert refs[p]==x["sha256"]
 baseline_tree=ast.parse((R/"work/fixed_flow_sensitivity_evaluation_20261024_v1/evaluate.py").read_text())
 identifiers=ast.literal_eval(next(x.value for x in ast.walk(baseline_tree) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="HIGH" for t in x.targets)))
 assert len(identifiers)==3
 prior_screen=B/"source_content_disposition_v1/SOURCE_CONTENT_DISPOSITION_RECEIPT.json"
 assert sha(prior_screen)=="70ea9809b5c15c9ad4657798532062d2aa498cce5f728a1256d882e68e55e363"
 prior=json.loads(prior_screen.read_text())
 for name,h in prior["outputs"].items():assert sha(prior_screen.parent/name)==h
 additional=[];known_identifier_files=set();source_bytes=0;python_files=0
 idfields={"record_key","excel_row","patient_id","patientunitstayid","subject_id","group_hash","case_key"}
 credentials=[r"hf_[A-Za-z0-9]{20,}",r"sk-[A-Za-z0-9_-]{20,}",r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",r"(?:AKIA|ASIA)[A-Z0-9]{16}"]
 literal_credential_count=0
 for p,h in sorted(refs.items()):
  path=R/p;assert not path.is_symlink();raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==h
  s=raw.decode();source_bytes+=len(raw)
  found=[i for i,line in enumerate(s.splitlines(),1) if any(re.search(r"(?<![A-Za-z0-9])"+str(v)+r"(?![A-Za-z0-9])",line) for v in identifiers)]
  if found:known_identifier_files.add(p)
  literal_credential_count+=sum(len(list(re.finditer(pattern,s))) for pattern in credentials)
  flags=[]
  if path.suffix==".py":
   python_files+=1;t=ast.parse(s)
   for n in ast.walk(t):
    if isinstance(n,ast.Constant) and isinstance(n.value,str):
     tests=[
      ("clinical_scalar_in_text",r"\b(?:Age|PreopAlb|PreopCr|PreopHb|PreopALT|Creatinine|Albumin)\s*[:=]\s*[\"']?[-+]?\d"),
      ("email_in_literal_REDACTED",r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
      ("long_numeric_identity_candidate_REDACTED",r"(?<![A-Za-z0-9])[1-9][0-9]{16}[0-9Xx](?![A-Za-z0-9])")]
     for kind,pattern in tests:
      if re.search(pattern,n.value,re.I):flags.append({"kind":kind,"line":n.lineno,"length":len(n.value)})
    if isinstance(n,ast.Dict):
     keys=[str(k.value) for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant) and str(k.value) in idfields and isinstance(v,ast.Constant) and isinstance(v.value,(str,int))]
     if keys:flags.append({"kind":"literal_identifier_dictionary","line":n.lineno,"keys":keys,"values_omitted":True})
    if isinstance(n,ast.Assign):
     names=[x.id for x in n.targets if isinstance(x,ast.Name)]
     if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>=8 and any(re.search(r"(?:^|_)(?:token|password|secret|cookie|api_key|authorization)(?:$|_)",a,re.I) for a in names):flags.append({"kind":"possible_credential_assignment_REDACTED","line":n.lineno,"names":names})
     if isinstance(n.value,(ast.List,ast.Set,ast.Tuple)) and any(re.search(r"(?:^|_)(?:ids?|pids?|idx|indices|rows?|patients?|records?|cases?)(?:$|_)",a,re.I) for a in names):
      vals=[x.value for x in n.value.elts if isinstance(x,ast.Constant)]
      if vals and all(isinstance(v,(int,float)) or isinstance(v,str) and re.fullmatch("[0-9]+",v) for v in vals):flags.append({"kind":"identifier_named_numeric_collection","line":n.lineno,"names":names,"count":len(vals)})
  if flags:additional.append({"path":p,"sha256":h,"flags":flags})
 assert known_identifier_files==set(private),sorted(known_identifier_files)
 assert literal_credential_count==0
 interpretations={
  "work/calculator_current_20261029_v1/native_route_qualification.py":"Line53 constructs the all-missing DEMO fixture, not an actual subject.",
  "work/calculator_current_20261029_v1/prepare_current.py":"Line55 constructs the explicitly synthetic DEMO key; clinical defaults are sourced only from the original synthetic_examples fixture, not a clinical row.",
  "work/process_resolution_current_20261029_v1/prepare.py":"Line16 renames field names for the public input schema. The literal is a column name, not an individual identifier."
 }
 assert {x["path"] for x in additional}==set(interpretations),additional
 roles=json.loads((B/"dependency_disposition_v3/ENTRYPOINT_DISPOSITIONS.json").read_text())
 role_doc=json.loads((B/"source_literal_role_review_v1/MANUAL_ROLE_DECISIONS.json").read_text())
 special={}
 for p,x in roles.items():special[p]={"role":x["role"],"reason":x["reason"],"replacement":x.get("replacement")}
 for x in role_doc["decisions"]:
  if x["classification"].startswith("SCREEN_FALSE_POSITIVE"):continue
  special.setdefault(x["path"],{"role":x["classification"],"reason":x["disposition"],"replacement":x.get("current_replacement")})
 x=role_doc["related_closeout"];special[x["path"]]={"role":x["classification"],"reason":x["reason"],"replacement":None}
 selections=[]
 for p,h in sorted(refs.items()):
  isprivate=p in private
  role=special.get(p,{"role":"SOURCE_REFERENCE_ONLY_NO_STANDALONE_EXECUTION_APPROVAL_FROM_CONTENT_REVIEW","reason":"Use the actual qualified current execution guide and dependency context; file presence/content eligibility does not authorize running its entrypoint.","replacement":None})
  selections.append({"source_relative":p,"archive_path":p,"sha256":h,"bytes":(R/p).stat().st_size,
   "classification":private[p]["classification"] if isprivate else "review_shareable",
   "source_content_decision":"EXCLUDE_FROM_SHARED_EXACT_SOURCE" if isprivate else "ELIGIBLE_CODE_CONTENT_FOR_INTERNAL_SHARED_REVIEW_SELECTION",
   "execution_role":role["role"],"execution_reason":role["reason"],"current_replacement":role["replacement"],
   "content_basis":"Exact bytes, whole-source AST/literal inspection and manually resolved risk contexts, plus authored review of the three additions and this classifier. Code content only; runtime inputs/outputs, fitted states and external data are excluded from this decision.",
   "origins":origins[p]})
 assert sum(x["classification"]=="review_shareable" for x in selections)==307
 OUT.mkdir(mode=0o700)
 save(OUT/"EXACT_SOURCE_CONTENT_ELIGIBILITY.json",selections)
 fields=["source_relative","archive_path","bytes","sha256","classification","source_content_decision","execution_role","current_replacement"]
 with (OUT/"EXACT_SOURCE_CONTENT_ELIGIBILITY.csv").open("x",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for x in selections:w.writerow({k:x[k] for k in fields})
 save(OUT/"ADDITIONAL_LITERAL_SCREEN_AND_MANUAL_DECISIONS.json",{"source_files":311,"python_files":python_files,"screen_findings":additional,"manual_interpretations":interpretations,"literal_credential_pattern_matches":0,"known_historical_identifier_values_not_copied":True,"known_identifier_bearing_source_files":sorted(known_identifier_files),"all_candidate_sources_unexecuted":True})
 doc="""# 登记源码的内容选择决定

本次合并307文件索引、全部现有源码增补，以及本内容审查程序，得到311份精确路径/SHA的登记源码。此前20处命中已逐项处置；补充检索了文本中的临床标量、联系方式/长数字候选、凭据赋值、标识符字典与数值集合，并实际审读三处新增命中。它们分别是两个明确构造的DEMO键和一次字段重命名，不是病例标识。

内容决定：307份可作为代码内容进入内部共享审查包的逐文件选择；4份含历史源表行标识的原件保持私有/历史私有。决定绑定当前文件SHA，字节变化即不能沿用。本表不选择运行产生的数据文件、真实提示、标签台账、训练态或外部论文全文，不批准对外发布。

代码内容适宜审阅与入口可运行是两件事。清单逐项保留旧入口、被替代辅助及来源复用的角色；其余源码也没有因此获得独立执行资格。旧171/122人口准备、旧报告/旧掩码和不合格重放入口仍受原处置限制。最后的归档仍需按真实用途选文件、加入已完成的新结果/图表/报告，并执行清洁解压与数值复现。

这完成的是当前登记311源码的有界内容选择审查，不证明所有未登记代码已覆盖，也不替代后续新增文件审查、全研究代码依赖闭合、数值/科学审查或PACKAGE_ACCEPTANCE。未逐行评价每个算法的科学含义；科学判断须在37项实际就绪后按原流程进行。本决定来自本任务对实际内容的审阅，不将“零匹配”本身作为普遍共享许可。
"""
 with (OUT/"CONTENT_SELECTION_DECISION_ZH.md").open("x",encoding="utf-8") as f:f.write(doc)
 for p,h in refs.items():assert sha(R/p)==h,p
 for name,h in addenda.items():assert sha(B/name)==h,name
 protection=json.loads((R/"outputs/post_reanalysis_process_review_v1/MANUSCRIPT_PROTECTION.json").read_text())["files"];assert len(protection)==92
 for p,h in protection.items():assert sha(p)==(h["sha256"] if isinstance(h,dict) else h)
 outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
 receipt={"status":"REGISTERED311_SOURCE_CODE_CONTENT_SELECTION_REVIEW_COMPLETE_NOT_WHOLE_PACKAGE_OR_SCIENCE",
  "time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_sha256":sha(__file__),"source_index_sha256":sha(INDEX),"prior_content_receipt_sha256":sha(prior_screen),"merged_addenda_sha256":addenda,
  "source_files":len(refs),"source_bytes":source_bytes,"code_content_eligible_for_internal_shared_review":307,"exact_private_sources":4,
  "candidate_source_execution":0,"new_data_or_fits":0,"original_source_bytes_unchanged":True,"protected92_hashes_unchanged":True,
  "whole_code_dependency_closure_not_claimed":True,"runtime_data_classification_not_inherited":True,"new_archive_not_built":True,
  "public_release_authorized":False,"scientific_review_complete":False,"outputs":outputs,
  "elapsed_seconds":time.monotonic()-began,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 save(OUT/"SOURCE_CONTENT_SELECTION_REVIEW_RECEIPT.json",receipt)
 print(json.dumps({"receipt":str(OUT/"SOURCE_CONTENT_SELECTION_REVIEW_RECEIPT.json"),"sha256":sha(OUT/"SOURCE_CONTENT_SELECTION_REVIEW_RECEIPT.json"),"sources":311,"eligible_code_sources":307,"private_sources":4,"elapsed_seconds":receipt["elapsed_seconds"],"peak_rss_bytes":receipt["peak_rss_bytes"]}))
if __name__=="__main__":main()
