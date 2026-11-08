from pathlib import Path
from copy import deepcopy
from difflib import SequenceMatcher
from datetime import datetime,timezone
import json,hashlib
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
W=Path(__file__).resolve().parents[1];P=W/'package';A=P/'03_Author_review'
baseline=W.parents[1]/'writing_runs/probabilistic_integration_20261105_v1/package/01_Journal_files/MANUSCRIPT.docx'
clean=P/'01_Journal_files/MANUSCRIPT.docx'
if not (W/'review/DOCUMENT_BUILD.json').exists():raise SystemExit('Build current clean manuscript first')
old=Document(baseline);new=Document(clean)
assert not old.tables and not new.tables,'Main manuscript expected to have no tables'
oldp=list(old.paragraphs);newp=list(new.paragraphs)
oldtext=[p.text for p in oldp];newtext=[p.text for p in newp]
tracked=Document(clean);body=tracked._element.body
sect=deepcopy(body.sectPr)
for child in list(body):body.remove(child)
date=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
counts={'inserted_paragraphs':0,'deleted_paragraphs':0,'unchanged_paragraphs':0};rid=1
def change(p,kind):
 global rid
 el=deepcopy(p._p)
 props=el.find(qn('w:pPr'))
 if props is None:props=OxmlElement('w:pPr');el.insert(0,props)
 rpr=props.find(qn('w:rPr'))
 if rpr is None:rpr=OxmlElement('w:rPr');props.append(rpr)
 mark=OxmlElement('w:'+kind);mark.set(qn('w:id'),str(rid));mark.set(qn('w:author'),'Editorial revision');mark.set(qn('w:date'),date);rid+=1;rpr.append(mark)
 wrap=OxmlElement('w:'+kind);wrap.set(qn('w:id'),str(rid));wrap.set(qn('w:author'),'Editorial revision');wrap.set(qn('w:date'),date);rid+=1
 for child in list(el):
  if child is props:continue
  el.remove(child)
  if kind=='del':
   for t in child.iter(qn('w:t')):t.tag=qn('w:delText')
  wrap.append(child)
 el.append(wrap);body.append(el)
 counts['inserted_paragraphs' if kind=='ins' else 'deleted_paragraphs']+=1
for tag,i,j,a,b in SequenceMatcher(a=oldtext,b=newtext,autojunk=False).get_opcodes():
 if tag=='equal':
  for p in newp[a:b]:body.append(deepcopy(p._p));counts['unchanged_paragraphs']+=1
 else:
  for p in oldp[i:j]:change(p,'del')
  for p in newp[a:b]:change(p,'ins')
body.append(sect)
settings=tracked.settings._element
if settings.find(qn('w:trackRevisions')) is None:settings.append(OxmlElement('w:trackRevisions'))
out=A/'MANUSCRIPT_TRACK_CHANGES.docx'
tracked.core_properties.author='';tracked.core_properties.last_modified_by='';tracked.core_properties.comments=''
tracked.save(out)
# Replay accept/reject semantics including paragraph marks; compare actual Word text.
def replay(accept):
 output=[]
 for el in tracked._element.body:
  if el.tag!=qn('w:p'):continue
  ppr=el.find(qn('w:pPr'));rpr=ppr.find(qn('w:rPr')) if ppr is not None else None
  if rpr is not None:
   if accept and rpr.find(qn('w:del')) is not None:continue
   if not accept and rpr.find(qn('w:ins')) is not None:continue
  def collect(node):
   if node.tag==qn('w:pPr'):return ''
   if (accept and node.tag==qn('w:del')) or (not accept and node.tag==qn('w:ins')):return ''
   if node.tag in [qn('w:t'),qn('w:delText')]:return node.text or ''
   if node.tag==qn('w:tab'):return '\t'
   if node.tag==qn('w:br'):return '' if node.get(qn('w:type')) in ['page','column'] else '\n'
   if node.tag==qn('w:cr'):return '\n'
   return ''.join(collect(x) for x in node)
  output.append(collect(el))
 return output
assert replay(True)==newtext,'Accepted changes do not reproduce clean manuscript'
assert replay(False)==oldtext,'Rejected changes do not reproduce immediate parent'
receipt={'status':'PASS_ACTUAL_OOXML_TRACK_CHANGES_REPLAY','baseline':str(baseline),'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'clean_sha256':hashlib.sha256(clean.read_bytes()).hexdigest(),'tracked_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'accept_replays_current_clean':True,'reject_replays_Nov5_parent':True,**counts}
(W/'review/TRACK_CHANGES_VERIFICATION.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
