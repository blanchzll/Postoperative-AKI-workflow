from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import ast,json,re,difflib,shutil
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Inches
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/probabilistic_integration_20261105_v1')
P=W/'package';S=P/'04_Editable_sources';OLD=W.parent/'discussion_deepening_20261104_v1/package'
D=P/'03_Author_review/probabilistic_integration_20261105';D.mkdir(exist_ok=True)
source=W.parent/'novelty_revision_20261101_v1/code/build_documents.py'
tree=ast.parse(source.read_text())
module=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[])
ns={};exec(compile(module,str(source),'exec'),ns)
style,add,write=ns['style_doc'],ns['add_md'],ns['write']
counts=json.loads((W/'review/WORD_COUNTS.json').read_text())
render=[]
def save(doc,rel):
 write(doc,P/rel);render.append(rel)
main=(S/'MANUSCRIPT_EN.md').read_text()
doc=style();doc.add_heading(main.splitlines()[0][2:],0)
for t in ['Authors, highest academic degrees, and affiliations: [AUTHOR INPUT NEEDED]','Corresponding author: [AUTHOR INPUT NEEDED: name, postal address, email, telephone]',f"Main text word count: {counts['main_body']}",'Article type: Original Investigation']:doc.add_paragraph(t)
add(doc,'\n'.join(main.splitlines()[1:]),break_main=True)
inside=False
for p in doc.paragraphs:
 if p.text=='Abstract':inside=True;continue
 if p.text=='Introduction':inside=False
 if inside:
  p.paragraph_format.space_after=ns['Pt'](0)
  for r in p.runs:r.font.size=ns['Pt'](10)
doc.add_page_break()
leg=(S/'FIGURE_LEGENDS.md').read_text()
add(doc,leg.split('## eFigure 1.',1)[0])
save(doc,'01_Journal_files/MANUSCRIPT.docx')
doc=style();add(doc,(S/'MANUSCRIPT_ZH.md').read_text());save(doc,'03_Author_review/MANUSCRIPT_ZH.docx')
for lang in ['EN','ZH']:
 for name in ['INTRODUCTION','METHODS','RESULTS','DISCUSSION','CONCLUSIONS']:
  rel=f'04_Editable_sources/Chapters/{name}_{lang}.docx'
  doc=style();add(doc,(P/rel).with_suffix('.md').read_text());save(doc,rel)
doc=style();add(doc,leg);save(doc,'01_Journal_files/FIGURE_LEGENDS.docx')
doc=style(letter=True);add(doc,(S/'COVER_LETTER.md').read_text().replace('# Cover letter\n\n',''));save(doc,'01_Journal_files/COVER_LETTER.docx')
supp=Document(OLD/'01_Journal_files/SUPPLEMENT_1.docx')
edits=json.loads((W/'review/SUPPLEMENT_EDIT_LOG.json').read_text())
def insert_before_heading(text,heading):
 temp=style(supp=True);add(temp,text)
 target=[p for p in supp.paragraphs if p.text==heading][-1]
 for el in temp._element.body:
  if el.tag!=qn('w:sectPr'):target._p.addprevious(deepcopy(el))
insert_before_heading(edits['methods'],'10. MIMIC-IV calendar-window sensitivity')
insert_before_heading(edits['results'],'Public-source transport and threshold burden')
temp=style(supp=True);add(temp,edits['panel'])
assert len(temp.tables)==1
table=temp.tables[0]
widths=[1.45,1.60,1.62,1.10,.90]
for col,width in zip(table.columns,widths):col.width=Inches(width)
for row in table.rows:
 for cell,width in zip(row.cells,widths):cell.width=Inches(width)
anchor=[p for p in supp.paragraphs if p.text=='eTable 21. Complete calibrated resource-grid region counts'][-1]
for el in temp._element.body:
 if el.tag!=qn('w:sectPr'):anchor._p.addprevious(deepcopy(el))
save(supp,'01_Journal_files/SUPPLEMENT_1.docx')
for rel in ['01_Journal_files/Supplementary_tables/EDITABLE_THREE_LINE_TABLES.docx','01_Journal_files/Supplementary_tables/SUPPLEMENTARY_TABLE_2.docx']:
 doc=Document(OLD/rel);doc.add_paragraph(edits['standalone_table_pointer']);save(doc,rel)
# Actual OOXML track changes against the immediate accepted writing parent.
counter=0
stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def change(tag,runs):
 global counter
 counter+=1;el=OxmlElement('w:'+tag)
 for k,v in [('id',str(counter)),('author','Codex'),('date',stamp)]:el.set(qn('w:'+k),v)
 for run in runs:
  c=deepcopy(run)
  if tag=='del':
   for t in c.iter(qn('w:t')):t.tag=qn('w:delText')
  el.append(c)
 return el
def changed_paragraph(template,oldp=None,newp=None,mark=None):
 p=deepcopy(template._p)
 for el in list(p):
  if el.tag!=qn('w:pPr'):p.remove(el)
 if oldp is not None:p.append(change('del',oldp._p.findall(qn('w:r'))))
 if newp is not None:p.append(change('ins',newp._p.findall(qn('w:r'))))
 if mark:
  pp=p.find(qn('w:pPr'))
  if pp is None:pp=OxmlElement('w:pPr');p.insert(0,pp)
  rp=pp.find(qn('w:rPr'))
  if rp is None:rp=OxmlElement('w:rPr');pp.append(rp)
  rp.append(change(mark,[]))
 return p
def view(doc,accept):
 out=[]
 for p in doc.paragraphs:
  if p._p.xpath('./w:pPr/w:rPr/w:ins') and not accept:continue
  if p._p.xpath('./w:pPr/w:rPr/w:del') and accept:continue
  def get(el):
   if el.tag==qn('w:pPr'):return ''
   if el.tag==qn('w:ins') and not accept:return ''
   if el.tag==qn('w:del') and accept:return ''
   if el.tag in [qn('w:t'),qn('w:delText')]:return el.text or ''
   return ''.join(get(c) for c in el)
  out.append(get(p._p))
 return out
receipts=[]
for lang,rel in [('EN','01_Journal_files/MANUSCRIPT.docx'),('ZH','03_Author_review/MANUSCRIPT_ZH.docx')]:
 old=Document(OLD/rel);new=Document(P/rel);track=Document(P/rel)
 op=old.paragraphs;np=new.paragraphs
 for el in list(track._element.body):
  if el.tag!=qn('w:sectPr'):track._element.body.remove(el)
 anchor=track._element.body.find(qn('w:sectPr'))
 def put(el):anchor.addprevious(el)
 for tag,a,b,c,d in difflib.SequenceMatcher(a=[p.text for p in op],b=[p.text for p in np],autojunk=False).get_opcodes():
  if tag=='equal':
   for p in np[c:d]:put(deepcopy(p._p))
  else:
   oldps=op[a:b];newps=np[c:d]
   common=min(len(oldps),len(newps))
   for k in range(common):put(changed_paragraph(newps[k],oldps[k],newps[k]))
   for p in oldps[common:]:put(changed_paragraph(p,oldp=p,mark='del'))
   for p in newps[common:]:put(changed_paragraph(p,newp=p,mark='ins'))
 setting=OxmlElement('w:trackRevisions');setting.set(qn('w:val'),'true');track.settings._element.append(setting)
 out=D/f'MANUSCRIPT_{lang}_TRACKED.docx';write(track,out);read=Document(out)
 assert view(read,True)==[p.text for p in new.paragraphs],lang+' accepted'
 assert view(read,False)==[p.text for p in old.paragraphs],lang+' rejected'
 receipts.append({'file':str(out.relative_to(P)),'accepted_view_matches_current':True,'rejected_view_matches_immediate_parent':True,'true_OOXML_revision_tags':True})
 before=(OLD/f'04_Editable_sources/MANUSCRIPT_{lang}.md').read_text()
 after=(S/f'MANUSCRIPT_{lang}.md').read_text()
 (D/f'MANUSCRIPT_{lang}_COMPARISON.html').write_text(difflib.HtmlDiff(wrapcolumn=90).make_file(before.splitlines(),after.splitlines(),fromdesc='Accepted November 4 discussion baseline',todesc='November 5 conditional resource integration',context=True,numlines=2))
 (D/f'MANUSCRIPT_{lang}_CHANGES.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='discussion_deepening_'+lang,tofile='probabilistic_integration_'+lang)))
(W/'review/TRACKED_REVISIONS_ACCEPTANCE.json').write_text(json.dumps(receipts,indent=2))
shutil.copy2(W/'review/TRACKED_REVISIONS_ACCEPTANCE.json',D)
(W/'review/RENDER_DOCUMENTS.json').write_text(json.dumps(render,indent=2))
build={'clean_documents_changed':render,'true_tracked_documents':receipts,'main_counts':counts,'main_tables':0,'main_figures':5,'numbered_eTables':24,'supplement_physical_tables':len(supp.tables),'figure_exports_preserved':True}
(W/'review/DOCUMENT_BUILD.json').write_text(json.dumps(build,indent=2))
print(json.dumps({'clean_documents_to_render':len(render),'true_tracked_replay':'PASS','supplement_physical_tables':len(supp.tables),'counts':counts}))
