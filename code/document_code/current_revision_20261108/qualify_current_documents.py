from pathlib import Path
import json,re,zipfile,hashlib
from pypdf import PdfReader
from docx import Document
from docx.oxml.ns import qn
W=Path(__file__).resolve().parents[1];P=W/'package';J=P/'01_Journal_files'
checks=[];issues=[]
for path in sorted(P.rglob('*.docx')):
 # Historical parent receipts contain JSON only; no older submission DOCX should be relabelled.
 d=Document(path);n=0
 for table in d.tables:
  pr=table._tbl.tblPr;bb=pr.find(qn('w:tblBorders'))
  ok=bb is not None
  for side in ['top','bottom']:
   node=bb.find(qn('w:'+side)) if bb is not None else None;ok=ok and node is not None and node.get(qn('w:val'))=='single'
  for side in ['left','right','insideH','insideV']:
   node=bb.find(qn('w:'+side)) if bb is not None else None;ok=ok and node is not None and node.get(qn('w:val')) in ['nil','none']
  # Third horizontal rule sits below the first row.
  for cell in table.rows[0].cells:
   borders=cell._tc.get_or_add_tcPr().find(qn('w:tcBorders'))
   edge=borders.find(qn('w:bottom')) if borders is not None else None
   ok=ok and edge is not None and edge.get(qn('w:val'))=='single'
  if not ok:issues.append('Table border '+str(path))
  n+=1
 with zipfile.ZipFile(path) as z:
  xml='\n'.join(z.read(n).decode('utf-8') for n in z.namelist() if n.endswith('.xml'))
  # Drawing/textpath watermarks differ from legitimate disclosure text.
  mark=bool(re.search(r'w:val="watermark"|<v:textpath\b|PowerPlusWaterMarkObject|WordPictureWatermark',xml,re.I))
  if mark:issues.append('Potential watermark '+str(path))
 checks.append({'docx':str(path.relative_to(P)),'tables_three_line':n,'watermark_object_found':mark})
pdfs=[];selected=[]
for path in [J/'MANUSCRIPT.pdf',J/'SUPPLEMENT_1.pdf',J/'FIGURE_LEGENDS.pdf',J/'COVER_LETTER.pdf']:
 pdf=PdfReader(path);texts=[page.extract_text() or '' for page in pdf.pages]
 if any(len(t.strip())<5 for t in texts):issues.append('Near-empty page '+str(path))
 pdfs.append({'file':str(path.relative_to(P)),'pages':len(texts),'bytes':path.stat().st_size,'text_characters':sum(map(len,texts))})
 if path.name=='MANUSCRIPT.pdf':
  alltext='\n'.join(texts)
  for token in ['2981','0.0717','7/59','TRIPOD']:
   if token not in alltext:issues.append('Missing main text token '+token)
  selected += [{'file':str(path.relative_to(W)),'page':i+1,'label':'main_'+str(i+1)} for i in sorted(set([0,1,2,len(texts)-2]))]
 if path.name=='SUPPLEMENT_1.pdf':
  for target in ['eTable 25.','eTable 28.','eTable 30.','eTable 31.','eFigure 16.','eFigure 17.']:
   # Omit table-of-contents appearance.
   indexes=[i for i,t in enumerate(texts) if target in t]
   if not indexes:issues.append('Supplement missing '+target)
   else:selected.append({'file':str(path.relative_to(W)),'page':indexes[-1]+1,'label':target.replace(' ','_').replace('.','')})
  for i in range(1,33):
   if 'eTable '+str(i)+'.' not in '\n'.join(texts):issues.append('Missing numbered table '+str(i))
 if path.name=='FIGURE_LEGENDS.pdf':
  for i in range(1,19):
   if 'eFigure '+str(i)+'.' not in '\n'.join(texts):issues.append('Missing figure legend '+str(i))
assert not issues,issues
r={'status':'PASS_DOCUMENT_STRUCTURE_TEXT_AND_THREE_LINE_BORDERS','checks':checks,'PDFs':pdfs,'selected_preview_pages':selected,'main_figures':len(list(J.glob('FIGURE_[1-5].pdf'))),'independent_panel_PDFs':len(list((P/'05_Figure_assets/Independent_panels').rglob('*.pdf'))),'issues':issues,'visual_page_review':'pending selected-page inspection','author_declarations_complete':False}
(W/'review/DOCUMENT_QUALIFICATION.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:r[k] for k in ['status','PDFs','main_figures','independent_panel_PDFs','issues']}))
