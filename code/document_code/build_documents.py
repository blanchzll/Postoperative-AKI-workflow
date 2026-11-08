from pathlib import Path
import re,json,shutil,hashlib,io
from docx import Document
from docx.shared import Inches,Pt,Mm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
W=Path(__file__).resolve().parents[1];B=W.parent;T=W/'work/text';P=W/'package';J=P/'01_Journal_files';A=P/'03_Author_review';S=P/'04_Editable_sources'
for d in [J,A,S,W/'review/rendered',W/'review/previews'] :d.mkdir(parents=True,exist_ok=True,mode=0o700)
def style_doc(supp=False,letter=False):
 d=Document();sec=d.sections[0];sec.page_width=Mm(210);sec.page_height=Mm(297)
 for k in ['top_margin','bottom_margin','left_margin','right_margin']:setattr(sec,k,Mm(20))
 for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','List Bullet']:
  st=d.styles[name];st.font.name='Times New Roman' if supp else 'Arial';st.font.size=Pt(10 if supp else 11);st.font.color.rgb=__import__('docx').shared.RGBColor(0,0,0)
  st.paragraph_format.line_spacing=1 if supp or letter else 2;st.paragraph_format.space_after=Pt(4 if supp or letter else 5)
  st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Serif CJK SC')
 for name in ['Title','Heading 1']:d.styles[name].font.size=Pt(12)
 for name in ['Heading 1','Heading 2','Heading 3']:d.styles[name].font.bold=True;d.styles[name].paragraph_format.keep_with_next=True
 sec.footer.paragraphs[0].alignment=1
 fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');sec.footer.paragraphs[0]._p.append(fld)
 d.core_properties.author='';d.core_properties.last_modified_by='';d.core_properties.comments='';d.core_properties.subject='';d.core_properties.keywords=''
 return d
def plain(s):return re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s).replace('**','').replace('`','')
def runs(p,s):
 for tok in re.split(r'(\*\*.*?\*\*|\[\d+(?:[,–-]\d+)*\])',s):
  if not tok:continue
  cite=bool(re.fullmatch(r'\[\d+(?:[,–-]\d+)*\]',tok))
  r=p.add_run(tok[1:-1] if cite else plain(tok));r.bold=tok.startswith('**') and tok.endswith('**')
  if cite:r.font.superscript=True
  r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Serif CJK SC')
def border(el,name,val='nil',sz='0'):
 b=OxmlElement('w:'+name);b.set(qn('w:val'),val)
 if val!='nil':b.set(qn('w:sz'),sz);b.set(qn('w:color'),'000000')
 el.append(b)
def three_line(t):
 # The top and bottom borders belong to the table; the third rule is below the header row.
 t.style='Normal Table';pr=t._tbl.tblPr
 for old in list(pr.findall(qn('w:tblBorders'))):pr.remove(old)
 bb=OxmlElement('w:tblBorders')
 for name in ['top','bottom','left','right','insideH','insideV']:border(bb,name,'single' if name in ['top','bottom'] else 'nil','10')
 pr.append(bb)
 for i,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
  if i==0:trpr.append(OxmlElement('w:tblHeader'))
  for c in row.cells:
   cp=c._tc.get_or_add_tcPr()
   for old in list(cp.findall(qn('w:tcBorders'))):cp.remove(old)
   if i==0:
    cb=OxmlElement('w:tcBorders');border(cb,'bottom','single','6');cp.append(cb)
   # Remove any inherited fill.
   for old in list(cp.findall(qn('w:shd'))):cp.remove(old)
   for p in c.paragraphs:
    p.paragraph_format.line_spacing=1;p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(2)
    for r in p.runs:r.font.size=Pt(10);r.bold=i==0
 return t
def display(s):
 s=s.replace('preop_full_champion_','27-input ').replace('preop_compact_champion_','9-input ').replace('endop_full_champion_','38-input ')
 s=s.replace('geography_preop','Geographic preop').replace('temporal_preop','Temporal preop').replace('geography_paired','Geographic paired').replace('temporal_paired','Temporal paired')
 return s.replace('_',' ')
def add_table(d,rows):
 n=len(rows[0]);t=d.add_table(rows=0,cols=n);t.autofit=False
 total=6.67
 widths=([2.0,1.25,.85,1.25,1.32] if n==5 else [1.5,1.47,.95,.95,.95,.85] if n==6 else [total/n]*n)
 for col,width in zip(t.columns,widths):col.width=Inches(width)
 for row in rows:
  cells=t.add_row().cells
  for i,txt in enumerate(row[:n]):cells[i].width=Inches(widths[i]);runs(cells[i].paragraphs[0],display(txt))
 three_line(t);return t
def add_md(d,md,break_tables=False,break_main=False):
 lines=md.splitlines();i=0;after_table=False
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+\|?$',lines[i+1]):
   rows=[[x.strip() for x in line.strip('|').split('|')]];i+=2
   while i<len(lines) and lines[i].startswith('|'):
    rows.append([x.strip() for x in lines[i].strip('|').split('|')]);i+=1
   add_table(d,rows);after_table=True;continue
  h=re.match(r'^(#{1,4})\s+(.*)',line)
  if h:
   title=h[2];after_table=False
   p=d.add_heading(level=min(len(h[1]),3));runs(p,title)
   if break_tables and re.match(r'(?:e)?Table \d+',title):p.paragraph_format.page_break_before=True
   if break_main and title in ['Key Points','Abstract','Introduction','References']:p.paragraph_format.page_break_before=True
  else:
   m=re.match(r'^\[(\d+)\]\s+(.*)',line)
   if m:line=m[1]+'. '+m[2];line=re.sub(r'https://doi.org/', 'doi:',line)
   p=d.add_paragraph(style='List Bullet' if line.startswith('- ') else None);runs(p,line[2:] if line.startswith('- ') else line)
   if after_table:
    p.paragraph_format.line_spacing=1
    for rr in p.runs:rr.font.size=Pt(10)
  i+=1
 return d
def write(d,p):d.core_properties.title=p.stem.replace('_',' ');d.save(p)
main=(T/'MANUSCRIPT_EN.md').read_text();title=main.splitlines()[0][2:]
body=main.split('## Introduction')[1].split('## Acknowledgment')[0];wc=len(('Introduction'+body).split())
d=style_doc();d.add_heading(title,0)
for v in ['Authors, highest academic degrees, and affiliations: [AUTHOR INPUT NEEDED]','Corresponding author: [AUTHOR INPUT NEEDED: name, postal address, email, telephone]',f'Main text word count: {wc}','Article type: Original Investigation']:d.add_paragraph(v)
add_md(d,'\n'.join(main.splitlines()[1:]),break_main=True)
inside_abstract=False
for pp in d.paragraphs:
 if pp.text=='Abstract':inside_abstract=True;continue
 if pp.text=='Introduction':inside_abstract=False
 if inside_abstract:
  pp.paragraph_format.space_after=Pt(0)
  for rr in pp.runs:rr.font.size=Pt(10)

d.add_page_break();add_md(d,(T/'MAIN_TABLES.md').read_text().replace('# Main tables\n',''),break_tables=True)
d.add_page_break();add_md(d,(T/'MAIN_FIGURE_LEGENDS.md').read_text())
write(d,J/'MANUSCRIPT.docx')
# Separate main tables are editable and use precisely the same cells as the manuscript.
mt=style_doc();add_md(mt,(T/'MAIN_TABLES.md').read_text().replace('# Main tables\n',''),break_tables=True);write(mt,J/'TABLES_1_2.docx')
for i,t in enumerate(mt.tables,1):
 dd=style_doc();chunk=re.split(r'\n## ',(T/'MAIN_TABLES.md').read_text())[i];add_md(dd,'## '+chunk);write(dd,J/f'TABLE_{i}.docx')
ld=style_doc();add_md(ld,(T/'FIGURE_LEGENDS.md').read_text());write(ld,J/'FIGURE_LEGENDS.docx')
cd=style_doc(letter=True);cover=(T/'COVER_LETTER.md').read_text().replace('# Cover letter\n\n','');add_md(cd,cover);write(cd,J/'COVER_LETTER.docx')
# One editable supplement, with its own reference numbering and all supporting illustrations embedded.
sd=style_doc(supp=True);sd.add_heading('Supplement 1',0);sd.add_paragraph(title)
sd.add_heading('Contents',1)
for v in ['eMethods. Detailed methods and development protocol','eResults. Supporting analyses']:sd.add_paragraph(v)
for line in (T/'ETABLES.md').read_text().splitlines():
 if line.startswith('## eTable '):sd.add_paragraph(line[3:])
entries=json.loads((T/'FIGURE_ENTRIES.json').read_text())
for e in entries[3:]:sd.add_paragraph(e['label']+'. '+e['title'])
sd.add_paragraph('eReferences. References cited in the supplementary methods')
for pp in sd.paragraphs:pp.paragraph_format.space_after=Pt(0)
sd.add_page_break();methods=(T/'EMETHODS.md').read_text();m,eref=methods.split('## eReferences')
add_md(sd,m);sd.add_page_break();add_md(sd,(T/'ERESULTS.md').read_text())
add_md(sd,(T/'ETABLES.md').read_text(),break_tables=True)
image_records=[]
for e in entries[3:]:
 sd.add_page_break();sd.add_heading(e['label']+'. '+e['title'],1)
 source=B/next(s for s in e['source']['exports'] if s.endswith('.png'))
 # Reuse the accepted raster export; no redrawing, cropping, or pixel editing.
 sd.add_picture(str(source),width=Mm(165))
 add_md(sd,e['caption'])
 image_records.append({'label':e['label'],'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
sd.add_page_break();add_md(sd,'# eReferences'+eref);write(sd,J/'SUPPLEMENT_1.docx')
# Main figure PDFs are the accepted vector files, byte-for-byte; all original exports go to the asset directory.
for e in entries[:3]:
 source=B/next(s for s in e['source']['exports'] if s.endswith('.pdf'));shutil.copy2(source,J/(e['label'].replace(' ','_').upper()+'.pdf'))
for e in entries:
 for rel in e['source']['exports']:
  dest=P/'05_Figure_assets'/e['label'].replace(' ','_')/Path(rel).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(B/rel,dest)
# Canonical sources contain only the current 14 references; obsolete sibling reference drafts are not carried forward.
for n in ['MANUSCRIPT_EN','MANUSCRIPT_ZH','COVER_LETTER','MAIN_TABLES','FIGURE_LEGENDS','EMETHODS','ERESULTS','ETABLES']:
 shutil.copy2(T/(n+'.md'),S/(n+'.md'))
for n in ['references.bib','references.ris']:shutil.copy2(B/'citation_audit'/n,S/n)
zd=style_doc();add_md(zd,(T/'MANUSCRIPT_ZH.md').read_text());write(zd,A/'MANUSCRIPT_ZH.docx')
(T/'AUTHOR_ITEMS_ZH.md').write_text((B/'manuscript_revision/AUTHOR_ITEMS_ZH.md').read_text()+'\n- 投稿信中确认全体作者批准、原创性/一稿多投状态、同研究相关稿件/预印本及既往投稿审阅情况。\n')
shutil.copy2(T/'AUTHOR_ITEMS_ZH.md',A/'AUTHOR_ITEMS_ZH.md')
(W/'review/DOCUMENT_BUILD.json').write_text(json.dumps({'main_body_words':wc,'main_title_characters':len(title),'main_tables':len(d.tables),'separate_main_tables':len(mt.tables),'supplement_tables':len(sd.tables),'eFigures':len(image_records),'image_sources':image_records,'watermarks_added':False,'tracked_revisions_added':False,'main_font_pt':11,'abstract_font_pt':10,'main_line_spacing':2,'supplement_font_pt':10,'supplement_line_spacing':1,'main_table_style':'editable three horizontal rules; no vertical/internal grid rules','renderer':'existing registered python-docx runtime; no package changes'},indent=2))
print(json.dumps({'main_body_words':wc,'supplement_tables':len(sd.tables),'word_files':len(list(P.rglob('*.docx'))),'output':str(P)}))
