from pathlib import Path
import ast,re,json,hashlib,shutil
from datetime import datetime
W=Path(__file__).resolve().parents[1]
P=W/'package';S=P/'04_Editable_sources';J=P/'01_Journal_files';A=P/'03_Author_review'
accepted=W.parents[1]/'correction_runs/clinical_supplement_20261108_v1/ACCEPTANCE.json'
if not accepted.exists():
 raise SystemExit("Current clinical supplement acceptance missing; keep current Word/PDF rebuild pending.")
helper=W.parents[1]/'writing_runs/novelty_revision_20261101_v1/code/build_documents.py'
tree=ast.parse(helper.read_text())
body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
ns={};exec(compile(ast.Module(body=body,type_ignores=[]),str(helper),'exec'),ns)
style_doc,add_md,write=[ns[n] for n in ['style_doc','add_md','write']]
from docx.shared import Mm,Pt
main=(S/'MANUSCRIPT_EN.md').read_text()
title=main.splitlines()[0][2:]
body=main.split('## Introduction\n',1)[1].split('## Acknowledgment\n',1)[0]
wc=len(body.split())
legends=(S/'FIGURE_LEGENDS.md').read_text()
parts=re.split(r'(?=^## (?:e)?Figure \d+\.)',legends,flags=re.M)[1:]
mains=[x for x in parts if x.startswith('## Figure ')]
supps=[x for x in parts if x.startswith('## eFigure ')]
assert len(mains)==5
assert wc<=3000,(wc,'main text over journal limit')
d=style_doc();d.add_heading(title,0)
for text in ['Authors, highest academic degrees, and affiliations: [AUTHOR INPUT NEEDED]','Corresponding author: [AUTHOR INPUT NEEDED: name, postal address, email, telephone]',f'Main text word count: {wc}','Article type: Original Investigation']:
 d.add_paragraph(text)
add_md(d,'\n'.join(main.splitlines()[1:]),break_main=True)
d.add_page_break();add_md(d,'# Figure legends\n\n'+'\n'.join(mains))
write(d,J/'MANUSCRIPT.docx')
ld=style_doc();add_md(ld,legends);write(ld,J/'FIGURE_LEGENDS.docx')
cd=style_doc(letter=True);add_md(cd,(S/'COVER_LETTER.md').read_text().replace('# Cover letter\n','',1));write(cd,J/'COVER_LETTER.docx')
mt=(S/'MAIN_TABLES.md').read_text()
td=style_doc();add_md(td,mt,break_tables=True)
dest=J/'Supplementary_tables';dest.mkdir(exist_ok=True)
write(td,dest/'EDITABLE_THREE_LINE_TABLES.docx')
for i,chunk in enumerate(re.split(r'(?=^## )',mt,flags=re.M)[1:],1):
 dd=style_doc();add_md(dd,chunk);write(dd,dest/f'SUPPLEMENTARY_TABLE_{i}.docx')
sd=style_doc(supp=True);sd.add_heading('Supplement 1',0);sd.add_paragraph(title)
sd.add_heading('Contents',1)
for text in ['eMethods. Detailed methods and development protocol','eResults. Supporting analyses']:
 sd.add_paragraph(text)
for line in (S/'ETABLES.md').read_text().splitlines():
 if line.startswith('## eTable '):sd.add_paragraph(line[3:])
for part in supps:sd.add_paragraph(part.splitlines()[0][3:])
methods=(S/'EMETHODS.md').read_text()
before,separator,refs=methods.partition('## eReferences')
sd.add_page_break();add_md(sd,before)
sd.add_page_break();add_md(sd,(S/'ERESULTS.md').read_text())
add_md(sd,(S/'ETABLES.md').read_text(),break_tables=True)
images=[]
for part in supps:
 label=re.match(r'## (eFigure \d+)\.',part).group(1)
 folder=P/'05_Figure_assets'/label.replace(' ','_')
 pngs=sorted(folder.glob('*.png'))
 assert pngs,(label,folder)
 image=pngs[0]
 sd.add_page_break();sd.add_heading(part.splitlines()[0][3:],1)
 sd.add_picture(str(image),width=Mm(165))
 add_md(sd,'\n'.join(part.splitlines()[1:]))
 images.append({'figure':label,'file':str(image.relative_to(P)),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
if separator:sd.add_page_break();add_md(sd,'# eReferences'+refs)
write(sd,J/'SUPPLEMENT_1.docx')
zd=style_doc();add_md(zd,(S/'MANUSCRIPT_ZH.md').read_text());write(zd,A/'MANUSCRIPT_ZH.docx')
for lang,md in [('EN',main),('ZH',(S/'MANUSCRIPT_ZH.md').read_text())]:
 sections=re.split(r'(?=^## )',md,flags=re.M)
 for cp in (S/'Chapters').glob('*_'+lang+'.md'):
  if cp.stem.rsplit('_',1)[0] not in {'INTRODUCTION','METHODS','RESULTS','DISCUSSION','CONCLUSIONS'}:continue
  heading=next((l for l in cp.read_text().splitlines() if l.startswith('## ')),None)
  if heading:
   found=[part for part in sections if part.startswith(heading+'\n')]
   assert len(found)==1,(cp,heading)
   cp.write_text(found[0].rstrip()+'\n')
   ch=style_doc();add_md(ch,cp.read_text());write(ch,cp.with_suffix('.docx'))
# All generated tables use the helper's actual table top/bottom and header rule.
receipt={'built_at':datetime.now().isoformat(),'main_body_words':wc,'main_title_characters':len(title),'main_figures':len(mains),'supplement_figures':len(supps),'supplement_tables':len(sd.tables),'standalone_three_line_tables':len(td.tables),'main_tables':len(d.tables),'image_sources':images,'main_font_pt':11,'main_line_spacing':2,'watermarks_added':False,'historical_parent_modified':False,'clinical_acceptance_sha256':hashlib.sha256(accepted.read_bytes()).hexdigest(),'sources':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in S.glob('*.md')}}
(W/'review/DOCUMENT_BUILD.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:receipt[k] for k in ['main_body_words','main_figures','supplement_figures','supplement_tables','standalone_three_line_tables']}))
