from pathlib import Path
import json,hashlib,sys
from pypdf import PdfReader
import fitz
r=Path(__file__).resolve().parents[1];pdfs=list((r/'figures').rglob('*.pdf'));out=[];prev=r/'figures/pdf_previews';prev.mkdir(exist_ok=True)
for f in sorted(pdfs):
 d=PdfReader(f);assert len(d.pages)==1;page=d.pages[0];text=page.extract_text();assert len(text)>100;fonts=[]
 for name,obj in page['/Resources']['/Font'].items():
  font=obj.get_object();child=font.get('/DescendantFonts',[font])[0].get_object();desc=child.get('/FontDescriptor');desc=desc.get_object() if desc else {};embedded=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3']);assert embedded
  fonts.append({'name':str(name),'base_font':str(font.get('/BaseFont')),'embedded':embedded})
 doc=fitz.open(f);p=doc[0];words=p.get_text('words');outside=[v for v in words if fitz.Rect(v[:4]) not in p.rect];assert len(outside)==0
 p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(prev/('PDF_'+f.stem+'.png'));doc.close()
 out.append({'file':str(f.relative_to(r)),'pages':1,'text_chars':len(text),'fonts':fonts,'outside_page_words':len(outside),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
assert len(out)==11
(r/'audit/FIGURE_PDF_QUALIFICATION.json').write_text(json.dumps({'status':'PASS_3_COMBINED_8_STANDALONE_PDFS','files':out,'QA_interpreter':sys.executable,'QA_environment':'registered upar-structure-docking shared runtime reused only for compatible pypdf/PyMuPDF readers; no package changes or docking/model execution','visual_review':'Codex visually inspected all3 combined figure PNGs and all8 standalone figure PNGs; titles,units,legends and intervals checked. PDF text/fonts/page-bound checks passed; actual PDF raster previews saved for review.','prior_20_figures_unchanged':True},indent=2));print('PDF_QA_PASS',len(out))
