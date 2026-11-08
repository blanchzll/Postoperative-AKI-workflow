"""JAMA Network Open supplement typography and native SVG insertion."""
from pathlib import Path
import re
from docx.shared import Pt,Mm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.part import Part
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml import etree
def set_run(run,font,size,bold=None):
 run.font.name=font;run.font.size=Pt(size)
 pr=run._element.get_or_add_rPr()
 ff=pr.find(qn("w:rFonts"))
 if ff is None:ff=OxmlElement("w:rFonts");pr.insert(0,ff)
 for k in ["ascii","hAnsi","cs","eastAsia"]:ff.set(qn("w:"+k),font)
 if bold is not None:run.font.bold=bold
def insert_svg(doc,svg,png,width_mm):
 """Keep the native SVG relationship so Word and the PDF renderer use vectors."""
 pic=doc.add_picture(str(png),width=Mm(width_mm))
 name=doc.part.package.next_partname("/word/media/image%d.svg")
 part=Part(name,"image/svg+xml",Path(svg).read_bytes(),doc.part.package)
 rid=doc.part.relate_to(part,RT.IMAGE)
 blip=pic._inline.xpath(".//a:blip")[0]
 original_png_rid=blip.get(qn("r:embed"))
 # Direct SVG also prevents LibreOffice choosing only the PNG compatibility bitmap.
 blip.set(qn("r:embed"),rid)
 extlst=OxmlElement("a:extLst");ext=OxmlElement("a:ext")
 ext.set("uri","{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
 ns="http://schemas.microsoft.com/office/drawing/2016/SVG/main"
 child=etree.Element("{"+ns+"}svgBlip",nsmap={"asvg":ns})
 child.set(qn("r:embed"),rid);ext.append(child);extlst.append(ext);blip.append(extlst)
 # The PNG was used only for dimensions, and is not part of the submitted graph.
 doc.part.drop_rel(original_png_rid)
 return pic
def style_supplement(doc):
 state="text"
 for p in doc.paragraphs:
  text=p.text.strip();heading=p.style.name.startswith("Heading")
  if heading and re.match(r"eTable\s+\d+\.",text):state="table";font,size,bold="Arial",12,True
  elif heading and re.match(r"eFigure\s+\d+\.",text):state="figure";font,size,bold="Arial",12,True
  elif heading:
   if text.startswith(("eMethods","eResults","eReferences")):state="text"
   font="Times New Roman" if state=="text" else "Arial"
   size=12 if p.style.name=="Heading 1" else 10;bold=True
  elif state in ["table","figure"]:font,size,bold="Arial",8,None
  else:font,size,bold="Times New Roman",10,None
  p.paragraph_format.line_spacing=1
  for r in p.runs:set_run(r,font,size,bold)
 for t in doc.tables:
  if len(t.columns)>=7:
   t.autofit=False
   for c in t.columns:c.width=Mm(257/len(t.columns))
   for row in t.rows:
    for c in row.cells:c.width=Mm(257/len(t.columns))
   tw=t._tbl.tblPr.find(qn("w:tblW"))
   if tw is not None:tw.set(qn("w:type"),"dxa");tw.set(qn("w:w"),str(Mm(257).twips))
  for i,row in enumerate(t.rows):
   for c in row.cells:
    for p in c.paragraphs:
     p.paragraph_format.line_spacing=1
     for r in p.runs:set_run(r,"Arial",10,True if i==0 else None)

def style_table_document(doc):
 for p in doc.paragraphs:
  p.paragraph_format.line_spacing=1
  for r in p.runs:set_run(r,"Arial",12 if p.style.name.startswith("Heading") else 8,True if p.style.name.startswith("Heading") else None)
 for t in doc.tables:
  for i,row in enumerate(t.rows):
   for cell in row.cells:
    for p in cell.paragraphs:
     p.paragraph_format.line_spacing=1
     for r in p.runs:set_run(r,"Arial",10,True if i==0 else None)
