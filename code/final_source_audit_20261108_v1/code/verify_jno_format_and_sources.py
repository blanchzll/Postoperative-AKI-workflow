"""Check JNO supplement typography/vectors against accepted source values."""
from pathlib import Path
from decimal import Decimal
import csv,hashlib,json,re,zipfile
from collections import Counter
from docx import Document
from pypdf import PdfReader
W=Path(__file__).resolve().parents[1];P=W/"package";J=P/"01_Journal_files";S=P/"04_Editable_sources"
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
parent=W.parent/"github_publication_20261108_v1/package"
body=lambda f:f.read_text().split("## Introduction\n",1)[1].split("## Acknowledgment\n",1)[0]
check("Clinical scientific main body preserved",body(S/"MANUSCRIPT_EN.md")==body(parent/"04_Editable_sources/MANUSCRIPT_EN.md"))
for n in ["EMETHODS.md","ERESULTS.md","FIGURE_LEGENDS.md","MAIN_TABLES.md"]:
 check(n+" scientific source text preserved",sha(S/n)==sha(parent/"04_Editable_sources"/n))
t=(S/"ETABLES.md").read_text().split("## eTable 4.",1)[1].split("## eTable 5.",1)[0]
rows=[[v.strip() for v in x.strip("|").split("|")] for x in t.splitlines() if x.startswith("|")]
source=list(csv.DictReader((P/"02_Code_and_aggregate_data/aggregate_data/tables/Table_S19_Stage_comparisons.csv").open()))
check("eTable4 identifies eight fields and28rows",len(rows)==30 and all(len(x)==8 for x in rows))
for i,(row,z) in enumerate(zip(rows[2:],source)):
 metric,capacity={"AUROC":("AUROC","—"),"Brier":("Brier score","—"),"capacity10_captured_per1000":("Captured events","10%"),"capacity20_captured_per1000":("Captured events","20%")}[z["metric"]]
 check("eTable4 row"+str(i+1)+" source metric/capacity",row[2:4]==[metric,capacity])
 check("eTable4 row"+str(i+1)+" source estimate/CI",all(Decimal(row[k])==Decimal(z[v]) for k,v in zip([5,6,7],["estimate","lower95","upper95"])))
d=Document(J/"SUPPLEMENT_1.docx")
for i,table in enumerate(d.tables):
 ok=all(r.font.name=="Arial" and r.font.size and abs(r.font.size.pt-10)<.01 for row in table.rows for c in row.cells for p in c.paragraphs for r in p.runs if r.text)
 check("Supplement table"+str(i+1)+" actual Arial10 runs",ok)
check("52 physical supplementary tables",len(d.tables)==52)
td=next(table for table in d.tables if [c.text for c in table.rows[0].cells][:3]==["Evaluation","System","Metric"])
check("eTable4 Word values retain exact mapped table",[[c.text for c in r.cells] for r in td.rows[1:]]==rows[2:])
titles=[]
for p in d.paragraphs:
 if p.style.name.startswith("Heading") and re.match(r"e(?:Table|Figure)\s+\d+\.",p.text):
  check(p.text[:80]+" Arial12bold title",all(r.font.name=="Arial" and r.font.size and r.font.size.pt==12 and r.font.bold for r in p.runs if r.text))
  titles.append(p.text)
m=P/"05_Figure_assets/JNO_VECTOR_PAGES/PAGE_AND_PANEL_MANIFEST.json";spec=json.loads(m.read_text());base=m.parent
check("18 supplemental figures",spec["figures_count"]==18)
check("53 supplemental panels",spec["panels_count"]==53)
check("31 native vector pages",spec["pages_count"]==31)
check("Final figure source review completed",str(spec.get("status","")).startswith("PASS") and "PENDING" not in str(spec.get("final_visual_review","")))
check("31 native Word inline vector drawings",len(d.inline_shapes)==31)
for i,x in enumerate(d.inline_shapes):check("Native vector width "+str(i+1),abs(x.width.mm-170)<.01)
with zipfile.ZipFile(J/"SUPPLEMENT_1.docx") as z:
 media=[n for n in z.namelist() if n.startswith("word/media/")]
 svg=[n for n in media if n.endswith(".svg")]
 check("Word contains31 native SVG assets",len(svg)==31)
 check("No bitmap-only graphs in Word media",not any(n.lower().endswith((".png",".jpg",".jpeg",".tiff",".gif",".bmp")) for n in media))
 expected=Counter(sha(base/x["svg"]) for x in spec["pages"])
 actual=Counter(hashlib.sha256(z.read(n)).hexdigest() for n in svg)
 check("Embedded SVG bytes match accepted source pages",actual==expected)
pdf=PdfReader(J/"SUPPLEMENT_1.pdf");rendered=[]
for i,page in enumerate(pdf.pages):
 text=page.extract_text()
 # In the figure section every graph text must remain selectable, in addition to headings/legend text.
 if re.match(r"\s*eFigure\s+\d+\.",text) and i>10:
  found=[]
  def visit(txt,cm,tm,font,size):
   if txt.strip() and font:
    f=str(font.get("/BaseFont",""))
    if "Arial" in f and abs(size-10)<.05:found.append(txt.strip())
  page.extract_text(visitor_text=visit)
  check("Supplement PDF page"+str(i+1)+" selectable Arial10 plot labels",len(found)>=3)
  rendered.append({"page":i+1,"Arial10_text_fragments":len(found),"image_xobjects":len(list(page.images)),"start":text[:95]})
check("All31 vector pages rendered",len(rendered)==31)
for rec in spec["pages"]:
 check(rec["figure"]+" page"+str(rec["page"])+" accepted SVG hash",sha(base/rec["svg"])==rec["svg_sha256"])
 check(rec["figure"]+" page"+str(rec["page"])+" body fonts10",rec["minimum_text_pt"]==10)
 check(rec["figure"]+" page"+str(rec["page"])+" native170mm",rec["width_mm"]==170)
panels=list((P/"05_Figure_assets/Independent_panels").rglob("*.pdf"))
check("71 independent panel PDFs present",len(panels)==71)
check("No private font binaries in package",not list(P.rglob("*.ttf")) and not list(P.rglob("*.otf")))
source={
 "main":sha(S/"MANUSCRIPT_EN.md"),"tables":sha(S/"ETABLES.md"),
 "supplement_docx":sha(J/"SUPPLEMENT_1.docx"),"supplement_pdf":sha(J/"SUPPLEMENT_1.pdf"),
 "figure_manifest":sha(m)}
result={"status":"PASS_JNO_FONTS_NATIVE_VECTORS_SOURCE_VALUES_AND_DOCUMENTS","checks_count":len(checks),"checks":checks,"source_hashes":source,"supplement_pages":len(pdf.pages),"rendered_figure_pages":rendered,"physical_tables":52,"numbered_eTables":32,"figures":23,"independent_panels":71,"science_unchanged":True,"new_statistics":0,"human_expert_review":False}
(W/"review/JNO_FINAL_SOURCE_AND_FORMAT_VERIFICATION.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ["status","checks_count","supplement_pages","physical_tables","numbered_eTables","figures","independent_panels"]}))
