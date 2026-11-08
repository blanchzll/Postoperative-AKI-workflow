"""Verify source-mapped reporting, frozen results and actual figure sizes."""
from pathlib import Path
from decimal import Decimal
import json,re,hashlib
from pypdf import PdfReader
from docx import Document
W=Path(__file__).resolve().parents[1];P=W/"package";parent=W.parent/"github_publication_20261108_v1/package"
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
a=W.parents[1]/"correction_runs/final_workbuddy_analysis_20261108_v1/ACCEPTANCE.json"
check("Fresh source/scientific acceptance identity",sha(a)=="e1bea0d540cbe005e694eda24da405a53382bc7315522202d7db657b690b6a6c")
s=P/"04_Editable_sources"
body=lambda path:path.read_text().split("## Introduction\n",1)[1].split("## Acknowledgment\n",1)[0]
check("Scientific manuscript body unchanged",body(s/"MANUSCRIPT_EN.md")==body(parent/"04_Editable_sources/MANUSCRIPT_EN.md"))
table=(s/"ETABLES.md").read_text().split("## eTable 4.",1)[1].split("## eTable 5.",1)[0]
rows=[[v.strip() for v in l.strip("|").split("|")] for l in table.splitlines() if l.startswith("|")]
check("All 28 rows explicitly label eight fields",len(rows)==30 and all(len(x)==8 for x in rows))
import csv
source=list(csv.DictReader((P/"02_Code_and_aggregate_data/aggregate_data/tables/Table_S19_Stage_comparisons.csv").open()))
for i,(row,original) in enumerate(zip(rows[2:],source)):
 expected={"AUROC":("AUROC","—"),"Brier":("Brier score","—"),"capacity10_captured_per1000":("Captured events","10%"),"capacity20_captured_per1000":("Captured events","20%")}[original["metric"]]
 check(f"eTable4 row {i+1} metric/capacity",tuple(row[2:4])==expected)
 check(f"eTable4 row {i+1} estimates/intervals",all(Decimal(row[k])==Decimal(original[field]) for k,field in zip([5,6,7],["estimate","lower95","upper95"])))
doc=Document(P/"01_Journal_files/SUPPLEMENT_1.docx")
dt=next(t for t in doc.tables if [c.text for c in t.rows[0].cells][:3]==["Evaluation","System","Metric"])
check("Rendered Word table retains 28 x 8 source-mapped cells",len(dt.rows)==29 and [[c.text for c in r.cells] for r in dt.rows[1:]]==rows[2:])
check("All 18 supplementary figures embedded",len(doc.inline_shapes)==18)
widths=[round(x.width.mm,2) for x in doc.inline_shapes]
check("Old supplemental figures at 170mm; new ones at native 160mm",widths==[170]*15+[160]*3)
check("Images fit supplement text width",all(x.width<=doc.sections[0].page_width-doc.sections[0].left_margin-doc.sections[0].right_margin for x in doc.inline_shapes))
def pdf_fonts(f):
 p=PdfReader(f).pages[0];sizes=[]
 def visit(text,cm,tm,font,sz):
  if text.strip():sizes.append(float(sz)*(cm[0]**2+cm[1]**2)**.5)
 p.extract_text(visitor_text=visit)
 return float(p.mediabox.width)*25.4/72,min(sizes) if sizes else None
figs=[]
for folder in sorted((P/"05_Figure_assets").iterdir()):
 if not folder.is_dir() or folder.name=="Independent_panels":continue
 f=next(folder.glob("*.pdf"));native,minimum=pdf_fonts(f)
 embed=160 if folder.name in ["eFigure_16","eFigure_17","eFigure_18"] else 170 if folder.name.startswith("e") else native
 effective=minimum*embed/native
 check(folder.name+" actual-size font >=6pt",effective>=6-1e-5)
 figs.append({"figure":folder.name,"native_width_mm":native,"native_min_font_pt":minimum,"embedding_width_mm":embed,"effective_min_font_pt":effective,"pdf_sha256":sha(f)})
 if folder.name not in ["eFigure_16","eFigure_17","eFigure_18"]:
  check(folder.name+" original numerical figure PDF unchanged",sha(f)==sha(parent/"05_Figure_assets"/folder.name/f.name))
panels=[]
for f in sorted((P/"05_Figure_assets/Independent_panels").rglob("*.pdf")):
 native,minimum=pdf_fonts(f);check(str(f.relative_to(P))+" native font >=6pt",minimum is not None and minimum>=6)
 panels.append({"file":str(f.relative_to(P)),"sha256":sha(f),"native_width_mm":native,"min_font_pt":minimum})
check("23 composite figures",len(figs)==23)
check("71 independent panel PDFs",len(panels)==71)
result={"status":"PASS_SOURCE_MAPPING_FROZEN_SCIENCE_ACTUAL_SIZES","checks":checks,"checks_count":len(checks),"figures":figs,"panels":panels,"new_statistics":0,"human_expert_panel":False,"scientific_source_acceptance_sha256":sha(a)}
(W/"review/FINAL_REPORTING_VERIFICATION.json").write_text(json.dumps(result,indent=2))
print(json.dumps({"status":result["status"],"checks":len(checks),"figures":len(figs),"panels":len(panels),"min_effective_composite_font_pt":min(x["effective_min_font_pt"] for x in figs)}))
