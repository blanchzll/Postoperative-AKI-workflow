
from pathlib import Path
import re,json,csv,hashlib,ast,zipfile,collections
from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader
W=Path("/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/novelty_revision_20261101_v1");P=W/"package";T=W/"work/text";R=W/"review"
checks=[]
def check(k,v):
 checks.append({"check":k,"pass":bool(v)})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((R/"BASELINE_PROTECTION.json").read_text());check("sealed_baseline_all_hashes",all(sha(p)==h for p,h in old.items()))
src=json.loads((R/"NEW_SCIENTIFIC_SOURCE_HASHES.json").read_text());check("new_statistical_sources_unchanged",all(sha(p)==h for p,h in src.items()))
m=(T/"MANUSCRIPT_EN.md").read_text();body=m.split("## Introduction")[1].split("## Acknowledgment")[0]
abstract=m.split("## Abstract")[1].split("## Introduction")[0];kp=m.split("## Key Points")[1].split("## Abstract")[0]
check("body_under_3000",len(body.split())<=3000);check("abstract_under_350",len(abstract.split())<=350);check("keypoints_under_100",len(kp.split())<=100);check("title_under_100",len(m.splitlines()[0][2:])<=100)
check("five_main_figure_files",len(list((P/"01_Journal_files").glob("FIGURE_[1-5].pdf")))==5)
check("no_main_table_files",not list((P/"01_Journal_files").glob("TABLE*.docx")))
check("no_tables_in_main_word",len(Document(P/"01_Journal_files/MANUSCRIPT.docx").tables)==0)
es=json.loads((T/"FIGURE_ENTRIES.json").read_text());check("20_legends",len(es)==20);check("main_fig_labels",[e["label"] for e in es[:5]]==[f"Figure {i}" for i in range(1,6)]);check("15_supplementary_figs",[e["label"] for e in es[5:]]==[f"eFigure {i}" for i in range(1,16)])
tabletitles=re.findall(r"^## eTable (\d+)\.",(T/"ETABLES.md").read_text(),re.M);check("24_supplementary_tables",tabletitles==[str(i) for i in range(1,25)])
# Each delivered figure is bound to the original accepted export; all four formats retained.
figure_checks=0
for e in es:
 for path in e["source"]["exports"]:
  target=P/"05_Figure_assets"/e["label"].replace(" ","_")/Path(path).name
  check(e["label"]+" "+Path(path).suffix+" hash",sha(target)==sha(path));figure_checks+=1
# Document integrity, clean markup and actual native PDF rendering.
docrows=[]
for p in P.rglob("*.docx"):
 d=Document(p);z=zipfile.ZipFile(p);xml=z.read("word/document.xml").decode()
 check(str(p.relative_to(P))+" clean revisions",not re.search(r"<w:(?:ins|del|commentRangeStart|commentRangeEnd)(?:\s|>)",xml))
 check(str(p.relative_to(P))+" no watermark",not re.search("watermark|PowerPlusWaterMarkObject",xml,re.I))
 check(str(p.relative_to(P))+" no comment part",not any(n.startswith("word/comments") for n in z.namelist()))
 pdf=R/"rendered"/(p.stem+".pdf");check(str(p.relative_to(P))+" rendered",pdf.exists())
 pages=len(PdfReader(pdf).pages) if pdf.exists() else None
 # Tables have only top/bottom and header-underlining rules.
 for i,t in enumerate(d.tables):
  borders=t._tbl.tblPr.find(qn("w:tblBorders"))
  check(p.name+" table"+str(i)+" three_line",borders is not None and all(borders.find(qn("w:"+k)).get(qn("w:val"))=="nil" for k in ["left","right","insideH","insideV"]) and all(borders.find(qn("w:"+k)).get(qn("w:val"))=="single" for k in ["top","bottom"]))
 docrows.append({"file":str(p.relative_to(P)),"tables":len(d.tables),"pages":pages})
# Frozen source code is parsed, never executed as a validation substitute.
parsed=0
for p in (P/"02_Code_and_aggregate_data").rglob("*.py"):
 try:ast.parse(p.read_text());parsed+=1
 except Exception as e:check("AST "+str(p),False)
check("no_patient_tables",not any("PRIVATE_" in p.name for p in P.rglob("*.csv")))
for p in P.rglob("*.zip"):
 with zipfile.ZipFile(p) as z:
  check(p.name+" CRC",z.testzip() is None);check(p.name+" no_private_record_names",not any("PRIVATE_" in n for n in z.namelist()))
# Compare source tables rather than handoff prose for the new primary numbers.
bind=json.loads((R/"NEW_NUMERIC_BINDING.json").read_text());check("30_new_values_source_bound",len(bind)==30 and all(r["present_in_main"] for r in bind))
check("MIMIC_final_current_counts","759 patients/59 events" in m and "56 events" not in m)
check("old_273_272_overlap_retained","273 records" in m and "272 groups" in m)
check("negative_Plattslope_retained","−0.276651" in m)
check("chapters_EN_ZH_present",len(list((P/"04_Editable_sources/Chapters").glob("*.docx")))==10)
# Every main citation resolves; first appearance order must be sequential.
front,refs=m.split("## References");seq=[];cited=[]
for match in re.finditer(r"\[(\d+(?:,\d+)*)\]",front):
 for n in map(int,match[1].split(",")):
  cited.append(n)
  if n not in seq:seq.append(n)
check("main_reference_order",seq==list(range(1,18)));check("17_main_references",len(re.findall(r"^\[\d+\]",refs,re.M))==17)
check("bib17_entries",len(re.findall(r"^@(?:article|misc)\{",(P/"04_Editable_sources/references.bib").read_text(),re.M))==17)
out={"status":"PASS" if all(c["pass"] for c in checks) else "FAIL","checks":checks,"document_pages":docrows,"body_words":len(("Introduction"+body).split()),"abstract_words":len(abstract.split()),"key_points_words":len(kp.split()),"baseline_files":len(old),"new_source_files":len(src),"figure_export_hashes":figure_checks,"python_sources_parsed":parsed,"checks_count":len(checks)}
(R/"FINAL_CHECKS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ["checks","document_pages"]},ensure_ascii=False));print("FAILED",json.dumps([c for c in checks if not c["pass"]],ensure_ascii=False))
