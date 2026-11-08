from pathlib import Path
import re,json,hashlib,difflib
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/workbuddy_final_revision_20261108_v1')
S=W/'package/04_Editable_sources'
tripod='Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378'
def shift(m):
 ids=[int(x) for x in m.group(1).split(',')]
 return '['+','.join(str(n+1 if n>=10 else n) for n in ids)+']'
changes=[]
for lang,refheading in [('EN','References'),('ZH','参考文献')]:
 p=S/f'MANUSCRIPT_{lang}.md';before=p.read_text()
 if '10.1136/bmj-2023-078378' in before:continue
 body,refs=before.split('## '+refheading+'\n',1)
 body=re.sub(r'\[(\d+(?:,\d+)*)\]',shift,body)
 if lang=='EN':
  body=body.replace('We reported AUROC, average precision, Brier score, calibration, captured/missed events, and positive predictive value.','We reported the area under the receiver operating characteristic curve (AUROC), average precision, Brier score, calibration, captured/missed events, and positive predictive value.')
  # AUROC first appears in the abstract, an independent reading unit.
  body=body.replace('Six-input model AUROCs were 0.614 and 0.621,','The six-input areas under the receiver operating characteristic curve were 0.614 and 0.621,')
  body=body.replace('Median age was 65 years (IQR, 56.0–71.8);','Median age was 65 years (interquartile range, 56.0 to 71.8).')
  marker='### Inputs and frozen systems'
  body=body.replace(marker,'Reporting was structured using the TRIPOD+AI guideline. [10]\n\n'+marker,1)
  body=body.replace('through November 5, 2026.','through November 8, 2026.')
 else:
  marker='### 输入与冻结系统'
  body=body.replace(marker,'报告结构依据TRIPOD+AI指南组织。[10]\n\n'+marker,1)
  body=body.replace('本轮记录更新至2026年11月5日。','本轮记录更新至2026年11月8日。')
 lines=[]
 for line in refs.splitlines():
  match=re.match(r'^\[(\d+)\] (.*)',line)
  if not match:continue
  n=int(match[1]);entry=match[2]
  entry=entry.replace('STARSurg Collaborative; EuroSurg Collaborative.','STARSurg Collaborative, EuroSurg Collaborative.')
  entry=entry.replace('Risk Stratification for Postoperative Acute Kidney Injury in Major Noncardiac Surgery Using Preoperative and Intraoperative Data.','Risk stratification for postoperative acute kidney injury in major noncardiac surgery using preoperative and intraoperative data.')
  entry=entry.replace('Intraoperative Data Enhance the Detection of High-Risk Acute Kidney Injury Patients When Added to a Baseline Prediction Model.','Intraoperative data enhance the detection of high-risk acute kidney injury patients when added to a baseline prediction model.')
  entry=entry.replace('Sci Data. 2023;10:1.','Sci Data. 2023;10(1):1.')
  entry=entry.replace('J Formos Med Assoc. 2026.','J Formos Med Assoc. Published online May 21, 2026.')
  entry=re.sub(r'(\d)–(?=\d)',r'\1-',entry)
  if n==10:lines.append('[10] '+tripod)
  lines.append('['+str(n+1 if n>=10 else n)+'] '+entry)
 after=body+'## '+refheading+'\n\n'+'\n\n'.join(lines)+'\n'
 p.write_text(after)
 changes.append({'language':lang,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'references':18})
 (W/f'review/EDITORIAL_FORMAT_{lang}.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='working_parent',tofile='format_revision')))
 sections=[('Introduction','INTRODUCTION'),('Methods','METHODS'),('Results','RESULTS'),('Discussion','DISCUSSION'),('Conclusions','CONCLUSIONS')] if lang=='EN' else [('前言','INTRODUCTION'),('方法','METHODS'),('结果','RESULTS'),('讨论','DISCUSSION'),('结论','CONCLUSIONS')]
 for heading,stem in sections:
  chunk='## '+heading+'\n'+after.split('## '+heading+'\n',1)[1].split('\n## ',1)[0]
  (S/'Chapters'/f'{stem}_{lang}.md').write_text(chunk.rstrip()+'\n')
# Figure letter case follows the same panel identity across sources.
p=S/'FIGURE_LEGENDS.md';t=p.read_text();t=re.sub(r'\*\*([a-d](?:,[a-d])*)\*\*',lambda m:'**'+m[1].upper()+'**',t)
p.write_text(t);(S/'Chapters/FIGURE_LEGENDS.md').write_text(t)
# Add the reporting reference to editable citation databases without changing keys.
with (S/'references.bib').open('a') as f:f.write('\n@article{Collins2024TRIPODAI,\n author = {Collins, Gary S. and Moons, Karel G. M. and Dhiman, Paula and others},\n title = {TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods},\n journal = {BMJ},\n year = {2024},\n volume = {385},\n pages = {e078378},\n doi = {10.1136/bmj-2023-078378}\n}\n')
with (S/'references.ris').open('a') as f:f.write('\nTY  - JOUR\nAU  - Collins, Gary S\nAU  - Moons, Karel G M\nAU  - Dhiman, Paula\nTI  - TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods\nJO  - BMJ\nPY  - 2024\nVL  - 385\nSP  - e078378\nDO  - 10.1136/bmj-2023-078378\nER  - \n')
(W/'review/EDITORIAL_FORMAT_REPAIR.json').write_text(json.dumps({'changes':changes,'new_reporting_guideline_source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/','online_first_source':'https://pubmed.ncbi.nlm.nih.gov/42168064/','author_manually_verified_all_references_claim_added':False,'new_numerical_analysis_added':False,'Word_and_PDF_rebuild':'pending accepted supplemental results'},ensure_ascii=False,indent=2))
print(json.dumps({'source_manuscripts_updated':len(changes),'references':18,'chapters_synchronized':10,'Word_render_pending':True}))
