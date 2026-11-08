from pathlib import Path
import re,json,difflib,shutil,hashlib
W=Path(__file__).resolve().parents[1];B=W.parent;T=W/'work/text';T.mkdir(exist_ok=True)
figmap={1:1,12:2,10:3,11:4,5:5,2:6,3:7,9:8,6:9,8:10,7:11,4:12,13:13,14:14}
tabmap={1:1,17:2,18:3,19:4,20:5,16:6,21:7,11:8,12:9,14:10,15:11,13:12,6:13,2:19,4:18,5:7}
changes=[]
s=(B/'manuscript_revision/MANUSCRIPT_EN.md').read_text()
def replace(old,new,issue):
 global s
 assert old in s,issue
 s=s.replace(old,new);changes.append({'id':issue,'before':old,'after':new})
replace('Same-record comparisons at fixed capacity quantify the capture cost of additional fields. Capacity increments and transferred thresholds quantify how many records enter the proposed follow-up workflow.', 'Same-record comparisons at fixed capacity quantify capture differences between input strategies. Capacity increments and transferred thresholds quantify how list size and event coverage change. Input collection and review of selected records impose separate demands on a service.', 'L01-distinct-input-and-review-demands')
replace('Studies incorporating intraoperative information have reported variable gains across populations and outcomes. [5–7]', 'Lei et al. compared perioperative stages using the highest-risk 20% of records, with 10% and 30% sensitivity analyses. [5] Other intraoperative studies have reported variable gains across populations and outcomes. [6,7]', 'N01-exact-nearest-top20-precedent')
replace('We asked how information requirements, assessment timing, and capacity trade off recorded AKI capture against anticipated review requirements.', 'The paired comparisons estimate the capture cost of reducing fields and the capture gain from operation-end updating. The capacity and threshold analyses estimate the resulting list sizes and event coverage.', 'L02-question-to-estimand-chain')
replace('Full procedures are in the Supplementary Methods.', 'Full procedures and input definitions are in eMethods and eTable 1 in Supplement 1.', 'J01-specific-methods-dictionary-reference')
replace('In gastrointestinal cancer surgery, selecting an AKI assessment strategy requires accounting for information requirements and review capacity. Nine preoperative inputs had observed discrimination close to that of 27 inputs, whereas operation-end gains varied across evaluations and capacities. Comparing event capture and selected-list size within evaluation samples quantifies the consequences of additional fields, risk updating, and capacity expansion.', 'At 20% review capacity, reducing preoperative inputs from 27 to nine changed capture by two geographic events and three temporal events, with paired intervals spanning gains and losses. Operation-end updating captured 15 additional geographic events and one temporal event. Expanding the nine-input list from 10% to 20% captured eight and nine additional events. These comparisons connect information requirements and assessment timing to the number of records selected and events captured.', 'L03-results-first-discussion')
replace('The present study estimates the capture cost of reducing fields within the same evaluation sample and capacity.', 'The present study estimates the capture cost of field reduction while holding evaluation records and review capacity constant.', 'N02-specific-added-value')
replace('Lei et al. assessed high-risk capture and reported modest discrimination gains from intraoperative information.', 'Lei et al. assessed capture in fixed highest-risk fractions and reported modest discrimination gains from intraoperative information.', 'N03-accurate-precedent-in-discussion')
replace('Compared systems incorporate model-family, development, and calibration differences whose contributions were not separated.', 'The stage contrast estimates the difference between complete fitted systems, including their model families, development procedures, and calibration.', 'L04-positive-estimand-definition')
replace('Our comparison estimates complete workflows, differing in model family, preprocessing, and released-code histories.', 'The public comparison estimates the performance of complete workflows with their respective model families, preprocessing, and released-code histories.', 'L05-workflow-definition')
replace('Nine routine preoperative inputs provide an assessment option with fewer fields, whose capture cost can be weighed against local collection burden and a justified loss tolerance. Operation-end updating, capacity expansion, and threshold transfer should be assessed jointly through event coverage and review requirements within the intended workflow.', 'Reducing preoperative inputs from 27 to nine changed observed capture by two to three events at 20% capacity, with intervals spanning gains and losses. Operation-end gains varied across evaluations. Reporting paired capture differences alongside list size and achieved threshold coverage can inform local choices about information collection and review capacity.', 'L06-concrete-conclusions')
replace('## Declarations for author completion','## Acknowledgment','J02-journal-disclosure-section')
replace('**AI assistance:** Codex (OpenAI) supported code checking, evidence synthesis, and manuscript drafting.', '**AI assistance:** Codex (OpenAI) supported code checking, evidence synthesis, manuscript drafting, language editing, and formatting through November 1, 2026.', 'D01-current-ai-use-scope')
s=re.sub(r'### 4\.\d+ ', '### ',s)
# Numbered display items follow the order of their first citation in the main text.
def convert(x):
 x=re.sub(r'Supplementary Figure (\d+)',lambda m:f'eFigure {figmap[int(m[1])]} in Supplement 1',x)
 x=re.sub(r'Tables S11[–-]S12', 'eTables 8 and 9 in Supplement 1',x)
 x=re.sub(r'Tables S(\d+)',lambda m:f'eTable {tabmap[int(m[1])]} in Supplement 1',x)
 x=re.sub(r'Table S(\d+)',lambda m:f'eTable {tabmap[int(m[1])]} in Supplement 1',x)
 x=x.replace('Tables 1 and S17','Table 1 and eTable 2 in Supplement 1')
 return x
s=convert(s)
s=s.replace('(Figure 2; Table 2).','(Figure 2; Table 2; eTable 4 in Supplement 1).')
s=s.replace('per 1,000; Figure 3).','per 1,000; Figure 3; eFigure 7 in Supplement 1).')
s=s.replace('All eight public views appear in eTable 6 in Supplement 1.','All eight public views and the main operating summaries appear in eTables 6 and 7 in Supplement 1.')
s=s.replace('Foundation-model conditions and contrasts remain in eTables 8 and 9 in Supplement 1.', 'Published-model and TabPFN benchmarks appear in eTable 8 in Supplement 1; medical-language-model conditions appear in eTable 9 and eFigure 11 in Supplement 1.')
s=s.replace('Body mass index and American Society', 'Calibration profiles are shown in eFigure 12 in Supplement 1. Body mass index and American Society')
s=s.replace('Their information-collection requirements differ.', 'Their information-collection requirements differ. Assumption-based workload scenarios are presented in eFigure 13 and eTable 18 in Supplement 1.')
s=s.replace("the three history indicators' contributions were not isolated.", "the three history indicators' contributions were not isolated. A same-family source-cohort comparison is provided in eFigure 14 and eTable 11 in Supplement 1.")
s=s.replace('**Conclusions and Relevance:** Nine inputs offer a lower-field option whose capture cost can be judged against collection burden and a justified loss tolerance. End-of-operation updating requires evaluation within its intended workflow. List-size and event-coverage estimates support choices about follow-up capacity.', '**Conclusions and Relevance:** Paired event-capture estimates quantify the consequences of reducing inputs, updating risk at operation end, and expanding review lists. Achieved coverage and list size after threshold transfer can inform local assessment workflows.')
s=s.replace('**Design, Setting, and Participants:** Retrospective cohort study of adult operations at five Chinese centers from December 4, 2017, through June 25, 2024, with geographic and temporal evaluation.', '**Design, Setting, and Participants:** Retrospective cohort study of adult operations at five Chinese centers from December 4, 2017, through June 25, 2024, with geographic and temporal evaluation of previously examined records.')
s=s.replace('; Figure 3).','; Figure 3; eFigure 7 in Supplement 1).')
q='Published-model and TabPFN benchmarks appear in eTable 8 in Supplement 1; medical-language-model conditions appear in eTable 9 and eFigure 11 in Supplement 1. '
s=s.replace(q,'').replace('(eFigure 10 in Supplement 1).','(eFigure 10 in Supplement 1). '+q.strip())
s=s.replace('hemoglobin. Overlapping predictor information','hemoglobin (eTable 10 in Supplement 1). Overlapping predictor information')
s=s.replace('eFigure 13 and eTable 18 in Supplement 1','eFigure 13 in Supplement 1')
s=s.replace('Review completion, staff time, additional assessment, and patient outcomes should be recorded.','Review completion, staff time, additional assessment, and patient outcomes should be recorded. Supporting analyses and complete paired metrics are reported in eTables 12–19 in Supplement 1.')
(T/'MANUSCRIPT_EN_scientific.md').write_text(s)
# Humanizer pass: remove noninformative drafting formulas, retain technical ranges and journal headings.
h=s.replace('Full procedures and input definitions are in', 'Procedures and input definitions are in')
h=h.replace('Supporting same-family and overlap-restricted comparisons', 'Same-family and overlap-restricted comparisons')
(T/'MANUSCRIPT_EN_humanizer.md').write_text(h)
# less-ai-tone is reviewed as a second pass; English text and scientific symbols are preserved under its explicit scope exclusions.
(T/'MANUSCRIPT_EN.md').write_text(h)
# Align the Chinese author-review manuscript with the same substantive revisions, without redoing unchanged translation.
z=(B/'manuscript_revision/MANUSCRIPT_ZH.md').read_text()
(T/'MANUSCRIPT_ZH_original.md').write_text(z)
# Chinese text will be revised after an exact paragraph read; preserve this as input until then.
# Cover letter contains only supported study results and explicitly incomplete author declarations.
title=h.splitlines()[0].lstrip('# ')
cover=f'''# Cover letter\n\nNovember 1, 2026\n\nEditors\nJAMA Network Open\n\nDear Editors:\n\nPlease consider our manuscript, “{title},” as an Original Investigation.\n\nServices planning postoperative kidney assessment must decide which information to collect, when to update risk, and how many records they can review. We evaluated these decisions in 3,710 gastrointestinal cancer operations from five Chinese centers, with geographic and temporal evaluations, and assessed transfer to 6,043 INSPIRE records.\n\nAt 20% review capacity, 27-input models captured two and three more recorded events than nine-input models; the paired intervals included gains and losses. Operation-end systems captured 15 additional geographic events and one temporal event. Expanding nine-input capacity from 10% to 20% added 116 and 168 selections and captured eight and nine additional events. Transferred source thresholds produced different list sizes and achieved coverage in the public cohort.\n\nThe study extends existing work on compact AKI models and high-risk stratification by estimating the paired capture cost of field reduction at equal capacity and connecting that comparison with operation-end updating and threshold-transfer consequences. These results are relevant to perioperative clinicians and services allocating postoperative assessment capacity.\n\nThe manuscript includes three main figures and two editable main tables. Supplement 1 contains the methods, supporting results, eTables, and eFigures. Analysis code and aggregate source tables accompany the submission materials.\n\nCodex (OpenAI) assisted code checking, evidence synthesis, drafting, editing, and formatting. A separate GPT-6.1 Sol (OpenAI) API workflow assisted extraction from prepared public-literature excerpts. The Acknowledgment section records this use. [AUTHOR INPUT NEEDED: confirm exact tool versions, complete dates and uses, and author responsibility.]\n\n[AUTHOR INPUT NEEDED: confirm all authors’ approval and accountability; originality and concurrent-submission status; related publications, preprints, or manuscripts from the same study; any prior editorial review; funding and competing interests. State the verified circumstances.]\n\nSincerely,\n\n[Corresponding author’s full name and highest academic degrees]\n[Affiliation and postal address]\n[Email address]\n[Telephone number]\n'''
(T/'COVER_LETTER_scientific.md').write_text(cover);(T/'COVER_LETTER_humanizer.md').write_text(cover);(T/'COVER_LETTER.md').write_text(cover)
# Main tables and all figure legends, with journal-specific citation labels.
main=convert((B/'MAIN_TABLES.md').read_text()).replace('Table 1 |','Table 1.').replace('Table 2 |','Table 2.')
main=main.replace('95% confidence interval','95% CI')
(T/'MAIN_TABLES.md').write_text(main)
figs=json.loads((B/'FIGURE_MANIFEST.json').read_text())
parts=re.split(r'\n## ',(B/'FIGURE_LEGENDS.md').read_text())[1:]
assert len(parts)==17
leg=[];entries=[]
for m,part in zip(figs,parts):
 head,caption=part.strip().split('\n',1)
 if m['figure'].startswith('Figure_'):
  num=int(m['figure'].split('_')[1]);label=f'Figure {num}';order=num
 else:
  num=int(m['figure'].split('_')[2]);order=3+figmap[num];label=f'eFigure {figmap[num]}'
 head=re.sub(r'^(?:Supplementary Figure|Figure) \d+\s*[|.]?\s*', '',head)
 caption=convert(caption).strip()
 leg.append((order,f'## {label}. {head}\n\n{caption}'))
 entries.append({'label':label,'order':order,'title':head,'caption':caption,'source':m})
leg.sort();entries.sort(key=lambda x:x['order'])
(T/'FIGURE_LEGENDS.md').write_text('# Figure legends\n\n'+'\n\n'.join(x[1] for x in leg)+'\n')
(T/'MAIN_FIGURE_LEGENDS.md').write_text('# Main figure legends\n\n'+'\n\n'.join(x[1] for x in leg[:3])+'\n')
(T/'FIGURE_ENTRIES.json').write_text(json.dumps(entries,indent=2))
(W/'review/CORRECTIONS.json').write_text(json.dumps(changes,indent=2))
(W/'review/NUMBERING_CROSSWALK.json').write_text(json.dumps({'figures_old_to_journal':figmap,'tables_old_to_journal':tabmap,'reason':'Journal first-citation order; original artwork and frozen source identifiers retained'},indent=2))
(W/'review/MANUSCRIPT_REVISIONS.diff').write_text(''.join(difflib.unified_diff((B/'manuscript_revision/MANUSCRIPT_EN.md').read_text().splitlines(True),h.splitlines(True),fromfile='received_manuscript',tofile='final_manuscript')))
print(json.dumps({'changes':len(changes),'title_characters':len(title),'figure_legends':len(leg),'text_dir':str(T)}))
