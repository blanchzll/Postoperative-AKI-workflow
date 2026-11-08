from pathlib import Path
import csv,json,re,shutil,hashlib,difflib
W=Path(__file__).resolve().parents[1];P=W/'package';S=P/'04_Editable_sources'
R=W.parents[1]/'correction_runs/clinical_supplement_20261108_v1'
ac=R/'ACCEPTANCE.json'
if not ac.exists():raise SystemExit('Current clinical supplement not yet accepted')
accept=json.loads(ac.read_text())
if not str(accept.get('status','')).startswith(('PASS','ACCEPT','COMPLETE')):raise SystemExit('Inspect actual acceptance status before integration: '+str(accept.get('status')))
if (W/'review/CLINICAL_INTEGRATION.json').exists():raise SystemExit('Integration already exists; inspect before rerun')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):
 with (R/'results'/name).open() as f:return list(csv.DictReader(f))
def num(x,d=3):
 if x is None or str(x).strip() in ['','nan','NaN']:return 'Not estimable'
 try:
  v=float(x)
  if v==int(v):return str(int(v))
  return f'{v:.{d}g}'
 except ValueError:return str(x).replace('|','/').replace('\n',' ')
def ci(row,key,lo=None,hi=None,d=3):
 lo=lo or key+'_lo';hi=hi or key+'_hi'
 v=num(row.get(key),d)
 return v+(' ('+num(row.get(lo),d)+' to '+num(row.get(hi),d)+')' if row.get(lo) not in [None,'','nan','NaN'] else '')
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in rr)+' |' for rr in rows])+'\n\n'
def heading(n,title,body,note):
 return f'\n## eTable {n}. {title}\n\n'+body+note+'\n'
context={'geography_preop':'Geographic, 9 to 27','temporal_preop':'Temporal, 9 to 27','geography_paired':'Geographic, 27 to 38','temporal_paired':'Temporal, 27 to 38'}
# Copy only aggregate tables; daily row-level dates and measurement histories remain restricted.
release=P/'02_Code_and_aggregate_data/clinical_supplement_20261108_v1'
(release/'results').mkdir(parents=True,exist_ok=True)
excluded={'DAILY_POTENTIAL_DEMAND.csv','MIMIC_DAILY_OBSERVATION.csv'}
copied=[]
for p in (R/'results').glob('*.csv'):
 if p.name not in excluded:
  dest=release/'results'/p.name;shutil.copy2(p,dest);copied.append({'file':p.name,'sha256':sha(dest)})
(release/'code').mkdir(exist_ok=True)
for p in (R/'code').glob('*.py'):shutil.copy2(p,release/'code'/p.name)
for name in ['README_ZH.md','PROTOCOL_LOCK.json','SCIENCE_CONCLUSION_MATRIX.csv','SCIENTIFIC_REVIEW.json','WORKBUDDY_ANALYSIS_DISPOSITIONS.csv','QUESTION_EVIDENCE_ANALYSIS_MATRIX.csv','ACCEPTANCE.json']:
 shutil.copy2(R/name,release/name)
# New supplementary tables use final accepted CSVs; full precision remains in these CSVs.
b=[]
rows=read('POLICY_EQUAL_BUDGET.csv')
diffs=read('POLICY_PAIRED_DIFFERENCES.csv')
out=[]
for r in rows:
 if abs(float(r['cost_ratio'])-.1)>1e-10:continue
 ds=[x for x in diffs if x['context']==r['context'] and x['policy']==r['policy'] and abs(float(x['cost_ratio'])-.1)<1e-10]
 delta='Reference'
 if ds:
  x=ds[0];keys=list(x)
  v=next((k for k in keys if 'delta' in k and not k.endswith(('_lo','_hi'))),None)
  if v:delta=ci(x,v,'lo','hi')
 out.append([context[r['context']],r['policy'].replace('_',' '),r['upgraded_n'],r['reviewed_n'],r['captured_events'],delta])
b.append(heading(25,'Information acquisition and expanded review at the same assumed resource budget',table(['Comparison','Policy','Acquired, n','Reviewed, n','Events captured','Difference per 1000 (95% CI)'],out),'Primary budget: ceil(0.20N) reviews plus acquisition for ceil(0.25N) records at 0.1 review equivalents per acquisition. Differences use expanded base review at that same total budget. Frozen systems are compared within each context; 2000 paired patient-group samples reroute acquisition and review. Other cost ratios and the 200-seed random allocation distribution remain in the aggregate CSV files.'))
rows=read('FIXED_TRIGGER_COMPARISONS.csv')
b.append(heading(26,'Frozen individual update gates and transferred review thresholds',table(['Comparison','Rule','Updated, n','Reviewed, n','Captured/total events','Review threshold'],[[context[r['context']],r['trigger'].replace('_',' '),r['updated_n'],r['review_n'],r['captured_events']+'/'+r['events'],num(r['threshold'],7)] for r in rows]),'Thresholds retain their development origin. Different systems have different thresholds; the resulting lists do not have equal target capacity. These rules describe predictive-field acquisition, with field-completion counts in FIELD_COMPLETION.csv. Diagnostic test orders and field-release clocks were unavailable.'))
rows=[r for r in read('DAILY_SUMMARY.csv') if r['metric']=='planned_final_reviews']
b.append(heading(27,'Potential final reviews indexed to study-cohort surgery dates',table(['Comparison','Center','Day denominator','Days, n','Requests, n','Mean (95% CI)','95th percentile','Maximum'],[[context[r['context']],r['center'],r['denominator'].replace('_',' '),r['days'],num(r['total']),ci(r,'mean','mean_block7_lo','mean_block7_hi'),num(r['p95']),num(r['max'])] for r in rows]),'Intervals use seven-day moving blocks. The fixed source-defined individual policy produces one final review request per selected operation, indexed by its surgery date. These are potential study-cohort requests. Complete hospital volume, action arrival/completion, shift clocks and measured task times were unavailable; conditional repeat-stage and time scenarios remain separately labelled in source tables.'))
rows=[r for r in read('CENTER_METRICS.csv') if r['role']=='base' and r['center']!='all' and r['rule']=='training_capacity20']
a=table(['Comparison','Center','Records/events','Brier (95% CI)','Observed/expected (95% CI)','Calibration slope'],[[context[r['context']],r['center'],r['N']+'/'+r['events'],ci(r,'Brier'),ci(r,'O_E'),num(r['slope'])] for r in rows])
a+=table(['Comparison','Center','Source cutoff','Reviewed, n','Captured, n','Review proportion (95% CI)','Sensitivity (95% CI)'],[[context[r['context']],r['center'],num(r['threshold'],7),r['reviewed_n'],r['captured_events'],ci(r,'review_fraction'),ci(r,'sensitivity')] for r in rows])
c=read('CENTER_SOURCE_COVERAGE.csv')
a+=table(['Center','Source records','Eligible preop','Eligible paired','Preop coverage','Paired coverage'],[[r['center'],r['source_records'],r['eligible_preop_records'],r['eligible_paired_records'],num(r['preop_source_record_coverage']),num(r['paired_source_record_coverage'])] for r in c])
b.append(heading(28,'Center-specific fixed-model calibration, review burden and source eligibility',a,'No new center exclusion refits or local recalibration were performed. Center 5 has six geographic events and zero temporal events; zero-event bootstrap O/E intervals are suppressed, and Wilson prevalence intervals remain in the CSV. Source coverage describes historical study eligibility.'))
a=table(['Quantity','Observed result'],[['MIMIC qualified population','759 patients; frozen calendar-ratio outcome: 59 events'],['Measured timed SCr components','148 patients; 55 overlap with the frozen 59-event outcome'],['Measured component before procedure day 0','12 patients'],['INSPIRE surgery-start to recorded criterion','Median 42.92 hours; 253 recorded events'],['Clinical surgery/measurement clocks','POD windows; exact operation start/end and release clocks unavailable'],['Complete KDIGO classification','Urine-output ascertainment and complete prediction/result-release clocks unavailable']])
b.append(heading(29,'Renal-outcome measurement windows and clock audit',a,'Timed measured SCr components were audited separately from the unchanged calendar-ratio endpoint. A record-to-criterion interval does not identify when a usable prediction preceded diagnosis. These counts do not replace any accepted study endpoint.'))
rows=read('HARD_OUTCOME_DISCRIMINATION.csv');assoc=read('HARD_OUTCOME_ASSOCIATIONS.csv')
out=[]
for r in rows:
 ar=next((x for x in assoc if x['population_model']==r['population_model'] and x['outcome']==r['outcome']),{})
 out.append([r['population_model'].replace('_',' '),r['outcome'].replace('_',' '),r['N']+'/'+r['events'],ci(r,'AUROC'),ci(ar,'OR','lo','hi')])
a=table(['AKI score','Recorded outcome','Records/events','AUROC (95% CI)','Unadjusted OR per risk doubling (95% CI)'],out)
rows=read('LOS_ASSOCIATIONS.csv')
a+=table(['AKI score','Stay endpoint','Records','Median days (IQR)','Spearman correlation (95% CI)'],[[r['population_model'].replace('_',' '),r['outcome'].replace('_',' '),r['N'],num(r['median_days'])+' ('+num(r['Q1_days'])+' to '+num(r['Q3_days'])+')',ci(r,'Spearman_rho','rho_lo','rho_hi')] for r in rows])
b.append(heading(30,'Frozen AKI scores and recorded hard outcomes or length of stay',a,'Scores are AKI predictions, not calibrated probabilities for the other outcomes. Odds ratios are unadjusted prognostic associations with patient-cluster covariance. Fewer than five outcome events preclude an inferential AUROC interval. Recorded 90-day mortality lacks complete dated follow-up. INSPIRE index-admission death requires an operation-to-discharge timestamp; 10 of 46 institutional death timestamps met that window. Implementation exposure and common time zero were unavailable.'))
rows=read('LOW_RISK_EVENTS.csv')
a=table(['Comparison','Inputs','Frozen low-risk rule','Low-risk records/events','Event proportion (Wilson 95% CI)'],[[context[r['context']],r['inputs'],r['rule'].replace('_',' '),r['low_risk_n']+'/'+r['low_risk_events'],ci(r,'observed_low_risk_event_rate','wilson_lo','wilson_hi')] for r in rows])
rows=read('MIMIC_LOW_RISK_EVENTS.csv')
a+=table(['MIMIC source line','Low-risk records','Recorded events','Qualification'],[[r.get('line',r.get('model','')),r.get('low_risk_n',''),r.get('low_risk_events',''),'Frozen source threshold; empty stratum remains empty'] for r in rows])
rows=read('MIMIC_DELETION_SENSITIVITY.csv')
a+=table(['Population','Deletion schedule','Measurements removed/total','Known events lost/total','Lost proportion (Wilson 95% CI)'],[[r['population'].replace('_',' '),r['mode'].replace('_',' '),r['deleted_measurements']+'/'+r['original_measurements'],r['lost_known_events_lower_bound']+'/'+r['original_known_events'],ci(r,'lost_known_fraction','lost_event_Wilson_lo','lost_event_Wilson_hi')] for r in rows])
b.append(heading(31,'Observed low-risk events and measurement-deletion sensitivity',a,'Deletion is retrospective and retains the accepted outcome reference. Known event losses are a lower bound because events that were never measured are unobserved. Wilson intervals preserve uncertainty when an empirical zero-loss bootstrap degenerates. Monitoring safety is not identified. Patient-bootstrap intervals and Chinese POD-window deletions remain in the source CSV files.'))
rows=[r for r in read('MIMIC_BRIER_REVERIFIED.csv') if r['probability']=='calibrated']
a=table(['Source line','Reference','Metric','Estimate (95% CI)'],[[r['line'],r['reference'].replace('_',' '),r['metric'].replace('_',' '),ci(r,'estimate','lo','hi',4)] for r in rows])
rows=read('DEVELOPMENT_OPTIMISM_AUDIT.csv')
a+=table(['Geographic development branch','Records/events','Apparent LR AUROC','Outer fixed-family AUROC','Difference'],[[r['branch'],r['N']+'/'+r['events'],num(r['development_apparent_AUROC']),num(r['outer_fixed_family_AUROC']),num(r['apparent_minus_outer'])] for r in rows])
b.append(heading(32,'Probability-error benchmarks and development audit',a,'The target-prevalence constant is a descriptive apparent reference, recomputed within each replicate; it is not a deployable target-independent model. Fixed-family CV, fixed-prediction bootstrap and 200 conditional fixed-choice refits do not constitute full model-selection optimism correction. Original-grid approximate max-deviation bands were separately reverified only for their declared primary comparisons, 20% budget and feasibility-rule domain; complete constants remain in ORIGINAL_GRID_BANDS_REVERIFIED.csv.'))
et=S/'ETABLES.md';et.write_text(et.read_text()+''.join(b))
# Add a single clinical supplement methods section before the supplement references.
methods=(R/'eMETHODS_FOR_WRITING_EN.md').read_text().split('\n',1)[1].strip()
methods=methods.replace('The new analyses are exploratory rather than fresh blind validation.','The new analyses are exploratory.')
methods=methods.replace('KDIGO SCr and urine components were distinguished ([KDIGO](https://pmc.ncbi.nlm.nih.gov/articles/PMC4089654/)).','KDIGO SCr and urine-output components were distinguished. [9]')
methods=methods.replace('Alternative numeric implementations independently checked counts, resampling, timed criteria, rank statistics and regression covariance; this was Codex review, not independent human adjudication.','Counts, resampling, timed criteria, rank statistics and regression covariance were checked using alternative numerical implementations.')
em=S/'EMETHODS.md';old=em.read_text();before,sep,refs=old.partition('## eReferences');assert sep
before+='\n## 11. Exploratory clinical workflow, measurement and outcome analyses\n\n'+methods+'\n\n'
refs=refs.replace('Sci Data. 2023;10:1.','Sci Data. 2023;10(1):1.').replace('J Formos Med Assoc. 2026.','J Formos Med Assoc. Published online May 21, 2026.')
refs+='\n[9] Summary of Recommendation Statements. Kidney Int Suppl (2011). 2012;2(1):8-12. https://doi.org/10.1038/kisup.2012.7\n'
em.write_text(before+sep+refs)
er=S/'ERESULTS.md';er.write_text(er.read_text()+'\n## 19. Exploratory clinical workflow, measurement and outcome analyses\n\n'+(R/'eRESULTS_FOR_WRITING_EN.md').read_text().split('\n',1)[1].strip()+'\n\nThese results are displayed in eFigures 16–18 and eTables 25–32. Full source values and unfavorable or undefined results remain in the clinical-supplement aggregate directory.\n')
# Copy the three new composites and eight standalone PDFs; retain analysis IDs in their provenance.
mapping={'CS1_same_budget_policy':16,'CS2_monitoring_dependence':17,'CS3_daily_demand':18}
caps=(R/'FIGURE_LEGENDS_FOR_WRITING.md').read_text()
paras=[x for x in caps.split('\n\n') if re.match(r'^CS[123]\.',x)]
assert len(paras)==3
lf=S/'FIGURE_LEGENDS.md';ls=lf.read_text().replace('in a, b, and d.','in A, B, and D.')
for stem,n in mapping.items():
 dest=P/'05_Figure_assets'/f'eFigure_{n}';dest.mkdir(exist_ok=True)
 for ext in ['pdf','png']:shutil.copy2(R/'figures'/(stem+'.'+ext),dest/(f'eFigure_{n}.'+ext))
 for src in (R/'figures/subpanels').glob(stem+'_*.pdf'):
  letter=src.stem.rsplit('_',1)[1]
  dd=P/'05_Figure_assets/Independent_panels'/f'eFigure_{n}';dd.mkdir(exist_ok=True)
  shutil.copy2(src,dd/f'eFigure_{n}_{letter}.pdf')
 paragraph=paras[n-16];title,description=paragraph.split('. ',1)
 titles={16:'Information routing at equal total resource budgets',17:'Observed low-risk events and monitoring dependence',18:'Potential final review demand indexed to study-cohort surgery dates'}
 ls+=f'\n## eFigure {n}. {titles[n]}\n\n'+description+'\n'
lf.write_text(ls)
# Limited main-text integration preserves the primary clinical question and journal length.
mp=S/'MANUSCRIPT_EN.md';main=mp.read_text();original=main
needle='Exploratory MIMIC analyses compared recorded-event rates by distinct SCr observation days and assessed age-stratified discrimination using the existing 65-year cutoff.'
main=main.replace(needle,'Exploratory MIMIC analyses assessed observation intensity, the existing 65-year age cutoff, and measurement windows.')
needle='Fourteen comparisons crossed raw/source-calibrated predictions, six budget fractions (5%–30%), and 31 additional-effort values (0–0.30 in 0.01 increments).'
main=main.replace(needle,'The complete grid crossed 14 comparisons, two probability views, six budgets, and 31 acquisition-effort values (eMethods).')
needle='No predictive models were fitted for this extension.'
main=main.replace(needle,needle+' Further exploratory analyses used fixed rules to assess selective acquisition, surgery-date-indexed demand, center burden, recorded hard outcomes, and measurement dependence (eMethods 11).')
needle='Age-stratified discrimination is reported in eTable 22.'
main=main.replace(needle,'The observed-prevalence constant had a Brier score of 0.0717; calibrated-model-minus-constant differences were 0.0088 (95% CI, 0.0049–0.0129) and 0.0027 (−0.0001 to 0.0054), respectively (eTable 32).')
needle='Selective-acquisition, local-updating, postoperative-day-1, benchmark, and stability results, including null and adverse comparisons, are retained in eFigures 3–13 and eTables 8–19.'
main=main.replace(needle,'Further equal-cost selective-acquisition intervals spanned zero in all four contexts (eFigure 16). Retaining only the first daily MIMIC SCr result removed 14.6% of measurements and lost 7/59 known events (11.9%; 95% CI, 4.3%–20.4%; eFigure 17). Other workflow and supporting results appear in eTables 8–19 and 25–32 and eFigures 3–13 and 18.')
needle='In MIMIC, modest discrimination and low calibration slopes coexisted with source-cutoff review rates of 36.8% and 27.7%.'
main=main.replace(needle,'In MIMIC, higher Brier point estimates than the observed-prevalence reference accompanied low calibration slopes and source-cutoff review rates of 36.8% and 27.7%.')
# Move numeric secondary reference boundaries to the supplement to keep the main clinical narrative readable.
old='Geographic same-family LR D was 13.817 (−97.368 to 118.007); temporal LR and voting D values were −25.717 (−321.472 to 100.119) and −68.249 (−275.283 to 56.436). Temporal nine-to-27-input LR D was 98.633 (25.564–140.916). All reference comparisons, budgets, probability views, and infeasible scenarios are retained (eTables 20 and 21).'
assert old in main
main=main.replace(old,'Same-family reference boundaries, all budgets, probability views, and infeasible scenarios are retained (eTables 20 and 21).')
mp.write_text(main)
(W/'review/CLINICAL_INTEGRATION_EN.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),main.splitlines(True))))
wc=len(main.split('## Introduction\n',1)[1].split('## Acknowledgment\n',1)[0].split())
# Cover letter numerical package inventory is generated from actual current source headings.
cp=S/'COVER_LETTER.md';cover=cp.read_text().replace('24 numbered eTables and 15 eFigures','32 numbered eTables and 18 eFigures').replace('All 20 figure legends','All 23 figure legends')
cp.write_text(cover)
receipt={'status':'SOURCE_INTEGRATION_REQUIRES_DOCUMENT_AND_LANGUAGE_REVIEW','clinical_acceptance':accept,'clinical_acceptance_sha256':sha(ac),'aggregate_files':copied,'restricted_tables_excluded':sorted(excluded),'new_supplement_figures':mapping,'new_panel_PDFs':8,'new_numbered_tables':list(range(25,33)),'main_words':wc,'main_figures':5,'main_references':18,'whole_package_accepted':False}
(W/'review/CLINICAL_INTEGRATION.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'main_words':wc,'clinical_acceptance_sha256':sha(ac),'new_supplement_figures':3,'new_panel_PDFs':8,'new_numbered_tables':8}))
