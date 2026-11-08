from pathlib import Path
import json,re,csv
W=Path(__file__).resolve().parents[1];B=W.parent;T=W/'work/text'
figmap={int(k):v for k,v in json.loads((W/'review/NUMBERING_CROSSWALK.json').read_text())['figures_old_to_journal'].items()}
tabmap={int(k):v for k,v in json.loads((W/'review/NUMBERING_CROSSWALK.json').read_text())['tables_old_to_journal'].items()}
def conv(s):
 s=re.sub(r'Supplementary (?:Figure|Fig\.) (\d+)',lambda m:f'eFigure {figmap[int(m[1])]}',s)
 s=re.sub(r'\bFig\. (\d+)',r'Figure \1',s)
 s=s.replace('Tables S11–S12','eTables 8 and 9').replace('Supplementary Methods','eMethods')
 s=re.sub(r'(?:Supplementary )?Table S(\d+)',lambda m:f'eTable {tabmap.get(int(m[1]),int(m[1]))}',s)
 return s
# Clean figure titles and cross-references without altering scientific captions or artwork.
entries=json.loads((T/'FIGURE_ENTRIES.json').read_text())
for e in entries:
 e['title']=re.sub(r'^(?:Supplementary )?Fig\. \d+\s*\|\s*','',e['title'])
 e['caption']=conv(e['caption'])
 for suffix in [' AUROC indicates area under the receiver operating characteristic curve; LR, logistic regression.',' AUROC indicates area under the receiver operating characteristic curve; RF, random forest; LR, logistic regression; SCr, serum creatinine.',' ALT indicates alanine aminotransferase. Evaluation counts are in eTable 2; the resampling procedure is specified in eMethods.']:
  e['caption']=e['caption'].replace(suffix,'')

 if e['label']=='Figure 1':e['caption']+=' AUROC indicates area under the receiver operating characteristic curve; LR, logistic regression.'
 if e['label']=='Figure 2':e['caption']=e['caption'].replace('2,000 patient-group bootstrap','2,000 hospital-stratified patient-group bootstrap');e['caption']+=' AUROC indicates area under the receiver operating characteristic curve; LR, logistic regression.'
 if e['label']=='Figure 3':e['caption']+=' AUROC indicates area under the receiver operating characteristic curve; RF, random forest; LR, logistic regression; SCr, serum creatinine.'
 if e['label']=='eFigure 3':e['caption']+=' ALT indicates alanine aminotransferase. Evaluation counts are in eTable 2; the resampling procedure is specified in eMethods.'
(T/'FIGURE_ENTRIES.json').write_text(json.dumps(entries,indent=2))
for name,es in [('FIGURE_LEGENDS.md',entries),('MAIN_FIGURE_LEGENDS.md',entries[:3])]:
 (T/name).write_text('# Figure legends\n\n'+'\n\n'.join(f"## {e['label']}. {e['title']}\n\n{e['caption']}" for e in es)+'\n')
# Consolidate supplementary methods: retain the complete technical S1-S7 specification and unique detailed-protocol paragraphs.
s=(B/'manuscript_revision/SUPPLEMENTARY_METHODS_EN.md').read_text();front=s.split('## Extended study protocol')[0];tail=s.split('## Extended study protocol')[1].split('## References')[0]
front=front.replace('# Supplementary Methods','# eMethods')
front=re.sub(r'## S(\d+)\. ',lambda m:'## '+m[1]+'. ',front)
# Remove duplicate extended headings/paragraphs whose information is fully specified above; retain the remainder verbatim.
keep=[];seen=[]
for p in tail.split('\n\n'):
 p=p.strip()
 if not p:continue
 if p.startswith('## '):keep.append(p);continue
 if p.startswith('The full definitions, original paired intervals'):continue
 if p.startswith('The source calibrator was'):continue
 keep.append(p)
methods=conv(front)+'\n## 8. Detailed cohort and development protocol\n\n'+'\n\n'.join(keep)+'\n'
# References are numbered in order of first supplementary citation, independently from the main article.
refs=(B/'manuscript_revision/MANUSCRIPT_EN.md').read_text().split('## References')[1];refdict={int(n):text for n,text in re.findall(r'^\[(\d+)\]\s+(.*)$',refs,re.M)}
assert len(refdict)==14
# Full manuscript references may use an opening heading only; source all exact records.
patterns=re.compile(r'\[(\d+(?:[,–-]\d+)*)\]')
def ids(token):
 out=[]
 for v in token.split(','):
  if re.search('[–-]',v):a,b=map(int,re.split('[–-]',v));out.extend(range(a,b+1))
  else:out.append(int(v))
 return out
order=[]
for m in patterns.finditer(methods):
 for n in ids(m[1]):
  if n not in order:order.append(n)
ren={old:i+1 for i,old in enumerate(order)}
methods=patterns.sub(lambda m:'['+','.join(str(ren[n]) for n in ids(m[1]))+']',methods)
methods+='\n## eReferences\n\n'+'\n\n'.join(f'[{ren[n]}] {refdict[n]}' for n in order)+'\n'
methods=conv(methods)
(T/'EMETHODS.md').write_text(methods)
(W/'review/SUPPLEMENT_REFERENCE_CROSSWALK.json').write_text(json.dumps(ren,indent=2))
res=conv((B/'supplement/SUPPLEMENTARY_RESULTS_EN.md').read_text().split('## Attachment structure')[0])
res=res.replace('# Supplementary results and attachment guide','# eResults')
res=re.sub(r'## S(\d+)\. ',lambda m:'## '+m[1]+'. ',res)
res=res.replace('S1–S16','Sections 1–16')
(T/'ERESULTS.md').write_text(res)
def read(rel):
 with (B/rel).open() as f:return list(csv.DictReader(f))
def table(headers,rows):
 def clean(v):
  s=str(v or '');return s.replace('|','/').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])
def fmt(x,n=3):
 try:
  v=float(x)
  if v!=v:return 'Not estimable'
  return f'{v:.{n}f}'
 except (ValueError,TypeError):return 'Not estimable' if not x else x
tables={};sources={}
def add(n,title,body,source):tables[n]=(title,body);sources[str(n)]=source
# Dictionary: one row for each original source variable, repeated route definitions remain in the CSV archive.
r=read('source_data/Table_S1_input_dictionary.csv');uniq={}
for x in r:uniq.setdefault(x['name'],x)
add(1,'Input dictionary',table(['Source field','Unit','Type'],[[k,v['unit'],v['type']] for k,v in uniq.items()])+'\n\nThe nine-input strategy uses age, sex, serum creatinine, blood urea nitrogen, serum albumin, hemoglobin, diabetes, hypertension, and other heart disease. Original route-specific codes and allowed values are supplied in the aggregate source tables.', ['source_data/Table_S1_input_dictionary.csv'])
# Reuse accepted complete displayed tables, preserving every source estimate and interval.
ext=(B/'supplement/SUPPLEMENTARY_TABLES_EN.md').read_text()
chunks=re.split(r'\n## ',ext)[1:]
by={}
for chunk in chunks:
 head,body=chunk.split('\n',1);m=re.search(r'S(\d+)([ABC]?)',head)
 if m:by.setdefault(int(m[1]),[]).append((head,body.strip()))
for old,new,title in [(17,2,'Cohort characteristics, missingness, and evaluation partitions'),(18,3,'Nine-input performance and the 27-input comparison'),(19,4,'Stage-system and same-family capture at 20% capacity'),(20,5,'Review capacity and incremental capture'),(21,7,'Public workflow performance and source-threshold coverage'),(13,12,'Exploratory capture-loss reference bounds')]:
 body='\n\n'.join(('### '+re.sub(r'^.*?\|\s*','',h)+'\n\n' if len(by[old])>1 else '')+conv(b) for h,b in by[old]);add(new,title,body,[f'supplement/SUPPLEMENTARY_TABLES_EN.md#S{old}'])
cols=['model','n','events','AUROC','AP','Brier','observed_expected','calibration_slope']
r=read('tables/Table_S16_POINT_ESTIMATES.csv')
add(6,'All eight public raw and source-calibrated views',table(['Workflow','Records','Events','AUROC','AP','Brier','O:E','Slope'],[[x[c] if c in ['model','n','events'] else fmt(x[c],6 if c=='Brier' else 3) for c in cols] for x in r])+'\n\nPublic workflows use the same 6,043 records and 253 serum-creatinine events. Original paired intervals, coded-history counts and the 54-category codebook are in the associated aggregate CSV files.', ['tables/Table_S16_POINT_ESTIMATES.csv'])
# Published-model and native TabPFN results, without selective omission of poorer conditions.
p='supplement/aggregate_tables/supplement_tables/S12_published/published_benchmark_comparison_20261024_v1/POINT_ESTIMATES.csv';r=read(p)
body=table(['Context','Workflow','Records','Events','AUROC','AP','Brier'],[[x['context'],x['model'],x['n'],x['events'],fmt(x['AUROC']),fmt(x['AP']),fmt(x['Brier'],6)] for x in r])
ss=[p]
for pp in sorted((B/'supplement/aggregate_tables/supplement_tables/S13_TabPFN').rglob('POINT_ESTIMATES.csv')):
 rr=read(str(pp.relative_to(B)));ss.append(str(pp.relative_to(B)))
 body+='\n\n### TabPFN: '+pp.parent.name+'\n\n'+table(['Workflow','Records','Events','AUROC','AP','Brier'],[[x['model'],x['n'],x['events'],fmt(x['AUROC']),fmt(x['AP']),fmt(x['Brier'],6)] for x in rr])
body+='\n\nPublished benchmarks use the eligibility rules and mappings in eMethods. The Chen comparison provides ranking because its original intercept was unavailable. Native TabPFN conditions are unrecalibrated. Complete intervals and paired contrasts are retained in the aggregate tables.'
add(8,'Published-model and TabPFN benchmarks',body,ss)
p='source_data/S4_LLM_all24_calibrated_points.csv';r=read(p)
add(9,'All 24 calibrated medical-language-model conditions',table(['Workflow','Records','Events','AUROC','AP','Brier'],[[x['model'],x['n'],x['events'],fmt(x['AUROC']),fmt(x['AP']),fmt(x['Brier'],6)] for x in r])+'\n\nBoth before- and after-adaptation conditions use development-label calibration. All 54 paired contrasts, raw scores and intervals accompany these point estimates in the aggregate tables.',[p])
p='source_data/Baseline_missingness_all27.csv';r=read(p)
add(10,'Observed missingness before imputation',table(['Cohort','Field','Records','Observed','Missing','Missing, %'],[[x[c] for c in ['cohort','variable','n','observed','missing','missing_percent']] for x in r]),[p])
p='tables/Table_S15_BRIDGE_POINT_ESTIMATES.csv';r=read(p)
cols=[c for c in ['context','line','model','n','events','AUROC','AP','Brier'] if c in r[0]]
add(11,'Six-, nine-, and 27-input source-cohort LR workflows',table(cols,[[fmt(x[c],6 if c=='Brier' else 3) if c in ['AUROC','AP','Brier'] else x[c] for c in cols] for x in r])+'\n\nAll raw and source-calibrated workflows and paired intervals are retained. Input-specific source-selected preprocessing and hyperparameters define the workflow comparison.',[p])
p='source_data/F4_allocation_16_scenarios.csv';r=read(p);cs=[c for c in r[0] if any(k in c.lower() for k in ['context','budget','capacity','selective','random','difference','lower','upper'])]
cs=['context','upgrade_fraction','review_fraction','selective_captured_n','random_mean_captured_n','selective_minus_random_mean_per1000','conditional_difference_low_per1000','conditional_difference_high_per1000']
add(13,'All 16 selective-versus-random acquisition comparisons',table(cs,[[fmt(x[c],4) if re.match(r'^-?\d*\.\d+',x[c] or '') else x[c] for c in cs] for x in r])+'\n\nRanges describe allocation variability from the fixed-size random comparisons, conditional on fitted systems and records. All original fields are included in the source CSV.',[p])
p='source_data/S2_calibration_update_points.csv';r=read(p)
add(14,'Target-local probability updates',table(['Context','Route','Update','Records','Events','AUROC','Brier'],[[x['context'],x['route'],x['method'],x['n'],x['events'],fmt(x['AUROC']),fmt(x['Brier'],6)] for x in r])+'\n\nOther target folds supply update outcomes. Paired bootstrap intervals refit update maps and adaptation thresholds. Complete intervals are in S2_calibration_update_support.csv.',[p])
p='source_data/S3_POD1_LR_points.csv';r=read(p)
add(15,'Conditional postoperative-day-1 LR comparisons',table(['Context','Workflow','Records','Events','AUROC','AP','Brier'],[[x['context'],x['model'],x['n'],x['events'],fmt(x['AUROC']),fmt(x['AP']),fmt(x['Brier'],8)] for x in r])+'\n\nThe principal Brier contrast compares all-POD1 with all-preoperative LR; all paired intervals are retained in S3_POD1_LR_differences.csv.',[p])
# A compact explicit index points to every retained subgroup/endpoint task without dumping hundreds of metric rows into Word.
rr=[]
for folder in ['S15_strata','S6_outcome','S6_fixedflow']:
 root=B/'supplement/aggregate_tables/supplement_tables'/folder
 groups={}
 for pp in root.rglob('*.csv'):
  groups.setdefault(str(pp.parent.relative_to(root)),[]).append(pp.name)
 for g,fs in sorted(groups.items()):rr.append([folder,g,', '.join(sorted(fs))])
add(16,'Subgroup and fixed-prediction sensitivity table index',table(['Analysis family','Task','Aggregate result files'],rr)+'\n\nThe archive contains all 44 stratified tasks, including sparse or zero-event strata, and the current corrected fixed-flow comparisons. Lists are constructed in the whole evaluation population before subgroup summaries.', ['supplement/aggregate_tables/supplement_tables/S15_strata','supplement/aggregate_tables/supplement_tables/S6_outcome','supplement/aggregate_tables/supplement_tables/S6_fixedflow'])
p='source_data/S5_conditional_stability.csv';r=read(p);cols=[c for c in r[0] if c in ['route','planned','valid','invalid'] or ('top20_overlap' in c and 'mean' in c)]
add(17,'Conditional model and review-list stability',table(cols,[[fmt(x[c],3) if 'mean' in c else x[c] for c in cols] for x in r])+'\n\nTwo hundred patient-group refits were performed per route with choices fixed, without hospital stratification. Complete risk-change, list-overlap and source-cut-flip summaries remain in the source CSV.',[p])
p='tables/Table_S4_Representative_workload_assumptions.csv';r=read(p)
body=table(['Context','Capacity rule','Minutes/review','Testing fraction','Hours/1000','Tests/1000'],[[x['context'],x['rule'].replace('assessment_capacity','')+'%',x['minutes_per_review'],x['additional_fraction'],fmt(x['review_hours_per1000'],2),fmt(x['additional_tests_per1000'],2)] for x in r])
p2='tables/Table_S4_Representative_effect_assumptions.csv';r2=read(p2)
body+='\n\n'+table(['Context','Capacity rule','Assumed RRR','Uptake','Hypothetical events averted/1000 (conditional 95% CI)'],[[x['context'],x['rule'].replace('assessment_capacity','')+'%',x['assumed_RRR'],x['assumed_uptake'],fmt(x['hypothetical_events_averted_per1000'],3)+' ('+fmt(x['conditional_sampling_lower95'],3)+', '+fmt(x['conditional_sampling_upper95'],3)+')'] for x in r2])
body+='\n\nThese are unmeasured workload and effect assumptions. Complete grids retain zero and harmful effect scenarios; sampling intervals condition on the assumptions. The study did not measure review completion or intervention effects.'
add(18,'Workload and intervention scenario assumptions',body,[p,p2])
p='tables/Table_S2_All78_paired_performance_metrics.csv';r=read(p)
rows=[[x['context'],x['model'],x['metric'],fmt(x['estimate'],6)+' ('+fmt(x['lower95'],6)+', '+fmt(x['upper95'],6)+')',x['valid_replicates']] for x in r]
add(19,'Complete 78-metric nine-versus-27-input paired results',table(['Context','Comparison','Metric','Difference (95% CI)','Valid replicates'],rows)+'\n\nThe original full-minus-nine contrast direction and metric-specific units are retained; capture metrics include counts, proportions and events per 1,000 records. Valid/requested resample counts and full precision are in the source CSV.',[p])
assert set(tables)==set(range(1,20)),sorted(tables)
(T/'ETABLES.md').write_text('# eTables\n\n'+'\n\n'.join(f'## eTable {n}. {title}\n\n{body}' for n,(title,body) in sorted(tables.items()))+'\n')
(W/'review/ETABLE_SOURCE_MAP.json').write_text(json.dumps(sources,indent=2))
print(json.dumps({'eTables':len(tables),'eMethods_words':len(methods.split()),'eResults_words':len(res.split()),'supplementary_references':len(ren),'eFigures':14}))
