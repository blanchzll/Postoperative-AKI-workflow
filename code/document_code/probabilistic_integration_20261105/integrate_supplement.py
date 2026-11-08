from pathlib import Path
import json,csv,shutil,hashlib
W=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/writing_runs/probabilistic_integration_20261105_v1')
P=W/'package';S=P/'04_Editable_sources';A=W.parents[1]/'correction_runs/probabilistic_resource_analysis_20261105_v1'
texts=json.loads((W/'work/SUPPLEMENT_EDITS.json').read_text())
methods=texts['methods'];results=texts['results']
p=S/'EMETHODS.md';t=p.read_text();assert methods not in t;t=t.replace('## 10. MIMIC-IV',methods+'\n## 10. MIMIC-IV',1);p.write_text(t)
p=S/'ERESULTS.md';t=p.read_text();marker='## Public-source transport and threshold burden';assert marker in t;t=t.replace(marker,results+'\n'+marker,1);p.write_text(t)
rows=list(csv.DictReader((A/'outputs/PRIMARY_20_PERCENT_SUMMARY.csv').open()))
selected=[r for r in rows if r['distribution']=='ratio_uniform_030']
selected += [r for r in rows if r['comparison']=='geography_27v38_champ' and r['distribution'] in ['ratio_low_030','ratio_high_030']]
names={'geography_9v27_champ':'Geographic 9→27','temporal_9v27_champ':'Temporal 9→27','geography_27v38_champ':'Geographic 27→38','temporal_27v38_champ':'Temporal 27→38'}
laws={'ratio_uniform_030':'Uniform[0,0.30]','ratio_low_030':'0.30×Beta(2,5)','ratio_high_030':'0.30×Beta(5,2)'}
def f(x):return f'{float(x):.2f}'.replace('-','−')
panel='### Conditional cost-weighted summaries at a 20% review-equivalent budget\n\n'
panel+='| Evaluation and cost law | More / equal / fewer / infeasible, % | Mean capture difference/1000 (95% CI) | More-capture share, 95% CI, % | Loss, base / additional |\n| --- | --- | --- | --- | --- |\n'
for r in selected:
 masses=' / '.join(f(100*float(r[k])) for k in ['p_win','p_tie','p_loss','p_infeasible'])
 mean=f(r['mean_delta_per1000_fallback'])+' ('+f(r['mean_delta_per1000_fallback_patient_lower95'])+' to '+f(r['mean_delta_per1000_fallback_patient_upper95'])+')'
 interval=f(100*float(r['p_win_patient_lower95']))+' to '+f(100*float(r['p_win_patient_upper95']))
 loss=f(r['expected_regret_base_per1000'])+' / '+f(r['expected_regret_extra_fallback_per1000'])
 panel+='| '+' | '.join([names[r['comparison']]+' '+laws[r['distribution']],masses,mean,interval,loss])+' |\n'
note='All cost laws were analyst-specified before their probability-weighted outcomes were calculated. Counts and model families match the primary comparisons above. More/equal/fewer compare recorded-event counts while acquisition is feasible. Infeasible settings form a separate fourth category, and all four percentages use the complete cost distribution. Mean differences and losses use an additional-information planning rule that reverts to baseline review before spending when acquisition is unaffordable. Loss is the expected captured-event shortfall per 1,000 eligible records from the better feasible path at each cost setting. Patient intervals are pointwise percentiles from the original 2,000 paired hospital-stratified patient-group bootstrap samples, conditional on each cost law and frozen systems. Costs are setting-level means in one interchangeable resource pool. Shares describe these assigned scenarios. Full feasible-conditional estimates, all five laws, raw-score views, other budgets, and fixed-minute sensitivities are retained in the aggregate tables. The original zero-acquisition comparisons remain above.'
panel+='\n'+note+'\n\n'
(W/'work/COST_WEIGHTED_PANEL.md').write_text(panel)
p=S/'ETABLES.md';t=p.read_text();t=t.replace('## eTable 21.',panel+'## eTable 21.',1);p.write_text(t)
legend='Cost-weighted summaries derived from the complete integer capture curves are reported in eTable 20. Those summaries use analyst-specified distributions and a pre-spending affordability rule detailed in eMethods.'
p=S/'FIGURE_LEGENDS.md';t=p.read_text();pre,rest=t.split('## Figure 2.',1);section,post=rest.split('## Figure 3.',1);section=section.rstrip()+' '+legend+'\n\n';p.write_text(pre+'## Figure 2.'+section+'## Figure 3.'+post)
(S/'Chapters/FIGURE_LEGENDS.md').write_text(p.read_text())
pointer='Conditional cost-weighted capture and loss summaries for these four comparisons are reported within eTable 20 in Supplement 1.'
p=S/'MAIN_TABLES.md';p.write_text(p.read_text().rstrip()+'\n\n'+pointer+'\n')
(S/'Chapters/MAIN_TABLES.md').write_text(p.read_text())
p=S/'MANUSCRIPT_ZH.md';t=p.read_text().replace('收益和损失各有多大；二者','收益和损失各有多大。二者').replace('正的D表示达到基础捕获数后仍有采集额度；负的D','正的D表示达到基础捕获数后仍有采集额度。负的D');p.write_text(t)
t=t.split('## 讨论\n',1)[1].split('\n## 结论',1)[0];(S/'Chapters/DISCUSSION_ZH.md').write_text('## 讨论\n'+t.rstrip()+'\n')
cover=(S/'COVER_LETTER.md').read_text().replace('November 1, 2026','November 5, 2026').replace('Dear Editors:','Dear Editors,')
start='The exploratory extension compared frozen workflows'
old=next(t for t in cover.split('\n\n') if t.startswith(start))
new='The exploratory extensions compared frozen workflows under a common acquisition-and-review budget. The original resource grid contained 5,208 scenarios and 168 equal-capture boundaries. Further probability weighting showed higher mean capture with expanded nine-input review than with acquisition of 27 inputs under five analyst-specified cost distributions. Geographic operation-end updating had a positive equal-capture boundary, while its weighted preference depended on acquisition costs. Temporal and same-family stage results were less consistent. In 759 MIMIC patients with 59 recorded creatinine events, frozen six-input RF discrimination and calibration were reduced, and transferred source-20% thresholds selected 36.8% and 27.7% of patients.'
cover=cover.replace(old,new)
(S/'COVER_LETTER.md').write_text(cover)
dest=P/'02_Code_and_aggregate_data/probabilistic_resource_analysis_20261105_v1';dest.mkdir()
for sub in ['code','aggregate_data','provenance']: (dest/sub).mkdir()
public_csv=['PARAMETER_SOURCE_QUALIFICATION.csv','PRIMARY_20_PERCENT_SUMMARY.csv','DISTRIBUTION_SENSITIVITY_SUMMARY.csv','NORMALIZED_ESTIMATES.csv','ABSOLUTE_TIME_ESTIMATES.csv','NORMALIZED_MC_VALIDATION.csv','INCLUSION_DECISIONS.csv','CLAIM_TABLE_VERSION_MAP.csv']
for n in public_csv:shutil.copy2(A/'outputs'/n,dest/'aggregate_data'/n)
for n in ['analyze_probability.py','verify_probability.py']:shutil.copy2(A/'work'/n,dest/'code'/n)
for n in ['PROTOCOL_LOCK.json','REQUEST.json','NUMERICAL_CODE_LOCK.json','INDEPENDENT_CODE_LOCK.json']:shutil.copy2(A/n,dest/'provenance'/n)
for n in ['PRECISION_SUMMARY.json','PARAMETER_QUALIFICATION_ACCEPTANCE.json','NUMERICAL_COMPUTATION_ACCEPTANCE.json','INDEPENDENT_NUMERICAL_VERIFICATION.json','INDEPENDENT_IO_REPAIR.json','DECISION_IDENTITIES_QA.json']:
 shutil.copy2(A/'outputs'/n,dest/'provenance'/n)
readme='# Conditional resource probability analysis\n\nThis directory adds aggregate results and frozen source code for AKI_PROBABILISTIC_RESOURCE_ANALYSIS_20261105_V1. Cost laws are analyst-specified conditional scenarios. Patient-level inputs, patient bootstrap capture curves, parameter-draw archives, and trained model materials remain in restricted server storage.\n\nThe entry points code/analyze_probability.py and code/verify_probability.py retain their accepted execution paths and source bindings. Full recomputation requires authorized access to the frozen clinical inputs, prediction banks and original paired bootstrap streams. Copying this directory alone supplies aggregate review and code inspection. Re-running predictive training is unnecessary for this extension.\n\nThe primary article summarizes 20% source-calibrated comparisons in eTable 20. All five normalized laws, six budgets, two score views and fixed-minute dependence sensitivities are preserved here. Mean differences ending in fallback assume affordability is judged before spending and baseline review is used when acquisition is infeasible. See the protocol for feasible-conditional alternatives and denominator definitions.\n\nThe existing resource grid and boundaries remain in aggregate_data elsewhere in the package. Historical acceptance files retain their original task scope. The writing integration acceptance at the package root identifies the current manuscript version.\n'
(dest/'README.md').write_text(readme)
rootread=P/'02_Code_and_aggregate_data/README.md';rootread.write_text(rootread.read_text().rstrip()+'\n\n## Probability-weighted resource extension\n\nSee probabilistic_resource_analysis_20261105_v1/README.md for the new aggregate results, code entry points, assumptions and restricted-input requirements.\n')
fig=P/'05_Figure_assets/Figure_2/PROBABILISTIC_SOURCE_MAP.json'
fig.write_text(json.dumps({'existing_figure_unchanged':True,'original_curves':'original N1_equal_budget_capture source data retained','new_aggregate_source':'02_Code_and_aggregate_data/probabilistic_resource_analysis_20261105_v1/aggregate_data/PRIMARY_20_PERCENT_SUMMARY.csv','supplement':'eTable 20, conditional cost-weighted panel','integration':'Figure legend pointer only; no new lines or cost laws added to the figure'},indent=2))
(W/'review/SUPPLEMENT_EDIT_LOG.json').write_text(json.dumps({'methods':methods,'results':results,'panel':panel,'figure_legend_addition':legend,'standalone_table_pointer':pointer,'public_csv':public_csv},ensure_ascii=False,indent=2))
print(json.dumps({'aggregate_files':len(public_csv),'new_panel_rows':len(selected),'new_numbered_tables':0,'private_data_copied':False}))
