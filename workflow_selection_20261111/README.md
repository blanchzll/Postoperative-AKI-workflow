# Resource-based selection of AKI review workflows

This extension compares frozen nine-input and 27-input preoperative workflows with a 38-input operation-end workflow after gastrointestinal cancer surgery. All three paths are evaluated on the same paired-eligible records. Geographic evaluation contains 1158 operations/72 events and temporal evaluation 1011/36.

## Selection measure

At budget K=ceil(qN), acquisition path k reviews floor(K-Nak) operations. The additional acquisition effort ak is expressed per eligible operation relative to available nine-input information, divided by the effort of one baseline review. Gk is 1000 times the difference in captured recorded events from nine-input review at K. The nested cost region is 0<=a27<=a38<=.30 in .01 steps. The complete grid has 3968 rows across two source lines, two budgets and raw/calibrated predictions.

The query returns the highest observed coverage, paired difference intervals and interval-supported choices. Cells where only nine-input review is affordable have their own label. Cost coordinates are service-planning scenarios. Information availability depends on assessment time.

## Query an evaluated scenario

Python 3.10 or later and its standard library can query the bundled ZIP directly.

~~~sh
python workflow_selection_20261111/select_workflow.py --evaluation geography --budget 20 --cost27 .15 --cost38 .15
~~~

In this scenario, nine/27/38 inputs capture 33/13/15 geographic events at the common 20% budget. The paired intervals favor nine-input review. The corresponding temporal counts are 20/9/9.

## Files

- WORKFLOW_CHOICE_AND_SHAP_CODE.zip contains analysis, plot and document code, complete grids, protocols and numerical checks.
- PRESENTATION_CODE_20261108_10.zip supplies the preceding reference, composite-figure and numeric-alignment source updates.
- FOUR_WORKFLOW_SHAP.csv supplies full raw/calibrated importance summaries for the four displayed models.
- STRATEGY_SELECTION_SUMMARY.json gives complete observed and interval-supported region summaries.

SHAP displays reuse accepted whole-pipeline explanations. Resource functions reuse frozen predictions and the original 2000 hospital-stratified patient-group weight samples. Source-dependent analyses require authorized inputs at their documented server paths. Aggregate querying works from this public folder.

The original study code and baseline release remain at the repository root. This release follows the repository's existing copyright and access arrangements.
