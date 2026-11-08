# Postoperative AKI workflow

Code for a multicentre retrospective study of perioperative kidney-risk assessment after gastrointestinal cancer surgery. The source cohort includes 3,710 operations at five Chinese centres and 205 recorded acute kidney injury events.

The study examines preoperative information requirements, operation-end updating, clinical review allocation, and transport to public sources. Resource comparisons measure recorded-event capture under explicit acquisition and review assumptions.

## Quick start

The root ANALYSIS_CODE.zip contains the same complete source snapshot as the expanded repository tree (excluding the archive itself).

Python 3.10 or later is sufficient for the aggregate arithmetic check:

~~~sh
git clone https://github.com/blanchzll/Postoperative-AKI-workflow.git
cd Postoperative-AKI-workflow
python3 verify_aggregates.py
~~~

The check uses the three supplied aggregate tables, verifies selection/capture arithmetic and the original post hoc reference-margin grid, and creates no patient records. It does not refit models or repeat patient-level resampling.

The repository name was verified against GitHub repository ID 1449983323.

## Source code and environments

The code/ directory mirrors every executable source file in the accepted manuscript package, including main model development/evaluation, public-source analyses, measurement sensitivity, probability-weighted resource comparisons, figure generation, and document preparation. STUDY_CODE_COVERAGE.csv binds each of the 385 package code paths to its published path and SHA256. The clinical_supplement/ directory additionally preserves the accepted November 8 handoff and its own source manifest. Dated baseline checks remain under provenance/historical/; SOURCE_RELEASE_PREFLIGHT.json and FINAL_PUBLIC_PREFLIGHT.json describe the current expanded release. The environment_specs/ directory records the versioned tabular, foundation-model, language-model and document environments used by the study; these are separate compatible environments.

Historical sources retain their original project paths. Patient-level reproduction requires an authorized copy of the source data and the input/model layout described by the corresponding analysis protocol. These source files are not presented as a single portable executable package. REPRODUCIBILITY.md separates the check performed here from data-dependent stages.

## Data access

Clinical records, identifiers, patient-level predictions, bootstrap index streams, access credentials and restricted trained-model artifacts are excluded from this release. INSPIRE and MIMIC-IV require the providers' applicable access permissions and terms. Contact and institutional access arrangements will be supplied after author confirmation.

## Release status

The November 8 clinical supplement has passed the request-bound numerical and scientific review. Its aggregate results, protocols, code, three figures and eight standalone panel PDFs are in clinical_supplement/. Manuscript files have been assembled for author review; author declarations remain pending. This repository publishes the accepted code and aggregate-results snapshot. The manuscript is under preparation; no publication DOI is assigned.

Results concern retrospective prediction and allocation under the specified conditions. Reported null and adverse comparisons remain part of the study. Source copyright is retained; a reuse licence has not yet been selected by the authors.

## Code coverage and entry points

STUDY_CODE_COVERAGE.csv covers all executable sources in the accepted manuscript code package. Historical source versions remain identified by their dated paths. Use each analysis protocol's current entry point; running every retained Python file would repeat superseded stages.

| Component | Source directory |
|---|---|
| Main clinical models, preprocessing, calibration and validation | code/analysis_sources/ |
| Joint acquisition-and-review budget and MIMIC calendar evaluation | code/analysis_sources/novelty_extension/ |
| Measurement-condition sensitivity | code/measurement_consistency_20261104_v1/code/ |
| Probability-weighted acquisition costs | code/probabilistic_resource_analysis_20261105_v1/code/ |
| Accepted November 8 clinical workflow, monitoring and hard-outcome supplement | clinical_supplement/code/ |
| Manuscript figures | code/figure_code/ |
| Earlier accepted document and standalone-panel exports | code/document_code/current_revision_20261108/ |
| Final table mapping, audit and typography corrections | final_source_audit/code/ |

The repository includes a small aggregate quick-start example and the accepted clinical-supplement aggregates. The full submission archive contains additional aggregate tables. Patient-level modelling and resampling require authorized source inputs; data-acquisition permissions and trained-state access remain subject to the data holders' terms.

## Final source and reporting update (November 8, 2026)

The final_source_audit/ directory adds the source-mapped eTable 4 reporting repair, the source/scientific audit and final-size figure export/verification code. The figures reuse accepted aggregate values; no models, bootstrap samples or workflow scenarios were added. The table now identifies AUROC, Brier score and event capture at 10% and 20% review capacity separately.

To render the three typography-corrected figures with the recorded plotting environment:

~~~sh
python3 final_source_audit/code/repair_final_size_figures.py --source-script clinical_supplement/code/plot_supplement.py --results clinical_supplement/results --output figures_final_size
~~~

This portable command reads aggregate CSV files. The source/scientific audit and document checks retain their server-workspace bindings and need the corresponding authorized sources. Previously dated review receipts remain historical; RELEASE_STATUS.json identifies this source snapshot.

## Journal-format source update

The current supplementary-figure export is in jno_supplement_figures/code/. It reconstructs the 18 supplemental figures as 53 native panels and 31 vector pages at 170 mm, with actual Arial 10-point text. The code reuses frozen plotting functions and aggregate results; no estimates or resampling were changed. The final_source_audit/code/ directory includes current Word formatting, native SVG insertion and final source/format verification. Earlier 160-mm exports remain dated intermediate sources.

The full accepted figure assets are supplied in the submission folder; the manifests here record their hashes. Running the current export requires the source/aggregate layout declared in FIGURE_CONTRACT.json and private installed Arial font files. Export scripts accept the documented project/output/font paths and stop if actual Arial is absent. Fonts are not redistributed. The standalone last-three-figure command above remains the earlier intermediate export, while the JNO source route is the current journal-format workflow.

The source and document checks are computational/editorial checks, not a human expert panel or a clinical implementation evaluation.

The final shared-legend source correction places the reference-grid key only on its applicable eFigure2B panel. All four global shared-legend calls were source-mapped once; numeric artists, intervals and annotations are unchanged. Earlier accepted exports remain in Git history.
