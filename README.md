# Postoperative AKI workflow

Code for a multicentre retrospective study of perioperative kidney-risk assessment after gastrointestinal cancer surgery. The source cohort includes 3,710 operations at five Chinese centres and 205 recorded acute kidney injury events.

The study examines preoperative information requirements, operation-end updating, clinical review allocation, and transport to public sources. Resource comparisons measure recorded-event capture under explicit acquisition and review assumptions.

## Quick start

Python 3.10 or later is sufficient for the aggregate arithmetic check:

~~~sh
git clone https://github.com/blanchzll/Postoperative-AKI-workflow.git
cd Postoperative-AKI-workflow
python3 verify_aggregates.py
~~~

The check uses the three supplied aggregate tables, verifies selection/capture arithmetic and the original post hoc reference-margin grid, and creates no patient records. It does not refit models or repeat patient-level resampling.

The repository name was verified against GitHub repository ID 1449983323.

## Source code and environments

The code/analysis_sources/ and code/figure_code/ directories preserve 319 accepted baseline Python source files byte for byte; the accepted supplement has its own source manifest. Exact source hashes are recorded in SOURCE_RELEASE_PREFLIGHT.json. The environment_specs/ directory records the versioned tabular, foundation-model, language-model and document environments used by the study; these are separate compatible environments.

Historical sources retain their original project paths. Patient-level reproduction requires an authorized copy of the source data and the input/model layout described by the corresponding analysis protocol. These source files are not presented as a single portable executable package. REPRODUCIBILITY.md separates the check performed here from data-dependent stages.

## Data access

Clinical records, identifiers, patient-level predictions, bootstrap index streams, access credentials and restricted trained-model artifacts are excluded from this release. INSPIRE and MIMIC-IV require the providers' applicable access permissions and terms. Contact and institutional access arrangements will be supplied after author confirmation.

## Release status

The November 8 clinical supplement has passed the request-bound numerical and scientific review. Its aggregate results, protocols, code, three figures and eight standalone panel PDFs are in clinical_supplement/. Manuscript files have been assembled for author review; author declarations remain pending. This repository publishes the accepted code and aggregate-results snapshot. The manuscript is under preparation; no publication DOI is assigned.

Results concern retrospective prediction and allocation under the specified conditions. Reported null and adverse comparisons remain part of the study. Source copyright is retained; a reuse licence has not yet been selected by the authors.
