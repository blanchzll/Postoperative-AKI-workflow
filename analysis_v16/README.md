# Current analysis code for postoperative AKI prioritization

This update contains the analysis source for the current article. The previous release at repository root provides the original model development, evaluation, and provenance material.

## Equal assessment capacity and equal AKI yield

Install the dependencies for the standalone analysis with `python3 -m pip install -r analysis_v16/requirements.txt`. The two frozen evaluation cohorts were verified with NumPy 2.5.4 and pandas 2.3.3. Use one CSV per evaluation cohort. Each eligible operation must have a unique operation identifier, a patient-group identifier, a hospital identifier, the recorded AKI endpoint (0 or 1), and three aligned frozen risk predictions. Input columns are:

```text
operation_id,patient_group,hospital,recorded_aki,original_nine,original_full27,original_endop38
```

Keep the original identifying order keys when independently verifying an authorized reconstruction. They resolve equal predicted risk. Run geographic and temporal evaluations separately.

```bash
python3 analysis_v16/capacity_grid.py --input /secure/path/paired_scores.csv --output ./capacity_results
```

The command ranks the same operations under all three models. It evaluates capacities from 5% to 30% and uses 20% to set the reference yield for the inverse comparison. The 2000 shared patient-group bootstrap samples are stratified by hospital. Intervals are pointwise and conditional on the fitted models. Outputs contain aggregate contrasts. Keep operation-level inputs under the data custodian's access terms.

## Source snapshot

- `fixed_input_scale/` contains the complete training-contained input-count sensitivity, fixed-count refitting, prediction evaluation, and aggregate summaries. The query adapter uses saved K-nearest-neighbor identities and direct Euclidean distances.
- `selection_and_refit/` records the nested training and calibration methods used by that extension.
- `paired_comparison/` records the original matched-record and same-family comparison.
- `capacity_grid/` records the original manuscript analysis with its application-specific source and integrity checks.

These snapshot scripts are retained for source inspection and source-layout reconstruction. They expect the originating project layout, approved source inputs, and fitted-model artifacts. Set `AKI_WORKSPACE` to the study workspace root for the scripts that access it. The portable capacity-grid command above runs from aligned risk predictions. The public download contains analysis code and documentation.
