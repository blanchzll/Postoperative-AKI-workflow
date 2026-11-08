# Reproduction scope

| Stage | Included | Verification in this release |
|---|---|---|
| Saved aggregate arithmetic | Script and three aggregate CSV files | Executed from a fresh server directory using standard Python |
| Clinical modelling, preprocessing and calibration | Accepted versioned Python sources and environment records | Preserved from the accepted analysis; private inputs and trained states are not redistributed |
| Patient-group resampling | Source implementation | Requires authorized patient/group inputs; no new resampling in this release |
| Frozen aggregate figures | Accepted source code | Historical path bindings remain; the final three supplemental figures have the aggregate-only command in README |
| November 8 supplement | Accepted code, aggregate results, protocols and figures | See clinical_supplement/ACCEPTANCE.json; patient-level input requirements remain |
| Clinical deployment and patient effects | No deployment trial | Not established by this code archive |

The aggregate check detects arithmetic/transcription inconsistencies in the supplied tables. It does not validate the complete modelling process or estimate intervention effects.

The final reporting update preserves all source estimates. It adds explicit table metrics and a 160-mm export for the last three supplementary figures, plus source and final-size checks. Raw review-conversation payloads and patient-level clinical records are excluded.

The current JNO export preserves all numeric artists and source annotations in 53 supplemental panels. The Word supplement embeds 31 native editable SVG pages; actual rendered PDF text was verified as Arial 10 point. Source-dependent workflows retain their declared server/project bindings. Font files and original patient-level sources are separate authorized runtime dependencies, not public code assets.
