# Reproduction scope

| Stage | Included | Verification in this release |
|---|---|---|
| Saved aggregate arithmetic | Script and three aggregate CSV files | Executed from a fresh server directory using standard Python |
| Clinical modelling, preprocessing and calibration | Accepted versioned Python sources and environment records | Preserved from the accepted analysis; private inputs and trained states are not redistributed |
| Patient-group resampling | Source implementation | Requires authorized patient/group inputs; no new resampling in this release |
| Frozen aggregate figures | Accepted source code | Historical path bindings remain; portable rendering entry point pending |
| November 8 supplement | Accepted code, aggregate results, protocols and figures | See clinical_supplement/ACCEPTANCE.json; patient-level input requirements remain |
| Clinical deployment and patient effects | No deployment trial | Not established by this code archive |

The aggregate check detects arithmetic/transcription inconsistencies in the supplied tables. It does not validate the complete modelling process or estimate intervention effects.
