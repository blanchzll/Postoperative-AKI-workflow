# Final WorkBuddy source audit and analysis handoff

This version verifies the latest public WorkBuddy review against the accepted corrected-data clinical analyses. It adds source review, deterministic arithmetic checks and the unique 28-row eTable4 metric/capacity/unit mapping. It does not fit prediction models, generate new bootstrap samples or add workflow simulations.

The canonical server root is `/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/final_workbuddy_analysis_20261108_v1`.

Run `python3 code/final_workbuddy_audit.py` on t630 where the original private source and frozen writing/analysis releases remain available. The script reads those sources and writes only this audit root. `--project-root` and `--audit-root` can select relocated equivalent server copies. Source hashes are checked before derived summaries. Patient data, sampling arrays, exact operation dates, shared raw tool outputs and credentials are excluded from this handoff. The accepted original research-code release remains a separate dependency.

Read `docs/FINAL_SCIENTIFIC_HANDOFF_ZH.md` for the clinical result interpretation and targeted repairs. `results/WORKBUDDY_DELTA_CLOSURE.csv` separates accepted evidence, corrections, unidentifiable estimands and the writing owner's remaining work. `docs/ETABLE4_REPLACEMENT.md` and its source-mapped CSV supply the repaired table. Metadata timestamps vary upon a rerun; scientific source checks and the source-mapped table are deterministic.

The new scientific acceptance covers this analysis audit. The overall user request additionally requires the writing owner to complete the updated manuscript, final-size figure repairs, individual panel PDFs, editorial review, submission package and verified code publication. No actual journal submission or human expert-panel approval is represented here.
