# Omics project preflight

- Project: `/home/lei/codex-workspaces/aki-prediction-manuscript/outputs/final_workbuddy_delivery_20261108_v1/figures_jno_20261108_v1`
- Text/code files scanned: 9
- Mode: read-only static scan; findings require project-aware review

## Signals

| Signal | Count | Example files |
|---|---:|---|
| absolute or working directory | 6 | `code/figure_reference.py`, `code/export_jno_panels.py`, `audit/pre_footer_visual_repair/export_jno_panels.py`, `audit/pre_eFigure2B_spacing/export_jno_panels.py`, `audit/pre_300dpi/export_jno_panels.py` |
| runtime package install | 0 | — |
| stochastic call | 0 | — |
| seed call | 0 | — |
| cell level de | 0 | — |
| sample aware de | 0 | — |
| legacy seurat slot | 0 | — |
| monocle2 | 0 | — |
| vector export | 30 | `code/export_jno_panels.py`, `code/compose_and_qualify_pages.py`, `audit/pre_footer_visual_repair/export_jno_panels.py`, `audit/pre_eFigure2B_spacing/export_jno_panels.py`, `audit/pre_300dpi/export_jno_panels.py` |
| raster export | 8 | `code/export_jno_panels.py`, `code/compose_and_qualify_pages.py`, `audit/pre_footer_visual_repair/export_jno_panels.py`, `audit/pre_eFigure2B_spacing/export_jno_panels.py`, `audit/pre_300dpi/export_jno_panels.py` |

## Interpretation gates

- Confirm biological replicate and statistical unit from metadata; static code cannot infer them reliably.
- If cell-level DE is present without a sample-aware method, review the principal inference before reuse.
- Stochastic calls are acceptable only when their seeds are fixed and recorded; the scanner reports calls and seeds separately.
- Absolute paths, runtime installs, legacy slots, and older APIs are portability risks rather than automatic errors.
- Inspect final composite figures separately; export-format strings do not prove final-size legibility.
