# Artwork manifest

Use this manifest when attaching artwork in Editorial Manager. The manuscript
uses related multi-panel images where the panels must be read together. Upload
the listed raster source files alongside the LaTeX package; TikZ figures remain
editable in the manuscript source.

## Main manuscript

| Figure | Source artwork | Pixels | Notes |
|---|---|---:|---|
| 1 | `figures/overview_architect.png` | 2978 × 1408 | Pipeline schematic; use the vector original instead if available. |
| 2a | `figures/ocr_page12_original.png` | 1588 × 2246 | Related panel. |
| 2b | `figures/ocr_page12_detection.png` | 1588 × 2246 | Related panel. |
| 2c | `figures/ocr_page12_recognition.png` | 1588 × 2246 | Related panel. |
| 2d | `figures/layout_page12.png` | 834 × 1179 | Related panel; printed at quarter-page width. |
| 3 | `figures/table_cells_page12.png` | 1280 × 662 | Table-cell output. |
| 4a | `figures/layout_calculus_p2_before.png` | 1037 × 1214 | Related before/after panel. |
| 4b | `figures/layout_calculus_p2_after.png` | 1037 × 1214 | Related before/after panel. |
| 5 | Embedded TikZ in `sections/03_methodology.tex` | Vector | Reading-order boundary expansion. |
| 6 | Embedded TikZ in `sections/03_methodology.tex` | Vector | Containment pruning. |
| 7 | Embedded TikZ in `sections/03_methodology.tex` | Vector | Orphan-line insertion. |
| 8 | Embedded TikZ in `sections/03_methodology.tex` | Vector | Overflow collision. |
| 9 | Embedded TikZ in `sections/03_methodology.tex` | Vector | Colour sampling. |
| 10a | `figures/source_1.png` | 2118 × 1072 | Related source/output panel. |
| 10b | `figures/source_3.png` | 920 × 1210 | Related source/output panel. |
| 10c | `figures/source_math.png` | 834 × 980 | Related source/output panel. |
| 10d | `figures/des_1.png` | 2108 × 1098 | Related source/output panel. |
| 10e | `figures/des_3.png` | 936 × 1204 | Related source/output panel. |
| 10f | `figures/des_math.png` | 844 × 978 | Related source/output panel. |

## Supplementary Material

| Figure | Source artwork | Pixels | Notes |
|---|---|---:|---|
| S1a | `figures/source_2.png` | 2096 × 1084 | Related source/output panel. |
| S1b | `figures/des_2.png` | 2114 × 1106 | Related source/output panel. |
| S2a | `figures/source_4.png` | 934 × 1194 | Related source/output panel. |
| S2b | `figures/des_4.png` | 932 × 1182 | Related source/output panel. |

## Pre-upload decisions

- Confirm that all screenshots and rendered pages are author-owned experimental
  outputs or that reuse permission is available.
- Preserve the original pixel dimensions; do not resample the PNG files.
- If the editable/vector source for Figure 1 exists, export it as PDF or EPS
  with embedded fonts and upload that version in preference to the PNG.
- `CDM.png`, `rec_det.png`, `ro_expand.png`, `ro_orphan.png`, and `ro_prune.png`
  are not referenced by the current manuscript and should not be uploaded.
