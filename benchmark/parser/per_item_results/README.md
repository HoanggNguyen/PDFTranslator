# Per-item parser evaluation records

Written in English rather than Vietnamese (unlike the rest of this repository) because
this directory is meant to be deposited in a data repository and read by reviewers and
readers of the article.

One row per evaluated item — per page for OmniDocBench, per table for PubTables-1M —
behind the parser results reported in the article. The report files under
`../eval_results/` give only slice-level aggregates, which cannot be checked or
re-sliced by a reader. These files can.

`verify.py` recomputes every headline number in the article from these CSVs and prints
computed against published. It uses the Python standard library only: no model weights,
no dataset download, no numpy.

```
sha256sum -c SHA256SUMS.txt
python3 verify.py
```

## What is *not* here

No content from either source dataset is redistributed. The rows carry identifiers
(image file names, table ids) and metric counters — no ground-truth boxes, no
ground-truth text, no OCR output, no predicted geometry. Re-running the evaluation from
scratch requires OmniDocBench and PubTables-1M from their own distributors.

## Files

| File | Rows | What one row is |
| --- | --- | --- |
| `omnidocbench_per_page_fine.csv` | 1,651 | one OmniDocBench page |
| `pubtables_per_table_<config>.csv` | 2,998 | one PubTables-1M table under one GT variant |
| `pubtables_per_pred_<config>.csv` | ~209,000 | one predicted cell box |

`<config>` is one of the seven cell-detector settings compared in the article:
`prune` (the configuration all headline numbers use), `noprune`, `nocontainer`,
`prefer-container`, `nocrop`, `thr0.2`, `thr0.5`.

## Why counters and not rates

Every metric in the article is a **micro** average over a slice: one ratio of two sums,
not the mean of per-item ratios. So the columns are raw numerators and denominators —
`edit_num`/`edit_den`, `tp@0.50`/`gt_cells` — and a reader reconstructs a metric by
summing each column over the rows they care about and dividing once at the end.

Averaging a per-item rate column instead would produce a macro average: a different
number, which will not match the article. That is the reason no per-page CER column and
no per-table F1 column exist.

## `omnidocbench_per_page_fine.csv`

Produced by `evaluation/eval_layout.py --gt-granularity fine` (the configuration behind
Table 1). 57 columns.

| Column | Meaning |
| --- | --- |
| `image_name` | OmniDocBench page image, the join key to `OmniDocBench.json` |
| `language`, `layout`, `subset`, `data_source` | GT page attributes; these define the slices reported in Appendix A. A page carrying several values for one attribute (it then contributes to several slices) has them joined with `\|`, so the row stays one-to-one with the page |
| `gt_boxes`, `pred_boxes` | region counts after dropping `ignore` regions and the `abandon`/`other` groups |
| `coco_tp_gt@<t>`, `coco_tp_pred@<t>` | true positives under strict COCO 1-1 matching, at IoU `<t>` ∈ {0.50, 0.55, …, 0.95} |
| `component_tp_gt@<t>`, `component_tp_pred@<t>` | true positives under the granularity-invariant *N*–*M* matcher, same thresholds |
| `cls_matched`, `cls_correct` | pairs matched at IoU 0.5, and how many carry the right label |
| `edit_num`, `edit_den` | normalised edit distance, summed (OmniDocBench `Edit_dist`: denominator is `max(len(pred), len(gt))`) |
| `cer_num`, `cer_den` | character error rate, summed (denominator is GT length) |
| `wer_num`, `wer_den` | word error rate, summed. Space-separated scripts only — CJK pages contribute 0/0 by design, since a word rate is meaningless without word boundaries |
| `ocr_pairs` | matched pairs that carried text on either side |
| `reading_order_ned` | normalised edit distance over the reading-order sequence. **Blank, not zero**, on the 150 pages with no matched pair: they have no sequence to score, and the article's 0.067 is the mean over the 1,501 pages that do. Scoring them 0 would pull the mean down with pages that were never measured |

Two matchers are reported because they answer different questions, and the gap between
them is a finding in the article rather than noise: `coco` demands one predicted region
per annotated region, so splitting one annotated block into several useful fragments is
counted as failure; `component` matches connected groups *N*–*M*, so it measures whether
the page area was recovered at all.

## `pubtables_per_table_<config>.csv`

Produced by `evaluation/eval_cells.py`. Both GT variants are in one file — filter on
`variant` before aggregating, or every count is doubled.

| Column | Meaning |
| --- | --- |
| `variant` | `merged` = spanning cells kept whole (the article's numbers); `unmerged` = every grid position its own cell |
| `table_id`, `xml` | PubTables-1M table identifier and its annotation file |
| `has_span`, `size_band` | sampling strata, from `data/pubtables1m/sample_1500.json` |
| `gt_cells`, `pred_cells` | cell counts; `pred_cells / gt_cells` summed gives the count ratio |
| `tp@0.50`, `tp@0.75` | matched pairs, optimal 1-1 Hungarian assignment on IoU. Precision and recall share this numerator because the assignment is 1-1 |
| `iou_sum_matched@0.50`, `n_matched@0.50` | sum and count of the IoU values of matched pairs — divide the sums to get mean IoU |
| `gt_spanning`, `gt_plain`, `tp_spanning`, `tp_plain` | the spanning-cell split, `merged` variant only; blank under `unmerged`, where spanning cells are not defined |

Note that `F1@0.50` here uses Hungarian matching while `AP@[.5:.95]` uses greedy
score-ordered matching, following COCO. The two are deliberately different — AP has to
answer "what is precision among the top-k by score", which forces the assignment to
depend on score — so the F1 in this table is not directly comparable to an F1 reported
under a COCO protocol.

## `pubtables_per_pred_<config>.csv`

| Column | Meaning |
| --- | --- |
| `variant`, `table_id` | as above |
| `pred_idx` | index into the `cells` array of that table in `../pubtables_results/cells_<config>.json` |
| `score` | detector confidence, the value AP ranks by |
| `tp@<t>` | 1 if this prediction is a true positive at IoU `<t>`, under COCO's greedy score-ordered assignment with each GT cell claimable once |

This file exists only so that `AP@[.5:.95]` can be reconstructed. AP is a set-level
quantity: it ranks every prediction across all 1,499 tables by score before measuring
precision, so there is no per-table AP that could be summed out of the per-table file.

## Provenance

| | |
| --- | --- |
| Predictions | `../parser_results/batch_*.json`, `../pubtables_results/cells_*.json` |
| Scoring code | `../evaluation/eval_layout.py`, `../evaluation/eval_cells.py` |
| Ground truth | OmniDocBench `OmniDocBench.json`; PubTables-1M test annotations, 1,499 tables sampled by `../evaluation/sample_pubtables.py` |

The two scoring scripts write these CSVs and their aggregate JSON reports in the same
pass, from the same counters, so the two cannot drift apart. Regenerating them
reproduces `../eval_results/eval_report_fine.json` and `../eval_results/eval_cells_*.json`
byte for byte:

```bash
python evaluation/eval_layout.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json --gt-granularity fine \
    --out eval_results/eval_report_fine.json \
    --per-page per_item_results/omnidocbench_per_page_fine.csv

python evaluation/eval_cells.py \
    --pred pubtables_results/cells_prune.json \
    --ann data/pubtables1m/test --sample data/pubtables1m/sample_1500.json \
    --out eval_results/eval_cells_prune.json \
    --per-table per_item_results/pubtables_per_table_prune.csv \
    --per-pred  per_item_results/pubtables_per_pred_prune.csv
```
