#!/usr/bin/env python3
"""Recompute the paper's headline parser numbers from the per-item CSVs.

Run this to check that the deposited per-page / per-table records actually add up to
the values printed in the article. Standard library only -- no numpy, no scipy, no
model weights, no dataset download. It reads nothing but the CSV files next to it.

    python3 verify.py            # -> prints a table, exits 0 if everything matches

Every metric in the article is a *micro* average over a whole slice: a ratio of two
sums, not the mean of per-item ratios. That is why these files carry raw numerators
and denominators (edit_num / edit_den, tp / gt_cells, ...) rather than a finished
per-page CER or per-table F1. Averaging a per-item ratio column would give a macro
average, which is a different number.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IOU_SWEEP = [round(0.5 + 0.05 * i, 2) for i in range(10)]
TOL = 5e-4  # the article rounds to 3 decimals


def f1(tp_gt: int, tp_pred: int, n_gt: int, n_pred: int) -> float:
    r = tp_gt / n_gt if n_gt else 0.0
    p = tp_pred / n_pred if n_pred else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


# --------------------------------------------------------------------------- #
# OmniDocBench -- Table 1 of the article
# --------------------------------------------------------------------------- #
def check_omnidocbench() -> list[tuple[str, float, float]]:
    rows = list(csv.DictReader((HERE / "omnidocbench_per_page_fine.csv").open()))
    col = lambda k: sum(float(r[k]) for r in rows)  # noqa: E731
    n_gt, n_pred = int(col("gt_boxes")), int(col("pred_boxes"))

    def loc(matcher: str, t: float) -> float:
        return f1(int(col(f"{matcher}_tp_gt@{t:.2f}")),
                  int(col(f"{matcher}_tp_pred@{t:.2f}")), n_gt, n_pred)

    # A page with no matched pair has no reading-order score at all. It is left blank,
    # not zero: scoring it 0 would drag the mean down with pages that were never scored.
    ro = [float(r["reading_order_ned"]) for r in rows if r["reading_order_ned"] != ""]

    return [
        ("pages evaluated",              float(len(rows)),                        1651.0),
        ("localisation F1@.5 (N-M)",     loc("component", 0.5),                   0.940),
        ("localisation F1@.5 (COCO 1-1)", loc("coco", 0.5),                       0.654),
        ("localisation mF1[.5:.95]",     sum(loc("component", t) for t in IOU_SWEEP) / 10, 0.700),
        ("label accuracy",               col("cls_correct") / col("cls_matched"), 0.889),
        ("text edit distance",           col("edit_num") / col("edit_den"),       0.105),
        ("text CER",                     col("cer_num") / col("cer_den"),         0.108),
        ("reading order NED",            sum(ro) / len(ro),                       0.067),
    ]


# --------------------------------------------------------------------------- #
# PubTables-1M -- Table 2 of the article (config "prune")
# --------------------------------------------------------------------------- #
def average_precision(records: list[tuple[float, int]], n_gt: int) -> float:
    """COCO 101-point interpolated AP. Mirrors evaluation/eval_cells.py in stdlib."""
    if not records or not n_gt:
        return float("nan")
    records.sort(key=lambda r: -r[0])
    prec, tp, fp = [], 0, 0
    rec = []
    for _, is_tp in records:
        tp += is_tp
        fp += 1 - is_tp
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    total, j = 0.0, 0
    for q in (i / 100 for i in range(101)):
        while j < len(rec) and rec[j] < q:
            j += 1
        total += prec[j] if j < len(prec) else 0.0
    return total / 101


def check_pubtables(config: str = "prune") -> list[tuple[str, float, float]]:
    tabs = [r for r in csv.DictReader((HERE / f"pubtables_per_table_{config}.csv").open())
            if r["variant"] == "merged"]
    col = lambda k: sum(int(r[k]) for r in tabs)  # noqa: E731
    n_gt, n_pred = col("gt_cells"), col("pred_cells")

    checks = [
        ("tables evaluated",       float(len(tabs)),                              1499.0),
        ("cells F1@IoU 0.50",      f1(col("tp@0.50"), col("tp@0.50"), n_gt, n_pred), 0.902),
        ("cells F1@IoU 0.75",      f1(col("tp@0.75"), col("tp@0.75"), n_gt, n_pred), 0.750),
        ("recall, ordinary cells", col("tp_plain") / col("gt_plain"),             0.945),
        ("recall, spanning cells", col("tp_spanning") / col("gt_spanning"),       0.504),
    ]

    # AP is a set-level quantity: it ranks every prediction across all 1,499 tables by
    # score before measuring precision. There is no per-table AP to sum, which is why
    # the per-prediction file exists. It is the one large file in this directory and is
    # kept out of the git repository, so check it only when present -- a clone verifies
    # everything else, the deposited copy verifies AP too.
    per_pred = HERE / f"pubtables_per_pred_{config}.csv"
    if per_pred.exists():
        preds = [r for r in csv.DictReader(per_pred.open()) if r["variant"] == "merged"]
        aps = [average_precision([(float(r["score"]), int(r[f"tp@{t:.2f}"])) for r in preds],
                                 n_gt) for t in IOU_SWEEP]
        checks.append(("cells AP@[.5:.95]", sum(aps) / len(aps), 0.529))
    else:
        print(f"  note: {per_pred.name} absent -- skipping AP@[.5:.95]")

    return checks


def main() -> int:
    failures = 0
    for title, build in (("OmniDocBench (Table 1)", check_omnidocbench),
                         ("PubTables-1M, config=prune (Table 2)", check_pubtables)):
        print(f"\n{title}")
        checks = build()          # after the header, so any note it prints lands here
        print(f"  {'metric':28s} {'recomputed':>12s} {'in article':>12s}   ")
        for name, got, want in checks:
            ok = abs(got - want) <= (TOL if want < 100 else 0.5)
            failures += not ok
            print(f"  {name:28s} {got:12.4f} {want:12.4f}   {'ok' if ok else 'MISMATCH'}")

    print(f"\n{'All values match the article.' if not failures else f'{failures} MISMATCH(ES)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
