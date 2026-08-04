"""Gom các report JSON (layout/OCR + formula) thành CSV để phân tích / vẽ biểu đồ.

Đọc report của eval_layout.py và eval_formula.py, làm phẳng mọi metric theo từng
lát cắt (slice) và xuất:
  * <out>_long.csv : (report, slice, metric, value)  -- tiện nhóm/vẽ.
  * <out>_wide.csv : mỗi hàng = 1 slice, mỗi cột = "<report>.<metric>".

Ví dụ
-----
    # chạy từ benchmark/parser/
    python evaluation/aggregate_reports.py \
        --layout fine=eval_results/eval_report_fine.json \
        --layout merged=eval_results/eval_report_merged.json \
        --formula eval_results/eval_report_formula.json \
        --report table=eval_results/eval_report_table.json \
        --out eval_results/eval_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(d: dict, prefix: str = "") -> dict:
    """Làm phẳng dict lồng nhau -> {'a.b.c': value} (bỏ qua list)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            continue
        else:
            out[key] = v
    return out


def rows_from_report(name: str, path: Path):
    rep = json.load(open(path, encoding="utf-8"))
    rows = []
    for slice_key, metrics in rep["slices"].items():
        flat = flatten(metrics)
        for metric, value in flat.items():
            rows.append({"report": name, "slice": slice_key,
                         "metric": metric, "value": value})
    return rows


def parse_kv(s: str, default_name: str) -> tuple[str, Path]:
    if "=" in s:
        name, path = s.split("=", 1)
        return name, Path(path)
    return default_name, Path(s)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", action="append", default=[],
                    help="report layout, dạng name=path (lặp nhiều lần). "
                         "VD: fine=../../eval_report_fine.json")
    ap.add_argument("--formula", action="append", default=[],
                    help="report formula, dạng [name=]path (lặp nhiều lần).")
    ap.add_argument("--report", action="append", default=[],
                    help="report BẤT KỲ (table/cdm/…), dạng name=path (lặp nhiều lần).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Tiền tố file ra: <out>_long.csv và <out>_wide.csv")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for spec in args.layout:
        name, path = parse_kv(spec, "layout")
        rows += rows_from_report(f"layout_{name}", path)
    for spec in args.formula:
        name, path = parse_kv(spec, "formula")
        rows += rows_from_report(name if "=" in spec else "formula", path)
    for spec in args.report:
        name, path = parse_kv(spec, "report")
        rows += rows_from_report(name, path)

    if not rows:
        print("Không có report nào. Truyền --layout/--formula.")
        return 1

    df = pd.DataFrame(rows)
    long_path = args.out.with_name(args.out.name + "_long.csv")
    df.to_csv(long_path, index=False)

    # wide: slice x (report.metric)
    df["col"] = df["report"] + "." + df["metric"]
    wide = df.pivot_table(index="slice", columns="col", values="value", aggfunc="first")
    # đưa 'all' lên đầu
    order = ["all"] + sorted(s for s in wide.index if s != "all")
    wide = wide.reindex([s for s in order if s in wide.index])
    wide_path = args.out.with_name(args.out.name + "_wide.csv")
    wide.to_csv(wide_path)

    print(f"[aggregate] {len(rows)} dòng metric, {df['slice'].nunique()} slice, "
          f"{df['report'].nunique()} report")
    print(f"  long -> {long_path}")
    print(f"  wide -> {wide_path}")

    # in nhanh vài cột chính cho 'all'
    key_cols = [c for c in wide.columns if any(
        k in c for k in ("f1@0.5", "CER", "WER", "reading_order.edit_distance_mean",
                          "coverage", "edit_matched_micro"))]
    if "all" in wide.index and key_cols:
        print("\n[all] các cột chính:")
        for c in sorted(key_cols):
            print(f"  {c:45s} {wide.loc['all', c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
