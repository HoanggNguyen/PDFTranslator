"""Đánh giá BẢNG theo hướng "coi cả bảng là một vùng OCR" (không đo cấu trúc HTML).

Lý do: GT OmniDocBench chỉ có ``html`` (không có bbox từng cell) và parser xuất
``cells`` phẳng (bbox + text, không có hàng/cột). Nên KHÔNG dựng HTML/TEDS; thay
vào đó nối text các cell theo thứ tự đọc (row-major) rồi so với text bóc từ GT html.

  * GT text : nối ``text_content`` của từng ``<td>/<th>`` (thứ tự tài liệu = row-major).
              Nếu bảng có nhiều bản html hợp lệ (html/html_2/html_3) -> lấy bản cho
              edit distance THẤP nhất (giống OmniDocBench chấp nhận đa đáp án).
  * pred text: gom cell của các Table phủ bởi GT, cụm thành hàng theo y, sort x,
              nối row-major.
  * Chuẩn hoá bằng norm_ocr (bóc thẻ, hợp nhất inline math ``$..$``/``<math>``).

Metric: Edit distance (/max, kiểu OmniDocBench) + CER, coverage, edit_matched /
edit_all. ĐO NỘI DUNG, KHÔNG đo cấu trúc hàng/cột (hạn chế đã biết).

Ví dụ
-----
    python eval_table.py --gt ../../OmniDocBench.json --pred ../../parser_json \
        --mapping ../../mapping.json --out ../../eval_report_table.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from lxml import html as LH

import eval_layout as E

SLICE_KEYS = ("language", "layout", "subset", "data_source")


def html_to_text(html_str: str) -> str:
    """Nối text từng cell <td>/<th> theo thứ tự tài liệu (row-major)."""
    try:
        tree = LH.fromstring(html_str)
    except Exception:
        return ""
    cells = tree.xpath("//td | //th")
    if cells:
        return " ".join(c.text_content() for c in cells)
    return tree.text_content()


def cluster_rows(cells: list[dict]) -> list[dict]:
    """Sắp cell theo thứ tự đọc row-major (cụm hàng theo y, rồi sort x)."""
    if not cells:
        return []
    boxes = [(c, c["bbox_pdf"]) for c in cells]
    boxes.sort(key=lambda cb: (cb[1][1] + cb[1][3]) / 2.0)   # theo y-center
    heights = sorted((b[3] - b[1]) for _, b in boxes)
    med_h = heights[len(heights) // 2] or 1.0
    rows, cur, cur_y = [], [], None
    for c, b in boxes:
        yc = (b[1] + b[3]) / 2.0
        if cur_y is None or abs(yc - cur_y) <= med_h * 0.6:
            cur.append((c, b))
            cur_y = yc if cur_y is None else (cur_y + yc) / 2.0
        else:
            rows.append(cur)
            cur, cur_y = [(c, b)], yc
    if cur:
        rows.append(cur)
    ordered = []
    for row in rows:
        row.sort(key=lambda cb: cb[1][0])      # trong hàng: theo x
        ordered.extend(c for c, _ in row)
    return ordered


def gt_tables(page: dict, drop_ignore: bool) -> list[dict]:
    out = []
    for d in page["dets"]:
        if d.get("category_type") != "table":
            continue
        if drop_ignore and d.get("ignore"):
            continue
        variants = [d[k] for k in ("html", "html_2", "html_3") if d.get(k)]
        out.append({
            "box": E.norm_box(E.poly_to_xyxy(d["poly"]), page["w"], page["h"]),
            "texts": [E.norm_ocr(html_to_text(h)) for h in variants] or [""],
        })
    return out


def pred_tables(page: dict) -> list[dict]:
    W, H = page.get("page_width"), page.get("page_height")
    out = []
    for e in page["elements"]:
        if e.get("label") != "Table":
            continue
        out.append({
            "box": E.norm_box(e["bbox_pdf"], W, H),
            "cells": [c for c in e.get("cells", []) if c.get("bbox_pdf")],
        })
    return out


class Acc:
    def __init__(self):
        self.n_gt = self.n_matched = self.n_pred = self.n_pred_unmatched = 0
        self.num_all = self.den_all = 0.0
        self.num_m = self.den_m = 0.0
        self.cer_num = self.cer_den = 0.0

    def summary(self):
        return {
            "gt_tables": self.n_gt,
            "coverage": round(self.n_matched / self.n_gt, 4) if self.n_gt else None,
            "pred_tables": self.n_pred, "pred_unmatched": self.n_pred_unmatched,
            "edit_all_micro": round(self.num_all / self.den_all, 4) if self.den_all else None,
            "edit_matched_micro": round(self.num_m / self.den_m, 4) if self.den_m else None,
            "score_matched": round(1 - self.num_m / self.den_m, 4) if self.den_m else None,
            "CER_matched": round(self.cer_num / self.cer_den, 4) if self.cer_den else None,
        }


def best_pair(pred_text: str, gt_texts: list[str]):
    """Lấy (dist, maxlen, gtlen) theo bản GT html cho edit distance nhỏ nhất."""
    best = None
    for gt in gt_texts:
        d = E._dist(pred_text, gt)
        if best is None or d < best[0]:
            best = (d, max(len(pred_text), len(gt)), len(gt))
    return best


def score_page(gts, preds, member_thr):
    used = [False] * len(preds)
    ps = {"n_gt": len(gts), "n_matched": 0, "n_pred": len(preds), "pairs": []}
    for g in gts:
        members, idxs = [], []
        for i, p in enumerate(preds):
            if E.contain_ratio(p["box"], g["box"]) >= member_thr:
                members.append(p)
                idxs.append(i)
        if members:
            for i in idxs:
                used[i] = True
            cells = [c for m in members for c in m["cells"]]
            pred_text = E.norm_ocr(" ".join(
                c.get("source_text", "") for c in cluster_rows(cells)))
            ps["n_matched"] += 1
            matched = True
        else:
            pred_text = ""
            matched = False
        d, mlen, gtlen = best_pair(pred_text, g["texts"])
        if mlen:
            ps["pairs"].append((d, mlen, gtlen, matched))
    ps["n_pred_unmatched"] = sum(1 for u in used if not u)
    return ps


def evaluate(gt_pages, pred_index, member_thr, drop_ignore):
    slices = defaultdict(Acc)
    for img_name, pred_page in pred_index.items():
        gt_page = gt_pages.get(img_name)
        if gt_page is None:
            continue
        gts = gt_tables(gt_page, drop_ignore)
        preds = pred_tables(pred_page)
        if not gts and not preds:
            continue
        ps = score_page(gts, preds, member_thr)
        keys = ["all"]
        for k in SLICE_KEYS:
            v = gt_page["attr"].get(k)
            if isinstance(v, list):
                keys += [f"{k}={x}" for x in v]
            elif v is not None:
                keys.append(f"{k}={v}")
        for key in keys:
            a = slices[key]
            a.n_gt += ps["n_gt"]
            a.n_matched += ps["n_matched"]
            a.n_pred += ps["n_pred"]
            a.n_pred_unmatched += ps["n_pred_unmatched"]
            for d, mlen, gtlen, matched in ps["pairs"]:
                a.num_all += d
                a.den_all += mlen
                if matched:
                    a.num_m += d
                    a.den_m += mlen
                    a.cer_num += d
                    a.cer_den += gtlen
    return slices


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--member-thr", type=float, default=0.5)
    ap.add_argument("--keep-ignore", action="store_true")
    ap.add_argument("--min-slice", type=int, default=10)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gt_pages = E.load_gt(args.gt, "merged")
    mapping = json.load(open(args.mapping, encoding="utf-8"))
    pred_index = E.build_pred_index(args.pred, mapping)
    print(f"[table] GT trang={len(gt_pages)}  pred trang={len(pred_index)}  "
          f"member_thr={args.member_thr}")

    slices = evaluate(gt_pages, pred_index, args.member_thr, not args.keep_ignore)
    report = {"config": {"member_thr": args.member_thr, "note": "OCR-only, no structure"},
              "slices": {k: a.summary() for k, a in slices.items()}}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    a = slices["all"].summary()
    print("\n===== TABLE (all) — đo NỘI DUNG, không đo cấu trúc =====")
    print(f"  GT bảng = {a['gt_tables']}   coverage = {a['coverage']}")
    print(f"  pred Table = {a['pred_tables']}   không khớp = {a['pred_unmatched']}")
    print(f"  edit_matched micro = {a['edit_matched_micro']}  -> score = {a['score_matched']}")
    print(f"  CER_matched        = {a['CER_matched']}")
    print(f"  edit_all micro     = {a['edit_all_micro']}  (gồm cả sót detect)")

    print("\n===== THEO LÁT CẮT (coverage / edit_matched / CER) =====")
    for key in sorted(slices):
        if key == "all":
            continue
        s = slices[key].summary()
        if s["gt_tables"] < args.min_slice:
            continue
        print(f"  {key:28s} cov={str(s['coverage']):>6}  "
              f"editM={str(s['edit_matched_micro']):>6}  "
              f"CER={str(s['CER_matched']):>6}  (gt={s['gt_tables']})")
    if args.out:
        print(f"\n[table] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
