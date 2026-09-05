"""Chấm CELL DETECTION trên PubTables-1M: Hungarian trên IoU -> P/R/F1 + AP.

Không dùng lưới logic, không heuristic. GT cell = ``row ∩ col`` (định nghĩa chính thức
của dataset, xem pubtables_gt.py); prediction là một TẬP box. Ghép hai tập bằng
**Hungarian** (``scipy.optimize.linear_sum_assignment``) trên ma trận IoU, nên không cần
biết ô nào thuộc hàng/cột nào.

Vì sao Hungarian chứ không greedy: greedy phụ thuộc thứ tự và có thể "hy sinh" một cặp tốt
để lấy một cặp kém hơn trước đó. Hungarian cho ghép tối ưu toàn cục, nên P/R không dao
động theo thứ tự box.

HAI matcher khác nhau, cố ý — phải ghi rõ khi báo cáo:

  * **P/R/F1 @ IoU cố định** dùng **Hungarian** (ghép 1-1 tối ưu toàn cục). Chọn vậy vì
    cell là quan hệ 1-1 tự nhiên và kết quả không phụ thuộc thứ tự box. Lưu ý: COCO dùng
    greedy, nên F1 ở đây **không** so trực tiếp được với F1 báo theo giao thức COCO.
  * **AP@[.5:.95]** dùng **greedy theo score giảm dần** (đúng giao thức COCO), vì AP phải
    trả lời "lấy top-k thì precision bao nhiêu" nên assignment buộc phụ thuộc score.

Nguồn của từng phần: GT cell = row ∩ col và quy tắc gộp supercell >50% lấy từ paper
PubTables-1M (Smock et al., CVPR 2022) và ``table_structure_to_cells``/``align_supercells``
trong microsoft/table-transformer. AP 101 điểm nội suy lấy từ giao thức COCO (Lin et al.,
2014). Hungarian là thuật toán chuẩn (Kuhn, 1955), cũng là cách PubTables-v2 ghép nhiều
bảng trên một trang. Phần phân tầng lấy mẫu là thống kê thông thường, không theo paper nào.

Độ đo:
  * P / R / F1 @ IoU 0,5 và 0,75
  * AP@[.5:.95] — COCO-style, xếp box theo ``score`` giảm dần (đây là lý do
    run_cells_pubtables.py phải giữ score)
  * mean IoU của các cặp đã ghép
  * count ratio = |pred| / |GT|
  * recall tách riêng ô **spanning** vs ô thường (chỉ ở biến thể GT merged)

Luôn chạy trên CẢ HAI biến thể GT (merged / unmerged) vì 42,0% bảng có span.

Ví dụ
-----
    # chạy từ benchmark/parser/
    python evaluation/eval_cells.py \
        --pred   pubtables_results/cells_prune.json \
        --ann    data/pubtables1m/test \
        --sample data/pubtables1m/sample_1500.json \
        --out    eval_results/eval_cells_prune.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import pubtables_gt as G

IOU_MAIN = (0.5, 0.75)
IOU_SWEEP = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # .50 .55 ... .95


def iou_matrix(pred: list[list[float]], gt: list[list[float]]) -> np.ndarray:
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)), dtype=float)
    p = np.asarray(pred, dtype=float)
    g = np.asarray(gt, dtype=float)
    x0 = np.maximum(p[:, None, 0], g[None, :, 0])
    y0 = np.maximum(p[:, None, 1], g[None, :, 1])
    x1 = np.minimum(p[:, None, 2], g[None, :, 2])
    y1 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = ap[:, None] + ag[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred: list[list[float]], gt: list[list[float]], thr: float):
    """Hungarian trên IoU; trả về [(pi, gi, iou)] các cặp đạt >= thr."""
    m = iou_matrix(pred, gt)
    if m.size == 0:
        return []
    pi, gi = linear_sum_assignment(-m)
    return [(int(a), int(b), float(m[a, b])) for a, b in zip(pi, gi) if m[a, b] >= thr]


def coco_tp_flags(pred: list[list[float]], scores: list[float],
                  gt: list[list[float]], thresholds: list[float]) -> dict[float, list[bool]]:
    """Gán TP/FP theo giao thức COCO, cho từng ngưỡng IoU.

    Duyệt prediction theo score giảm dần; mỗi prediction lấy GT chưa bị chiếm có IoU
    cao nhất và >= ngưỡng. Ràng buộc **một GT chỉ ghép một lần** là thứ khiến box trùng
    lặp bị tính FP thay vì TP.

    Khác với Hungarian dùng cho P/R/F1: AP cần thứ tự theo score (phải trả lời "nếu chỉ
    lấy top-k thì precision bao nhiêu"), nên assignment phải phụ thuộc score, không phải
    tối ưu toàn cục.

    Trả về {ngưỡng: [is_tp theo ĐÚNG thứ tự của ``pred``]}.
    """
    out: dict[float, list[bool]] = {t: [False] * len(pred) for t in thresholds}
    if not pred or not gt:
        return out
    m = iou_matrix(pred, gt)
    order = sorted(range(len(pred)), key=lambda i: -scores[i])
    for t in thresholds:
        taken = [False] * len(gt)
        flags = out[t]
        for pi in order:
            best_gi, best_iou = -1, t
            for gi in range(len(gt)):
                if taken[gi]:
                    continue
                if m[pi, gi] >= best_iou:
                    best_gi, best_iou = gi, m[pi, gi]
            if best_gi >= 0:
                taken[best_gi] = True
                flags[pi] = True
    return out


def average_precision(records: list[tuple[float, bool]], n_gt: int) -> float:
    """AP kiểu COCO (nội suy 101 điểm) từ [(score, is_tp)] đã gộp toàn tập."""
    if n_gt == 0 or not records:
        return float("nan")
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([1 if r[1] else 0 for r in records], dtype=float)
    fp = np.cumsum([0 if r[1] else 1 for r in records], dtype=float)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # precision đơn điệu giảm
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    qs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, qs, side="left")
    vals = [prec[i] if i < len(prec) else 0.0 for i in idx]
    return float(np.mean(vals))


class Acc:
    """Cộng dồn micro theo một lát cắt."""

    def __init__(self):
        self.n_gt = self.n_pred = 0
        self.tp = {t: 0 for t in IOU_MAIN}
        self.iou_sum = 0.0
        self.iou_n = 0
        self.n_tab = 0
        self.gt_span = self.tp_span = 0
        self.gt_plain = self.tp_plain = 0
        self.sweep: dict[float, list[tuple[float, bool]]] = {t: [] for t in IOU_SWEEP}

    def summary(self) -> dict:
        out = {"tables": self.n_tab, "gt_cells": self.n_gt, "pred_cells": self.n_pred}
        for t in IOU_MAIN:
            tp = self.tp[t]
            p = tp / self.n_pred if self.n_pred else None
            r = tp / self.n_gt if self.n_gt else None
            f1 = (2 * p * r / (p + r)) if (p and r) else (0.0 if (p is not None and r is not None) else None)
            key = f"{t:.2f}"
            out[f"P@{key}"] = round(p, 4) if p is not None else None
            out[f"R@{key}"] = round(r, 4) if r is not None else None
            out[f"F1@{key}"] = round(f1, 4) if f1 is not None else None
        aps = [average_precision(list(self.sweep[t]), self.n_gt) for t in IOU_SWEEP]
        aps = [a for a in aps if not math.isnan(a)]
        out["AP@[.5:.95]"] = round(float(np.mean(aps)), 4) if aps else None
        out["mean_IoU_matched"] = round(self.iou_sum / self.iou_n, 4) if self.iou_n else None
        out["count_ratio"] = round(self.n_pred / self.n_gt, 4) if self.n_gt else None
        if self.gt_span:
            out["R@0.50_spanning"] = round(self.tp_span / self.gt_span, 4)
        if self.gt_plain:
            out["R@0.50_plain"] = round(self.tp_plain / self.gt_plain, 4)
        return out


def evaluate(pred_doc: dict, ann_dir: Path, sample: dict, variant: str) -> dict:
    by_name = {it["name"]: it for it in sample["items"]}
    slices: dict[str, Acc] = defaultdict(Acc)

    for name, pr in pred_doc["results"].items():
        it = by_name.get(name)
        if it is None:
            continue
        try:
            gt = G.gt_for(ann_dir / pr["xml"])
        except Exception:
            continue
        gcells = gt["cells_merged"] if variant == "merged" else gt["cells_unmerged"]
        gboxes = [c["box"] for c in gcells]
        pboxes = [c["box"] for c in pr["cells"]]
        pscores = [c["score"] for c in pr["cells"]]

        keys = ["all",
                f"has_span={it['has_span']}",
                f"size={it.get('size_band', '?')}"]

        pairs_main = {t: match(pboxes, gboxes, t) for t in IOU_MAIN}

        # Cho AP: dùng giao thức COCO — duyệt pred theo score GIẢM DẦN, mỗi GT chỉ
        # được ghép MỘT lần. Bắt buộc phải có ràng buộc 1-1 này, nếu không thì hai box
        # trùng nhau trên cùng một GT đều tính là TP và AP bị thổi lên (đã kiểm: nhân
        # đôi mọi box -> precision thật 0,5 nhưng AP ra 1,0).
        tp_flags = coco_tp_flags(pboxes, pscores, gboxes, IOU_SWEEP)

        for key in keys:
            a = slices[key]
            a.n_tab += 1
            a.n_gt += len(gboxes)
            a.n_pred += len(pboxes)
            for t in IOU_MAIN:
                a.tp[t] += len(pairs_main[t])
            for pi, gi, v in pairs_main[0.5]:
                a.iou_sum += v
                a.iou_n += 1
                if variant == "merged":
                    if gcells[gi].get("is_spanning"):
                        a.tp_span += 1
                    else:
                        a.tp_plain += 1
            if variant == "merged":
                a.gt_span += sum(1 for c in gcells if c.get("is_spanning"))
                a.gt_plain += sum(1 for c in gcells if not c.get("is_spanning"))
            for t in IOU_SWEEP:
                flags = tp_flags[t]
                for i in range(len(pboxes)):
                    a.sweep[t].append((pscores[i], flags[i]))

    return {k: v.summary() for k, v in slices.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pred_doc = json.loads(args.pred.read_text(encoding="utf-8"))
    sample = json.loads(args.sample.read_text(encoding="utf-8"))

    report = {"pred_config": pred_doc.get("config", {}),
              "sample_meta": sample.get("meta", {}),
              "variants": {}}
    for variant in ("merged", "unmerged"):
        report["variants"][variant] = evaluate(pred_doc, args.ann, sample, variant)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cfg = pred_doc.get("config", {})
    print(f"=== CELL DETECTION | prune={cfg.get('prune')} "
          f"crop_to_table={cfg.get('crop_to_table')} thr={cfg.get('threshold')} ===")
    for variant in ("merged", "unmerged"):
        a = report["variants"][variant].get("all")
        if not a:
            continue
        print(f"\n-- GT {variant} --  bảng={a['tables']} gt={a['gt_cells']} pred={a['pred_cells']}")
        print(f"   IoU 0.50 : P={a['P@0.50']} R={a['R@0.50']} F1={a['F1@0.50']}")
        print(f"   IoU 0.75 : P={a['P@0.75']} R={a['R@0.75']} F1={a['F1@0.75']}")
        print(f"   AP@[.5:.95] = {a['AP@[.5:.95]']}   mean IoU = {a['mean_IoU_matched']}")
        print(f"   count ratio = {a['count_ratio']}")
        if "R@0.50_spanning" in a:
            print(f"   R@0.50  spanning={a['R@0.50_spanning']}  plain={a['R@0.50_plain']}")
        print("   --- theo lát cắt ---")
        for k in sorted(report["variants"][variant]):
            if k == "all":
                continue
            s = report["variants"][variant][k]
            print(f"     {k:22s} F1@.5={str(s['F1@0.50']):>7}  "
                  f"AP={str(s['AP@[.5:.95]']):>7}  (bảng={s['tables']})")
    if args.out:
        print(f"\n[cells] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
