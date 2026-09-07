"""Legibility metric: box遮挡 dẫn tới chữ KHÔNG ĐỌC ĐƯỢC.

Collisions đếm cặp box giao nhau — nhưng "chạm mép 2%" và "che kín 90%" cùng
đếm 1. Metric này đo mức che trực tiếp theo IoS (Intersection over Smaller):

    IoS = area(A ∩ B) / min(area(A), area(B))

IoS ≥ 0.5 ⇒ box nhỏ hơn bị che hầu hết mặt ⇒ chữ trong nó khó/không đọc.
Báo cáo per system/doc:
  * occluded_rate  — % box output bị che ≥50% (chữ mờ/đè)
  * occluding_rate — % box CHE box khác (gây lỗi)
  * mean_ios       — IoS trung bình các cặp giao (mức độ đè tổng thể)

Chạy local trên detector JSON đã pull về (`_layout/docling/`), không cần GPU
hay HF. Ưu điểm: tái chấm lại bao nhiêu lần cũng được.

    python -m benchmark.e2e.metrics.eval_readability \
        --out benchmark/e2e/work/eval-t1-202/out --langs vi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inter_area(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def ios(a, b) -> float:
    smaller = min(area(a), area(b))
    if smaller <= 0:
        return 0.0
    return inter_area(a, b) / smaller


def score_page(elements: list[dict], occl_thresh: float) -> dict:
    boxes = [e["bbox_norm"] for e in elements]
    n = len(boxes)
    occluded = set()      # bị che ≥ threshold
    occluding = set()     # che box khác ≥ threshold
    ios_vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            inter = inter_area(a, b)
            if inter <= 0:
                continue
            # Bỏ quan hệ lồng nhau (nested): Docling phát cả text-block lẫn
            # text-line con bên trong — box con nằm gọn box cha là hierarchy
            # hợp lệ, KHÔNG phải chữ đè chữ. Chỉ tính che một PHẦN hai bên.
            if inter >= 0.99 * min(area(a), area(b)):
                continue
            v = inter / min(area(a), area(b))
            ios_vals.append(v)
            if v >= occl_thresh:
                occluded.add(i)
                occluded.add(j)
                if area(a) >= area(b):
                    occluding.add(i)
                else:
                    occluding.add(j)
    return {
        "n_boxes": n,
        "n_occluded": len(occluded),
        "n_occluding": len(occluding),
        "mean_ios": round(sum(ios_vals) / len(ios_vals), 4) if ios_vals else 0.0,
        "n_pairs": len(ios_vals),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--langs", default="vi")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="IoS coi là 'bị che' (default 0.5).")
    a = p.parse_args()

    root = a.out / "_layout" / "docling"
    if not root.is_dir():
        raise SystemExit(f"!! không tìm thấy {root} — pull artifact trước")

    summary = {}
    for system_dir in sorted(root.iterdir()):
        if not system_dir.is_dir():
            continue
        system = system_dir.name
        for lang in filter(None, a.langs.split(",")):
            lang_dir = system_dir / lang
            if not lang_dir.is_dir():
                continue
            per_doc = {}
            for doc_dir in sorted(lang_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue
                pages = []
                for pg_file in sorted(doc_dir.glob("p*.json")):
                    d = json.loads(pg_file.read_text())
                    pages.append(score_page(d["elements"], a.threshold))
                n_boxes = sum(x["n_boxes"] for x in pages)
                per_doc[doc_dir.name] = {
                    "pages": len(pages),
                    "occluded_rate": (round(sum(x["n_occluded"] for x in pages) / n_boxes, 4)
                                      if n_boxes else None),
                    "occluding_rate": (round(sum(x["n_occluding"] for x in pages) / n_boxes, 4)
                                       if n_boxes else None),
                    "mean_ios": (round(sum(x["mean_ios"] for x in pages) / len(pages), 4)
                                 if pages else None),
                }
            if per_doc:
                total_boxes = sum(v["pages"] for v in per_doc.values())
                docs = list(per_doc.values())
                summary[f"{system}.{lang}"] = {
                    "threshold": a.threshold,
                    "docs": per_doc,
                    "overall_occluded_rate": round(
                        sum(v["occluded_rate"] or 0 for v in docs) / len(docs), 4),
                    "overall_mean_ios": round(
                        sum(v["mean_ios"] or 0 for v in docs) / len(docs), 4),
                }

    dest = a.out / "_metrics" / "readability.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"{'system':<26}{'occluded≥{:.0%}'.format(a.threshold):>12}{'mean_IoS':>10}")
    for k, v in summary.items():
        print(f"{k:<26}{v['overall_occluded_rate']:>12.1%}{v['overall_mean_ios']:>10.3f}")
    print(f"\nchi tiết: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
