"""Nhóm A — layout preservation. Trục chính của luận văn.

Phân biệt cho rõ, vì hai thứ này giống hệt nhau về hình học mà khác hẳn về ngữ nghĩa:

* ``benchmark/parser/evaluation/eval_layout.py`` đo **"parser tìm box giỏi tới đâu"**
  — box của parser so với GT của cùng trang đó.
* Module này đo **"PDF đầu ra giữ layout của trang nguồn tới đâu"** — box detector
  tìm thấy trên trang *đã dịch*, so với GT người vẽ của trang *nguồn*.

Nên lõi hình học (``iou``, ``area``, ``contain_ratio``) import thẳng từ file kia,
còn phần ghép và phần metric thì viết riêng.

Bốn quyết định đáng giải thích:

1. **Ghép bằng Hungarian, cost ``1 − IoU``, ràng buộc cùng nhóm nhãn.** Không dùng
   greedy: greedy phụ thuộc thứ tự duyệt nên đổi thứ tự box là đổi điểm. Ràng buộc
   cùng nhóm để một cái bảng bị dịch trôi không được "ghép tạm" vào một đoạn text
   gần đó rồi ăn điểm IoU.

2. **``Anchor-IoU`` tính riêng trên {figure, table, formula, furniture}.** Đây là
   những thứ **phải đứng yên tuyệt đối**: dịch xong thì chữ nở ra hay co lại là
   chuyện bình thường, nhưng cái hình không được nhảy chỗ. Tách ra thì tín hiệu
   sạch, không bị nhiễu bởi độ giãn ngôn ngữ. **BabelDOC không đo cái này.**

3. **``collision`` và ``margin`` chỉ nhìn phía đầu ra, không cần GT.** Chữ đè lên
   chữ và chữ tràn ra lề là lỗi *nhìn thấy được*, và mIoU giấu chúng hoàn toàn —
   một trang có thể có mIoU cao mà vẫn xấu. Lồng nhau (caption nằm trong figure)
   **không** tính là va chạm, nếu không thì trang nào cũng "va".

4. **Tài liệu nào ``page_inflation != 1`` thì bị loại, áp cho mọi hệ như nhau.**
   Không phải để ưu ái ai: metric ở đây neo theo *từng trang nguồn*, hệ nào reflow
   làm đổi số trang thì không tồn tại phép ghép trang nào đúng. Số doc bị loại được
   báo cáo tường minh — với DeepL đó chính là một phát hiện, không phải một lỗ hổng.

Hai hàng chuẩn, bắt buộc có trong mọi bảng:

* ``_source`` (**Source ceiling**) — detector chạy trên PDF **nguồn chưa dịch**, chấm
  với GT người vẽ. Đây là **trần thật**; mọi điểm phải đọc *tương đối* so với nó.
  Không có hàng này thì không biết 0.72 là "translator làm hỏng" hay "detector chỉ
  giỏi đến thế".
* ``identity`` — copy PDF nguồn làm đầu ra. Phải trùng khít hàng ceiling; lệch là
  lỗi của chính harness.

Ví dụ
-----
    python -m benchmark.e2e.metrics.eval_preserve \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi --detector docling
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.e2e.parse.run_detectors import (ANCHOR_GROUPS, DOCLAYNET_GROUP,
                                               GROUPS, SOURCE_KEY, reading_order)
from benchmark.parser.evaluation.eval_layout import area, contain_ratio, inter_area, iou

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # .50 .. .95
TEXT_GROUPS = ("text", "title", "list", "caption")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None,
                   help="Mặc định: mọi hệ có box trong _layout/<detector>/.")
    p.add_argument("--detector", default="docling",
                   help="Detector chấm. Đổi sang 'surya'/'doclayout' để kiểm "
                        "robustness §P6 — thứ hạng đổi là phải nói ra.")
    p.add_argument("--collision-eps", type=float, default=0.05,
                   help="Hai box coi là va chạm khi phần chồng vượt tỉ lệ này so "
                        "với diện tích box NHỎ hơn.")
    p.add_argument("--nest-thr", type=float, default=0.9,
                   help="Chồng ở mức này trở lên coi là LỒNG NHAU (caption trong "
                        "figure), không phải va chạm.")
    p.add_argument("--margin-pad", type=float, default=0.01,
                   help="Nới content-box của trang nguồn ra ngần này (theo tỉ lệ "
                        "khổ trang) trước khi tính tràn lề.")
    p.add_argument("--margin-out", type=float, default=0.05,
                   help="Box bị tính là tràn khi có hơn ngần này diện tích nằm "
                        "ngoài content-box đã nới.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Ghép                                                                          #
# --------------------------------------------------------------------------- #
def hungarian(gts: list[dict], preds: list[dict]) -> list[tuple[int, int, float]]:
    """Ghép 1-1 tối ưu trong từng nhóm nhãn. Trả (i_gt, i_pred, iou), iou > 0."""
    if not gts or not preds:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:                    # pragma: no cover
        raise SystemExit(f"!! cần numpy + scipy: {exc}") from exc

    n, m = len(gts), len(preds)
    # 2.0 = "không được phép ghép": mọi cost hợp lệ nằm trong [0, 1], nên nghiệm tối
    # ưu chỉ chạm ô cấm khi không còn lựa chọn nào khác — và những ô đó bị lọc bỏ
    # ngay sau đó bằng điều kiện iou > 0.
    cost = np.full((n, m), 2.0, dtype=float)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if g["group"] != p["group"]:
                continue
            cost[i, j] = 1.0 - iou(g["bbox_norm"], p["bbox_norm"])

    rows, cols = linear_sum_assignment(cost)
    matched = []
    for i, j in zip(rows, cols):
        if cost[i, j] >= 1.0:                      # IoU = 0 hoặc khác nhóm
            continue
        matched.append((int(i), int(j), 1.0 - float(cost[i, j])))
    return matched


# --------------------------------------------------------------------------- #
# Metric từng trang — trả về SỐ ĐẾM THÔ, phần chia để dồn ở mức trên              #
# --------------------------------------------------------------------------- #
def kendall_tau(a: list[int], b: list[int]) -> float | None:
    if len(a) < 2:
        return None
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def score_page(gt_page: dict, pred_page: dict, args: argparse.Namespace) -> dict:
    gts, preds = gt_page["elements"], pred_page["elements"]
    matched = hungarian(gts, preds)

    ious = [m[2] for m in matched]
    tp_at = {t: sum(1 for v in ious if v >= t) for t in IOU_THRESHOLDS}

    # Anchor: ghép lại RIÊNG trên tập con. Không lọc từ kết quả ghép chung — ghép
    # chung có thể đã "tiêu" một cái bảng vào một cặp kém hơn vì phải tối ưu toàn cục.
    a_gts = [g for g in gts if g["group"] in ANCHOR_GROUPS]
    a_preds = [p for p in preds if p["group"] in ANCHOR_GROUPS]
    a_matched = hungarian(a_gts, a_preds)
    a_ious = [m[2] for m in a_matched]

    # Text containment: bao nhiêu phần của box đầu ra còn nằm trong khung gốc.
    # Chữ nở tràn ra ngoài khung ⇒ tụt, kể cả khi IoU vẫn còn khá.
    contain = [contain_ratio(preds[j]["bbox_norm"], gts[i]["bbox_norm"])
               for i, j, _ in matched if gts[i]["group"] in TEXT_GROUPS]

    retention = {g: [0, 0] for g in GROUPS}        # [n_gt, n_pred]
    for g in gts:
        retention[g["group"]][0] += 1
    for p in preds:
        retention[p["group"]][1] += 1

    # Va chạm: chỉ nhìn phía đầu ra. Lồng nhau không tính.
    collisions = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            bi, bj = preds[i]["bbox_norm"], preds[j]["bbox_norm"]
            inter = inter_area(bi, bj)
            if inter <= 0:
                continue
            smaller = min(area(bi), area(bj))
            if smaller <= 0 or inter / smaller <= args.collision_eps:
                continue
            if max(contain_ratio(bi, bj), contain_ratio(bj, bi)) >= args.nest_thr:
                continue
            collisions += 1

    # Tràn lề: so với content-box của TRANG NGUỒN (union box GT), nới ra một chút.
    if gts:
        cx0 = max(0.0, min(g["bbox_norm"][0] for g in gts) - args.margin_pad)
        cy0 = max(0.0, min(g["bbox_norm"][1] for g in gts) - args.margin_pad)
        cx1 = min(1.0, max(g["bbox_norm"][2] for g in gts) + args.margin_pad)
        cy1 = min(1.0, max(g["bbox_norm"][3] for g in gts) + args.margin_pad)
        content = [cx0, cy0, cx1, cy1]
        outside = sum(1 for p in preds
                      if 1.0 - contain_ratio(p["bbox_norm"], content) > args.margin_out)
    else:
        outside = 0

    # Thứ tự đọc: cùng một hàm sắp xếp cho cả hai phía (xem run_detectors).
    tau = kendall_tau([gts[i].get("reading_order", i) for i, _, _ in matched],
                      [preds[j].get("reading_order", j) for _, j, _ in matched])

    return {
        "n_gt": len(gts), "n_pred": len(preds), "n_matched": len(matched),
        "sum_iou": round(sum(ious), 6),
        "tp": {str(t): tp_at[t] for t in IOU_THRESHOLDS},
        "n_gt_anchor": len(a_gts), "n_pred_anchor": len(a_preds),
        "n_matched_anchor": len(a_matched), "sum_iou_anchor": round(sum(a_ious), 6),
        "n_contain": len(contain), "sum_contain": round(sum(contain), 6),
        "retention": retention,
        "n_collisions": collisions, "n_margin_out": outside,
        "tau": round(tau, 4) if tau is not None else None,
    }


# --------------------------------------------------------------------------- #
# Nạp dữ liệu                                                                   #
# --------------------------------------------------------------------------- #
def load_gt(corpus: Path, tiers: list[str]) -> dict[str, list[dict]]:
    """doc_id -> danh sách trang, mỗi trang có elements đã gắn `group` + thứ tự đọc."""
    docs: dict[str, list[dict]] = {}
    for tier in tiers:
        path = corpus / tier / "gt.json"
        if not path.exists():
            print(f"  [bỏ qua] {tier}: chưa có gt.json", flush=True)
            continue
        gt = json.loads(path.read_text(encoding="utf-8"))
        for doc in gt["docs"]:
            pages = []
            for page in doc["pages"]:
                elements = [{"class": e["class"],
                             "group": DOCLAYNET_GROUP.get(e["class"], "text"),
                             "bbox_norm": e["bbox_norm"]}
                            for e in page["elements"]]
                reading_order(elements)
                pages.append({"page": page["page"], "elements": elements})
            docs[doc["doc_id"]] = pages
    return docs


def load_pred(layout_dir: Path) -> dict[int, dict] | None:
    if not layout_dir.is_dir():
        return None
    pages = {}
    for path in sorted(layout_dir.glob("p*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        pages[int(rec["page"])] = rec
    return pages or None


def evaluate(system: str, lang: str | None, gt_docs: dict, layout_root: Path,
             out_root: Path, args: argparse.Namespace) -> list[dict]:
    records = []
    for doc_id, gt_pages in sorted(gt_docs.items()):
        base = (layout_root / system / doc_id if lang is None
                else layout_root / system / lang / doc_id)
        rec = {"system": system, "lang": lang, "doc_id": doc_id,
               "skipped": None, "pages": []}

        preds = load_pred(base)
        if preds is None:
            rec["skipped"] = "chưa có box detector"
            records.append(rec)
            continue

        # Cửa chặn số trang: metric neo theo TỪNG trang nguồn.
        if len(preds) != len(gt_pages):
            rec["skipped"] = (f"số trang lệch: nguồn {len(gt_pages)}, "
                              f"đầu ra {len(preds)} (reflow)")
            records.append(rec)
            continue

        for gt_page in gt_pages:
            pred_page = preds.get(gt_page["page"])
            if pred_page is None:
                continue
            rec["pages"].append(score_page(gt_page, pred_page, args))
        records.append(rec)
    return records


def summarize(records: list[dict]) -> dict:
    """Dồn số đếm thô rồi mới chia — trung bình của các tỉ lệ khác tỉ lệ của các tổng."""
    scored = [r for r in records if not r["skipped"]]
    pages = [p for r in scored for p in r["pages"]]
    if not pages:
        return {"n_docs": len(records), "n_docs_scored": 0,
                "n_docs_skipped": len(records), "n_pages": 0}

    n_gt = sum(p["n_gt"] for p in pages)
    n_pred = sum(p["n_pred"] for p in pages)
    n_matched = sum(p["n_matched"] for p in pages)
    sum_iou = sum(p["sum_iou"] for p in pages)

    ga = sum(p["n_gt_anchor"] for p in pages)
    pa = sum(p["n_pred_anchor"] for p in pages)
    ma = sum(p["n_matched_anchor"] for p in pages)
    sum_iou_a = sum(p["sum_iou_anchor"] for p in pages)

    f1 = {}
    for t in IOU_THRESHOLDS:
        tp = sum(p["tp"][str(t)] for p in pages)
        prec = tp / n_pred if n_pred else 0.0
        rec_ = tp / n_gt if n_gt else 0.0
        f1[str(t)] = round(2 * prec * rec_ / (prec + rec_), 6) if prec + rec_ else 0.0

    retention = defaultdict(lambda: [0, 0])
    for p in pages:
        for g, (a, b) in p["retention"].items():
            retention[g][0] += a
            retention[g][1] += b

    n_contain = sum(p["n_contain"] for p in pages)
    taus = [p["tau"] for p in pages if p["tau"] is not None]

    return {
        "n_docs": len(records), "n_docs_scored": len(scored),
        "n_docs_skipped": len(records) - len(scored), "n_pages": len(pages),
        "n_gt": n_gt, "n_pred": n_pred, "n_matched": n_matched,
        # mIoU trung bình trên cặp ĐÃ GHÉP — đây là con số so trực tiếp với BIoU
        # của paper. Nó bỏ qua box không ghép được, nên LUÔN đọc kèm F1.
        "mIoU": round(sum_iou / n_matched, 4) if n_matched else None,
        "F1@0.5": f1["0.5"],
        "mF1@[.5:.95]": round(sum(f1.values()) / len(f1), 4),
        "F1_by_threshold": f1,
        "anchor_mIoU": round(sum_iou_a / ma, 4) if ma else None,
        "anchor_n_gt": ga, "anchor_n_pred": pa, "anchor_n_matched": ma,
        "text_containment": (round(sum(p["sum_contain"] for p in pages) / n_contain, 4)
                             if n_contain else None),
        "element_retention": {g: round(v[1] / v[0], 4)
                              for g, v in sorted(retention.items()) if v[0]},
        "collisions_per_page": round(sum(p["n_collisions"] for p in pages) / len(pages), 4),
        "margin_violation_rate": (round(sum(p["n_margin_out"] for p in pages) / n_pred, 4)
                                  if n_pred else None),
        "reading_order_tau": round(sum(taus) / len(taus), 4) if taus else None,
    }


def main() -> int:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    layout_root = args.out / "_layout" / args.detector
    if not layout_root.is_dir():
        print(f"!! chưa có {layout_root} — chạy parse.run_detectors trước")
        return 1

    gt_docs = load_gt(args.corpus, tiers)
    if not gt_docs:
        print(f"!! không nạp được gt.json nào từ {args.corpus} (tiers {tiers})")
        return 1
    print(f">>> GT: {len(gt_docs)} tài liệu, "
          f"{sum(len(p) for p in gt_docs.values())} trang, "
          f"{sum(len(pg['elements']) for p in gt_docs.values() for pg in p)} box")

    systems = ([s.strip() for s in args.systems.split(",") if s.strip()]
               if args.systems else
               sorted(d.name for d in layout_root.iterdir()
                      if d.is_dir() and d.name != SOURCE_KEY))

    dest = args.out / "_metrics" / "layout"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []

    # Hàng chuẩn TRƯỚC: đọc mọi hàng khác tương đối so với nó.
    ceiling = evaluate(SOURCE_KEY, None, gt_docs, layout_root, args.out, args)
    summary = summarize(ceiling)
    summary["detector"] = args.detector
    (dest / f"source_ceiling.{args.detector}.json").write_text(
        json.dumps({"summary": summary, "records": ceiling}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    rows.append(("Source ceiling", "—", summary))

    for system in systems:
        for lang in langs:
            if not (layout_root / system / lang).is_dir():
                continue
            records = evaluate(system, lang, gt_docs, layout_root, args.out, args)
            s = summarize(records)
            s["detector"] = args.detector
            (dest / f"{system}.{lang}.{args.detector}.json").write_text(
                json.dumps({"summary": s, "records": records}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
            rows.append((system, lang, s))

    hdr = (f"{'system':22} {'lang':4} {'docs':>7} {'mIoU':>7} {'anchor':>7} "
           f"{'F1@.5':>7} {'mF1':>7} {'contain':>8} {'collis':>7} {'margin':>7} {'tau':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for system, lang, s in rows:
        def f(v, spec=".3f"):
            return format(v, spec) if isinstance(v, (int, float)) else "—"
        print(f"{system:22} {lang or '—':4} "
              f"{s.get('n_docs_scored', 0):>3}/{s.get('n_docs', 0):<3} "
              f"{f(s.get('mIoU')):>7} {f(s.get('anchor_mIoU')):>7} "
              f"{f(s.get('F1@0.5')):>7} {f(s.get('mF1@[.5:.95]')):>7} "
              f"{f(s.get('text_containment')):>8} "
              f"{f(s.get('collisions_per_page'), '.2f'):>7} "
              f"{f(s.get('margin_violation_rate')):>7} "
              f"{f(s.get('reading_order_tau'), '.2f'):>6}")

    print(f"\ndetector = {args.detector}   |   chi tiết: {dest}/")
    print("Đọc mIoU TƯƠNG ĐỐI so với hàng 'Source ceiling'. Doc bị loại vì lệch số "
          "trang được ghi trong trường 'skipped' của từng record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
