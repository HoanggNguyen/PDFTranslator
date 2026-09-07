"""Nhóm A-bis — layout preservation đo bằng bbox NATIVE (PyMuPDF), bỏ qua detector.

Bản đối chứng cho ``eval_preserve.py``: thay vì chạy Docling detector trên ảnh
render, module này đọc thẳng text layer của PDF đầu ra và so với GT DocLayNet
(người vẽ) của trang nguồn. Ba quyết định thiết kế:

1. **Matcher là ``component``/``coco`` của benchmark/parser, KHÔNG phải Hungarian
   như eval_preserve.** Lý do: native text layer không có nhãn ngữ nghĩa nên không
   ràng buộc nhóm được, và mức granularity (line/span) vốn phân mảnh hơn GT block
   — chính là bài toán many-to-one mà ``component_tp_by_threshold`` (union hai phía
   của thành phần liên thông rồi so IoU) được viết ra để xử lý. ``coco`` (greedy
   1-1) được báo kèm như "thước nghiêm": khe hở component-F1 trừ coco-F1 đo mức
   over/under-segmentation của từng pipeline.

2. **Class-agnostic hoàn toàn.** Không map nhãn, không Anchor-IoU, không Kendall
   tau — text layer không cho những thứ đó một cách đáng tin. Module này chỉ trả
   lời một câu: "vùng chữ nằm đúng chỗ không".

3. **``_source`` là trần định nghĩa-phân-đoạn, không phải trần detector.** Text
   layer của chính file nguồn so với GT: con số này đo GT block vs PDF-native
   line khác nhau bao nhiêu. Nếu _source chỉ đạt ~0.7 mF1 thì khoảng còn lại là
   do định nghĩa box, không phải do translator. ``identity`` phải khớp _source
   hệt nhau (cùng file) — lệch là lỗi harness.

Hạn chế đã biết (đọc kết quả phải nhớ): text bị clip/overflow vẫn có bbox "đẹp"
trong text layer; trang scanned không có text layer sẽ đóng góp 0 box. Metric này
vì thế LẠC QUAN hơn thực tế nhìn thấy — dùng nó để chặn dưới detector noise,
không dùng thay detector.

Ví dụ
-----
    python -m benchmark.e2e.metrics.eval_native \
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
        --tier T1 --lang vi --granularity line \
        --out-json out/_metrics/native_layout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.e2e.metrics.eval_preserve import load_gt
from benchmark.parser.evaluation.eval_layout import (
    IOU_THRESHOLDS,
    coco_tp_by_threshold,
    component_tp_by_threshold,
)

MATCHERS = ("coco", "component")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Thư mục out/ của một run (chứa <system>/<lang>/<doc>/output.pdf).")
    p.add_argument("--tier", default="T1")
    p.add_argument("--lang", default="vi")
    p.add_argument("--systems", default=None,
                   help="Mặc định: mọi thư mục con của out/ không bắt đầu bằng '_'.")
    p.add_argument("--granularity", choices=["block", "line", "span"], default="line",
                   help="Mức box trích từ text layer. 'line' là mặc định: span quá "
                        "phân mảnh, block tuỳ generator PDF quyết định.")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Trích box native                                                              #
# --------------------------------------------------------------------------- #
def native_boxes(pdf_path: Path, granularity: str) -> dict[int, list[dict]]:
    """page_no (0-based) -> list {'box': [x0,y0,x1,y1] đã chuẩn hoá [0,1]}."""
    import fitz  # PyMuPDF

    pages: dict[int, list[dict]] = {}
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc):
            w, h = page.rect.width, page.rect.height
            if not w or not h:
                pages[pno] = []
                continue
            boxes = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:                 # bỏ block ảnh
                    continue
                if granularity == "block":
                    txt = "".join(s.get("text", "")
                                  for ln in block.get("lines", [])
                                  for s in ln.get("spans", []))
                    if txt.strip():
                        boxes.append(_nb(block["bbox"], w, h))
                    continue
                for line in block.get("lines", []):
                    if granularity == "line":
                        txt = "".join(s.get("text", "") for s in line.get("spans", []))
                        if txt.strip():
                            boxes.append(_nb(line["bbox"], w, h))
                        continue
                    for span in line.get("spans", []):
                        if (span.get("text") or "").strip():
                            boxes.append(_nb(span["bbox"], w, h))
            pages[pno] = boxes
    return pages


def _nb(bbox, w: float, h: float) -> dict:
    return {"box": [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]}


# --------------------------------------------------------------------------- #
# Chấm                                                                          #
# --------------------------------------------------------------------------- #
def score_doc(gt_pages: list[dict], pred_pages: dict[int, list[dict]]) -> dict | None:
    """Dồn TP thô của cả doc. None nếu reflow (số trang lệch) — cùng cửa chặn
    với eval_preserve: metric neo theo từng trang nguồn."""
    if len(pred_pages) != len(gt_pages):
        return None
    tp = {m: {t: [0, 0] for t in IOU_THRESHOLDS} for m in MATCHERS}
    n_gt = n_pred = 0
    for gt_page in gt_pages:
        gts = [{"box": e["bbox_norm"]} for e in gt_page["elements"]]
        preds = pred_pages.get(gt_page["page"], [])   # gt 'page' 0-based = PyMuPDF pno
        n_gt += len(gts)
        n_pred += len(preds)
        co = coco_tp_by_threshold(gts, preds, IOU_THRESHOLDS)
        cm = component_tp_by_threshold(gts, preds, IOU_THRESHOLDS)
        for t in IOU_THRESHOLDS:
            tp["coco"][t][0] += co[t][0]
            tp["coco"][t][1] += co[t][1]
            tp["component"][t][0] += cm[t][0]
            tp["component"][t][1] += cm[t][1]
    return {"n_gt": n_gt, "n_pred": n_pred, "tp": tp}


def prf(n_tp_gt: int, n_tp_pred: int, n_gt: int, n_pred: int) -> tuple[float, float, float]:
    r = n_tp_gt / n_gt if n_gt else 0.0
    p = n_tp_pred / n_pred if n_pred else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def summarize(acc: dict) -> dict:
    out = {"n_gt": acc["n_gt"], "n_pred": acc["n_pred"], "localization": {}}
    for m in MATCHERS:
        p50, r50, f50 = prf(*acc["tp"][m][0.5], acc["n_gt"], acc["n_pred"])
        _, _, f75 = prf(*acc["tp"][m][0.75], acc["n_gt"], acc["n_pred"])
        mf = sum(prf(*acc["tp"][m][t], acc["n_gt"], acc["n_pred"])[2]
                 for t in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)
        out["localization"][m] = {
            "precision@0.5": round(p50, 4), "recall@0.5": round(r50, 4),
            "f1@0.5": round(f50, 4), "f1@0.75": round(f75, 4),
            "mF1@[.5:.95]": round(mf, 4),
        }
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    gt_docs = load_gt(args.corpus, [args.tier])
    if not gt_docs:
        print("!! không nạp được GT", file=sys.stderr)
        return 1

    if args.systems:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    else:
        systems = sorted(d.name for d in args.out.iterdir()
                         if d.is_dir() and not d.name.startswith("_")
                         and d.name != "report")
    # _source: text layer của chính PDF nguồn trong corpus — trần định nghĩa-phân-đoạn.
    systems = ["_source"] + [s for s in systems if s != "_source"]

    # system -> doc_id -> acc thô
    per_doc: dict[str, dict[str, dict]] = defaultdict(dict)
    skipped: dict[str, dict[str, str]] = defaultdict(dict)
    for system in systems:
        for doc_id, gt_pages in sorted(gt_docs.items()):
            pdf = (args.corpus / args.tier / f"{doc_id}.pdf" if system == "_source"
                   else args.out / system / args.lang / doc_id / "output.pdf")
            if not pdf.exists():
                skipped[system][doc_id] = "không có output.pdf"
                continue
            try:
                pred_pages = native_boxes(pdf, args.granularity)
            except Exception as exc:                        # PDF hỏng / không text layer
                skipped[system][doc_id] = f"lỗi đọc text layer: {exc}"
                continue
            acc = score_doc(gt_pages, pred_pages)
            if acc is None:
                skipped[system][doc_id] = (f"reflow: nguồn {len(gt_pages)} trang, "
                                           f"đầu ra {len(pred_pages)} trang")
                continue
            per_doc[system][doc_id] = acc

    # Dồn: per (system, doc) + per system (all docs)
    report = {"config": {"corpus": str(args.corpus), "out": str(args.out),
                         "tier": args.tier, "lang": args.lang,
                         "granularity": args.granularity,
                         "iou_thresholds": IOU_THRESHOLDS, "matchers": list(MATCHERS)},
              "systems": {}}
    for system in systems:
        docs = per_doc.get(system, {})
        total = {"n_gt": 0, "n_pred": 0,
                 "tp": {m: {t: [0, 0] for t in IOU_THRESHOLDS} for m in MATCHERS}}
        for acc in docs.values():
            total["n_gt"] += acc["n_gt"]
            total["n_pred"] += acc["n_pred"]
            for m in MATCHERS:
                for t in IOU_THRESHOLDS:
                    total["tp"][m][t][0] += acc["tp"][m][t][0]
                    total["tp"][m][t][1] += acc["tp"][m][t][1]
        report["systems"][system] = {
            "all": summarize(total) if docs else None,
            "per_doc": {d: summarize(a) for d, a in sorted(docs.items())},
            "skipped": skipped.get(system, {}),
        }

    # In bảng: per system × per domain, cả hai matcher
    hdr = (f"{'system':<18} {'domain':<26} {'gt':>5} {'pred':>5} "
           f"{'coco F1@.5':>10} {'coco mF1':>9} {'comp F1@.5':>10} {'comp mF1':>9}")
    print(hdr)
    print("-" * len(hdr))
    for system in systems:
        rep = report["systems"][system]
        for doc_id, s in rep["per_doc"].items():
            c, m = s["localization"]["coco"], s["localization"]["component"]
            print(f"{system:<18} {doc_id:<26} {s['n_gt']:>5} {s['n_pred']:>5} "
                  f"{c['f1@0.5']:>10.4f} {c['mF1@[.5:.95]']:>9.4f} "
                  f"{m['f1@0.5']:>10.4f} {m['mF1@[.5:.95]']:>9.4f}")
        if rep["all"]:
            c, m = rep["all"]["localization"]["coco"], rep["all"]["localization"]["component"]
            print(f"{system:<18} {'== TỔNG ==':<26} {rep['all']['n_gt']:>5} "
                  f"{rep['all']['n_pred']:>5} {c['f1@0.5']:>10.4f} "
                  f"{c['mF1@[.5:.95]']:>9.4f} {m['f1@0.5']:>10.4f} "
                  f"{m['mF1@[.5:.95]']:>9.4f}")
        for doc_id, why in rep["skipped"].items():
            print(f"{system:<18} {doc_id:<26} [bỏ qua] {why}")
        print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"[native] report -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
