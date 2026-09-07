"""Nhóm B-bis — IC-harm (ink collision), phiên bản pixel của "chữ chồng chữ".

Động lực: OF-harm chỉ bắt dòng tràn *khỏi owner sang element khác* và cần ≥30%
containment để gán owner — nó mù với chồng lấn *bên trong* cùng một vùng text
(dòng đè dòng trong một paragraph), và mù hoàn toàn với_va chạm nhìn thấy được
mà bbox vẫn nằm trong owner. Metric này nhìn thẳng vào mực:

* Binarize mực của trang render (ngưỡng luminance), giới hạn trong vùng text GT
  (mặt nạ từ nguồn — cùng nguyên tắc anchor-theo-nguồn của eval_visual).
* Connected-component trên mực đã giãn nhẹ: chữ chồng chữ làm các dòng chảy vào
  nhau thành blob CAO bất thường (≈ 2 lần chiều cao chữ).
* ``frac_tall`` = tỉ lệ pixel mực nằm trong component cao hơn ``h_ratio`` lần
  chiều cao chữ trung vị CỦA TRANG NGUỒN (neo theo nguồn để hệ phóng to font
  hợp lệ không bị phạt oan).
* **IC-harm = max(0, frac_tall(dst) − frac_tall(src))** — phần *vượt mức* so với
  chính trang nguồn. Nhờ vậy ``identity`` = 0 bằng construction, và các blob cao
  có sẵn ở nguồn (bảng, công thức lọt vào vùng text) không bị tính oan.

Không dùng detector, không dùng text layer — chỉ pixel. Hàng ``identity`` phải
là 0; khác 0 là lỗi harness.

Ví dụ
-----
    python -m benchmark.e2e.metrics.eval_ink \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.e2e.metrics.eval_visual import SOURCE_KEY, TEXT_CLASSES, text_mask

RENDER_KEY = "_render"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--ink-thr", type=float, default=0.5,
                   help="Pixel là mực khi luminance < ngưỡng này [0,1].")
    p.add_argument("--dilate-px", type=int, default=2,
                   help="Giãn mực ngần này pixel trước khi tìm component để chữ "
                        "cùng từ dính vào nhau (mặc định theo DPI render ~150).")
    p.add_argument("--h-ratio", type=float, default=1.8,
                   help="Component cao hơn h_ratio lần chiều cao chữ trung vị "
                        "của trang NGUỒN được coi là blob chồng lấn.")
    p.add_argument("--min-comp-px", type=int, default=6,
                   help="Bỏ component nhỏ hơn (nhiễu anti-alias).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Lõi pixel                                                                     #
# --------------------------------------------------------------------------- #
def load_gray(path: Path, size: tuple[int, int] | None = None):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None and im.size != size:
            im = im.resize(size, Image.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0


def tall_ink_fraction(gray, mask, ink_thr: float, dilate_px: int,
                      h_ratio: float, min_comp_px: int,
                      ref_height: float | None = None):
    """Trả (frac_tall, median_comp_height). frac_tall = phần mực nằm trong
    component cao hơn h_ratio × ref_height (mặc định: trung vị chính trang)."""
    import numpy as np
    from scipy import ndimage

    ink = (gray < ink_thr) & mask
    if not ink.any():
        return None, None
    if dilate_px > 0:
        ink = ndimage.binary_dilation(ink, iterations=dilate_px)
    labels, n = ndimage.label(ink)
    if n == 0:
        return None, None
    slices = ndimage.find_objects(labels)
    heights, tall_ids = [], []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp = labels[sl] == i
        if int(comp.sum()) < min_comp_px:
            continue
        h = sl[0].stop - sl[0].start
        heights.append(h)
    if not heights:
        return None, None
    heights = np.asarray(heights)
    med = float(np.median(heights))
    anchor = ref_height if ref_height is not None else med
    tall = heights > h_ratio * anchor
    if not tall.any():
        return 0.0, med
    # Đếm pixel mực (trước giãn) trong các component cao.
    ink_raw = (gray < ink_thr) & mask
    tall_mask = np.isin(labels, [i + 1 for i, t in enumerate(tall) if t]) & ink_raw
    total = int(ink_raw.sum())
    return (float(tall_mask.sum()) / total if total else None), med


# --------------------------------------------------------------------------- #
def load_gt_pages(corpus: Path, tiers: list[str]) -> dict[str, dict[int, list[dict]]]:
    docs: dict[str, dict[int, list[dict]]] = {}
    for tier in tiers:
        path = corpus / tier / "gt.json"
        if not path.exists():
            continue
        gt = json.loads(path.read_text(encoding="utf-8"))
        for doc in gt["docs"]:
            docs[doc["doc_id"]] = {p["page"]: p["elements"] for p in doc["pages"]}
    return docs


def evaluate(system: str, lang: str | None, gt_docs: dict, render_root: Path,
             args: argparse.Namespace) -> list[dict]:
    records = []
    for doc_id, gt_pages in sorted(gt_docs.items()):
        base = (render_root / system / doc_id if lang is None
                else render_root / system / lang / doc_id)
        rec = {"system": system, "lang": lang, "doc_id": doc_id,
               "skipped": None, "pages": []}
        pngs = sorted(base.glob("p*.png"))
        if not pngs:
            rec["skipped"] = "chưa có ảnh render"
            records.append(rec)
            continue
        if len(pngs) != len(gt_pages):
            rec["skipped"] = (f"số trang lệch: nguồn {len(gt_pages)}, "
                              f"đầu ra {len(pngs)} (reflow)")
            records.append(rec)
            continue
        src_dir = render_root / SOURCE_KEY / doc_id
        for page_no, elements in sorted(gt_pages.items()):
            src_png = src_dir / f"p{page_no:03d}.png"
            dst_png = base / f"p{page_no:03d}.png"
            if not src_png.exists() or not dst_png.exists():
                continue
            src = load_gray(src_png)
            dst = load_gray(dst_png, size=(src.shape[1], src.shape[0]))
            mask = text_mask(src.shape, elements, args.dilate_px)
            frac_src, h_src = tall_ink_fraction(
                src, mask, args.ink_thr, args.dilate_px,
                args.h_ratio, args.min_comp_px)
            frac_dst, _ = tall_ink_fraction(
                dst, mask, args.ink_thr, args.dilate_px,
                args.h_ratio, args.min_comp_px, ref_height=h_src)
            if frac_src is None or frac_dst is None:
                continue
            rec["pages"].append({
                "page": page_no,
                "ic_harm": round(max(0.0, frac_dst - frac_src), 6),
                "tall_src": round(frac_src, 6), "tall_dst": round(frac_dst, 6),
                "src_glyph_h": round(h_src, 2) if h_src else None,
            })
        records.append(rec)
    return records


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if not r["skipped"]]
    pages = [p for r in scored for p in r["pages"]]
    return {
        "n_docs": len(records), "n_docs_scored": len(scored),
        "n_docs_skipped": len(records) - len(scored),
        "n_pages": len(pages),
        "ic_harm_mean": (round(sum(p["ic_harm"] for p in pages) / len(pages), 6)
                         if pages else None),
    }


def main() -> int:
    args = parse_args()
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    gt_docs = load_gt_pages(args.corpus, tiers)
    if not gt_docs:
        print("!! không nạp được GT", file=sys.stderr)
        return 1

    render_root = args.out / RENDER_KEY
    dest = args.out / "_metrics" / "ink"
    dest.mkdir(parents=True, exist_ok=True)

    # Mọi hệ có render + identity (copy nguồn) làm đối chứng; _source là mask gốc.
    systems = sorted({d.name for d in render_root.iterdir()
                      if d.is_dir() and d.name not in (SOURCE_KEY,)})
    for system in systems:
        for lang in langs:
            records = evaluate(system, lang, gt_docs, render_root, args)
            name = f"{system}.{lang}.json"
            (dest / name).write_text(
                json.dumps({"config": vars(args) | {"corpus": str(args.corpus),
                                                    "out": str(args.out)},
                            "summary": summarize(records), "records": records},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            s = summarize(records)
            print(f"[ink] {name:36s} docs={s['n_docs_scored']}/{s['n_docs']} "
                  f"pages={s['n_pages']:4d} ic_harm={s['ic_harm_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
