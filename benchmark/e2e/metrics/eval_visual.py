"""Nhóm B — trục pixel, **không cần detector**.

Vì sao đáng làm dù đã có nhóm A: nhóm A phụ thuộc một mô hình học máy (detector),
nên ai cũng có quyền hỏi "kết quả có phải do detector không". Nhóm B không có mô
hình nào cả — chỉ có pixel. **Hai trục độc lập mà đồng thuận thì kết luận rất mạnh.**

Ba metric, vai trò khác hẳn nhau:

* **Masked-SSIM ⭐ — tín hiệu chính.** Che toàn bộ vùng text theo GT người vẽ (nới
  2 px) rồi so phần còn lại. Phần còn lại là hình, bảng, đường kẻ, logo — những thứ
  **phải đứng nguyên từng pixel** dù dịch sang ngôn ngữ nào. Gần như không mơ hồ:
  1.0 nghĩa là mọi thứ không phải chữ đều nằm y nguyên chỗ cũ.
* **Ink-profile distance** — chiếu mật độ mực lên trục ngang và trục dọc rồi tính
  khoảng cách Wasserstein 1D. Rẻ, và **không phụ thuộc hệ chữ viết**: nó đo "mực
  phân bố ở đâu trên trang", không đo "chữ trông thế nào".
* **Full-page SSIM** — báo cáo **chỉ để chứng minh nó là metric tồi ở đây.** Glyph
  tiếng Việt khác glyph tiếng Anh nên SSIM toàn trang sụp bất kể layout tốt hay
  xấu; ai dùng nó làm metric chính là đang đo sự khác biệt của bảng chữ cái. Một
  đoạn phân tích ngắn, có ích cho luận văn.

Hai chi tiết kỹ thuật dễ làm sai:

1. **Trang đầu ra khác khổ trang nguồn thì phải scale, và phải ghi lại là đã scale.**
   Không scale thì SSIM báo lỗi shape; scale mà im lặng thì một hệ đổi khổ giấy sẽ
   trông như không có chuyện gì.
2. **Mặt nạ lấy từ GT của trang NGUỒN**, không phải từ box detector trên trang đích.
   Nếu lấy theo đích thì hệ nào làm chữ tràn ra ngoài sẽ tự che luôn phần nó làm
   hỏng — tự chấm điểm cho mình.

Ví dụ
-----
    python -m benchmark.e2e.metrics.eval_visual \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_KEY = "_source"

# Nhóm bị che khi tính Masked-SSIM: mọi thứ là CHỮ. Phần còn lại là thứ phải đứng yên.
TEXT_CLASSES = {"Text", "Title", "Section-header", "List-item", "Caption", "Footnote"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None)
    p.add_argument("--dilate-px", type=int, default=2,
                   help="Nới mặt nạ text ra ngần này pixel. Box GT bám sát chữ nên "
                        "không nới thì viền glyph vẫn lọt vào phần được chấm.")
    p.add_argument("--ssim-win", type=int, default=7,
                   help="Cửa sổ SSIM (lẻ).")
    return p.parse_args()


def text_mask(shape: tuple[int, int], elements: list[dict], dilate: int):
    """True = pixel THUỘC vùng chữ (sẽ bị loại khỏi Masked-SSIM)."""
    import numpy as np

    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for e in elements:
        if e["class"] not in TEXT_CLASSES:
            continue
        x0, y0, x1, y1 = e["bbox_norm"]
        i0 = max(0, int(y0 * h) - dilate)
        i1 = min(h, int(y1 * h) + dilate)
        j0 = max(0, int(x0 * w) - dilate)
        j1 = min(w, int(x1 * w) + dilate)
        if i1 > i0 and j1 > j0:
            mask[i0:i1, j0:j1] = True
    return mask


def load_gray(path: Path, size: tuple[int, int] | None = None):
    """Ảnh xám float [0,1]. `size` = (w, h) để ép về khổ trang nguồn."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None and im.size != size:
            im = im.resize(size, Image.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0


def ink_profile_distance(src, dst) -> dict:
    """Wasserstein-1 giữa hai phân bố mật độ mực chiếu lên từng trục.

    Mực = 1 − độ sáng. Chuẩn hoá thành phân bố xác suất rồi lấy khoảng cách giữa
    hai hàm phân phối tích luỹ, chia cho chiều dài trục ⇒ số nằm trong [0,1] và so
    được giữa các trang khác khổ.
    """
    import numpy as np

    def one_axis(a, b, axis):
        pa = (1.0 - a).sum(axis=axis)
        pb = (1.0 - b).sum(axis=axis)
        sa, sb = pa.sum(), pb.sum()
        if sa <= 0 or sb <= 0:
            return None
        ca = np.cumsum(pa / sa)
        cb = np.cumsum(pb / sb)
        return float(np.abs(ca - cb).sum() / len(ca))

    dx = one_axis(src, dst, 0)      # chiếu lên trục ngang
    dy = one_axis(src, dst, 1)      # chiếu lên trục dọc
    vals = [v for v in (dx, dy) if v is not None]
    return {"ink_x": round(dx, 6) if dx is not None else None,
            "ink_y": round(dy, 6) if dy is not None else None,
            "ink_mean": round(sum(vals) / len(vals), 6) if vals else None}


def score_page(src_png: Path, dst_png: Path, elements: list[dict],
               args: argparse.Namespace) -> dict:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    with Image.open(src_png) as im:
        src_size = im.size                      # (w, h)
    src = load_gray(src_png)
    dst = load_gray(dst_png, size=src_size)     # ép về khổ nguồn nếu lệch
    with Image.open(dst_png) as im:
        resized = im.size != src_size

    full = float(ssim(src, dst, data_range=1.0, win_size=args.ssim_win))

    # Masked-SSIM: SSIM cục bộ theo từng pixel, rồi chỉ lấy trung bình ở NGOÀI vùng
    # chữ. Không thể xoá pixel rồi mới tính — SSIM cần lân cận liên tục; xoá tạo ra
    # cạnh giả và điểm sẽ sai.
    _, ssim_map = ssim(src, dst, data_range=1.0, win_size=args.ssim_win, full=True)
    mask = text_mask(src.shape, elements, args.dilate_px)
    keep = ~mask
    # Bỏ viền: SSIM không xác định trong nửa cửa sổ ở rìa ảnh.
    pad = args.ssim_win // 2
    border = np.zeros_like(keep)
    border[pad:-pad or None, pad:-pad or None] = True
    keep = keep & border

    n_keep = int(keep.sum())
    masked = float(ssim_map[keep].mean()) if n_keep else None

    return {"ssim_full": round(full, 6),
            "ssim_masked": round(masked, 6) if masked is not None else None,
            "n_pixels_kept": n_keep,
            "frac_pixels_kept": round(n_keep / keep.size, 4),
            "resized": resized,
            **ink_profile_distance(src, dst)}


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


def evaluate(system: str, lang: str, gt_docs: dict, render_root: Path,
             args: argparse.Namespace) -> list[dict]:
    records = []
    for doc_id, gt_pages in sorted(gt_docs.items()):
        src_dir = render_root / SOURCE_KEY / doc_id
        dst_dir = render_root / system / lang / doc_id
        rec = {"system": system, "lang": lang, "doc_id": doc_id,
               "skipped": None, "pages": []}

        if not dst_dir.is_dir():
            rec["skipped"] = "chưa render đầu ra"
            records.append(rec)
            continue
        src_pages = sorted(src_dir.glob("p*.png"))
        dst_pages = sorted(dst_dir.glob("p*.png"))
        if len(src_pages) != len(dst_pages):
            rec["skipped"] = (f"số trang lệch: nguồn {len(src_pages)}, "
                              f"đầu ra {len(dst_pages)} (reflow)")
            records.append(rec)
            continue

        for src_png, dst_png in zip(src_pages, dst_pages):
            page = int(src_png.stem[1:])
            try:
                rec["pages"].append({
                    "page": page,
                    **score_page(src_png, dst_png, gt_pages.get(page, []), args)})
            except Exception as exc:  # noqa: BLE001 — một trang hỏng không giết cả lượt
                rec["pages"].append({"page": page,
                                     "error": f"{type(exc).__name__}: {exc}"})
        records.append(rec)
    return records


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if not r["skipped"]]
    pages = [p for r in scored for p in r["pages"] if "error" not in p]

    def mean(key):
        vals = [p[key] for p in pages if p.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_docs": len(records), "n_docs_scored": len(scored),
        "n_docs_skipped": len(records) - len(scored), "n_pages": len(pages),
        "n_pages_error": sum(1 for r in scored for p in r["pages"] if "error" in p),
        # Tín hiệu chính.
        "masked_ssim": mean("ssim_masked"),
        "ink_distance": mean("ink_mean"),
        # Chỉ để chỉ ra nó vô dụng ở đây — đừng xếp hạng bằng cột này.
        "full_ssim": mean("ssim_full"),
        "frac_pixels_kept": mean("frac_pixels_kept"),
        "n_pages_resized": sum(1 for p in pages if p.get("resized")),
    }


def main() -> int:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    render_root = args.out / "_render"
    if not render_root.is_dir():
        print(f"!! chưa có {render_root} — chạy parse.render_pages trước")
        return 1

    gt_docs = load_gt_pages(args.corpus, tiers)
    if not gt_docs:
        print(f"!! không nạp được gt.json nào từ {args.corpus}")
        return 1

    systems = ([s.strip() for s in args.systems.split(",") if s.strip()]
               if args.systems else
               sorted(d.name for d in render_root.iterdir()
                      if d.is_dir() and d.name != SOURCE_KEY))

    dest = args.out / "_metrics" / "visual"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for system in systems:
        for lang in langs:
            if not (render_root / system / lang).is_dir():
                continue
            print(f">>> {system}/{lang}", flush=True)
            records = evaluate(system, lang, gt_docs, render_root, args)
            s = summarize(records)
            (dest / f"{system}.{lang}.json").write_text(
                json.dumps({"summary": s, "records": records}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
            rows.append((system, lang, s))

    if not rows:
        print(f"!! không thấy ảnh render của hệ nào dưới {render_root}/")
        return 1

    hdr = (f"{'system':22} {'lang':4} {'docs':>7} {'masked-SSIM':>12} "
           f"{'ink-dist':>9} {'full-SSIM':>10} {'%pixel chấm':>12}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for system, lang, s in rows:
        def f(v, spec=".4f"):
            return format(v, spec) if isinstance(v, (int, float)) else "—"
        print(f"{system:22} {lang:4} {s['n_docs_scored']:>3}/{s['n_docs']:<3} "
              f"{f(s['masked_ssim']):>12} {f(s['ink_distance']):>9} "
              f"{f(s['full_ssim']):>10} "
              f"{f(s['frac_pixels_kept'], '.1%'):>12}")
    print(f"\nchi tiết: {dest}/")
    print("Xếp hạng bằng masked-SSIM. Cột full-SSIM ở đây CHỈ để chứng minh nó là "
          "metric tồi khi đổi hệ chữ viết — đừng dùng nó để kết luận.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
