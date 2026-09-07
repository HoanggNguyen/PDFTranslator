"""Nhóm B — bảo toàn pixel và overflow gây hại, **không cần detector**.

Vì sao đáng làm dù đã có nhóm A: nhóm A phụ thuộc một mô hình học máy (detector),
nên ai cũng có quyền hỏi "kết quả có phải do detector không". Nhóm B không có mô
hình nào cả — chỉ có pixel. **Hai trục độc lập mà đồng thuận thì kết luận rất mạnh.**

Bốn metric headline, đều đo trên PDF đầu ra thật:

* **NT-PPR** (non-text pixel preservation): tỷ lệ pixel không đổi bên ngoài các
  vùng chữ GT được phép dịch.
* **IO-PPR** (immutable-object pixel preservation): cùng phép đo nhưng chỉ trong
  Picture và Formula. Không gộp Table hay header/footer vì nội dung chữ ở đó có
  thể được dịch hợp lệ.
* **OF-harm**: tỷ lệ dòng chữ thật trích từ PDF đầu ra vừa tràn khỏi GT owner, vừa
  đi vào element khác, và vùng giao có pixel thật sự thay đổi.
* **Page-fail rate**: tỷ lệ trang vi phạm ít nhất một ngưỡng PPR/OF-harm. Đây là
  metric đuôi phân phối để vài trang hỏng nặng không biến mất trong trung bình.

Masked-SSIM, ink-profile và full-page SSIM vẫn được lưu làm chẩn đoán, nhưng không
làm headline. Full-page SSIM bị nhiễu bởi việc glyph bắt buộc thay đổi khi dịch.

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

# Mọi vùng chữ được phép thay đổi. Header/footer cũng là chữ; để chúng trong mask
# bảo toàn sẽ phạt một bản dịch đúng. Table được xử như owner cho OF-harm, nhưng
# không che toàn bộ bảng khỏi NT-PPR vì đường kẻ và nền bảng vẫn phải được giữ.
TEXT_CLASSES = {
    "Text", "Title", "Section-header", "List-item", "Caption", "Footnote",
    "Page-header", "Page-footer",
}
OWNER_CLASSES = TEXT_CLASSES | {"Table"}
IMMUTABLE_CLASSES = {"Picture", "Formula"}


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
    p.add_argument("--pixel-tolerance", type=int, default=8,
                   help="Pixel được coi là giữ nguyên khi lệch grayscale không "
                        "quá giá trị này trên thang 0..255.")
    p.add_argument("--harm-overlap", type=float, default=0.05,
                   help="Phần bbox dòng chữ phải tràn vào element khác để xét harm.")
    p.add_argument("--harm-change", type=float, default=0.05,
                   help="Phần pixel trong vùng giao phải thật sự đổi để xét harm.")
    p.add_argument("--page-fail-ppr", type=float, default=0.95,
                   help="Trang fail nếu NT-PPR hoặc IO-PPR thấp hơn ngưỡng này.")
    p.add_argument("--page-fail-harm", type=float, default=0.05,
                   help="Trang fail nếu OF-harm cao hơn ngưỡng này.")
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


def immutable_mask(shape: tuple[int, int], elements: list[dict], text):
    """Pixel của Picture/Formula, trừ phần giao với vùng chữ được phép dịch."""
    import numpy as np

    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for e in elements:
        if e["class"] not in IMMUTABLE_CLASSES:
            continue
        x0, y0, x1, y1 = e["bbox_norm"]
        i0, i1 = max(0, int(y0 * h)), min(h, int(y1 * h + 0.999999))
        j0, j1 = max(0, int(x0 * w)), min(w, int(x1 * w + 0.999999))
        if i1 > i0 and j1 > j0:
            mask[i0:i1, j0:j1] = True
    return mask & ~text


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter_box(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), min(a[3], b[3])]


def extract_text_lines(pdf_path: Path) -> dict[int, list[list[float]]]:
    """Đọc bbox dòng chữ *nhìn thấy* từ PDF; không dùng layout detector.

    Một số pipeline giữ text nguồn ở render-mode 3 / alpha=0 để phục vụ tìm kiếm.
    PyMuPDF vẫn trả những span đó trong ``get_text('dict')``; gộp bbox dòng thô sẽ
    biến text ẩn thành overflow giả. Vì vậy bbox được dựng lại chỉ từ span alpha>0.
    """
    import fitz

    pages: dict[int, list[list[float]]] = {}
    with fitz.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf):
            w, h = page.rect.width, page.rect.height
            lines = []
            for block in page.get_text("dict", sort=True).get("blocks", []):
                for line in block.get("lines", []):
                    visible = [s for s in line.get("spans", [])
                               if s.get("alpha", 255) > 0 and s.get("text", "").strip()]
                    if not visible:
                        continue
                    x0 = min(s["bbox"][0] for s in visible)
                    y0 = min(s["bbox"][1] for s in visible)
                    x1 = max(s["bbox"][2] for s in visible)
                    y1 = max(s["bbox"][3] for s in visible)
                    if w > 0 and h > 0 and x1 > x0 and y1 > y0:
                        lines.append([x0 / w, y0 / h, x1 / w, y1 / h])
            pages[page_no] = lines
    return pages


def harmful_overflow(lines, elements, changed, min_overlap: float,
                     min_change: float) -> tuple[int, int]:
    """Đếm dòng overflow có thay đổi pixel trong element bị xâm lấn.

    Owner là GT text/table chứa phần lớn dòng. Chỉ phần nằm *ngoài owner* mới
    được xét; nhờ vậy caption hợp lệ nằm trong vùng giao Picture không tự thành
    lỗi. Kiểm tra pixel loại các giao bbox hình thức không tạo hỏng thật.
    """
    import numpy as np

    h, w = changed.shape
    owners = [e for e in elements if e["class"] in OWNER_CLASSES]
    targets = [e for e in elements
               if e["class"] in OWNER_CLASSES | IMMUTABLE_CLASSES]
    harmful = assigned = 0
    for line in lines:
        line_area = _area(line)
        if line_area <= 0 or not owners:
            continue
        contain = [_area(_inter_box(line, e["bbox_norm"])) / line_area
                   for e in owners]
        if not contain or max(contain) < 0.30:
            continue
        # Nếu Table và một text box cùng chứa dòng, ưu tiên box nhỏ hơn.
        oi = max(range(len(owners)),
                 key=lambda i: (contain[i], -_area(owners[i]["bbox_norm"])))
        owner = owners[oi]
        owner_box = owner["bbox_norm"]
        assigned += 1
        for target in targets:
            if target is owner:
                continue
            overlap = _inter_box(line, target["bbox_norm"])
            if _area(overlap) / line_area <= min_overlap:
                continue
            x0, y0 = max(0, int(overlap[0] * w)), max(0, int(overlap[1] * h))
            x1 = min(w, int(overlap[2] * w + 0.999999))
            y1 = min(h, int(overlap[3] * h + 0.999999))
            if x1 <= x0 or y1 <= y0:
                continue
            allowed = np.ones((y1 - y0, x1 - x0), dtype=bool)
            ox0, oy0 = max(x0, int(owner_box[0] * w)), max(y0, int(owner_box[1] * h))
            ox1 = min(x1, int(owner_box[2] * w + 0.999999))
            oy1 = min(y1, int(owner_box[3] * h + 0.999999))
            if ox1 > ox0 and oy1 > oy0:
                allowed[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = False
            n = int(allowed.sum())
            if n >= 4 and int((changed[y0:y1, x0:x1] & allowed).sum()) / n > min_change:
                harmful += 1
                break
    return harmful, assigned


def load_gray(path: Path, size: tuple[int, int] | None = None):
    """Ảnh xám float [0,1]. `size` = (w, h) để ép về khổ trang nguồn."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None and im.size != size:
            im = im.resize(size, Image.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0


def load_rgb(path: Path, size: tuple[int, int] | None = None):
    """Ảnh RGB float [0,1] cho PPR; giữ màu thay vì chỉ so luminance."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
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


def score_page(src_png: Path, dst_png: Path, elements: list[dict], lines,
               args: argparse.Namespace) -> dict:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    with Image.open(src_png) as im:
        src_size = im.size                      # (w, h)
    src = load_gray(src_png)
    dst = load_gray(dst_png, size=src_size)     # ép về khổ nguồn nếu lệch
    src_rgb = load_rgb(src_png)
    dst_rgb = load_rgb(dst_png, size=src_size)
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

    changed = np.max(np.abs(src_rgb - dst_rgb), axis=2) > args.pixel_tolerance / 255.0
    nt_ppr = float((~changed)[keep].mean()) if n_keep else None
    immutable = immutable_mask(src.shape, elements, mask) & border
    n_immutable = int(immutable.sum())
    io_ppr = float((~changed)[immutable].mean()) if n_immutable else None
    n_harm, n_lines = harmful_overflow(
        lines, elements, changed, args.harm_overlap, args.harm_change)
    of_harm = n_harm / n_lines if n_lines else None
    page_fail = bool(
        (nt_ppr is not None and nt_ppr < args.page_fail_ppr)
        or (io_ppr is not None and io_ppr < args.page_fail_ppr)
        or (of_harm is not None and of_harm > args.page_fail_harm)
    )

    return {"ssim_full": round(full, 6),
            "ssim_masked": round(masked, 6) if masked is not None else None,
            "nt_ppr": round(nt_ppr, 6) if nt_ppr is not None else None,
            "io_ppr": round(io_ppr, 6) if io_ppr is not None else None,
            "of_harm": round(of_harm, 6) if of_harm is not None else None,
            "page_fail": int(page_fail),
            "n_harmful_lines": n_harm,
            "n_assigned_lines": n_lines,
            "n_pixels_kept": n_keep,
            "n_immutable_pixels": n_immutable,
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

        pdf_path = args.out / system / lang / doc_id / "output.pdf"
        try:
            lines_by_page = extract_text_lines(pdf_path) if pdf_path.is_file() else {}
        except Exception as exc:  # pixel metric vẫn chạy nếu text layer hỏng
            lines_by_page = {}
            rec["text_layer_error"] = f"{type(exc).__name__}: {exc}"

        for src_png, dst_png in zip(src_pages, dst_pages):
            page = int(src_png.stem[1:])
            try:
                rec["pages"].append({
                    "page": page,
                    **score_page(src_png, dst_png, gt_pages.get(page, []),
                                 lines_by_page.get(page, []), args)})
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
        # Tín hiệu headline, độc lập detector.
        "nt_ppr": mean("nt_ppr"),
        "io_ppr": mean("io_ppr"),
        "of_harm": mean("of_harm"),
        "page_fail_rate": mean("page_fail"),
        # Chẩn đoán phụ.
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
    config = {
        "dilate_px": args.dilate_px,
        "ssim_win": args.ssim_win,
        "pixel_tolerance_255": args.pixel_tolerance,
        "ppr_color_rule": "max absolute sRGB channel difference <= tolerance",
        "harm_overlap": args.harm_overlap,
        "harm_change": args.harm_change,
        "page_fail_ppr": args.page_fail_ppr,
        "page_fail_harm": args.page_fail_harm,
        "immutable_classes": sorted(IMMUTABLE_CLASSES),
    }
    rows = []
    for system in systems:
        for lang in langs:
            if not (render_root / system / lang).is_dir():
                continue
            print(f">>> {system}/{lang}", flush=True)
            records = evaluate(system, lang, gt_docs, render_root, args)
            s = summarize(records)
            (dest / f"{system}.{lang}.json").write_text(
                json.dumps({"config": config, "summary": s, "records": records}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
            rows.append((system, lang, s))

    if not rows:
        print(f"!! không thấy ảnh render của hệ nào dưới {render_root}/")
        return 1

    hdr = (f"{'system':22} {'lang':4} {'docs':>7} {'NT-PPR':>8} "
           f"{'IO-PPR':>8} {'OF-harm':>8} {'page-fail':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for system, lang, s in rows:
        def f(v, spec=".4f"):
            return format(v, spec) if isinstance(v, (int, float)) else "—"
        print(f"{system:22} {lang:4} {s['n_docs_scored']:>3}/{s['n_docs']:<3} "
              f"{f(s['nt_ppr']):>8} {f(s['io_ppr']):>8} "
              f"{f(s['of_harm']):>8} {f(s['page_fail_rate'], '.1%'):>10}")
    print(f"\nchi tiết: {dest}/")
    print("NT-PPR/IO-PPR/OF-harm/Page-fail là headline không cần detector; "
          "Masked/full SSIM và ink-profile chỉ là chẩn đoán phụ trong JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
