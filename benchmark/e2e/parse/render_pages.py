"""PDF -> PNG 150 DPI cho trang nguồn và output của mọi hệ.

Một bước riêng vì ảnh này có **hai** người dùng: detector (`run_detectors`) và
Masked-SSIM (`metrics/eval_visual`). Render trong từng module là render hai lần cho
cùng một trang, và tệ hơn là hai module có thể lỡ dùng DPI khác nhau — lúc đó box
của detector và pixel của SSIM không còn nằm trong cùng một hệ toạ độ.

Ba lựa chọn cố định, đừng đổi giữa chừng:

* **150 DPI.** Đủ để RT-DETR nhìn rõ chữ nhỏ trong bảng, mà một trang A4 vẫn chỉ
  ~1240×1754 px. 300 DPI làm ảnh nặng gấp 4 và không cải thiện box.
* **Không alpha, nền trắng.** PDF trang trắng để alpha=0 thì SSIM đọc thành đen và
  mọi so sánh sai lệch hệ thống.
* **Tên file theo chỉ số trang 3 chữ số** (`p000.png`), không theo nhãn trang in
  trên giấy. Ghép trang nguồn với trang đích là ghép theo *chỉ số*.

Memo hoá theo ``sha256`` của PDF: chạy lại sau khi thêm một hệ mới thì các hệ cũ
không bị render lại. Xoá cả ``_render/`` là an toàn — nó sinh lại được.

Ví dụ
-----
    python -m benchmark.e2e.parse.render_pages \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out --tiers T1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DPI = 150
SOURCE_KEY = "_source"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None,
                   help="Mặc định: mọi thư mục hệ tìm thấy dưới <out>/. Nhờ vậy "
                        "thêm một hệ mới không phải sửa lệnh.")
    p.add_argument("--dpi", type=int, default=DPI,
                   help=f"Đổi là mọi số visual cũ hết so được (default {DPI}).")
    p.add_argument("--force", action="store_true",
                   help="Render lại kể cả khi sha256 không đổi.")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(pdf: Path, dest: Path, dpi: int, force: bool) -> dict:
    """Render mọi trang. Trả về stamp; bỏ qua nếu stamp cũ còn khớp."""
    import fitz  # PyMuPDF

    stamp_path = dest / "stamp.json"
    digest = sha256(pdf)
    if not force and stamp_path.exists():
        try:
            old = json.loads(stamp_path.read_text(encoding="utf-8"))
            if old.get("sha256") == digest and old.get("dpi") == dpi:
                n = len(list(dest.glob("p*.png")))
                if n == old.get("n_pages"):
                    return {**old, "cached": True}
        except Exception:  # noqa: BLE001 — stamp hỏng thì render lại, không phải lỗi
            pass

    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("p*.png"):
        stale.unlink()

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    sizes = []
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc):
            # alpha=False ⇒ nền trắng thật. Trang trong suốt mà để alpha sẽ thành
            # đen khi đổ xuống mảng numpy, làm SSIM sai lệch hệ thống.
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(dest / f"p{i:03d}.png")
            sizes.append([pix.width, pix.height])

    stamp = {"sha256": digest, "dpi": dpi, "n_pages": len(sizes), "sizes": sizes,
             "src": pdf.name, "cached": False}
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    return stamp


def discover_docs(corpus: Path, tiers: list[str]) -> list[Path]:
    docs = []
    for tier in tiers:
        tier_dir = corpus / tier
        if tier_dir.is_dir():
            docs += sorted(tier_dir.glob("*.pdf"))
    return docs


def discover_systems(out: Path, explicit: str | None) -> list[str]:
    if explicit:
        return [s.strip() for s in explicit.split(",") if s.strip()]
    # Thư mục hệ = thư mục không bắt đầu bằng '_' (những cái đó là vùng làm việc).
    return sorted(d.name for d in out.iterdir()
                  if d.is_dir() and not d.name.startswith("_"))


def main() -> int:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    docs = discover_docs(args.corpus, tiers)
    if not docs:
        print(f"!! không thấy PDF nào dưới {args.corpus} cho tiers {tiers}")
        return 1
    if not args.out.is_dir():
        print(f"!! chưa có {args.out} — chạy runner trước")
        return 1

    render_root = args.out / "_render"
    n_new = n_cached = 0

    print(f">>> trang nguồn ({len(docs)} PDF) @ {args.dpi} DPI")
    for pdf in docs:
        stamp = render(pdf, render_root / SOURCE_KEY / pdf.stem, args.dpi, args.force)
        n_cached += stamp["cached"]
        n_new += not stamp["cached"]
        print(f"  {'·' if stamp['cached'] else '+'} {pdf.stem:34} "
              f"{stamp['n_pages']:3d} trang", flush=True)

    systems = discover_systems(args.out, args.systems)
    for system in systems:
        for lang in langs:
            base = args.out / system / lang
            if not base.is_dir():
                continue
            print(f"\n>>> {system}/{lang}")
            for pdf in docs:
                out_pdf = base / pdf.stem / "output.pdf"
                if not out_pdf.exists():
                    print(f"  ! {pdf.stem:34} thiếu output.pdf", flush=True)
                    continue
                stamp = render(out_pdf, render_root / system / lang / pdf.stem,
                               args.dpi, args.force)
                n_cached += stamp["cached"]
                n_new += not stamp["cached"]
                print(f"  {'·' if stamp['cached'] else '+'} {pdf.stem:34} "
                      f"{stamp['n_pages']:3d} trang", flush=True)

    print(f"\n[render] {n_new} tài liệu render mới, {n_cached} dùng lại "
          f"(· = cache). Ảnh: {render_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
