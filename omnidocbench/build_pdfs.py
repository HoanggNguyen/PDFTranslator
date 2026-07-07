"""Group OmniDocBench page images into multi-page PDFs (default 32 images/PDF).

Why: we want to benchmark the *real* ``StageAParser.parse_pdf`` path. parse_pdf
takes a PDF, so we pack the individual page images into PDFs — each image
becomes one page, at native resolution (PyMuPDF embeds it, no re-encode).

A ``mapping.json`` records exactly which image landed on which page of which
PDF, so the per-PDF parser output can be split back to per-image results after
the run.

Output layout::

    <out>/
      batch_00000.pdf        # 32 pages = 32 images
      batch_00001.pdf
      ...
      mapping.json

``mapping.json``::

    {
      "per_pdf": 32,
      "num_images": 1651,
      "num_pdfs": 52,
      "pdfs": [
        {"pdf": "batch_00000.pdf", "images": ["imgA.jpg", "imgB.jpg", ...]},
        ...
      ]
    }

Page index (0-based) of an image == its position in that PDF's ``images`` list.

Example
-------
    python build_pdfs.py \
        --images /media/lhbac32/OmniDocBench_data/images \
        --out    /media/lhbac32/OmniDocBench_data/pdfs \
        --per-pdf 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path,
                        help="Folder of page images (one image per page).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output folder for the batched PDFs + mapping.json.")
    parser.add_argument("--per-pdf", type=int, default=32,
                        help="Number of images packed into each PDF (default 32).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N images (quick test).")
    return parser.parse_args()


def find_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise NotADirectoryError(f"--images is not a folder: {images_dir}")
    files = [p for p in sorted(images_dir.iterdir())
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    return files


def image_to_pdf_pages(doc, image_path: Path) -> bool:
    """Append the image as one page to ``doc``. Returns True on success."""
    import fitz  # PyMuPDF

    try:
        with fitz.open(image_path) as img_doc:
            pdf_bytes = img_doc.convert_to_pdf()
        with fitz.open("pdf", pdf_bytes) as img_pdf:
            doc.insert_pdf(img_pdf)
        return True
    except Exception as exc:
        print(f"  [skip] {image_path.name}: {exc!r}", flush=True)
        return False


def main() -> int:
    args = parse_args()

    import fitz  # PyMuPDF

    images = find_images(args.images)
    if args.limit is not None:
        images = images[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)

    per_pdf = max(1, args.per_pdf)
    num_pdfs = (len(images) + per_pdf - 1) // per_pdf
    width = max(5, len(str(num_pdfs - 1)))

    print(f"[build_pdfs] {len(images)} images -> {num_pdfs} PDFs "
          f"({per_pdf} images/PDF) in {args.out}", flush=True)

    pdf_entries: list[dict] = []
    for pdf_idx in range(num_pdfs):
        chunk = images[pdf_idx * per_pdf: (pdf_idx + 1) * per_pdf]
        pdf_name = f"batch_{pdf_idx:0{width}d}.pdf"
        out_pdf = args.out / pdf_name

        doc = fitz.open()
        used_images: list[str] = []
        try:
            for image_path in chunk:
                if image_to_pdf_pages(doc, image_path):
                    used_images.append(image_path.name)
            if len(doc) == 0:
                print(f"  [warn] {pdf_name}: no valid pages, skipped", flush=True)
                continue
            doc.save(out_pdf)
        finally:
            doc.close()

        pdf_entries.append({"pdf": pdf_name, "images": used_images})
        print(f"  [{pdf_idx + 1}/{num_pdfs}] {pdf_name}  {len(used_images)} pages",
              flush=True)

    mapping = {
        "per_pdf": per_pdf,
        "num_images": sum(len(e["images"]) for e in pdf_entries),
        "num_pdfs": len(pdf_entries),
        "pdfs": pdf_entries,
    }
    mapping_path = args.out / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print(f"\n[build_pdfs] done. {mapping['num_pdfs']} PDFs, "
          f"{mapping['num_images']} pages total.", flush=True)
    print(f"  mapping: {mapping_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
