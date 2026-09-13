"""Generate a small synthetic PDF + GT for connectivity smoke tests, NOT research eval."""

import argparse
import json
from pathlib import Path

from benchmark.e2e.manifest import sha256


def create(dest: Path):
    import fitz

    if dest.exists():
        raise ValueError("Destination exists; choose a new corpus directory")
    tier = dest / "T1"
    tier.mkdir(parents=True)
    path = tier / "smoke_demo.pdf"
    gt_pages, map_pages = [], []
    with fitz.open() as doc:
        for i in range(2):
            page = doc.new_page(width=595, height=842)
            page.insert_text(
                (50, 65), f"Document translation test {i + 1}", fontsize=20
            )
            prose = (
                "This document tests whether a translation pipeline preserves the structure "
                "of a complete PDF. The text should be translated into Vietnamese while "
                "numbers and the illustration remain readable. In 2025 the project processed "
                "123 documents. The target for the next period is 456 documents. "
                "A successful result contains all paragraphs and keeps the page order."
            )
            assert (
                page.insert_textbox(fitz.Rect(50, 100, 545, 245), prose, fontsize=12)
                >= 0
            )
            assert (
                page.insert_textbox(
                    fitz.Rect(50, 430, 545, 560),
                    "The illustration above contains three bars. Compare its position and size "
                    "before and after translation. This synthetic example is only a smoke test; "
                    "it does not represent the variety of documents in a research dataset.",
                    fontsize=12,
                )
                >= 0
            )
            elements = []
            for block in page.get_text("blocks"):
                box = list(block[:4])
                elements.append(
                    {
                        "class": "Title" if box[1] < 80 else "Text",
                        "bbox_norm": [
                            box[0] / 595,
                            box[1] / 842,
                            box[2] / 595,
                            box[3] / 842,
                        ],
                    }
                )
            with fitz.open() as chart:
                canvas = chart.new_page(width=300, height=120)
                for x, h in [(30, 30), (110, 60), (190, 90)]:
                    canvas.draw_rect(
                        fitz.Rect(x, 110 - h, x + 40, 110),
                        color=(0.1, 0.3, 0.6),
                        fill=(0.2, 0.5, 0.8),
                    )
                pix = canvas.get_pixmap()
                image_box = fitz.Rect(120, 270, 420, 390)
                page.insert_image(image_box, pixmap=pix)
            elements.append(
                {
                    "class": "Picture",
                    "bbox_norm": [120 / 595, 270 / 842, 420 / 595, 390 / 842],
                }
            )
            gt_pages.append(
                {"page": i, "width": 595, "height": 842, "elements": elements}
            )
            map_pages.append(
                {
                    "page": i,
                    "original_filename": "synthetic-demo",
                    "n_chars": len(page.get_text()),
                }
            )
        doc.save(path)
    common = {"doc_id": path.stem, "pdf": path.name}
    (tier / "gt.json").write_text(
        json.dumps(
            {
                "dataset": "SYNTHETIC-SMOKE-ONLY",
                "docs": [{**common, "pages": gt_pages}],
            },
            indent=2,
        )
    )
    (tier / "mapping.json").write_text(
        json.dumps(
            {
                "dataset": "SYNTHETIC-SMOKE-ONLY",
                "docs": [
                    {
                        **common,
                        "pages": map_pages,
                        "sha256": sha256(path),
                        "n_pages": 2,
                        "n_chars": sum(p["n_chars"] for p in map_pages),
                        "n_source_docs": 1,
                    }
                ],
            },
            indent=2,
        )
    )
    print(f"Synthetic smoke corpus: {dest} (never use for research conclusions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    create(parser.parse_args().out)
