#!/usr/bin/env python3
"""Draw bboxes from a translated JSON onto the original PDF for visual inspection.

Usage:
    python test/draw_bbox.py --pdf input.pdf --json output.translated.json --output bbox.pdf
    python test/draw_bbox.py --pdf input.pdf --json output.json --output bbox.pdf --pages 0-9
"""

import argparse
import json
from pathlib import Path

import fitz

COLORS = {
    "FLOWING_TEXT": (0.0, 0.0, 1.0),  # blue
    "IN_PLACE": (0.0, 0.7, 0.0),  # green
    "EQUATION": (1.0, 0.5, 0.0),  # orange
    "TABLE": (0.5, 0.0, 0.8),  # purple
    "BYPASS": (1.0, 0.0, 0.0),  # red
}
CELL_COLOR = (0.8, 0.7, 0.0)  # yellow
EQ_LINE_COLOR = (0.0, 0.8, 0.8)  # cyan — equation_text_lines


def parse_pages(s: str) -> list[int] | None:
    if not s:
        return None
    result = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pages", default=None, help="e.g. 0-9 or 0,1,5")
    args = p.parse_args()

    pages_filter = parse_pages(args.pages)

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    doc = fitz.open(args.pdf)

    for page_idx, page_data in enumerate(data.get("pages", [])):
        if pages_filter is not None and page_idx not in pages_filter:
            continue
        if page_idx >= doc.page_count:
            break

        page = doc[page_idx]

        for elem in page_data.get("elements", []):
            cat = elem.get("category", "")
            label = elem.get("label", "")
            bbox = elem.get("bbox_pdf")
            if not bbox:
                continue

            color = COLORS.get(cat, (0.5, 0.5, 0.5))
            rect = fitz.Rect(*bbox)
            page.draw_rect(rect, color=color, width=1.5)
            page.insert_text(
                fitz.Point(bbox[0], bbox[1] - 1),
                f"{cat[:2]} {label}",
                fontsize=6,
                color=color,
            )

            if cat == "TABLE":
                for cell in elem.get("cells", []):
                    cb = cell.get("bbox_pdf")
                    if cb:
                        page.draw_rect(fitz.Rect(*cb), color=CELL_COLOR, width=0.8)

            if cat == "EQUATION":
                for line in elem.get("equation_text_lines", []) or []:
                    lb = line.get("bbox_pdf")
                    if not lb:
                        continue
                    page.draw_rect(fitz.Rect(*lb), color=EQ_LINE_COLOR, width=0.8)
                    label_text = line.get("text", "")[:20]
                    page.insert_text(
                        fitz.Point(lb[0], lb[1] - 1),
                        label_text,
                        fontsize=5,
                        color=EQ_LINE_COLOR,
                    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    print(f"Saved → {out}")
    print(
        "Legend: blue=FLOWING_TEXT  green=IN_PLACE  orange=EQUATION  purple=TABLE  "
        "red=BYPASS  yellow=cell  cyan=equation_text_line"
    )


if __name__ == "__main__":
    main()
