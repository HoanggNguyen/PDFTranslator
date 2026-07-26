"""Review-layer helpers for the human-in-the-loop checkpoints (Gradio-agnostic).

Phase 1 (after OCR) and Phase 3 (after render) let the user click a page
preview to select an element and edit it. This module holds the pure logic:
rasterize a page with element boxes drawn on top, hit-test a click to an
element, and apply edits to the parsed / translated dicts. No Gradio imports,
so it is unit-testable in isolation.

Coordinate note: ``bbox_pdf`` is top-left origin, Y-down, same axes as the
rasterized image (see parser/utils/bbox.py) — only a uniform ``dpi/72`` scale
separates PDF points from image pixels. Element boxes therefore map onto both
the original page (Phase 1) and the rendered output page (Phase 3), which keep
the original page dimensions.
"""

from __future__ import annotations

import html
from typing import Any

import fitz
from PIL import Image, ImageDraw

from pdf2zh.parser.enums import DEFAULT_CATEGORY, SURYA_LABEL_MAP, SuryaLabel
from pdf2zh.parser.utils.image import _fitz_render

# Labels offered in the Phase-1 label dropdown (Surya layout labels).
LABEL_CHOICES = [
    SuryaLabel.TEXT,
    SuryaLabel.LIST_ITEM,
    SuryaLabel.FOOTNOTE,
    SuryaLabel.SECTION_HEADER,
    SuryaLabel.PAGE_HEADER,
    SuryaLabel.PAGE_FOOTER,
    SuryaLabel.CAPTION,
    SuryaLabel.TABLE,
    SuryaLabel.TABLE_OF_CONTENTS,
    SuryaLabel.PICTURE,
    SuryaLabel.FIGURE,
    SuryaLabel.FORM,
    SuryaLabel.EQUATION,
    SuryaLabel.CODE,
]

# Outline colors by category (RGB).
_COLOR_TRANSLATABLE = (0, 120, 255)
_COLOR_BYPASS = (150, 150, 150)
_COLOR_HIGHLIGHT = (230, 30, 30)


def _category_for_label(label: str) -> str:
    """Derive the ElementCategory string for a raw layout label."""
    return SURYA_LABEL_MAP.get(label, DEFAULT_CATEGORY).value


def hex_to_rgb(value: str | None) -> list[int] | None:
    """Parse a CSS color string into ``[r, g, b]`` ints, or None if unparseable.

    Accepts ``#rgb`` / ``#rrggbb`` and ``rgb(...)`` / ``rgba(...)`` forms (what
    ``gr.ColorPicker`` emits). Alpha is ignored. None lets the caller fall back
    to auto-sampling instead of crashing on a stray value.
    """
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            return None
        try:
            return [int(h[i : i + 2], 16) for i in (0, 2, 4)]
        except ValueError:
            return None
    if s.lower().startswith(("rgb(", "rgba(")):
        try:
            inner = s[s.index("(") + 1 : s.rindex(")")]
            parts = inner.split(",")
            if len(parts) < 3:
                return None
            return [max(0, min(255, round(float(p)))) for p in parts[:3]]
        except ValueError:
            return None
    return None


def render_page_with_boxes(
    pdf_path: str,
    page_index: int,
    elements: list[dict],
    dpi: int = 150,
    highlight_idx: int | None = None,
    highlight_cell: tuple[int, int] | None = None,
) -> tuple[Image.Image, list[dict], float]:
    """Rasterize page ``page_index`` and draw each element's bbox on top.

    A TABLE element with cells is drawn cell-by-cell inside a thin outer
    border (translation happens per cell, not per whole table); every other
    element (and a TABLE with no cells) draws a single box. Returns
    ``(image, boxes, scale)`` where ``boxes`` is a list of ``{"elem_idx",
    "cell_idx", "bbox_img", "category"}`` (``cell_idx`` is None for
    element-level boxes) and ``scale = dpi/72``. Each box's tag shows its
    ``label`` (e.g. "Table", "Caption"), not its index. Uses ``_fitz_render``
    directly (deterministic scale) — NOT ``render_page_to_image`` (which may
    return a native-res embedded image).
    """
    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"page_index {page_index} out of range")
        img = _fitz_render(doc[page_index], dpi).convert("RGB")
    finally:
        doc.close()

    scale = dpi / 72.0
    draw = ImageDraw.Draw(img)
    boxes: list[dict] = []
    for elem_idx, elem in enumerate(elements):
        category = elem.get("category", "")
        cells = elem.get("cells", [])
        label = elem.get("label", "")
        if category == "TABLE" and cells:
            outer = elem.get("bbox_pdf")
            if outer and len(outer) == 4:
                ox0, oy0, ox1, oy1 = (v * scale for v in outer)
                outer_color = (
                    _COLOR_HIGHLIGHT
                    if elem_idx == highlight_idx
                    else _COLOR_TRANSLATABLE
                )
                draw.rectangle([ox0, oy0, ox1, oy1], outline=outer_color, width=1)
                draw.text((ox0 + 2, max(0, oy0 - 12)), label, fill=outer_color)
            for cell_idx, cell in enumerate(cells):
                bbox_pdf = cell.get("bbox_pdf")
                if not bbox_pdf or len(bbox_pdf) != 4:
                    continue
                x0, y0, x1, y1 = (v * scale for v in bbox_pdf)
                boxes.append(
                    {
                        "elem_idx": elem_idx,
                        "cell_idx": cell_idx,
                        "bbox_img": [x0, y0, x1, y1],
                        "category": category,
                    }
                )
                if highlight_cell == (elem_idx, cell_idx):
                    color, width = _COLOR_HIGHLIGHT, 3
                else:
                    color, width = _COLOR_TRANSLATABLE, 1
                draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            continue

        bbox_pdf = elem.get("bbox_pdf")
        if not bbox_pdf or len(bbox_pdf) != 4:
            continue
        x0, y0, x1, y1 = (v * scale for v in bbox_pdf)
        boxes.append(
            {
                "elem_idx": elem_idx,
                "cell_idx": None,
                "bbox_img": [x0, y0, x1, y1],
                "category": category,
            }
        )

        if elem_idx == highlight_idx:
            color, width = _COLOR_HIGHLIGHT, 3
        elif category == "BYPASS":
            color, width = _COLOR_BYPASS, 1
        else:
            color, width = _COLOR_TRANSLATABLE, 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        # Label tag at the top-left corner.
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color)

    return img, boxes, scale


def render_page_plain(
    pdf_path: str,
    page_index: int,
    dpi: int = 150,
) -> tuple[Image.Image, tuple[int, int]]:
    """Rasterize page ``page_index`` with no boxes drawn.

    Returns ``(image, (width, height))`` in image pixels. Uses ``_fitz_render``
    directly (deterministic ``dpi/72`` scale), matching ``render_page_with_boxes``.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"page_index {page_index} out of range")
        img = _fitz_render(doc[page_index], dpi).convert("RGB")
    finally:
        doc.close()
    return img, (img.width, img.height)


def overlay_svg(
    elements: list[dict],
    scale: float,
    width: int,
    height: int,
    highlight_idx: int | None = None,
    highlight_cell: tuple[int, int] | None = None,
) -> tuple[str, list[dict]]:
    """Build an SVG overlay drawing each element's bbox + label, and the boxes list.

    Mirrors the drawing loop of ``render_page_with_boxes`` so ``boxes`` is
    identical (same ``bbox_img``/``elem_idx``/``cell_idx``/``category``) —
    ``hit_test`` is unchanged. A TABLE element with cells is drawn cell-by-cell
    inside a thin dashed outer border (translation happens per cell, not per
    whole table); every other element (and a TABLE with no cells) draws a
    single box. ``highlight_idx`` marks a selected element (the whole table,
    for label/bypass editing); ``highlight_cell`` marks a single selected cell
    as ``(elem_idx, cell_idx)``. Each box's tag shows its ``label`` (e.g.
    "Table", "Caption"), not its index — escaped since, unlike the fixed
    colors/coordinates, it's arbitrary text. The svg box equals the ``<img>``
    box exactly (viewBox aspect == natural aspect), so rects align
    pixel-for-pixel with the base raster.
    """
    boxes: list[dict] = []
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'preserveAspectRatio="none" '
        f'style="display:block;width:100%;height:auto">'
    ]
    for elem_idx, elem in enumerate(elements):
        category = elem.get("category", "")
        cells = elem.get("cells", [])
        label = html.escape(elem.get("label", ""))
        if category == "TABLE" and cells:
            outer = elem.get("bbox_pdf")
            if outer and len(outer) == 4:
                ox0, oy0, ox1, oy1 = (v * scale for v in outer)
                outer_color = (
                    _COLOR_HIGHLIGHT
                    if elem_idx == highlight_idx
                    else _COLOR_TRANSLATABLE
                )
                outer_rgb = f"rgb({outer_color[0]},{outer_color[1]},{outer_color[2]})"
                parts.append(
                    f'<rect x="{ox0}" y="{oy0}" width="{ox1 - ox0}" '
                    f'height="{oy1 - oy0}" fill="none" stroke="{outer_rgb}" '
                    f'stroke-width="1" stroke-dasharray="4 3"/>'
                )
                parts.append(
                    f'<text x="{ox0 + 2}" y="{max(0, oy0 - 2)}" font-size="12" '
                    f'fill="{outer_rgb}">{label}</text>'
                )
            for cell_idx, cell in enumerate(cells):
                bbox_pdf = cell.get("bbox_pdf")
                if not bbox_pdf or len(bbox_pdf) != 4:
                    continue
                x0, y0, x1, y1 = (v * scale for v in bbox_pdf)
                boxes.append(
                    {
                        "elem_idx": elem_idx,
                        "cell_idx": cell_idx,
                        "bbox_img": [x0, y0, x1, y1],
                        "category": category,
                    }
                )
                if highlight_cell == (elem_idx, cell_idx):
                    color, stroke = _COLOR_HIGHLIGHT, 3
                else:
                    color, stroke = _COLOR_TRANSLATABLE, 1
                rgb = f"rgb({color[0]},{color[1]},{color[2]})"
                parts.append(
                    f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
                    f'fill="none" stroke="{rgb}" stroke-width="{stroke}"/>'
                )
            continue

        bbox_pdf = elem.get("bbox_pdf")
        if not bbox_pdf or len(bbox_pdf) != 4:
            continue
        x0, y0, x1, y1 = (v * scale for v in bbox_pdf)
        boxes.append(
            {
                "elem_idx": elem_idx,
                "cell_idx": None,
                "bbox_img": [x0, y0, x1, y1],
                "category": category,
            }
        )

        if elem_idx == highlight_idx:
            color, stroke = _COLOR_HIGHLIGHT, 3
        elif category == "BYPASS":
            color, stroke = _COLOR_BYPASS, 1
        else:
            color, stroke = _COLOR_TRANSLATABLE, 2
        rgb = f"rgb({color[0]},{color[1]},{color[2]})"
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
            f'fill="none" stroke="{rgb}" stroke-width="{stroke}"/>'
        )
        parts.append(
            f'<text x="{x0 + 2}" y="{max(0, y0 - 2)}" font-size="12" '
            f'fill="{rgb}">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts), boxes


def hit_test(
    boxes: list[dict], x_img: float, y_img: float
) -> tuple[int, int | None] | None:
    """Return ``(elem_idx, cell_idx)`` of the smallest box containing the click.

    ``cell_idx`` is None when the hit box is a whole element (not a table
    cell). ``x_img, y_img`` are in the same image-pixel space as ``bbox_img``.
    Smallest-area-wins resolves overlapping boxes (e.g. caption inside figure,
    or a cell within its table's outer bounds).
    """
    best: tuple[int, int | None] | None = None
    best_area = float("inf")
    for box in boxes:
        x0, y0, x1, y1 = box["bbox_img"]
        if x0 <= x_img <= x1 and y0 <= y_img <= y1:
            area = (x1 - x0) * (y1 - y0)
            if area < best_area:
                best_area = area
                best = (box["elem_idx"], box.get("cell_idx"))
    return best


def _get_element(doc: dict, page_i: int, elem_i: int) -> dict | None:
    try:
        return doc["pages"][page_i]["elements"][elem_i]
    except (KeyError, IndexError, TypeError):
        return None


def _get_cell(doc: dict, page_i: int, elem_i: int, cell_i: int) -> dict | None:
    elem = _get_element(doc, page_i, elem_i)
    if elem is None:
        return None
    try:
        return elem["cells"][cell_i]
    except (KeyError, IndexError, TypeError):
        return None


def apply_phase1_edit(
    parsed: dict,
    page_i: int,
    elem_i: int,
    label: str,
    source_text: str,
    bypass: bool,
) -> str | None:
    """Apply a Phase-1 edit in place. Returns a warning message, or None on success.

    - ``bypass=True`` sets category=BYPASS (excluded from translation), keeping
      the label so unchecking restores the derived category.
    - Otherwise sets label + derived category + source_text.
    - Changing to/from ``Table`` is blocked (cell structure can't be rebuilt);
      the source_text edit still applies.
    """
    elem = _get_element(parsed, page_i, elem_i)
    if elem is None:
        return "Không tìm thấy element."

    if bypass:
        elem["category"] = "BYPASS"
        elem["source_text"] = source_text
        return None

    old_cat = elem.get("category")
    if label == SuryaLabel.TABLE or old_cat == "TABLE":
        elem["source_text"] = source_text
        if label != elem.get("label"):
            return "Giữ nguyên nhãn Table — không dựng lại cấu trúc ô."
        # A table un-bypassed: restore its TABLE category.
        elem["category"] = "TABLE"
        return None

    elem["label"] = label
    elem["category"] = _category_for_label(label)
    elem["source_text"] = source_text
    return None


def apply_phase2_edit(
    translated: dict,
    page_i: int,
    elem_i: int,
    translated_text: str,
) -> str | None:
    """Set an element's translated_text in place (used at Phase 3)."""
    elem = _get_element(translated, page_i, elem_i)
    if elem is None:
        return "Không tìm thấy element."
    elem["translated_text"] = translated_text
    return None


def apply_phase1_cell_edit(
    parsed: dict,
    page_i: int,
    elem_i: int,
    cell_i: int,
    source_text: str,
) -> str | None:
    """Set a TABLE cell's source_text in place (used at Phase 1).

    Cells have no label/category of their own — translation runs per cell, so
    only the OCR text is editable here.
    """
    cell = _get_cell(parsed, page_i, elem_i, cell_i)
    if cell is None:
        return "Không tìm thấy cell."
    cell["source_text"] = source_text
    return None


def apply_phase2_cell_edit(
    translated: dict,
    page_i: int,
    elem_i: int,
    cell_i: int,
    translated_text: str,
) -> str | None:
    """Set a TABLE cell's translated_text in place (used at Phase 3)."""
    cell = _get_cell(translated, page_i, elem_i, cell_i)
    if cell is None:
        return "Không tìm thấy cell."
    cell["translated_text"] = translated_text
    return None


def add_element(
    parsed: dict,
    page_i: int,
    bbox_pdf: list[float],
    label: str,
    source_text: str,
    bg_color: list[int] | None = None,
    text_color: list[int] | None = None,
) -> int:
    """Append a new element to a page (for a missed region). Returns its elem_idx.

    Appending keeps existing element indices — and therefore color uids — stable.
    ``bg_color`` / ``text_color`` (RGB ints), when given, override the render's
    per-element color sampling (see ``renderer._sample_colors``).
    """
    elements = parsed["pages"][page_i]["elements"]
    elem = {
        "label": label,
        "category": _category_for_label(label),
        "bbox_pdf": [float(v) for v in bbox_pdf],
        "source_text": source_text,
        "translated_text": "",
        "cells": [],
    }
    if bg_color is not None:
        elem["bg_color"] = [int(c) for c in bg_color]
    if text_color is not None:
        elem["text_color"] = [int(c) for c in text_color]
    elements.append(elem)
    return len(elements) - 1


def output_page_position(pages: list[int] | None, page_index: int) -> int | None:
    """Position (0-based) of a source page within the translated output.

    The output PDF contains only the translated pages in ascending order, so a
    source ``page_index`` maps to its rank in ``sorted(pages)``. When ``pages``
    is None the whole document is rendered, so the position equals page_index.
    """
    if pages is None:
        return page_index
    sel = sorted(pages)
    return sel.index(page_index) if page_index in sel else None


def page_index_of(parsed: dict, page_i: int) -> int:
    """True (0-based) document page number of the ``page_i``-th parsed page."""
    page = parsed["pages"][page_i]
    return int(page.get("page_index", page_i))


def normalize_click(index: Any) -> tuple[float, float] | None:
    """Coerce a Gradio Image ``.select`` index into (x, y) image pixels.

    Accepts (x, y) tuples/lists; returns None for anything unexpected so the
    caller can ignore a stray event instead of crashing.
    """
    if isinstance(index, (list, tuple)) and len(index) >= 2:
        try:
            return float(index[0]), float(index[1])
        except (TypeError, ValueError):
            return None
    return None
