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


def render_page_with_boxes(
    pdf_path: str,
    page_index: int,
    elements: list[dict],
    dpi: int = 150,
    highlight_idx: int | None = None,
) -> tuple[Image.Image, list[dict], float]:
    """Rasterize page ``page_index`` and draw each element's bbox on top.

    Returns ``(image, boxes, scale)`` where ``boxes`` is a list of
    ``{"elem_idx", "bbox_img", "category"}`` (bbox_img in image pixels) and
    ``scale = dpi/72``. Uses ``_fitz_render`` directly (deterministic scale) —
    NOT ``render_page_to_image`` (which may return a native-res embedded image).
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
        bbox_pdf = elem.get("bbox_pdf")
        if not bbox_pdf or len(bbox_pdf) != 4:
            continue
        x0, y0, x1, y1 = (v * scale for v in bbox_pdf)
        category = elem.get("category", "")
        boxes.append(
            {"elem_idx": elem_idx, "bbox_img": [x0, y0, x1, y1], "category": category}
        )

        if elem_idx == highlight_idx:
            color, width = _COLOR_HIGHLIGHT, 3
        elif category == "BYPASS":
            color, width = _COLOR_BYPASS, 1
        else:
            color, width = _COLOR_TRANSLATABLE, 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        # Element number tag at the top-left corner.
        draw.text((x0 + 2, max(0, y0 - 12)), str(elem_idx), fill=color)

    return img, boxes, scale


def hit_test(boxes: list[dict], x_img: float, y_img: float) -> int | None:
    """Return the elem_idx of the smallest box containing the click, else None.

    ``x_img, y_img`` are in the same image-pixel space as ``bbox_img``.
    Smallest-area-wins resolves overlapping boxes (e.g. caption inside figure).
    """
    best_idx: int | None = None
    best_area = float("inf")
    for box in boxes:
        x0, y0, x1, y1 = box["bbox_img"]
        if x0 <= x_img <= x1 and y0 <= y_img <= y1:
            area = (x1 - x0) * (y1 - y0)
            if area < best_area:
                best_area = area
                best_idx = box["elem_idx"]
    return best_idx


def _get_element(doc: dict, page_i: int, elem_i: int) -> dict | None:
    try:
        return doc["pages"][page_i]["elements"][elem_i]
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


def add_element(
    parsed: dict,
    page_i: int,
    bbox_pdf: list[float],
    label: str,
    source_text: str,
) -> int:
    """Append a new element to a page (for a missed region). Returns its elem_idx.

    Appending keeps existing element indices — and therefore color uids — stable.
    """
    elements = parsed["pages"][page_i]["elements"]
    elements.append(
        {
            "label": label,
            "category": _category_for_label(label),
            "bbox_pdf": [float(v) for v in bbox_pdf],
            "source_text": source_text,
            "translated_text": "",
            "cells": [],
        }
    )
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
