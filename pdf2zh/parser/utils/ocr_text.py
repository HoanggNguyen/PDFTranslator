"""OCR text cleaning and extraction utilities.

This module provides functions to clean and process OCR output from Surya,
including handling common OCR artifacts and extracting text for specific regions.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from pdf2zh.parser.utils.bbox import bbox_area, bbox_intersection, polygon_to_bbox

logger = logging.getLogger(__name__)


def adjust_cell_bbox(
    matching_cell_lines: list[Any],
    cell_bbox_pdf: list[float],
    cell_bbox_image: list[float],
    padding: float = 2.0,
) -> list[float]:
    """
    Co nhỏ cell_bbox_pdf lại để ôm sát vào phân vùng chứa textlines thực tế,
    sau đó bổ sung thêm một lượng padding.

    Args:
        matching_cell_lines: Danh sách các dòng OCR tìm thấy trong ô
        cell_bbox_pdf: Bounding box của ô ở hệ PDF [x0, y0, x1, y1]
        cell_bbox_image: Bounding box của ô ở hệ ảnh [x0, y0, x1, y1]
        padding: Khoảng cách đệm thêm vào các cạnh (đơn vị: points)

    Returns:
        Bounding box mới hệ PDF [x0, y0, x1, y1] đã được điều chỉnh ôm sát text
    """
    if not matching_cell_lines:
        return cell_bbox_pdf

    text_x0 = float("inf")
    text_y0 = float("inf")
    text_x1 = float("-inf")
    text_y1 = float("-inf")

    for line in matching_cell_lines:
        line_bbox = _get_ocr_bbox(line)
        if line_bbox is None:
            continue
        text_x0 = min(text_x0, line_bbox[0])
        text_y0 = min(text_y0, line_bbox[1])
        text_x1 = max(text_x1, line_bbox[2])
        text_y1 = max(text_y1, line_bbox[3])

    if text_x0 == float("inf"):
        return cell_bbox_pdf

    img_w = cell_bbox_image[2] - cell_bbox_image[0]
    img_h = cell_bbox_image[3] - cell_bbox_image[1]
    pdf_w = cell_bbox_pdf[2] - cell_bbox_pdf[0]
    pdf_h = cell_bbox_pdf[3] - cell_bbox_pdf[1]

    scale_x = pdf_w / img_w if img_w > 0 else 1.0
    scale_y = pdf_h / img_h if img_h > 0 else 1.0

    dx0 = max(0.0, text_x0 - cell_bbox_image[0])
    dy0 = max(0.0, text_y0 - cell_bbox_image[1])
    dx1 = max(0.0, cell_bbox_image[2] - text_x1)
    dy1 = max(0.0, cell_bbox_image[3] - text_y1)

    new_pdf_x0 = cell_bbox_pdf[0] + max(0.0, dx0 * scale_x - padding)
    new_pdf_y0 = cell_bbox_pdf[1] + max(0.0, dy0 * scale_y - padding)
    new_pdf_x1 = cell_bbox_pdf[2] - max(0.0, dx1 * scale_x - padding)
    new_pdf_y1 = cell_bbox_pdf[3] - max(0.0, dy1 * scale_y - padding)

    final_x0 = max(cell_bbox_pdf[0], min(new_pdf_x0, cell_bbox_pdf[2]))
    final_y0 = max(cell_bbox_pdf[1], min(new_pdf_y0, cell_bbox_pdf[3]))
    final_x1 = max(final_x0, min(new_pdf_x1, cell_bbox_pdf[2]))
    final_y1 = max(final_y0, min(new_pdf_y1, cell_bbox_pdf[3]))

    return [final_x0, final_y0, final_x1, final_y1]


def clean_ocr_text(text: str) -> str:
    """Clean OCR text by removing artifacts and normalizing whitespace.

    Processing steps:
    1. Normalize Unicode (NFC form)
    2. Remove control characters except newlines and tabs
    3. Fix common OCR artifacts (ligatures, smart quotes, etc.)
    4. Normalize whitespace (collapse multiple spaces, trim lines)
    5. Remove empty lines at start/end

    Args:
        text: Raw OCR text from Surya

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Step 1: Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Remove control characters except newlines and tabs
    cleaned_chars = []
    for char in text:
        if char in ("\n", "\t"):
            cleaned_chars.append(char)
        elif unicodedata.category(char)[0] != "C":
            cleaned_chars.append(char)
    text = "".join(cleaned_chars)

    # # Step 3: Fix common OCR artifacts
    # # Ligatures
    # text = text.replace("\ufb01", "fi")
    # text = text.replace("\ufb02", "fl")
    # text = text.replace("\ufb00", "ff")
    # text = text.replace("\ufb03", "ffi")
    # text = text.replace("\ufb04", "ffl")

    # # Smart quotes to straight quotes
    # text = text.replace("\u2018", "'")  # Left single quote
    # text = text.replace("\u2019", "'")  # Right single quote
    # text = text.replace("\u201c", '"')  # Left double quote
    # text = text.replace("\u201d", '"')  # Right double quote

    # # Dashes
    # text = text.replace("\u2013", "-")  # En dash
    # text = text.replace("\u2014", "-")  # Em dash
    # text = text.replace("\u2212", "-")  # Minus sign

    # # Other common artifacts
    # text = text.replace("\u00a0", " ")  # Non-breaking space
    # text = text.replace("\u2026", "...")  # Ellipsis

    text = text.replace("<br>", "\n")  # Line break tags

    # Step 4: Normalize whitespace

    # Collapse multiple spaces into one
    text = re.sub(r" +", " ", text)

    # Trim each line
    lines = text.split("\n")
    lines = [line.strip() for line in lines]

    # Step 5: Remove empty lines at start and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)


def collect_ocr_text(ocr_result: Any) -> str:
    """Collect all text lines from an OCR result into a single string.

    Used after crop-then-OCR: the entire OCR result belongs to one layout
    region, so we simply concatenate all detected text lines.

    Args:
        ocr_result: Surya OCR result with ``text_lines`` attribute

    Returns:
        Cleaned concatenated text
    """
    if not hasattr(ocr_result, "text_lines"):
        return ""

    lines = []
    for line in ocr_result.text_lines:
        if hasattr(line, "text") and line.text:
            lines.append(line.text)

    return clean_ocr_text(" ".join(lines))


def smart_join_text_lines(lines: list[Any]) -> str:
    if not lines:
        return ""

    result = []
    last_valid_text = ""  # Lưu lại văn bản của dòng có chữ gần nhất

    for line in lines:
        current_text = getattr(line, "text", "").strip()

        if not current_text:
            continue

        # Nếu là dòng chứa chữ đầu tiên, chỉ cần thêm vào kết quả
        if not result:
            result.append(current_text)
            last_valid_text = current_text
            continue

        ends_with_punctuation = last_valid_text[-1] in {".", "!", "?"}
        starts_with_upper = current_text[0].isupper()
        ends_with_hyphen = last_valid_text.endswith("-")

        if not ends_with_punctuation and starts_with_upper:
            result.append("\n" + current_text)
        elif ends_with_hyphen:
            result.append(current_text)
        else:
            result.append(" " + current_text)

        last_valid_text = current_text

    return clean_ocr_text("".join(result))


def sort_text_lines(lines: list[Any]) -> list[Any]:
    """
    Sort OCR text lines in reading order (top-to-bottom, left-to-right).
    """
    if not lines:
        return []

    first_line = lines[0]
    if hasattr(first_line, "bbox") and first_line.bbox:

        def get_full_bbox(line):
            b = line.bbox
            return b[0], b[1], b[2], b[3]

    elif hasattr(first_line, "polygon"):

        def get_full_bbox(line):
            poly = line.polygon
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            return min(xs), min(ys), max(xs), max(ys)

    else:
        return lines

    boxes = []
    for line in lines:
        x_min, y_min, x_max, y_max = get_full_bbox(line)
        y_center = (y_min + y_max) / 2.0

        boxes.append((y_min, y_center, x_min, y_max, line))

    boxes.sort()

    rows = []
    current_row = []
    anchor_y_center = None

    for box in boxes:
        y_min, y_center, x_min, y_max, line = box

        if not current_row:
            current_row.append((x_min, line))
            anchor_y_center = y_center
        else:
            if y_min <= anchor_y_center <= y_max:
                current_row.append((x_min, line))
            else:
                rows.append(current_row)
                current_row = [(x_min, line)]
                anchor_y_center = y_center

    if current_row:
        rows.append(current_row)

    sorted_lines = []
    for row in rows:
        row.sort()
        for _, line in row:
            sorted_lines.append(line)

    return sorted_lines


def extract_text_for_region(
    ocr_result: Any,
    region_bbox: list[float],
    overlap_threshold: float = 0.5,
) -> list[Any]:
    """Extract OCR text that falls within a region.

    Finds all text lines from the OCR result that overlap significantly
    with the given region and concatenates them.

    Args:
        ocr_result: Surya OCR result object with text_lines attribute
        region_bbox: [x0, y0, x1, y1] in image pixels
        image_width: Image width for coordinate validation
        image_height: Image height for coordinate validation
        overlap_threshold: Minimum overlap ratio to include a line

    Returns:
        Concatenated text from overlapping lines and estimated font size
    """
    if not hasattr(ocr_result, "text_lines"):
        return []

    matching_lines = _collect_region_matches(
        getattr(ocr_result, "text_lines", []),
        region_bbox,
        overlap_threshold,
    )
    return sort_text_lines(matching_lines)


def _collect_region_matches(
    items: list[Any],
    region_bbox: list[float],
    overlap_threshold: float,
) -> list[Any]:
    matching_items: list[Any] = []

    for item in items:
        if not hasattr(item, "text"):
            continue

        item_bbox = _get_ocr_bbox(item)
        if item_bbox is None:
            continue

        intersection = bbox_intersection(region_bbox, item_bbox)
        if intersection is None:
            continue

        item_area = max(1.0, bbox_area(item_bbox))
        overlap_ratio = bbox_area(intersection) / item_area
        if overlap_ratio >= overlap_threshold:
            matching_items.append(item)

    return matching_items


def _get_ocr_bbox(item: Any) -> list[float] | None:
    item_bbox = getattr(item, "bbox", None)
    if item_bbox is not None:
        return list(item_bbox)

    if hasattr(item, "polygon"):
        return polygon_to_bbox(item.polygon)

    return None


def log_toc_hints(elements: list[Any], page_index: int) -> None:
    """Log potential Table of Contents entries for debugging.

    Looks for Section-header elements that might indicate chapter structure
    and logs them for manual review during development.

    Args:
        elements: List of ElementData objects from the page
        page_index: 0-based page number for logging context
    """
    toc_hints = []

    for elem in elements:
        label = getattr(elem, "label", "")

        # Look for section headers and TOC elements
        if label in ("Section-header", "Table-of-contents"):
            text = getattr(elem, "source_text", "")
            if text:
                # Truncate long text
                display_text = text[:80] + "..." if len(text) > 80 else text
                toc_hints.append(f"  [{label}] {display_text}")

    if toc_hints:
        logger.debug(f"Page {page_index} TOC hints:\n" + "\n".join(toc_hints))


def join_raw_text(elements: list[Any]) -> str:
    """Concatenate ``source_text`` from translatable layout elements.

    Collects source text from every :class:`~pdf2zh.scanned.enums.ElementCategory`
    that carries translatable content (``FLOWING_TEXT`` and ``IN_PLACE``) and
    joins them with newlines to form the ``raw_text`` field of
    :class:`~pdf2zh.scanned.models.PageData`.

    BYPASS, TABLE, and EQUATION categories are intentionally excluded:
    BYPASS has no text; TABLE text is stored per-cell; EQUATION text is a
    placeholder handled separately.

    Args:
        elements: Ordered list of :class:`~pdf2zh.scanned.models.ElementData`
                  objects (or any object with ``category`` and ``source_text``
                  attributes) for a single page.

    Returns:
        Single string with element texts joined by ``"\n"``,
        or an empty string if no translatable elements are present.
    """
    from pdf2zh.parser.enums import ElementCategory

    text_parts = []

    for elem in elements:
        category = getattr(elem, "category", None)

        # Only include FLOWING_TEXT and IN_PLACE categories
        if category in (ElementCategory.FLOWING_TEXT, ElementCategory.IN_PLACE):
            source_text = getattr(elem, "source_text", "")
            if source_text:
                text_parts.append(source_text)

    return "\n".join(text_parts)
