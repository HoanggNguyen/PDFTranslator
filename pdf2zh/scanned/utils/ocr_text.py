"""OCR text cleaning and extraction utilities.

This module provides functions to clean and process OCR output from Surya,
including handling common OCR artifacts and extracting text for specific regions.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)


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

    # Step 3: Fix common OCR artifacts
    # Ligatures
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = text.replace("\ufb00", "ff")
    text = text.replace("\ufb03", "ffi")
    text = text.replace("\ufb04", "ffl")

    # Smart quotes to straight quotes
    text = text.replace("\u2018", "'")  # Left single quote
    text = text.replace("\u2019", "'")  # Right single quote
    text = text.replace("\u201c", '"')  # Left double quote
    text = text.replace("\u201d", '"')  # Right double quote

    # Dashes
    text = text.replace("\u2013", "-")  # En dash
    text = text.replace("\u2014", "-")  # Em dash
    text = text.replace("\u2212", "-")  # Minus sign

    # Other common artifacts
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u2026", "...")  # Ellipsis

    # Step 4: Normalize whitespace
    # Replace tabs with spaces
    text = text.replace("\t", " ")

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


def extract_text_for_region(
    ocr_result: Any,
    region_bbox: list[float],
    image_width: float,
    image_height: float,
    overlap_threshold: float = 0.5,
) -> str:
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
        Concatenated text from overlapping lines
    """
    if not hasattr(ocr_result, "text_lines"):
        return ""

    rx0, ry0, rx1, ry1 = region_bbox
    region_area = max(1.0, (rx1 - rx0) * (ry1 - ry0))

    matching_lines = []

    for line in ocr_result.text_lines:
        if not hasattr(line, "bbox") or not hasattr(line, "text"):
            continue

        lx0, ly0, lx1, ly1 = line.bbox

        # Calculate intersection
        ix0 = max(rx0, lx0)
        iy0 = max(ry0, ly0)
        ix1 = min(rx1, lx1)
        iy1 = min(ry1, ly1)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        intersection_area = (ix1 - ix0) * (iy1 - iy0)
        line_area = max(1.0, (lx1 - lx0) * (ly1 - ly0))

        # Check if enough of the line is within the region
        line_overlap = intersection_area / line_area
        if line_overlap >= overlap_threshold:
            matching_lines.append((ly0, line.text))

    # Sort by vertical position and join
    matching_lines.sort(key=lambda x: x[0])
    text = " ".join(line_text for _, line_text in matching_lines)

    return clean_ocr_text(text)


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
        category = getattr(elem, "category", None)

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
    """Join source_text from FLOWING_TEXT and IN_PLACE elements.

    This creates the raw_text field for PageData by concatenating
    text from translatable elements in order.

    Args:
        elements: List of ElementData objects

    Returns:
        Joined text with single newline between elements
    """
    from pdf2zh.scanned.enums import ElementCategory

    text_parts = []

    for elem in elements:
        category = getattr(elem, "category", None)

        # Only include FLOWING_TEXT and IN_PLACE categories
        if category in (ElementCategory.FLOWING_TEXT, ElementCategory.IN_PLACE):
            source_text = getattr(elem, "source_text", "")
            if source_text:
                text_parts.append(source_text)

    return "\n".join(text_parts)
