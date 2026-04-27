"""Font-size estimation helpers for scanned Stage A output."""

from __future__ import annotations

from statistics import mean, median
from typing import Any

import numpy as np

from pdf2zh.scanned.enums import ElementCategory

FONT_SIZE_BUCKETS = ("xs", "sm", "md", "lg", "xl")
HEADER_LIKE_LABELS = {
    "PageHeader",
    "SectionHeader",
    "Caption",
    "TableOfContents",
}
FOOTER_LIKE_LABELS = {
    "Footnote",
    "PageFooter",
}


def extract_text_lines_for_region(
    ocr_result: Any,
    region_bbox: list[float],
    overlap_threshold: float = 0.5,
) -> list[Any]:
    """Return OCR text lines whose bbox overlaps the requested region."""

    if not hasattr(ocr_result, "text_lines"):
        return []

    rx0, ry0, rx1, ry1 = region_bbox
    matched_lines: list[Any] = []

    for line in ocr_result.text_lines:
        if not hasattr(line, "bbox") or not hasattr(line, "text"):
            continue

        lx0, ly0, lx1, ly1 = line.bbox
        ix0 = max(rx0, lx0)
        iy0 = max(ry0, ly0)
        ix1 = min(rx1, lx1)
        iy1 = min(ry1, ly1)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        intersection_area = (ix1 - ix0) * (iy1 - iy0)
        line_area = max(1.0, (lx1 - lx0) * (ly1 - ly0))
        if (intersection_area / line_area) >= overlap_threshold:
            matched_lines.append(line)

    return matched_lines


def compute_line_height_pt(
    line_bbox: list[float],
    image_height_px: float,
    page_height_pt: float,
) -> float:
    """Convert text-line bbox height from OCR image pixels to PDF points."""

    if image_height_px <= 0 or page_height_pt <= 0 or len(line_bbox) != 4:
        return 0.0

    line_height_px = max(0.0, float(line_bbox[3]) - float(line_bbox[1]))
    return (line_height_px / image_height_px) * page_height_pt


def filter_valid_line_heights(line_heights: list[float]) -> set[float]:
    """Filter impossible and noisy line heights using a robust page-level range."""

    valid = [height for height in line_heights if 3.0 <= height <= 72.0]
    if not valid:
        return set()

    if len(valid) < 4:
        return set(valid)

    arr = np.array(valid, dtype=float)
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = max(0.1, q3 - q1)
    lower = p10 - (1.5 * iqr)
    upper = p90 + (1.5 * iqr)

    return {height for height in valid if lower <= height <= upper}


def estimate_raw_font_size(line_heights: list[float]) -> float | None:
    """Estimate a raw font size from filtered line heights."""

    if not line_heights:
        return None
    if len(line_heights) >= 3:
        return float(median(line_heights))
    return float(mean(line_heights))


def build_font_size_profile(body_font_size_pt: float) -> dict[str, float]:
    """Generate stable page-level font-size buckets from the page body size."""

    md = clamp_font_size(body_font_size_pt or 10.5)
    profile = {
        "xs": clamp_font_size(round(md * 0.76, 1)),
        "sm": clamp_font_size(round(md * 0.88, 1)),
        "md": md,
        "lg": clamp_font_size(round(md * 1.25, 1)),
        "xl": clamp_font_size(round(md * 1.55, 1)),
    }
    return profile


def nearest_bucket(
    raw_size: float,
    profile: dict[str, float],
) -> str:
    """Map a raw size to the nearest bucket using midpoint thresholds."""

    xs = profile["xs"]
    sm = profile["sm"]
    md = profile["md"]
    lg = profile["lg"]
    xl = profile["xl"]

    thresholds = [
        (xs + sm) / 2.0,
        (sm + md) / 2.0,
        (md + lg) / 2.0,
        (lg + xl) / 2.0,
    ]

    if raw_size < thresholds[0]:
        return "xs"
    if raw_size < thresholds[1]:
        return "sm"
    if raw_size < thresholds[2]:
        return "md"
    if raw_size < thresholds[3]:
        return "lg"
    return "xl"


def normalize_font_size_bucket(
    raw_size: float,
    profile: dict[str, float],
    *,
    category: ElementCategory,
    label: str,
) -> str:
    """Normalize a raw size to a stable bucket with light semantic constraints."""

    bucket = nearest_bucket(raw_size, profile)

    if label in HEADER_LIKE_LABELS and bucket == "xs":
        bucket = "sm"
    elif category == ElementCategory.TABLE and bucket in {"lg", "xl"}:
        if raw_size <= profile["lg"]:
            bucket = "md"

    return bucket


def clamp_font_size(value: float, minimum: float = 6.0, maximum: float = 24.0) -> float:
    """Clamp a font size to the supported rendering range."""

    return float(max(minimum, min(maximum, value)))
