"""Utility modules for the scanned PDF pipeline.

This package contains:
- bbox: Bounding box conversion and manipulation
- ocr_text: OCR text cleaning and extraction
- image: PDF page rendering and cropping
- hardware: Device detection and hardware profiles
"""

from pdf2zh.scanned.utils.bbox import (
    clamp_bbox,
    convert_bbox,
    image_bbox_to_pdf,
    is_degenerate,
    offset_bbox,
    polygon_to_bbox,
)
from pdf2zh.scanned.utils.hardware import (
    HardwareProfile,
    resolve_hardware,
)
from pdf2zh.scanned.utils.ocr_text import (
    clean_ocr_text,
    collect_ocr_text,
    extract_text_for_region,
    log_toc_hints,
)

__all__ = [
    # bbox
    "convert_bbox",
    "clamp_bbox",
    "offset_bbox",
    "is_degenerate",
    "polygon_to_bbox",
    "image_bbox_to_pdf",
    # ocr_text
    "clean_ocr_text",
    "collect_ocr_text",
    "extract_text_for_region",
    "log_toc_hints",
    # hardware
    "resolve_hardware",
    "HardwareProfile",
]
