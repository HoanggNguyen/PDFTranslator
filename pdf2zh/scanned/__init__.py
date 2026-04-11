"""Scanned PDF translation pipeline - Stage A.

This package provides parsing and analysis for scanned (image-based) PDFs
using Surya for layout detection and OCR.

Main exports:
- ParsedDocument: Complete parsed document structure
- PageData, ElementData, CellData: Page and element data models
- ChapterInfo: Chapter metadata (filled by Stage B)
- ElementCategory: Element category enum (BYPASS, FLOWING_TEXT, etc.)
- PDFTypeDetector: Detect if PDF is scanned, digital, or mixed
- StageAParser: Main parser for Stage A processing
"""

from pdf2zh.scanned.enums import (
    ElementCategory,
    SuryaLabel,
    SURYA_LABEL_MAP,
    DEFAULT_CATEGORY,
)
from pdf2zh.scanned.models import (
    CellData,
    ElementData,
    PageData,
    ChapterInfo,
    ParsedDocument,
)
from pdf2zh.scanned.detector import PDFTypeDetector
from pdf2zh.scanned.parser import StageAParser

__all__ = [
    # Enums and mappings
    "ElementCategory",
    "SuryaLabel",
    "SURYA_LABEL_MAP",
    "DEFAULT_CATEGORY",
    # Data models
    "CellData",
    "ElementData",
    "PageData",
    "ChapterInfo",
    "ParsedDocument",
    # Detector
    "PDFTypeDetector",
    "StageAParser",
]
