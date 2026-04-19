"""Data models for the scanned PDF pipeline.

This module defines the core dataclasses used throughout Stage A:
- CellData: Individual table cell with bbox, row/col indices, and text
- ElementData: A layout element (text block, figure, table, etc.)
- PageData: A single page with dimensions, elements, and metadata
- ChapterInfo: Chapter metadata (filled by Stage B)
- ParsedDocument: The complete parsed document structure
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from PIL import Image

from pdf2zh.scanned.enums import ElementCategory


@dataclass
class CellData:
    """A single cell within a TABLE element.

    Attributes:
        bbox_pdf: [x0, y0, x1, y1] in absolute PDF points (page-level coordinates)
        row_id: 0-based row index
        col_id: 0-based column index
        source_text: OCR text content; empty string for empty cells
        translated_text: Empty string after Stage A; filled by Stage C
    """

    bbox_pdf: list[float]
    row_id: int
    col_id: int
    source_text: str
    translated_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this cell to a JSON-compatible dictionary.

        Returns:
            Dict with keys ``bbox_pdf``, ``row_id``, ``col_id``,
            ``source_text``, and ``translated_text``.
        """
        return {
            "bbox_pdf": self.bbox_pdf,
            "row_id": self.row_id,
            "col_id": self.col_id,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CellData:
        """Deserialize a :class:`CellData` from a plain dictionary.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.  ``translated_text``
                  defaults to ``""`` if absent (backward compatibility).

        Returns:
            New :class:`CellData` instance.
        """
        return cls(
            bbox_pdf=data["bbox_pdf"],
            row_id=data["row_id"],
            col_id=data["col_id"],
            source_text=data["source_text"],
            translated_text=data.get("translated_text", ""),
        )


@dataclass
class ElementData:
    """A layout element detected by Surya.

    Attributes:
        label: Raw Surya label (e.g., "Text", "Section-header", "Table")
        category: One of the 5 ElementCategory values determining handling
        bbox_pdf: [x0, y0, x1, y1] in PDF points; x0 < x1, y0 < y1
        source_text: OCR text; always "" for BYPASS and EQUATION categories
        translated_text: Empty string after Stage A; filled by Stage C
        latex: "[EQUATION_PLACEHOLDER]" for EQUATION category; "" otherwise
        cells: Non-empty only for TABLE category; empty list otherwise
    """

    label: str
    category: ElementCategory
    bbox_pdf: list[float]
    source_text: str
    translated_text: str = ""
    latex: str = ""
    cells: list[CellData] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this element to a JSON-compatible dictionary.

        The ``category`` field is stored as its string value so the output is
        JSON-serializable without custom encoders.

        Returns:
            Dict with keys ``label``, ``category``, ``bbox_pdf``,
            ``source_text``, ``translated_text``, ``latex``, and ``cells``.
        """
        return {
            "label": self.label,
            "category": (
                self.category.value
                if isinstance(self.category, ElementCategory)
                else self.category
            ),
            "bbox_pdf": self.bbox_pdf,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "latex": self.latex,
            "cells": [c.to_dict() for c in self.cells],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElementData:
        """Deserialize an :class:`ElementData` from a plain dictionary.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.  Optional keys
                  ``translated_text``, ``latex``, and ``cells`` default to
                  ``""``, ``""``, and ``[]`` respectively.

        Returns:
            New :class:`ElementData` instance.
        """
        return cls(
            label=data["label"],
            category=ElementCategory(data["category"]),
            bbox_pdf=data["bbox_pdf"],
            source_text=data["source_text"],
            translated_text=data.get("translated_text", ""),
            latex=data.get("latex", ""),
            cells=[CellData.from_dict(c) for c in data.get("cells", [])],
        )


@dataclass
class PageData:
    """Data for a single PDF page.

    Attributes:
        page_index: 0-based page number
        page_width: Width in PDF points (from page.rect.width)
        page_height: Height in PDF points (from page.rect.height)
        elements: Layout elements in top-to-bottom reading order
        raw_text: Joined source_text of FLOWING_TEXT and IN_PLACE elements
        chapter_id: Empty string after Stage A; filled by Stage B
    """

    page_index: int
    page_width: float
    page_height: float
    elements: list[ElementData] = field(default_factory=list)
    raw_text: str = ""
    chapter_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this page to a JSON-compatible dictionary.

        Returns:
            Dict with keys ``page_index``, ``page_width``, ``page_height``,
            ``elements``, ``raw_text``, and ``chapter_id``.
        """
        return {
            "page_index": self.page_index,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "elements": [e.to_dict() for e in self.elements],
            "raw_text": self.raw_text,
            "chapter_id": self.chapter_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageData:
        """Deserialize a :class:`PageData` from a plain dictionary.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.  Optional keys
                  ``elements``, ``raw_text``, and ``chapter_id`` default to
                  ``[]``, ``""``, and ``""`` respectively.

        Returns:
            New :class:`PageData` instance.
        """
        return cls(
            page_index=data["page_index"],
            page_width=data["page_width"],
            page_height=data["page_height"],
            elements=[ElementData.from_dict(e) for e in data.get("elements", [])],
            raw_text=data.get("raw_text", ""),
            chapter_id=data.get("chapter_id", ""),
        )


@dataclass
class ChapterInfo:
    """Chapter metadata (empty after Stage A, filled by Stage B).

    Attributes:
        chapter_id: Identifier like "ch_0", "ch_1", etc.
        title: Chapter heading text; empty if not found
        start_page: 0-based inclusive start page
        end_page: 0-based inclusive end page (end_page >= start_page)
        summary: LLM-generated summary; empty initially
        glossary: {term: definition}; empty initially
    """

    chapter_id: str
    title: str
    start_page: int
    end_page: int
    summary: str = ""
    glossary: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this chapter info to a JSON-compatible dictionary.

        Returns:
            Dict with keys ``chapter_id``, ``title``, ``start_page``,
            ``end_page``, ``summary``, and ``glossary``.
        """
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "glossary": self.glossary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterInfo:
        """Deserialize a :class:`ChapterInfo` from a plain dictionary.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.  Optional keys
                  ``summary`` and ``glossary`` default to ``""`` and ``{}``.

        Returns:
            New :class:`ChapterInfo` instance.
        """
        return cls(
            chapter_id=data["chapter_id"],
            title=data["title"],
            start_page=data["start_page"],
            end_page=data["end_page"],
            summary=data.get("summary", ""),
            glossary=data.get("glossary", {}),
        )


@dataclass
class ParsedDocument:
    """Complete parsed document from Stage A.

    Attributes:
        pdf_path: Path to the source PDF file
        pages: List of PageData, one per page, 0-based order
        chapters: Empty list after Stage A; filled by Stage B
        glossary: Empty dict after Stage A; filled by Stage B
    """

    pdf_path: str
    pages: list[PageData] = field(default_factory=list)
    chapters: list[ChapterInfo] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full document to a JSON-compatible dictionary.

        Returns:
            Dict with keys ``pdf_path``, ``pages``, ``chapters``,
            and ``glossary``.
        """
        return {
            "pdf_path": self.pdf_path,
            "pages": [p.to_dict() for p in self.pages],
            "chapters": [c.to_dict() for c in self.chapters],
            "glossary": self.glossary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedDocument:
        """Deserialize a :class:`ParsedDocument` from a plain dictionary.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.  Optional keys
                  ``pages``, ``chapters``, and ``glossary`` default to
                  ``[]``, ``[]``, and ``{}`` respectively.

        Returns:
            New :class:`ParsedDocument` instance.
        """
        return cls(
            pdf_path=data["pdf_path"],
            pages=[PageData.from_dict(p) for p in data.get("pages", [])],
            chapters=[ChapterInfo.from_dict(c) for c in data.get("chapters", [])],
            glossary=data.get("glossary", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize the document to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation (default 2).

        Returns:
            Pretty-printed JSON string with UTF-8 characters unescaped
            (``ensure_ascii=False``).
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ParsedDocument:
        """Deserialize a :class:`ParsedDocument` from a JSON string.

        Args:
            json_str: JSON string as produced by :meth:`to_json`.

        Returns:
            New :class:`ParsedDocument` instance.
        """
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str | Path) -> None:
        """Save the document to a JSON file on disk.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path (``str`` or :class:`~pathlib.Path`).
        """
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ParsedDocument:
        """Load a :class:`ParsedDocument` from a JSON file on disk.

        Args:
            path: Source file path (``str`` or :class:`~pathlib.Path`).

        Returns:
            New :class:`ParsedDocument` instance parsed from the file.
        """
        path = Path(path)
        return cls.from_json(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class LayoutBlockResult:
    """A layout block with stable IDs and coordinates in all required spaces."""

    block_id: str
    page_index: int
    position: int
    label: str
    category: ElementCategory
    bbox_layout: list[float]
    bbox_image: list[float]
    bbox_pdf: list[float]


@dataclass(slots=True)
class LayoutPageResult:
    """Layout output for one page."""

    page_index: int
    page_width: float
    page_height: float
    layout_image_bbox: list[float]
    image_bbox: list[float]
    blocks: list[LayoutBlockResult] = field(default_factory=list)


@dataclass(slots=True)
class LayoutParseResult:
    """Full layout phase output."""

    pdf_path: str
    pages: list[LayoutPageResult] = field(default_factory=list)

    def page_map(self) -> dict[int, LayoutPageResult]:
        return {page.page_index: page for page in self.pages}

    def block_map(self) -> dict[str, LayoutBlockResult]:
        return {
            block.block_id: block
            for page in self.pages
            for block in page.blocks
        }


@dataclass(slots=True)
class OCRPageResult:
    """OCR output for one full page."""

    page_index: int
    image_bbox: list[float]
    ocr_result: Any

    @property
    def image_width(self) -> float:
        return self.image_bbox[2] - self.image_bbox[0]

    @property
    def image_height(self) -> float:
        return self.image_bbox[3] - self.image_bbox[1]


@dataclass(slots=True)
class OCRParseResult:
    """Full-page OCR phase output."""

    pdf_path: str
    pages: list[OCRPageResult] = field(default_factory=list)

    def page_map(self) -> dict[int, OCRPageResult]:
        return {page.page_index: page for page in self.pages}


@dataclass(slots=True)
class TableBlockResult:
    """Merged table output for one layout table block."""

    block_id: str
    source_text: str
    cells: list[CellData] = field(default_factory=list)
    used_fallback_ocr: bool = False


@dataclass(slots=True)
class TableParseResult:
    """Table phase output indexed by layout block id."""

    pdf_path: str
    tables: dict[str, TableBlockResult] = field(default_factory=dict)


@dataclass(slots=True)
class EquationBlockResult:
    """Equation phase output for one layout equation block."""

    block_id: str
    latex: str


@dataclass(slots=True)
class EquationParseResult:
    """Equation phase output indexed by layout block id."""

    pdf_path: str
    equations: dict[str, EquationBlockResult] = field(default_factory=dict)


@dataclass(slots=True)
class _DocumentContext:
    """Immutable page selection and geometry for one PDF parse request."""

    pdf_path: Path
    page_indices: list[int]
    page_dims: dict[int, tuple[float, float]]


@dataclass(slots=True)
class _TableJob:
    """Bookkeeping for one table crop inside a batch."""

    block: LayoutBlockResult
    page_width: float
    page_height: float
    table_crop: Image.Image
    highres_crop: Image.Image
    page_ocr: OCRPageResult | None