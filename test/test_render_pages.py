"""Regression tests for arbitrary page-range rendering + output-only-translated.

The render layer used to conflate the compacted-list enumerate index with the
original page number, so any range not starting at page 0 silently rendered
nothing. These tests pin the fixed behavior:
  - a range like [2, 3, 4] (not starting at 0) renders correctly, and
  - the output PDF contains ONLY the translated pages, in order.
"""

import shutil
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.render.config import RenderConfig
from pdf2zh.render.renderer import render_document

TYPST = shutil.which("typst")


def _make_pdf(path: Path, n_pages: int) -> None:
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=300, height=200)
        page.insert_text((20, 40), f"Original page {i} native text")
    doc.save(str(path))
    doc.close()


def _compacted_parsed(page_indices: list[int]) -> dict:
    """A parsed doc compacted to `page_indices` (mirrors parse_pdf(pages=...))."""
    return {
        "pages": [
            {
                "page_index": pi,
                "page_width": 300,
                "page_height": 200,
                "elements": [
                    {
                        "category": "FLOWING_TEXT",
                        "label": "Text",
                        "bbox_pdf": [20, 30, 280, 55],
                        "source_text": f"Original page {pi} native text",
                        "translated_text": f"Trang dịch số {pi}",
                    }
                ],
            }
            for pi in page_indices
        ]
    }


@pytest.mark.skipif(TYPST is None, reason="typst binary not installed")
class TestArbitraryPageRange:
    def test_midrange_renders_and_outputs_only_selected(self, tmp_path):
        pdf_path = tmp_path / "src.pdf"
        _make_pdf(pdf_path, n_pages=6)

        pages = [2, 3, 4]  # 0-based, NOT starting at 0
        parsed = _compacted_parsed(pages)
        cfg = RenderConfig(typst_binary=TYPST)
        cfg.pages = pages
        out_pdf = tmp_path / "out.pdf"

        stats = render_document(pdf_path, parsed, out_pdf, cfg)

        # Rendered all 3 selected pages (the pre-fix bug rendered 0).
        assert stats["pages"] == 3
        assert stats["elements_rendered"] == 3

        # Output contains ONLY the translated pages, not the full 6-page doc.
        out = fitz.open(str(out_pdf))
        try:
            assert out.page_count == 3
            # Translated text present; the out-of-range page-0 text must be gone.
            assert "Trang dịch số 2" in out[0].get_text()
            assert "Original page 0" not in out[0].get_text()
        finally:
            out.close()

    def test_pages_none_keeps_all_pages(self, tmp_path):
        pdf_path = tmp_path / "src.pdf"
        _make_pdf(pdf_path, n_pages=3)

        parsed = _compacted_parsed([0, 1, 2])
        cfg = RenderConfig(typst_binary=TYPST)
        cfg.pages = None  # translate whole document
        out_pdf = tmp_path / "out.pdf"

        render_document(pdf_path, parsed, out_pdf, cfg)

        out = fitz.open(str(out_pdf))
        try:
            assert out.page_count == 3
        finally:
            out.close()


_PARA = (
    "This paragraph is long enough in the source that autofit picks a modest "
    "font size and the box is treated as multi line rather than single line."
)
_LONG = _PARA + " " + _PARA + " " + _PARA  # translated overflows the tight box


def _para_span_stats(pdf: Path):
    doc = fitz.open(str(pdf))
    try:
        sizes, bottoms = [], []
        for b in doc[0].get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    if "paragraph" in s["text"] or "multi" in s["text"]:
                        sizes.append(s["size"])
                        bottoms.append(s["bbox"][3])
        return (max(sizes) if sizes else 0.0), (max(bottoms) if bottoms else 0.0)
    finally:
        doc.close()


@pytest.mark.skipif(TYPST is None, reason="typst binary not installed")
class TestCollisionAwareSizing:
    """Text keeps its size and overflows into empty space, but shrinks to avoid
    colliding with a neighbor below (mirrors the sizing heuristic)."""

    def _render(self, tmp_path, with_neighbor):
        pdf_path = tmp_path / f"src_{with_neighbor}.pdf"
        doc = fitz.open()
        doc.new_page(width=400, height=400)
        doc.save(str(pdf_path))
        doc.close()

        els = [
            {
                "label": "Text",
                "category": "FLOWING_TEXT",
                "bbox_pdf": [40, 40, 360, 110],
                "source_text": _PARA,
                "translated_text": _LONG,
                "cells": [],
            }
        ]
        if with_neighbor:
            els.append(
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "bbox_pdf": [40, 116, 360, 150],
                    "source_text": "N",
                    "translated_text": "Neighbor line",
                    "cells": [],
                }
            )
        parsed = {
            "pages": [
                {
                    "page_index": 0,
                    "page_width": 400,
                    "page_height": 400,
                    "elements": els,
                }
            ]
        }
        out = tmp_path / f"out_{with_neighbor}.pdf"
        render_document(pdf_path, parsed, out, RenderConfig(typst_binary=TYPST))
        return _para_span_stats(out)

    def test_no_neighbor_keeps_larger_size_and_overflows_down(self, tmp_path):
        size_free, bottom_free = self._render(tmp_path, with_neighbor=False)
        size_near, _ = self._render(tmp_path, with_neighbor=True)
        # Empty space below → keeps a bigger font than when a neighbor forces a shrink.
        assert size_free > size_near + 0.5
        # And the text is allowed to overflow below the tight bbox (y1 = 110).
        assert bottom_free > 110

    _LONG_TITLE = (
        "A very long translated chapter title that would run past the page number "
        "column if it were allowed to expand all the way to the right page margin"
    )

    def _title_right_edge(self, tmp_path, with_number):
        pdf_path = tmp_path / f"toc_{with_number}.pdf"
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        doc.save(str(pdf_path))
        doc.close()

        els = [
            {
                "label": "Text",
                "category": "FLOWING_TEXT",
                "bbox_pdf": [80, 40, 290, 56],  # single-line title
                "source_text": "Short",
                "translated_text": self._LONG_TITLE,
                "cells": [],
            }
        ]
        if with_number:
            els.append(
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "bbox_pdf": [527, 40, 543, 53],  # page number, same row
                    "source_text": "ii",
                    "translated_text": "ii",
                    "cells": [],
                }
            )
        parsed = {
            "pages": [
                {
                    "page_index": 0,
                    "page_width": 595,
                    "page_height": 842,
                    "elements": els,
                }
            ]
        }
        out = tmp_path / f"toc_out_{with_number}.pdf"
        render_document(pdf_path, parsed, out, RenderConfig(typst_binary=TYPST))
        doc = fitz.open(str(out))
        try:
            right = 0.0
            for b in doc[0].get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for s in line["spans"]:
                        if s["text"].strip() != "ii":
                            right = max(right, s["bbox"][2])
            return right
        finally:
            doc.close()

    def test_single_line_title_stops_before_right_neighbor(self, tmp_path):
        with_num = self._title_right_edge(tmp_path, with_number=True)
        without_num = self._title_right_edge(tmp_path, with_number=False)
        # With a page number on the right, the title must not overrun its left edge.
        assert with_num <= 527.5
        # Without it, the title is free to use the rest of the page width.
        assert without_num > with_num


class TestColorOverride:
    """User-added boxes may carry explicit bg/text colors (review.add_element)."""

    def test_sample_colors_honors_element_overrides(self, tmp_path):
        from pdf2zh.render.renderer import _sample_colors

        pdf = tmp_path / "p.pdf"
        _make_pdf(pdf, 1)
        parsed = {
            "pages": [
                {
                    "page_index": 0,
                    "page_width": 300,
                    "page_height": 200,
                    "elements": [
                        {
                            "category": "FLOWING_TEXT",
                            "label": "Text",
                            "bbox_pdf": [20, 30, 120, 55],
                            "source_text": "x",
                            "translated_text": "y",
                            "bg_color": [10, 20, 30],
                            "text_color": [200, 100, 50],
                        }
                    ],
                }
            ]
        }
        bg_colors: dict = {}
        text_colors: dict = {}
        stats = {"bg_samples": 0}
        _sample_colors(pdf, parsed, RenderConfig(), {}, bg_colors, text_colors, stats)
        # Override used verbatim — no sampling from the page pixels.
        assert bg_colors["p0:e0"] == (10, 20, 30)
        assert text_colors["p0:e0"] == (200, 100, 50)
