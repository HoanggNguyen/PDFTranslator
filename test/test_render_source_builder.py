"""Unit tests for table-cell rendering decisions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.render.config import RenderConfig
from pdf2zh.render.source_builder import (
    _downward_avail_height,
    _rightward_avail_width,
    build_typst_source,
)


class TestDownwardAvailHeight:
    """Collision-aware block height: overflow into empty space, not into neighbors."""

    BBOX = [40, 40, 360, 70]  # tight height = 30

    def test_no_neighbor_expands_by_max(self):
        # No element below → expand down by max_expand (bounded by page).
        h = _downward_avail_height(
            self.BBOX, [self.BBOX], page_height=800, max_expand=80
        )
        assert h == 30 + 80  # (70 + 80) - 40

    def test_neighbor_below_caps_expansion(self):
        neighbor = [40, 120, 360, 160]  # starts at y=120, overlaps horizontally
        h = _downward_avail_height(
            self.BBOX, [self.BBOX, neighbor], page_height=800, max_expand=80
        )
        assert h == 120 - 40  # capped at the neighbor's top

    def test_non_overlapping_neighbor_ignored(self):
        # A box below but in a different column must not cap the expansion.
        side = [400, 120, 500, 160]
        h = _downward_avail_height(
            self.BBOX, [self.BBOX, side], page_height=800, max_expand=80
        )
        assert h == 30 + 80

    def test_page_bottom_bounds_expansion(self):
        h = _downward_avail_height(
            self.BBOX, [self.BBOX], page_height=90, max_expand=80
        )
        assert h == 90 - 40  # page bottom closer than max_expand


class TestRightwardAvailWidth:
    """Single-line width: extend right into empty space, stop at a right neighbor."""

    BBOX = [80, 40, 290, 56]  # a TOC title on the left

    def test_no_neighbor_expands_to_page_edge(self):
        w = _rightward_avail_width(self.BBOX, [self.BBOX], page_width=595)
        assert w == 595 - 80

    def test_right_neighbor_caps_width(self):
        page_num = [527, 40, 543, 53]  # right-hand page number, same row
        w = _rightward_avail_width(self.BBOX, [self.BBOX, page_num], page_width=595)
        assert w == 527 - 80  # stops at the page number's left edge

    def test_neighbor_on_other_row_ignored(self):
        below = [527, 200, 543, 213]  # to the right but a different row
        w = _rightward_avail_width(self.BBOX, [self.BBOX, below], page_width=595)
        assert w == 595 - 80


def test_table_cell_without_source_text_is_not_rendered():
    parsed = {
        "pages": [
            {
                "page_width": 200,
                "page_height": 100,
                "elements": [
                    {
                        "category": "TABLE",
                        "label": "Table",
                        "bbox_pdf": [0, 0, 200, 100],
                        "cells": [
                            {
                                "bbox_pdf": [0, 0, 100, 20],
                                "source_text": "Source",
                                "translated_text": "Translated",
                            },
                            {
                                "bbox_pdf": [100, 0, 200, 20],
                                "translated_text": "Must remain original",
                            },
                        ],
                    }
                ],
            }
        ]
    }

    source = build_typst_source(
        parsed,
        {"p0:e0:c0": 10, "p0:e0:c1": 10},
        {},
        {},
        RenderConfig(),
    )

    assert "Translated" in source
    assert "Must remain original" not in source


def test_native_typst_content_is_embedded_without_eval():
    parsed = {
        "pages": [
            {
                "page_width": 200,
                "page_height": 100,
                "elements": [
                    {
                        "category": "EQUATION",
                        "label": "Equation",
                        "bbox_pdf": [0, 0, 100, 20],
                        "source_text": "Source",
                        "translated_text": "<math>arrow.r.double</math>",
                    }
                ],
            }
        ]
    }

    source = build_typst_source(parsed, {"p0:e0": 10}, {}, {}, RenderConfig())

    assert 'eval(markup, mode: "markup")' not in source
    assert "#let e0_0_tm = [$arrow.r.double$]" in source
