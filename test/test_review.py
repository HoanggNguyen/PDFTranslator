"""Unit tests for the Phase-1/Phase-3 review helpers (pdf2zh/webapp/review.py)."""

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.webapp.review import (
    add_element,
    apply_phase1_cell_edit,
    apply_phase1_edit,
    apply_phase2_cell_edit,
    apply_phase2_edit,
    hex_to_rgb,
    hit_test,
    normalize_click,
    output_page_position,
    overlay_svg,
    render_page_plain,
    render_page_with_boxes,
)


def _doc_with(elements):
    return {
        "pages": [
            {
                "page_index": 0,
                "page_width": 300,
                "page_height": 200,
                "elements": elements,
            }
        ]
    }


class TestHitTest:
    def test_smallest_box_wins_on_overlap(self):
        boxes = [
            {"elem_idx": 0, "bbox_img": [0, 0, 300, 200], "category": "BYPASS"},
            {"elem_idx": 1, "bbox_img": [10, 10, 60, 40], "category": "IN_PLACE"},
        ]
        # inside both -> smaller (idx 1); neither box has cell_idx -> None
        assert hit_test(boxes, 30, 25) == (1, None)
        assert hit_test(boxes, 200, 150) == (0, None)  # only the big one
        assert hit_test(boxes, 999, 999) is None  # outside all

    def test_hits_a_table_cell(self):
        boxes = [
            {
                "elem_idx": 2,
                "cell_idx": 0,
                "bbox_img": [0, 0, 50, 20],
                "category": "TABLE",
            },
            {
                "elem_idx": 2,
                "cell_idx": 1,
                "bbox_img": [50, 0, 100, 20],
                "category": "TABLE",
            },
        ]
        assert hit_test(boxes, 10, 10) == (2, 0)
        assert hit_test(boxes, 60, 10) == (2, 1)


class TestApplyPhase1Edit:
    def test_reclassify_picture_to_pageheader_sets_category(self):
        doc = _doc_with(
            [
                {
                    "label": "Picture",
                    "category": "BYPASS",
                    "bbox_pdf": [0, 0, 10, 10],
                    "source_text": "",
                    "translated_text": "",
                }
            ]
        )
        msg = apply_phase1_edit(doc, 0, 0, "PageHeader", "Chapter 1", bypass=False)
        assert msg is None
        elem = doc["pages"][0]["elements"][0]
        assert elem["label"] == "PageHeader"
        assert elem["category"] == "FLOWING_TEXT"
        assert elem["source_text"] == "Chapter 1"

    def test_bypass_sets_bypass_category(self):
        doc = _doc_with(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "bbox_pdf": [0, 0, 10, 10],
                    "source_text": "x",
                    "translated_text": "",
                }
            ]
        )
        apply_phase1_edit(doc, 0, 0, "Text", "x", bypass=True)
        assert doc["pages"][0]["elements"][0]["category"] == "BYPASS"

    def test_unbypass_restores_derived_category(self):
        doc = _doc_with(
            [
                {
                    "label": "SectionHeader",
                    "category": "BYPASS",
                    "bbox_pdf": [0, 0, 10, 10],
                    "source_text": "x",
                    "translated_text": "",
                }
            ]
        )
        apply_phase1_edit(doc, 0, 0, "SectionHeader", "x", bypass=False)
        assert doc["pages"][0]["elements"][0]["category"] == "FLOWING_TEXT"

    def test_changing_to_table_is_blocked(self):
        doc = _doc_with(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "bbox_pdf": [0, 0, 10, 10],
                    "source_text": "x",
                    "translated_text": "",
                }
            ]
        )
        msg = apply_phase1_edit(doc, 0, 0, "Table", "x", bypass=False)
        assert msg is not None
        assert doc["pages"][0]["elements"][0]["label"] == "Text"  # unchanged

    def test_missing_element_returns_message(self):
        doc = _doc_with([])
        assert apply_phase1_edit(doc, 0, 5, "Text", "x", bypass=False) is not None


class TestApplyPhase2Edit:
    def test_sets_translated_text(self):
        doc = _doc_with(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "bbox_pdf": [0, 0, 10, 10],
                    "source_text": "hi",
                    "translated_text": "old",
                }
            ]
        )
        apply_phase2_edit(doc, 0, 0, "xin chào")
        assert doc["pages"][0]["elements"][0]["translated_text"] == "xin chào"


def _table_doc():
    return _doc_with(
        [
            {
                "label": "Table",
                "category": "TABLE",
                "bbox_pdf": [0, 0, 100, 20],
                "source_text": "",
                "translated_text": "",
                "cells": [
                    {
                        "bbox_pdf": [0, 0, 50, 20],
                        "bbox_text": [0, 0, 50, 20],
                        "source_text": "hi",
                        "translated_text": "",
                    },
                    {
                        "bbox_pdf": [50, 0, 100, 20],
                        "bbox_text": [50, 0, 100, 20],
                        "source_text": "there",
                        "translated_text": "",
                    },
                ],
            }
        ]
    )


class TestApplyPhase1CellEdit:
    def test_sets_cell_source_text(self):
        doc = _table_doc()
        msg = apply_phase1_cell_edit(doc, 0, 0, 1, "xin")
        assert msg is None
        assert doc["pages"][0]["elements"][0]["cells"][1]["source_text"] == "xin"

    def test_missing_cell_returns_message(self):
        doc = _table_doc()
        assert apply_phase1_cell_edit(doc, 0, 0, 5, "x") is not None


class TestApplyPhase2CellEdit:
    def test_sets_cell_translated_text(self):
        doc = _table_doc()
        msg = apply_phase2_cell_edit(doc, 0, 0, 0, "chào")
        assert msg is None
        assert doc["pages"][0]["elements"][0]["cells"][0]["translated_text"] == "chào"

    def test_missing_cell_returns_message(self):
        doc = _table_doc()
        assert apply_phase2_cell_edit(doc, 0, 0, 5, "x") is not None


class TestAddElement:
    def test_append_keeps_indices_and_derives_category(self):
        doc = _doc_with(
            [
                {
                    "label": "Text",
                    "category": "FLOWING_TEXT",
                    "bbox_pdf": [0, 0, 10, 10],
                    "source_text": "a",
                    "translated_text": "",
                }
            ]
        )
        idx = add_element(doc, 0, [20, 20, 80, 40], "Caption", "một chú thích")
        assert idx == 1
        elem = doc["pages"][0]["elements"][1]
        assert elem["category"] == "IN_PLACE"  # Caption -> IN_PLACE
        assert elem["source_text"] == "một chú thích"
        assert elem["translated_text"] == ""

    def test_no_color_keys_when_not_provided(self):
        doc = _doc_with([])
        add_element(doc, 0, [0, 0, 10, 10], "Text", "x")
        elem = doc["pages"][0]["elements"][0]
        assert "bg_color" not in elem
        assert "text_color" not in elem

    def test_stores_color_overrides_when_provided(self):
        doc = _doc_with([])
        add_element(
            doc,
            0,
            [0, 0, 10, 10],
            "Text",
            "x",
            bg_color=[255, 200, 0],
            text_color=[10, 20, 30],
        )
        elem = doc["pages"][0]["elements"][0]
        assert elem["bg_color"] == [255, 200, 0]
        assert elem["text_color"] == [10, 20, 30]


class TestHexToRgb:
    def test_six_digit_hex(self):
        assert hex_to_rgb("#ffcc00") == [255, 204, 0]

    def test_three_digit_hex_expands(self):
        assert hex_to_rgb("#fc0") == [255, 204, 0]

    def test_rgb_and_rgba_forms(self):
        assert hex_to_rgb("rgb(255, 204, 0)") == [255, 204, 0]
        assert hex_to_rgb("rgba(255, 204, 0, 0.5)") == [255, 204, 0]

    def test_bad_values_return_none(self):
        assert hex_to_rgb(None) is None
        assert hex_to_rgb("") is None
        assert hex_to_rgb("#12") is None
        assert hex_to_rgb("not-a-color") is None
        assert hex_to_rgb("rgb(1, 2)") is None


class TestOutputPagePosition:
    def test_range(self):
        assert output_page_position([2, 3, 4], 3) == 1
        assert output_page_position([4, 2, 3], 4) == 2  # sorted -> [2,3,4]
        assert output_page_position([2, 3, 4], 5) is None

    def test_none_is_identity(self):
        assert output_page_position(None, 7) == 7


class TestNormalizeClick:
    def test_tuple_ok(self):
        assert normalize_click((12, 34)) == (12.0, 34.0)
        assert normalize_click([12, 34, 0]) == (12.0, 34.0)

    def test_bad_returns_none(self):
        assert normalize_click(None) is None
        assert normalize_click(5) is None


class TestRenderPageWithBoxes:
    def test_boxes_scaled_to_image_pixels(self, tmp_path):
        pdf_path = tmp_path / "p.pdf"
        doc = fitz.open()
        doc.new_page(width=300, height=200)
        doc.save(str(pdf_path))
        doc.close()

        elements = [
            {
                "label": "Text",
                "category": "FLOWING_TEXT",
                "bbox_pdf": [10, 20, 110, 60],
                "source_text": "",
                "translated_text": "",
            },
            {
                "label": "Picture",
                "category": "BYPASS",
                "bbox_pdf": [0, 0, 300, 200],
                "source_text": "",
                "translated_text": "",
            },
        ]
        # dpi=72 -> scale 1.0 -> bbox_img == bbox_pdf, image == page size.
        img, boxes, scale = render_page_with_boxes(str(pdf_path), 0, elements, dpi=72)
        assert scale == 1.0
        assert img.size == (300, 200)
        assert len(boxes) == 2
        assert boxes[0]["bbox_img"] == [10, 20, 110, 60]
        # hit_test on the returned boxes selects the small element.
        assert hit_test(boxes, 50, 40) == (0, None)

    def test_table_draws_one_box_per_cell(self, tmp_path):
        pdf_path = tmp_path / "p.pdf"
        doc = fitz.open()
        doc.new_page(width=300, height=200)
        doc.save(str(pdf_path))
        doc.close()

        elements = [
            {
                "label": "Table",
                "category": "TABLE",
                "bbox_pdf": [0, 0, 100, 20],
                "source_text": "",
                "translated_text": "",
                "cells": [
                    {
                        "bbox_pdf": [0, 0, 50, 20],
                        "bbox_text": [0, 0, 50, 20],
                        "source_text": "hi",
                        "translated_text": "",
                    },
                    {
                        "bbox_pdf": [50, 0, 100, 20],
                        "bbox_text": [50, 0, 100, 20],
                        "source_text": "there",
                        "translated_text": "",
                    },
                ],
            }
        ]
        _, boxes, _ = render_page_with_boxes(str(pdf_path), 0, elements, dpi=72)
        assert len(boxes) == 2  # per-cell, not one box for the whole table
        assert [b["cell_idx"] for b in boxes] == [0, 1]
        assert all(b["elem_idx"] == 0 for b in boxes)
        assert hit_test(boxes, 10, 10) == (0, 0)
        assert hit_test(boxes, 60, 10) == (0, 1)


_OVERLAY_ELEMENTS = [
    {
        "label": "Text",
        "category": "FLOWING_TEXT",
        "bbox_pdf": [10, 20, 110, 60],
        "source_text": "",
        "translated_text": "",
    },
    {
        "label": "Picture",
        "category": "BYPASS",
        "bbox_pdf": [0, 0, 300, 200],
        "source_text": "",
        "translated_text": "",
    },
]


class TestRenderPagePlain:
    def test_size_matches_render_page_with_boxes(self, tmp_path):
        pdf_path = tmp_path / "p.pdf"
        doc = fitz.open()
        doc.new_page(width=300, height=200)
        doc.save(str(pdf_path))
        doc.close()

        # dpi=72 -> scale 1.0 -> image == page size, and matches the boxed render.
        img, size = render_page_plain(str(pdf_path), 0, dpi=72)
        assert size == (300, 200)
        assert img.size == (300, 200)
        boxed, _, _ = render_page_with_boxes(str(pdf_path), 0, [], dpi=72)
        assert img.size == boxed.size

    def test_out_of_range_raises(self, tmp_path):
        pdf_path = tmp_path / "p.pdf"
        doc = fitz.open()
        doc.new_page(width=100, height=100)
        doc.save(str(pdf_path))
        doc.close()
        import pytest

        with pytest.raises(IndexError):
            render_page_plain(str(pdf_path), 5, dpi=72)


class TestOverlaySvg:
    def test_boxes_identical_to_render_page_with_boxes(self, tmp_path):
        pdf_path = tmp_path / "p.pdf"
        doc = fitz.open()
        doc.new_page(width=300, height=200)
        doc.save(str(pdf_path))
        doc.close()

        _, ref_boxes, scale = render_page_with_boxes(
            str(pdf_path), 0, _OVERLAY_ELEMENTS, dpi=72
        )
        _, boxes = overlay_svg(_OVERLAY_ELEMENTS, scale, 300, 200)
        assert boxes == ref_boxes
        # hit_test behaves the same on both box lists.
        assert hit_test(boxes, 50, 40) == (0, None)

    def test_svg_has_rect_per_element_and_highlight_stroke(self):
        svg, boxes = overlay_svg(_OVERLAY_ELEMENTS, 1.0, 300, 200, highlight_idx=0)
        assert svg.startswith('<svg viewBox="0 0 300 200"')
        assert svg.count("<rect") == len(_OVERLAY_ELEMENTS) == 2
        # index tag per element.
        assert svg.count("<text") == 2
        # highlighted element -> red stroke; bypass element -> gray stroke.
        assert "rgb(230,30,30)" in svg
        assert "rgb(150,150,150)" in svg

    def test_skips_elements_without_valid_bbox(self):
        elements = [
            {"category": "FLOWING_TEXT"},  # no bbox_pdf
            {"category": "FLOWING_TEXT", "bbox_pdf": [1, 2, 3]},  # wrong length
            {"category": "FLOWING_TEXT", "bbox_pdf": [0, 0, 10, 10]},
        ]
        svg, boxes = overlay_svg(elements, 1.0, 50, 50)
        assert len(boxes) == 1
        assert boxes[0]["elem_idx"] == 2

    def test_table_draws_one_rect_per_cell_plus_dashed_outer_border(self):
        elements = [
            {
                "label": "Table",
                "category": "TABLE",
                "bbox_pdf": [0, 0, 100, 20],
                "source_text": "",
                "translated_text": "",
                "cells": [
                    {
                        "bbox_pdf": [0, 0, 50, 20],
                        "bbox_text": [0, 0, 50, 20],
                        "source_text": "hi",
                        "translated_text": "",
                    },
                    {
                        "bbox_pdf": [50, 0, 100, 20],
                        "bbox_text": [50, 0, 100, 20],
                        "source_text": "there",
                        "translated_text": "",
                    },
                ],
            }
        ]
        svg, boxes = overlay_svg(elements, 1.0, 100, 20)
        assert len(boxes) == 2  # per-cell boxes, not one for the whole table
        assert [b["cell_idx"] for b in boxes] == [0, 1]
        # 2 cell rects + 1 dashed outer border rect.
        assert svg.count("<rect") == 3
        assert "stroke-dasharray" in svg

    def test_table_cell_highlight_reddens_only_that_cell(self):
        elements = [
            {
                "label": "Table",
                "category": "TABLE",
                "bbox_pdf": [0, 0, 100, 20],
                "source_text": "",
                "translated_text": "",
                "cells": [
                    {
                        "bbox_pdf": [0, 0, 50, 20],
                        "bbox_text": [0, 0, 50, 20],
                        "source_text": "hi",
                        "translated_text": "",
                    },
                    {
                        "bbox_pdf": [50, 0, 100, 20],
                        "bbox_text": [50, 0, 100, 20],
                        "source_text": "there",
                        "translated_text": "",
                    },
                ],
            }
        ]
        svg, _ = overlay_svg(elements, 1.0, 100, 20, highlight_cell=(0, 1))
        assert "rgb(230,30,30)" in svg  # the selected cell turns red
        assert 'stroke-width="3"' in svg
