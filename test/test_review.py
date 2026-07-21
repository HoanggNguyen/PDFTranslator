"""Unit tests for the Phase-1/Phase-3 review helpers (pdf2zh/webapp/review.py)."""

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.webapp.review import (
    add_element,
    apply_phase1_edit,
    apply_phase2_edit,
    hit_test,
    normalize_click,
    output_page_position,
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
        assert hit_test(boxes, 30, 25) == 1  # inside both -> smaller (idx 1)
        assert hit_test(boxes, 200, 150) == 0  # only the big one
        assert hit_test(boxes, 999, 999) is None  # outside all


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
        assert hit_test(boxes, 50, 40) == 0
