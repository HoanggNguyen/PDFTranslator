import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image

from pdf2zh.scanned.enums import ElementCategory
from pdf2zh.scanned.models import (
    LayoutPageResult,
    LayoutParseResult,
    OCRPageResult,
    OCRParseResult,
    TableBlockResult,
    TableParseResult,
)
from pdf2zh.scanned.parser import StageAParser


class _DummyLayoutModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def __call__(self, images, batch_size=None, auto_unload=False):
        return self.predictions


class _DummyTableModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def __call__(self, images, batch_size=None, auto_unload=False):
        self.calls.append(
            {
                "count": len(images),
                "batch_size": batch_size,
                "sizes": [image.size for image in images],
            }
        )
        return self.predictions


class TestSparseTextTableRelabel(unittest.TestCase):
    def setUp(self):
        self.parser = object.__new__(StageAParser)
        self.parser.hardware = SimpleNamespace(
            layout_batch_size=4,
            table_batch_size=2,
        )

    def _make_line(self, bbox, text):
        return SimpleNamespace(bbox=bbox, text=text)

    def _make_ocr_page(self, lines):
        return OCRPageResult(
            page_index=0,
            image_bbox=[0, 0, 100, 100],
            ocr_result=SimpleNamespace(text_lines=lines),
        )

    def _make_layout_prediction(self, label, bbox=None):
        if bbox is None:
            bbox = [10, 10, 90, 70]
        return [
            SimpleNamespace(
                image_bbox=[0, 0, 100, 100],
                bboxes=[SimpleNamespace(label=label, bbox=bbox, position=0)],
            )
        ]

    def _parse_single_block(self, label, lines):
        self.parser.layout_model = _DummyLayoutModel(
            self._make_layout_prediction(label)
        )
        images = [Image.new("RGB", (100, 100), color="white")]
        layout_pages = self.parser._parse_layout_batch(
            [0],
            {0: (100.0, 100.0)},
            images,
            ocr_pages=[self._make_ocr_page(lines)],
        )
        return layout_pages[0].blocks[0]

    def test_paragraph_text_stays_flowing_text(self):
        block = self._parse_single_block(
            "Text",
            [
                self._make_line([12, 12, 86, 20], "Paragraph one"),
                self._make_line([12, 24, 84, 32], "Paragraph two"),
                self._make_line([12, 36, 82, 44], "Paragraph three"),
            ],
        )

        self.assertEqual(block.label, "Text")
        self.assertEqual(block.category, ElementCategory.FLOWING_TEXT)

    def test_sparse_text_is_relabelled_to_table(self):
        block = self._parse_single_block(
            "Text",
            [
                self._make_line([12, 12, 30, 20], "R1C1"),
                self._make_line([58, 12, 80, 20], "R1C2"),
                self._make_line([14, 28, 34, 36], "R2C1"),
                self._make_line([60, 28, 82, 36], "R2C2"),
            ],
        )

        self.assertEqual(block.label, "Table")
        self.assertEqual(block.category, ElementCategory.TABLE)

    def test_real_table_label_stays_table(self):
        block = self._parse_single_block(
            "Table",
            [
                self._make_line([12, 12, 86, 20], "Header"),
                self._make_line([12, 24, 84, 32], "Row"),
                self._make_line([12, 36, 82, 44], "Footer"),
            ],
        )

        self.assertEqual(block.label, "Table")
        self.assertEqual(block.category, ElementCategory.TABLE)

    def test_fewer_than_three_lines_do_not_relabel(self):
        block = self._parse_single_block(
            "Text",
            [
                self._make_line([12, 12, 30, 20], "A"),
                self._make_line([58, 28, 80, 36], "B"),
            ],
        )

        self.assertEqual(block.label, "Text")
        self.assertEqual(block.category, ElementCategory.FLOWING_TEXT)

    def test_relabelled_block_is_sent_to_table_detection(self):
        block = self._parse_single_block(
            "Text",
            [
                self._make_line([12, 12, 30, 20], "R1C1"),
                self._make_line([58, 12, 80, 20], "R1C2"),
                self._make_line([14, 28, 34, 36], "R2C1"),
                self._make_line([60, 28, 82, 36], "R2C2"),
            ],
        )
        page = LayoutPageResult(
            page_index=0,
            page_width=100.0,
            page_height=100.0,
            layout_image_bbox=[0, 0, 100, 100],
            image_bbox=[0, 0, 100, 100],
            blocks=[block],
        )

        table_model = _DummyTableModel([[[5.0, 5.0, 45.0, 25.0]]])
        self.parser.table_model = table_model

        result = self.parser._parse_tables_batch(
            [page],
            [Image.new("RGB", (100, 100), color="white")],
        )

        self.assertIn(block.block_id, result.tables)
        self.assertEqual(table_model.calls[0]["count"], 1)

    def test_relabelled_block_without_cells_falls_back_to_region_text(self):
        lines = [
            self._make_line([12, 12, 30, 20], "R1C1"),
            self._make_line([58, 12, 80, 20], "R1C2"),
            self._make_line([14, 28, 34, 36], "R2C1"),
            self._make_line([60, 28, 82, 36], "R2C2"),
        ]
        block = self._parse_single_block("Text", lines)
        layout_result = LayoutParseResult(
            pdf_path="",
            pages=[
                LayoutPageResult(
                    page_index=0,
                    page_width=100.0,
                    page_height=100.0,
                    layout_image_bbox=[0, 0, 100, 100],
                    image_bbox=[0, 0, 100, 100],
                    blocks=[block],
                )
            ],
        )
        ocr_result = OCRParseResult(pdf_path="", pages=[self._make_ocr_page(lines)])
        table_result = TableParseResult(
            pdf_path="",
            tables={
                block.block_id: TableBlockResult(
                    block_id=block.block_id,
                    cells_bbox=[],
                    crop_size=(80.0, 60.0),
                )
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            layout_result.pdf_path = tmp_pdf.name
            ocr_result.pdf_path = tmp_pdf.name
            parsed = self.parser.merge_results(
                tmp_pdf.name,
                layout_result,
                ocr_result,
                table_result=table_result,
            )

        self.assertEqual(parsed.pages[0].elements[0].category, ElementCategory.TABLE)
        self.assertEqual(
            parsed.pages[0].elements[0].source_text,
            "R1C1 R1C2 R2C1 R2C2",
        )


if __name__ == "__main__":
    unittest.main()
