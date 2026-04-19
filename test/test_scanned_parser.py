import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pdf2zh.scanned.enums import ElementCategory
from pdf2zh.scanned.parser import (
    LayoutBlockResult,
    LayoutPageResult,
    LayoutParseResult,
    OCRPageResult,
    OCRParseResult,
    StageAParser,
    _DocumentContext,
)


class DummyLine:
    def __init__(self, text, bbox):
        self.text = text
        self.bbox = bbox


class DummyOCRResult:
    def __init__(self, text_lines, image_bbox=None):
        self.text_lines = text_lines
        self.image_bbox = image_bbox or [0, 0, 50, 50]


class DummyCell:
    def __init__(self, bbox, row_id=0, col_id=0):
        self.bbox = bbox
        self.row_id = row_id
        self.col_id = col_id


class DummyTablePrediction:
    def __init__(self, cells):
        self.cells = cells


class DummyLayoutBox:
    def __init__(self, label, bbox, position):
        self.label = label
        self.bbox = bbox
        self.position = position


class DummyLayoutPrediction:
    def __init__(self, bboxes, image_bbox=None):
        self.bboxes = bboxes
        self.image_bbox = image_bbox or [0, 0, 50, 50]


class TestScannedParser(unittest.TestCase):
    def setUp(self):
        self.sample_pdf = (
            Path(__file__).parent / "file" / "translate.cli.plain.text.pdf"
        )
        self.parser = StageAParser(
            device="cpu",
            page_batch_size=1,
            layout_batch_size=1,
            detection_batch_size=1,
            ocr_batch_size=1,
            table_batch_size=1,
            equation_batch_size=1,
            allow_parallel_phases=False,
        )

    def test_merge_results_assigns_text_by_overlap(self):
        layout_result = LayoutParseResult(
            pdf_path=str(self.sample_pdf),
            pages=[
                LayoutPageResult(
                    page_index=0,
                    page_width=50,
                    page_height=50,
                    layout_image_bbox=[0, 0, 50, 50],
                    image_bbox=[0, 0, 50, 50],
                    blocks=[
                        LayoutBlockResult(
                            block_id="0:0",
                            page_index=0,
                            position=0,
                            label="Text",
                            category=ElementCategory.FLOWING_TEXT,
                            bbox_layout=[10, 10, 30, 20],
                            bbox_image=[10, 10, 30, 20],
                            bbox_pdf=[10, 10, 30, 20],
                        ),
                        LayoutBlockResult(
                            block_id="0:1",
                            page_index=0,
                            position=1,
                            label="Text",
                            category=ElementCategory.FLOWING_TEXT,
                            bbox_layout=[38, 10, 42, 20],
                            bbox_image=[38, 10, 42, 20],
                            bbox_pdf=[38, 10, 42, 20],
                        ),
                    ],
                )
            ],
        )
        ocr_result = OCRParseResult(
            pdf_path=str(self.sample_pdf),
            pages=[
                OCRPageResult(
                    page_index=0,
                    image_bbox=[0, 0, 50, 50],
                    ocr_result=DummyOCRResult(
                        [
                            DummyLine("Hello world", [11, 11, 29, 19]),
                            DummyLine("Ignore me", [28, 10, 48, 20]),
                        ]
                    ),
                )
            ],
        )

        merged = self.parser.merge_results(self.sample_pdf, layout_result, ocr_result)

        self.assertEqual(merged.pages[0].elements[0].source_text, "Hello world")
        self.assertEqual(merged.pages[0].elements[1].source_text, "")
        self.assertEqual(merged.pages[0].raw_text, "Hello world")

    def test_parse_tables_falls_back_to_crop_ocr_when_page_ocr_is_empty(self):
        layout_result = LayoutParseResult(
            pdf_path=str(self.sample_pdf),
            pages=[
                LayoutPageResult(
                    page_index=0,
                    page_width=50,
                    page_height=50,
                    layout_image_bbox=[0, 0, 50, 50],
                    image_bbox=[0, 0, 50, 50],
                    blocks=[
                        LayoutBlockResult(
                            block_id="0:0",
                            page_index=0,
                            position=0,
                            label="Table",
                            category=ElementCategory.TABLE,
                            bbox_layout=[0, 0, 50, 50],
                            bbox_image=[0, 0, 50, 50],
                            bbox_pdf=[0, 0, 50, 50],
                        )
                    ],
                )
            ],
        )
        ocr_result = OCRParseResult(
            pdf_path=str(self.sample_pdf),
            pages=[
                OCRPageResult(
                    page_index=0,
                    image_bbox=[0, 0, 50, 50],
                    ocr_result=DummyOCRResult([]),
                )
            ],
        )

        self.parser._table_predictor = MagicMock(
            return_value=[DummyTablePrediction([DummyCell([0, 0, 50, 50])])]
        )
        self.parser._recognition_predictor = MagicMock(
            return_value=[DummyOCRResult([DummyLine("Cell text", [0, 0, 50, 50])])]
        )
        self.parser._detection_predictor = object()

        with patch.object(
            self.parser,
            "_load_page_images",
            return_value=(
                [Image.new("RGB", (50, 50), "white")],
                [Image.new("RGB", (100, 100), "white")],
            ),
        ):
            table_result = self.parser.parse_tables(
                self.sample_pdf,
                layout_result,
                ocr_result,
            )

        table_block = table_result.tables["0:0"]
        self.assertTrue(table_block.used_fallback_ocr)
        self.assertEqual(table_block.source_text, "Cell text")
        self.assertEqual(table_block.cells[0].source_text, "Cell text")
        self.parser._recognition_predictor.assert_called_once()

    def test_parse_equations_returns_real_latex_when_enabled(self):
        layout_result = LayoutParseResult(
            pdf_path=str(self.sample_pdf),
            pages=[
                LayoutPageResult(
                    page_index=0,
                    page_width=50,
                    page_height=50,
                    layout_image_bbox=[0, 0, 50, 50],
                    image_bbox=[0, 0, 50, 50],
                    blocks=[
                        LayoutBlockResult(
                            block_id="0:0",
                            page_index=0,
                            position=0,
                            label="Formula",
                            category=ElementCategory.EQUATION,
                            bbox_layout=[0, 0, 50, 50],
                            bbox_image=[0, 0, 50, 50],
                            bbox_pdf=[0, 0, 50, 50],
                        )
                    ],
                )
            ],
        )

        self.parser._recognition_predictor = MagicMock(
            return_value=[DummyOCRResult([DummyLine("x^2 + y^2", [0, 0, 50, 10])])]
        )

        with patch.object(
            self.parser,
            "_load_page_images",
            return_value=(
                [Image.new("RGB", (50, 50), "white")],
                [Image.new("RGB", (100, 100), "white")],
            ),
        ):
            equation_result = self.parser.parse_equations(
                self.sample_pdf,
                layout_result,
                enable_latex=True,
            )

        self.assertEqual(equation_result.equations["0:0"].latex, "x^2 + y^2")

    def test_phase_workflow_builds_parsed_document(self):
        image = Image.new("RGB", (50, 50), "white")
        highres_image = Image.new("RGB", (100, 100), "white")
        context = _DocumentContext(
            pdf_path=self.sample_pdf,
            page_indices=[0],
            page_dims={0: (50, 50)},
        )

        self.parser._layout_predictor = MagicMock(
            return_value=[
                DummyLayoutPrediction(
                    [
                        DummyLayoutBox("Text", [0, 0, 50, 15], 0),
                        DummyLayoutBox("Table", [0, 15, 50, 35], 1),
                        DummyLayoutBox("Formula", [0, 35, 50, 50], 2),
                    ]
                )
            ]
        )
        self.parser._recognition_predictor = MagicMock(
            side_effect=[
                [
                    DummyOCRResult(
                        [
                            DummyLine("Body text", [0, 0, 50, 15]),
                            DummyLine("Table text", [0, 15, 50, 35]),
                        ]
                    )
                ],
                [DummyOCRResult([DummyLine("E=mc^2", [0, 0, 50, 10])])],
            ]
        )
        self.parser._table_predictor = MagicMock(
            return_value=[DummyTablePrediction([DummyCell([0, 0, 50, 20])])]
        )
        self.parser._detection_predictor = object()

        with patch.object(
            self.parser, "_prepare_document_context", return_value=context
        ):
            with patch.object(
                self.parser,
                "_load_page_images",
                return_value=([image], [highres_image]),
            ):
                layout_result = self.parser.parse_layout(self.sample_pdf, pages=[0])
                ocr_result = self.parser.parse_ocr(self.sample_pdf, pages=[0])
                table_result = self.parser.parse_tables(
                    self.sample_pdf,
                    layout_result,
                    ocr_result,
                )
                equation_result = self.parser.parse_equations(
                    self.sample_pdf,
                    layout_result,
                    enable_latex=True,
                )
                parsed_doc = self.parser.merge_results(
                    self.sample_pdf,
                    layout_result,
                    ocr_result,
                    table_result=table_result,
                    equation_result=equation_result,
                )

        self.assertEqual(len(parsed_doc.pages), 1)
        self.assertEqual(len(parsed_doc.pages[0].elements), 3)
        self.assertEqual(parsed_doc.pages[0].elements[0].source_text, "Body text")
        self.assertEqual(
            parsed_doc.pages[0].elements[1].cells[0].source_text, "Table text"
        )
        self.assertEqual(parsed_doc.pages[0].elements[2].latex, "E=mc^2")
        self.assertEqual(parsed_doc.pages[0].raw_text, "Body text")


if __name__ == "__main__":
    unittest.main()
