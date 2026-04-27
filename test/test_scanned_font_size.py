import unittest
from pathlib import Path

from pdf2zh.scanned.enums import ElementCategory
from pdf2zh.scanned.models import (
    CellData,
    LayoutBlockResult,
    LayoutPageResult,
    LayoutParseResult,
    OCRPageResult,
    OCRParseResult,
    TableBlockResult,
    TableParseResult,
)
from pdf2zh.scanned.parser import StageAParser
from pdf2zh.scanned.schema import validate_stage_output
from pdf2zh.scanned.utils.font_size import (
    build_font_size_profile,
    compute_line_height_pt,
    normalize_font_size_bucket,
)


class DummyLine:
    def __init__(self, text, bbox):
        self.text = text
        self.bbox = bbox


class DummyOCRResult:
    def __init__(self, text_lines, image_bbox=None):
        self.text_lines = text_lines
        self.image_bbox = image_bbox or [0, 0, 200, 200]


class TestScannedFontSize(unittest.TestCase):
    def setUp(self):
        self.sample_pdf = (
            Path(__file__).parent / "file" / "translate.cli.plain.text.pdf"
        )
        self.parser = object.__new__(StageAParser)

    def test_compute_line_height_pt_scales_with_page_height(self):
        height_pt = compute_line_height_pt([0, 10, 50, 30], 200, 100)
        self.assertEqual(height_pt, 10.0)

    def test_bucket_normalization_prefers_stable_sizes(self):
        profile = build_font_size_profile(10.0)
        self.assertEqual(
            normalize_font_size_bucket(
                9.7, profile, category=ElementCategory.FLOWING_TEXT, label="Text"
            ),
            "md",
        )
        self.assertEqual(
            normalize_font_size_bucket(
                6.8, profile, category=ElementCategory.IN_PLACE, label="Section-header"
            ),
            "sm",
        )
        self.assertEqual(
            normalize_font_size_bucket(
                12.0, profile, category=ElementCategory.TABLE, label="Table"
            ),
            "md",
        )

    def test_merge_results_assigns_font_sizes_and_schema_passes(self):
        layout_result = LayoutParseResult(
            pdf_path=str(self.sample_pdf),
            pages=[
                LayoutPageResult(
                    page_index=0,
                    page_width=100.0,
                    page_height=100.0,
                    layout_image_bbox=[0, 0, 200, 200],
                    image_bbox=[0, 0, 200, 200],
                    blocks=[
                        LayoutBlockResult(
                            block_id="0:0",
                            page_index=0,
                            position=0,
                            label="Section-header",
                            category=ElementCategory.IN_PLACE,
                            bbox_layout=[0, 0, 200, 30],
                            bbox_image=[0, 0, 200, 30],
                            bbox_pdf=[0, 0, 100, 15],
                        ),
                        LayoutBlockResult(
                            block_id="0:1",
                            page_index=0,
                            position=1,
                            label="Text",
                            category=ElementCategory.FLOWING_TEXT,
                            bbox_layout=[0, 30, 200, 110],
                            bbox_image=[0, 30, 200, 110],
                            bbox_pdf=[0, 15, 100, 55],
                        ),
                        LayoutBlockResult(
                            block_id="0:2",
                            page_index=0,
                            position=2,
                            label="Table",
                            category=ElementCategory.TABLE,
                            bbox_layout=[0, 110, 200, 170],
                            bbox_image=[0, 110, 200, 170],
                            bbox_pdf=[0, 55, 100, 85],
                        ),
                        LayoutBlockResult(
                            block_id="0:3",
                            page_index=0,
                            position=3,
                            label="Picture",
                            category=ElementCategory.BYPASS,
                            bbox_layout=[0, 170, 200, 200],
                            bbox_image=[0, 170, 200, 200],
                            bbox_pdf=[0, 85, 100, 100],
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
                    image_bbox=[0, 0, 200, 200],
                    ocr_result=DummyOCRResult(
                        [
                            DummyLine("Header", [0, 0, 100, 24]),
                            DummyLine("Body line one", [0, 32, 180, 52]),
                            DummyLine("Body line two", [0, 58, 180, 78]),
                            DummyLine("Body line three", [0, 84, 180, 104]),
                            DummyLine("Table body", [0, 118, 180, 138]),
                        ]
                    ),
                )
            ],
        )
        table_result = TableParseResult(
            pdf_path=str(self.sample_pdf),
            tables={
                "0:2": TableBlockResult(
                    block_id="0:2",
                    source_text="Table body",
                    cells=[
                        CellData(
                            bbox_pdf=[0, 55, 100, 85],
                            row_id=0,
                            col_id=0,
                            source_text="Table body",
                            translated_text="",
                        )
                    ],
                )
            },
        )

        parsed = self.parser.merge_results(
            self.sample_pdf,
            layout_result,
            ocr_result,
            table_result=table_result,
            equation_result=None,
        )

        page = parsed.pages[0]
        self.assertGreater(page.body_font_size_pt, 0.0)
        self.assertEqual(
            set(page.font_size_profile.keys()), {"xs", "sm", "md", "lg", "xl"}
        )

        header = page.elements[0]
        body = page.elements[1]
        table = page.elements[2]
        bypass = page.elements[3]

        self.assertEqual(body.font_size_bucket, "md")
        self.assertEqual(body.font_size_pt, page.font_size_profile["md"])
        self.assertIn(header.font_size_bucket, {"sm", "md", "lg", "xl"})
        self.assertEqual(table.font_size_bucket, "md")
        self.assertEqual(bypass.font_size_pt, 0.0)
        self.assertEqual(bypass.font_size_bucket, "")

        validation = validate_stage_output(
            parsed.to_dict(), stage="A", skip_json_schema=False
        )
        self.assertTrue(validation.valid, validation.errors)


if __name__ == "__main__":
    unittest.main()
