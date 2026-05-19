import tempfile
import unittest
from types import SimpleNamespace

from pdf2zh.scanned.enums import ElementCategory
from pdf2zh.scanned.models import (
    EquationWordData,
    LayoutBlockResult,
    LayoutPageResult,
    LayoutParseResult,
    OCRPageResult,
    OCRParseResult,
)
from pdf2zh.scanned.parser import StageAParser
from pdf2zh.scanned.schema import validate_stage_output
from pdf2zh.scanned.utils.ocr_text import extract_text_for_region


class TestScannedEquationWorkflow(unittest.TestCase):
    def setUp(self):
        self.parser = object.__new__(StageAParser)
        self.parser.hardware = SimpleNamespace()

    def _make_block(
        self,
        block_id: str,
        label: str,
        category: ElementCategory,
        bbox: list[float],
    ) -> LayoutBlockResult:
        return LayoutBlockResult(
            block_id=block_id,
            page_index=0,
            position=0,
            label=label,
            category=category,
            bbox_layout=list(bbox),
            bbox_image=list(bbox),
            bbox_pdf=list(bbox),
        )

    def _make_page(self, blocks: list[LayoutBlockResult]) -> LayoutPageResult:
        return LayoutPageResult(
            page_index=0,
            page_width=1000.0,
            page_height=1000.0,
            layout_image_bbox=[0.0, 0.0, 1000.0, 1000.0],
            image_bbox=[0.0, 0.0, 1000.0, 1000.0],
            blocks=blocks,
        )

    def test_extract_text_for_region_uses_text_lines_only(self):
        ocr_result = SimpleNamespace(
            text_lines=[
                SimpleNamespace(
                    text="outside line",
                    bbox=[200.0, 200.0, 260.0, 220.0],
                    words=[
                        SimpleNamespace(
                            text="inside-word",
                            bbox=[110.0, 110.0, 150.0, 125.0],
                            bbox_valid=True,
                        )
                    ],
                )
            ]
        )

        result = extract_text_for_region(ocr_result, [100.0, 100.0, 160.0, 130.0])
        self.assertEqual(result, [])

    def test_merge_results_collects_equation_words_and_skips_invalid_word_boxes(self):
        block = self._make_block(
            "eq-0",
            "Equation",
            ElementCategory.EQUATION,
            [100.0, 200.0, 280.0, 250.0],
        )
        layout_result = LayoutParseResult(pdf_path="", pages=[self._make_page([block])])
        ocr_result = OCRParseResult(
            pdf_path="",
            pages=[
                OCRPageResult(
                    page_index=0,
                    image_bbox=[0.0, 0.0, 1000.0, 1000.0],
                    ocr_result=SimpleNamespace(
                        text_lines=[
                            SimpleNamespace(
                                text="where x count",
                                bbox=[105.0, 205.0, 250.0, 225.0],
                                words=[
                                    SimpleNamespace(
                                        text="where",
                                        bbox=[105.0, 205.0, 145.0, 225.0],
                                        bbox_valid=True,
                                    ),
                                    SimpleNamespace(
                                        text="<math>x</math>",
                                        bbox=[150.0, 205.0, 180.0, 225.0],
                                        bbox_valid=False,
                                    ),
                                    SimpleNamespace(
                                        text="count",
                                        bbox=[185.0, 205.0, 230.0, 225.0],
                                        bbox_valid=True,
                                    ),
                                ],
                            )
                        ]
                    ),
                )
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            layout_result.pdf_path = tmp_pdf.name
            ocr_result.pdf_path = tmp_pdf.name
            parsed = self.parser.merge_results(tmp_pdf.name, layout_result, ocr_result)

        element = parsed.pages[0].elements[0]
        self.assertEqual(element.source_text, "where x count")
        self.assertEqual(
            element.equation_words,
            [
                EquationWordData(
                    text="where",
                    bbox_image=[105.0, 205.0, 145.0, 225.0],
                    bbox_pdf=[105.0, 205.0, 145.0, 225.0],
                ),
                EquationWordData(
                    text="count",
                    bbox_image=[185.0, 205.0, 230.0, 225.0],
                    bbox_pdf=[185.0, 205.0, 230.0, 225.0],
                ),
            ],
        )

        validation = validate_stage_output(parsed.to_dict(), skip_json_schema=True)
        self.assertTrue(validation.valid, validation.errors)

    def test_orphan_line_is_reassigned_to_equation_blocks_by_words(self):
        left = self._make_block(
            "eq-left",
            "Equation",
            ElementCategory.EQUATION,
            [100.0, 200.0, 180.0, 240.0],
        )
        right = self._make_block(
            "eq-right",
            "Equation",
            ElementCategory.EQUATION,
            [320.0, 200.0, 400.0, 240.0],
        )
        layout_result = LayoutParseResult(
            pdf_path="", pages=[self._make_page([left, right])]
        )
        ocr_result = OCRParseResult(
            pdf_path="",
            pages=[
                OCRPageResult(
                    page_index=0,
                    image_bbox=[0.0, 0.0, 1000.0, 1000.0],
                    ocr_result=SimpleNamespace(
                        text_lines=[
                            SimpleNamespace(
                                text="cos sec",
                                bbox=[90.0, 210.0, 430.0, 235.0],
                                words=[
                                    SimpleNamespace(
                                        text="cos",
                                        bbox=[105.0, 210.0, 150.0, 235.0],
                                        bbox_valid=True,
                                    ),
                                    SimpleNamespace(
                                        text="sec",
                                        bbox=[330.0, 210.0, 370.0, 235.0],
                                        bbox_valid=True,
                                    ),
                                ],
                            )
                        ]
                    ),
                )
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            layout_result.pdf_path = tmp_pdf.name
            ocr_result.pdf_path = tmp_pdf.name
            parsed = self.parser.merge_results(tmp_pdf.name, layout_result, ocr_result)

        self.assertEqual(len(parsed.pages[0].elements), 2)
        left_element, right_element = parsed.pages[0].elements
        self.assertEqual(left_element.source_text, "cos")
        self.assertEqual(right_element.source_text, "sec")
        self.assertEqual([word.text for word in left_element.equation_words], ["cos"])
        self.assertEqual([word.text for word in right_element.equation_words], ["sec"])

    def test_expand_layout_blocks_uses_centered_text_line_coverage(self):
        block = self._make_block(
            "text-0",
            "Text",
            ElementCategory.FLOWING_TEXT,
            [100.0, 100.0, 150.0, 120.0],
        )
        page_ocr = OCRPageResult(
            page_index=0,
            image_bbox=[0.0, 0.0, 300.0, 300.0],
            ocr_result=SimpleNamespace(
                text_lines=[
                    SimpleNamespace(text="centered", bbox=[98.0, 100.0, 180.0, 120.0]),
                    SimpleNamespace(text="far", bbox=[151.0, 100.0, 220.0, 120.0]),
                ]
            ),
        )

        expanded = self.parser._expand_layout_blocks(
            [block],
            page_ocr,
            [0.0, 0.0, 300.0, 300.0],
            300.0,
            300.0,
        )

        self.assertEqual(expanded[0].bbox_image, [98.0, 100.0, 180.0, 120.0])
        self.assertEqual(expanded[0].bbox_pdf, [98.0, 100.0, 180.0, 120.0])


if __name__ == "__main__":
    unittest.main()
