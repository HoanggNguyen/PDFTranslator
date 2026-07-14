"""Tests for the self-healing Typst compile pipeline.

Layer 1 (markup sanitizer) outputs must compile with the real typst binary.
Layer 2 (renderer repair loop) must map compile errors back to element vars
and downgrade them to plain text instead of failing the whole document.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.render.config import RenderConfig
from pdf2zh.render.markup import to_typst_native
from pdf2zh.render.renderer import _failing_element_vars, render_document
from pdf2zh.render.source_builder import build_typst_source

TYPST = shutil.which("typst")


class TestFailingElementVars:
    SOURCE = "\n".join(
        [
            "#set page(width: 100pt, height: 100pt)",  # line 1
            "#let e0_0_tm = [good markup]",  # line 2
            "#let e0_1_tm = [broken",  # line 3
            "  still broken]",  # line 4
            '#let e0_1_c2_md = "cell"',  # line 5
        ]
    )

    def test_maps_error_line_to_element_var(self):
        stderr = "error: expected expression\n  ┌─ ../tmp/x/overlay.typ:4:2\n"
        assert _failing_element_vars(self.SOURCE, stderr) == {"e0_1"}

    def test_maps_cell_var(self):
        stderr = "┌─ overlay.typ:5:1\n"
        assert _failing_element_vars(self.SOURCE, stderr) == {"e0_1_c2"}

    def test_error_outside_elements_not_attributed(self):
        stderr = "┌─ overlay.typ:1:1\n"
        assert _failing_element_vars(self.SOURCE, stderr) == set()

    def test_multiple_errors_collected(self):
        stderr = "┌─ overlay.typ:3:5\n...\n┌─ overlay.typ:5:1\n"
        assert _failing_element_vars(self.SOURCE, stderr) == {"e0_1", "e0_1_c2"}


def _parsed_with(translated: str, category: str = "FLOWING_TEXT") -> dict:
    return {
        "pages": [
            {
                "page_width": 200,
                "page_height": 100,
                "elements": [
                    {
                        "category": category,
                        "label": "Text",
                        "bbox_pdf": [10, 10, 190, 40],
                        "source_text": "Source",
                        "translated_text": translated,
                    }
                ],
            }
        ]
    }


class TestFallbackVars:
    def test_fallback_var_uses_plain_markdown_path(self):
        parsed = _parsed_with("<typst>broken #let ( markup</typst>")
        cfg = RenderConfig()
        source = build_typst_source(parsed, {"p0:e0": 10}, {}, {}, cfg)
        assert "#let e0_0_tm = [" in source

        source_fb = build_typst_source(
            parsed, {"p0:e0": 10}, {}, {}, cfg, fallback_vars={"e0_0"}
        )
        assert "#let e0_0_tm = [" not in source_fb
        assert '#let e0_0_md = "' in source_fb

    def test_equation_fallback_uses_markdown_path(self):
        parsed = _parsed_with("x <math>a+b</math>", category="EQUATION")
        cfg = RenderConfig()
        source_fb = build_typst_source(
            parsed, {"p0:e0": 10}, {}, {}, cfg, fallback_vars={"e0_0"}
        )
        assert '#let e0_0_md = "' in source_fb


@pytest.mark.skipif(TYPST is None, reason="typst binary not installed")
class TestSanitizerOutputCompiles:
    @pytest.mark.parametrize(
        "text",
        [
            "<math>page_index = k</math>",
            '<math>upright("page_index") = k</math>',
            '<math>TP(t) = #{"pairs" "with" IoU >= t}</math>',
            "<math>_(x)</math>",
            "<math>x_</math>",
            '<math>x = "unclosed</math>',
            "<typst>page $page_index$ = k</typst>",
        ],
    )
    def test_output_compiles(self, text, tmp_path):
        typ = tmp_path / "t.typ"
        typ.write_text(to_typst_native(text) + "\n", encoding="utf-8")
        result = subprocess.run(
            [TYPST, "compile", str(typ), str(tmp_path / "t.pdf")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.skipif(TYPST is None, reason="typst binary not installed")
class TestRepairLoopEndToEnd:
    def test_broken_element_downgraded_instead_of_failing(self, tmp_path):
        import fitz

        pdf_path = tmp_path / "src.pdf"
        doc = fitz.open()
        doc.new_page(width=200, height=100)
        doc.save(str(pdf_path))
        doc.close()

        # Raw <typst> passthrough with invalid markup: survives the static
        # sanity gates, only the compiler can catch it.
        parsed = _parsed_with("<typst>broken #let ( markup</typst>")
        cfg = RenderConfig(typst_binary=TYPST)
        out_pdf = tmp_path / "out.pdf"

        stats = render_document(pdf_path, parsed, out_pdf, cfg)

        assert out_pdf.exists()
        assert stats["elements_fallback"] == 1
