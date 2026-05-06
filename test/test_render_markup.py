"""Unit tests for pdf2zh.render.markup — HTML/LaTeX → Typst markup conversion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2zh.render.markup import parse_toc_line, to_typst_markup


class TestMathConversion:
    def test_display_math(self):
        out = to_typst_markup('<math display="block">\\frac{a}{b}</math>')
        assert "$$\\frac{a}{b}$$" in out

    def test_inline_math(self):
        out = to_typst_markup("<math>x^2</math>")
        assert "$x^2$" in out

    def test_multiple_math_blocks(self):
        out = to_typst_markup("See <math>a+b</math> and <math>c-d</math>")
        assert "$a+b$" in out
        assert "$c-d$" in out

    def test_math_preserved_in_equation_mode(self):
        out = to_typst_markup("$a=b$", is_equation=True)
        # Already has $...$, should stay
        assert "$a=b$" in out

    def test_bare_latex_wrapped_in_equation_mode(self):
        out = to_typst_markup("We get \\frac{a}{b}", is_equation=True)
        assert "$" in out
        assert "\\frac{a}{b}" in out

    def test_bare_latex_not_wrapped_outside_equation(self):
        # In non-equation mode, bare LaTeX is not wrapped
        out = to_typst_markup("Some text \\frac{a}{b}", is_equation=False)
        # No wrapping — just passes through (stripped as unknown tag or kept)
        assert "\\frac{a}{b}" in out


class TestHtmlFormatting:
    def test_bold(self):
        assert "**hello**" in to_typst_markup("<b>hello</b>")

    def test_strong(self):
        assert "**world**" in to_typst_markup("<strong>world</strong>")

    def test_italic(self):
        assert "_hi_" in to_typst_markup("<i>hi</i>")

    def test_em(self):
        assert "_em_" in to_typst_markup("<em>em</em>")

    def test_superscript_outside_math(self):
        out = to_typst_markup("x<sup>2</sup>")
        assert "^2^" in out

    def test_subscript_outside_math(self):
        out = to_typst_markup("H<sub>2</sub>O")
        assert "~2~" in out

    def test_bold_with_italic_inside(self):
        out = to_typst_markup("<b>bold <i>and italic</i></b>")
        assert "**" in out
        assert "_" in out


class TestLiteralCharacters:
    def test_less_than_escaped(self):
        out = to_typst_markup("If a < b then")
        assert "\\<" in out

    def test_italic_variable_with_comparison(self):
        out = to_typst_markup("If <i>a</i> < <i>b</i>")
        assert "_a_" in out
        assert "_b_" in out
        assert "\\<" in out

    def test_plain_text_hash_escaped(self):
        out = to_typst_markup("Price: #100")
        assert "\\#100" in out

    def test_plain_text_at_escaped(self):
        out = to_typst_markup("Email @user")
        assert "\\@user" in out

    def test_no_false_tag_match(self):
        # A tag-like that isn't a known formatting tag
        out = to_typst_markup("<unknown>text</unknown>")
        # Unknown tags stripped
        assert "<unknown>" not in out
        assert "text" in out


class TestMathAndProseEquation:
    def test_mixed_prose_and_math(self):
        out = to_typst_markup(
            'Nếu ax<sup>2</sup> + bx + c = 0, thì x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}.',
            is_equation=True,
        )
        # Vietnamese prose preserved
        assert "Nếu" in out
        assert "thì" in out
        # Math wrapped
        assert "$" in out

    def test_equation_with_html_math_tags(self):
        out = to_typst_markup(
            'a(b + c) = ab + ac <math display="block">\\frac{a+c}{b}</math>',
            is_equation=True,
        )
        assert "$$\\frac{a+c}{b}$$" in out
        assert "a(b + c) = ab + ac" in out


class TestTocLineParsing:
    def test_simple_entry(self):
        result = parse_toc_line("Introduction 1")
        assert result == ("Introduction", "1")

    def test_entry_with_dots(self):
        result = parse_toc_line("Chapter 1: Overview ....... 15")
        assert result is not None
        assert result[1] == "15"
        assert "Chapter 1" in result[0]

    def test_entry_with_bold_markup(self):
        result = parse_toc_line("<b>Derivatives</b> 174")
        assert result is not None
        assert result[1] == "174"

    def test_entry_no_page_number(self):
        result = parse_toc_line("Just a heading")
        assert result is None

    def test_empty_line(self):
        assert parse_toc_line("") is None

    def test_multipart_number(self):
        result = parse_toc_line("3.1 Derivatives of Polynomials 174")
        assert result is not None
        assert result[1] == "174"
