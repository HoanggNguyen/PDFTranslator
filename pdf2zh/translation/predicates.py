from __future__ import annotations

import re

_MATH_TAG = re.compile(r"<math\b[^>]*>.*?</math>", re.DOTALL | re.IGNORECASE)
_TYPST_TAG = re.compile(r"<typst\b[^>]*>.*?</typst>", re.DOTALL | re.IGNORECASE)
_DOLLAR_BLOCK = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_DOLLAR_INLINE = re.compile(r"\$[^$\n]*\$")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+(?:\s*\{[^{}]*\})*")
_HTML_TAG = re.compile(r"<[^>]+>")
_EQ_LABEL = re.compile(r"\(\d+(?:\.\d+)*[a-z]?\)")
# Two or more consecutive letters across major Unicode scripts
_LETTER_RUN = re.compile(
    r"[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF"
    r"\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u2E80-\u9FFF]{2,}"
)


def _strip_math(s: str) -> str:
    s = _MATH_TAG.sub("", s)
    s = _TYPST_TAG.sub("", s)
    s = _DOLLAR_BLOCK.sub("", s)
    s = _DOLLAR_INLINE.sub("", s)
    s = _LATEX_CMD.sub("", s)
    s = _EQ_LABEL.sub("", s)
    s = _HTML_TAG.sub("", s)
    return s


def is_plain_text(s: str) -> bool:
    """True if there is translatable prose (real word) outside math."""
    return bool(_LETTER_RUN.search(_strip_math(s)))


def is_equation_only(s: str) -> bool:
    """True if string is pure math (no translatable text remaining)."""
    return not _strip_math(s).strip()
