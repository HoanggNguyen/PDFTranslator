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
    r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF"  # Latin + Vietnamese diacritics
    r"\u0370-\u03FF\u0400-\u04FF"
    r"\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u2E80-\u9FFF]{2,}"
)

# Math vocabulary words that are NOT natural-language prose.
# If an equation text line consists only of these, it should not be translated.
_MATH_WORDS: frozenset[str] = frozenset(
    {
        # Trig functions and inverses
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "arccot",
        "arcsec",
        "arccsc",
        "sinh",
        "cosh",
        "tanh",
        "coth",
        "sech",
        "csch",
        # Common math functions
        "log",
        "ln",
        "exp",
        "det",
        "dim",
        "ker",
        "gcd",
        "lcm",
        "max",
        "min",
        "lim",
        "sup",
        "inf",
        "mod",
        "deg",
        "arg",
        # Greek letter names (lower and upper)
        "pi",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
        "Lambda",
        "Mu",
        "Nu",
        "Xi",
        "Omicron",
        "Pi",
        "Rho",
        "Sigma",
        "Tau",
        "Upsilon",
        "Phi",
        "Chi",
        "Psi",
        "Omega",
        # Variant Greek
        "varepsilon",
        "varphi",
        "vartheta",
        "varrho",
        "varsigma",
        # Math units and abbreviations
        "rad",
        "radian",
        "radians",
        "rpm",
        "rps",
        "mm",
        "cm",
        "km",
        "kg",
        "mg",
        "ml",
        "ms",
        "ns",
        "us",
        "hz",
        "khz",
        "mhz",
        "ghz",
        # Math operator words
        "plus",
        "minus",
        "times",
        "over",
        "div",
        # Single-letter identifiers that appear as standalone words (a-z, A-Z)
        *list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    }
)

_WORD_RE = re.compile(r"[A-Za-z]{2,}")


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


def has_prose_for_equation(s: str) -> bool:
    """Like is_plain_text but also filters out math vocabulary words.

    Used for equation_words: returns True only if there are real
    natural-language words (e.g. 'where', 'if', 'then', 'means') beyond
    math symbols spelled out as text (e.g. 'pi', 'theta', 'rad', 'sin').
    """
    stripped = _strip_math(s)
    words = _WORD_RE.findall(stripped)
    return any(w.lower() not in _MATH_WORDS for w in words)


def is_equation_only(s: str) -> bool:
    """True if string is pure math (no translatable text remaining)."""
    return not _strip_math(s).strip()
