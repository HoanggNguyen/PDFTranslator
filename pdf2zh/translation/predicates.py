from __future__ import annotations

import re

_MATH_TAG = re.compile(r"<math>.*?</math>", re.DOTALL)
_EQ_LABEL = re.compile(r"\(\d+(\.\d+)*[a-z]?\)")
# Two or more consecutive letters across major Unicode scripts
_LETTER_RUN = re.compile(
    r"[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF"
    r"\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u2E80-\u9FFF]{2,}"
)


def is_plain_text(s: str) -> bool:
    without_math = _MATH_TAG.sub("", s)
    return bool(_LETTER_RUN.search(without_math))


def is_equation_only(s: str) -> bool:
    without_math = _MATH_TAG.sub("", s)
    without_labels = _EQ_LABEL.sub("", without_math)
    return not without_labels.strip()
