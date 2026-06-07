from __future__ import annotations

import re

_HTML_TAG = re.compile(r"</?(?:b|i|u|strong|em)>", re.IGNORECASE)


def strip_html_tags(s: str) -> str:
    return _HTML_TAG.sub("", s)


def inflate(bbox: list[float], pad: float) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [x0 - pad, y0 - pad, x1 + pad, y1 + pad]


def clamp(bbox: list[float], pw: float, ph: float) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [max(0.0, x0), max(0.0, y0), min(pw, x1), min(ph, y1)]
