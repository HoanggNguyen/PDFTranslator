"""Vision-based equation translator.

For EQUATION elements whose source_text contains natural-language prose
(e.g. "where", "if", "means"), crops the bbox region from the PDF page,
sends it to a vision LLM, and writes the result to translated_text.

Pure-math equations (only symbols, Greek letters, operators) are skipped —
they need no overlay.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import fitz

from .config import TranslatorConfig
from .gateway import Gateway
from .predicates import has_prose_for_equation

logger = logging.getLogger(__name__)

_VISION_SYSTEM = """\
You are a mathematical text extractor and translator.

Given a cropped image of an equation/formula region from a PDF page:

1. Read ALL visible text — both natural language and math symbols.
2. Translate natural-language words to {target_language}. Leave math untouched.
3. Wrap math expressions in <math>...</math> using Typst syntax (no backslash commands):
   - frac(a, b)  for fractions
   - sqrt(x)     for square roots
   - plus.minus  for ±
   - overline(x) for x̄
   - x^2, x_n   for superscripts/subscripts (no curly braces)
   - sum_(i=0)^n, integral_a^b
   - pi, theta, alpha, beta, gamma, delta, sigma, omega (no backslash)
4. Return ONLY the formatted translated text. No explanation, no markdown fences.\
"""


def _crop_bbox_image(pdf_path: str, page_idx: int, bbox_pdf: list, dpi: int = 150) -> str:
    """Crop bbox from a PDF page and return base64 PNG."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        pad = 4.0
        x0, y0, x1, y1 = bbox_pdf
        clip = fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pm = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        return base64.b64encode(pm.tobytes("png")).decode("ascii")
    finally:
        doc.close()


async def _run(doc: dict, cfg: TranslatorConfig) -> None:
    pdf_path = doc.get("pdf_path", "")
    if not pdf_path or not Path(pdf_path).exists():
        logger.warning("equation_vision: pdf_path '%s' not found, skipping", pdf_path)
        return

    targets: list[tuple[int, dict]] = [
        (page_idx, elem)
        for page_idx, page in enumerate(doc.get("pages", []))
        for elem in page.get("elements", [])
        if (
            elem.get("category") == "EQUATION"
            and has_prose_for_equation(elem.get("source_text", ""))
        )
    ]

    if not targets:
        logger.info("equation_vision: no prose EQUATION elements found")
        return

    logger.info("equation_vision: processing %d elements", len(targets))
    system = _VISION_SYSTEM.format(target_language=cfg.target_language)

    async with Gateway(cfg) as gw:
        async def _process(page_idx: int, elem: dict) -> None:
            try:
                img = _crop_bbox_image(pdf_path, page_idx, elem["bbox_pdf"])
                result = await gw.call_vision(system, "Translate this equation region:", img)
                elem["translated_text"] = result
                logger.debug("equation_vision p%d: %r", page_idx, result[:60])
            except Exception as exc:
                logger.warning("equation_vision: p%d failed: %s", page_idx, exc)

        await asyncio.gather(*[_process(pi, elem) for pi, elem in targets])


def equation_vision_pass(doc: dict, cfg: TranslatorConfig) -> None:
    """Translate prose-containing EQUATION elements via vision LLM."""
    asyncio.run(_run(doc, cfg))
