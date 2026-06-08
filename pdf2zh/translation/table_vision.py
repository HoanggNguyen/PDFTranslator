"""Vision-based table OCR verifier.

For each TABLE element, crops the bbox from the PDF page and sends it to a
vision LLM together with the current OCR cell data (text + positions).

The LLM checks whether the OCR is accurate:
- If correct → skip (cells are left unchanged, translation proceeds normally).
- If incorrect → update source_text and/or bbox_pdf on each affected cell
  before translation runs.

This pass runs BEFORE phase-2 translation so corrections feed into the
translated output rather than being applied after.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

import fitz
import json_repair

from .config import TranslatorConfig
from .gateway import Gateway

logger = logging.getLogger(__name__)

_VISION_SYSTEM = """\
You are a table OCR verifier for a PDF translation pipeline.

You receive:
1. A cropped image of a table from a PDF page.
2. The current OCR output as JSON: a list of cells, each with:
   - "idx": cell index (integer, used to identify the cell)
   - "source_text": OCR-extracted text
   - "bbox_pdf": [x0, y0, x1, y1] position in PDF points

Your task:
- Compare the visible text in the image against the OCR data.
- If ALL cells are accurate: return {"correct": true, "cells": []}
- If ANY cell has wrong text or clearly wrong position: return {"correct": false, "cells": [...]}

In the corrected cells list, include ONLY cells that need fixing.
Each corrected cell must have "idx" plus the fields to update ("source_text" and/or "bbox_pdf").

Return ONLY valid JSON. No explanation, no markdown fences.\
"""


def _crop_bbox_image(
    pdf_path: str, page_idx: int, bbox_pdf: list, dpi: int = 150
) -> str:
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
        logger.warning("table_vision: pdf_path '%s' not found, skipping", pdf_path)
        return

    tables: list[tuple[int, dict]] = [
        (page_idx, elem)
        for page_idx, page in enumerate(doc.get("pages", []))
        for elem in page.get("elements", [])
        if elem.get("category") == "TABLE" and elem.get("cells")
    ]

    if not tables:
        logger.info("table_vision: no TABLE elements found")
        return

    logger.info("table_vision: verifying %d tables", len(tables))

    async with Gateway(cfg) as gw:

        async def _process(page_idx: int, elem: dict) -> None:
            cells = elem.get("cells", [])
            if not cells:
                return
            try:
                img = _crop_bbox_image(pdf_path, page_idx, elem["bbox_pdf"])
                cells_data = [
                    {
                        "idx": i,
                        "source_text": c.get("source_text", ""),
                        "bbox_pdf": c.get("bbox_pdf", []),
                    }
                    for i, c in enumerate(cells)
                ]
                prompt = (
                    f"Current OCR cells:\n{json.dumps(cells_data, ensure_ascii=False)}"
                )
                raw = await gw.call_vision(_VISION_SYSTEM, prompt, img)
                result = json_repair.loads(raw)
                if not isinstance(result, dict) or result.get("correct", True):
                    return
                for corr in result.get("cells", []):
                    idx = corr.get("idx")
                    if not isinstance(idx, int) or idx >= len(cells):
                        continue
                    if "source_text" in corr:
                        cells[idx]["source_text"] = corr["source_text"]
                    if "bbox_pdf" in corr:
                        cells[idx]["bbox_pdf"] = corr["bbox_pdf"]
                logger.debug(
                    "table_vision p%d: corrected %d cells",
                    page_idx,
                    len(result.get("cells", [])),
                )
            except Exception as exc:
                logger.warning("table_vision: p%d failed: %s", page_idx, exc)

        await asyncio.gather(*[_process(pi, elem) for pi, elem in tables])


def table_vision_pass(doc: dict, cfg: TranslatorConfig) -> None:
    """Verify and correct TABLE cell OCR before translation."""
    asyncio.run(_run(doc, cfg))
