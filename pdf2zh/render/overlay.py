from __future__ import annotations

import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def composite_overlay(
    original_pdf: Path,
    overlay_pdf: Path,
    output_pdf: Path,
    pages: list[int] | None,
) -> None:
    """Stamp overlay_pdf transparently onto original_pdf.

    For scanned PDFs the original page image is the background;
    the overlay PDF carries translated text and cover rects with sampled bg colors.
    PyMuPDF's show_pdf_page() composites them in a single vector operation.
    """
    src = fitz.open(str(original_pdf))
    ov = fitz.open(str(overlay_pdf))

    ov_page_iter = iter(range(ov.page_count))

    for src_idx in range(src.page_count):
        if pages is not None and src_idx not in pages:
            continue
        try:
            ov_idx = next(ov_page_iter)
        except StopIteration:
            logger.warning("Overlay has fewer pages than original (stopped at %d)", src_idx)
            break

        src_page = src[src_idx]
        src_page.show_pdf_page(
            src_page.rect,
            ov,
            ov_idx,
            overlay=True,
        )
        logger.debug("Composited overlay page %d → source page %d", ov_idx, src_idx)

    ov.close()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    src.save(str(output_pdf), garbage=4, deflate=True, deflate_images=True,
             deflate_fonts=True, use_objstms=1, clean=True)
    src.close()
    logger.info("Saved %s", output_pdf)
