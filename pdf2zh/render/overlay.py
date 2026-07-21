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
    """Stamp overlay_pdf transparently onto the selected pages of original_pdf.

    The output contains ONLY the translated pages (those in ``pages``), in
    ascending page order — not the whole original document. When ``pages`` is
    None every page is kept. For scanned PDFs the original page image is the
    background; the overlay PDF carries translated text and cover rects with
    sampled bg colors. PyMuPDF's show_pdf_page() composites them in a single
    vector operation.
    """
    src = fitz.open(str(original_pdf))
    ov = fitz.open(str(overlay_pdf))

    if pages is None:
        selected = list(range(src.page_count))
    else:
        selected = sorted(p for p in pages if 0 <= p < src.page_count)

    # Overlay page i corresponds to the i-th selected source page: source_builder
    # emits rendered pages in ascending page_index order, matching sorted(pages).
    # Guard a shorter overlay so a missing page never raises.
    n = min(len(selected), ov.page_count)
    if n < len(selected):
        logger.warning(
            "Overlay has fewer pages (%d) than selected (%d); stopping early",
            ov.page_count,
            len(selected),
        )

    out = fitz.open()
    for i in range(n):
        src_idx = selected[i]
        out.insert_pdf(src, from_page=src_idx, to_page=src_idx)
        out[i].show_pdf_page(out[i].rect, ov, i, overlay=True)
        logger.debug(
            "Composited overlay page %d → output page %d (source %d)", i, i, src_idx
        )

    ov.close()
    src.close()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    out.save(
        str(output_pdf),
        garbage=4,
        deflate=True,
        deflate_images=True,
        deflate_fonts=True,
        use_objstms=1,
        clean=True,
    )
    out.close()
    logger.info("Saved %s (%d pages)", output_pdf, n)
