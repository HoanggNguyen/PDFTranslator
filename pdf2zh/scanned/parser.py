"""Stage A Parser for scanned PDFs.

This module provides StageAParser, which orchestrates:
1. PDF page loading via Surya's native loader (``surya.input.load``)
2. Surya layout detection
3. Per-region crop → batch OCR across all pages (no full-page OCR)
4. Table structure recognition
5. Coordinate conversion and data assembly

Image loading is delegated to Surya's ``load_from_file`` so that the
correct DPI / preprocessing is applied automatically, matching the
behaviour of ``surya_gui`` and Surya's own CLI tools.

OCR crops from all pages in a batch are pooled into a single
``recognition_predictor`` call for maximum GPU utilization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import gc

from pdf2zh.scanned.models import (
    PageData,
    ParsedDocument,
)
from pdf2zh.scanned.batch_processing import (
    cleanup_batch,
    collect_batch_block_infos,
    load_batch_images,
    resolve_batch_sizes,
    run_batch_ocr,
)
from pdf2zh.scanned.predictors import SuryaPredictors
from pdf2zh.scanned.table_processing import process_table_batch
from pdf2zh.scanned.utils.hardware import (
    resolve_hardware,
    set_torch_device_env,
)
from pdf2zh.scanned.utils.image import get_page_dimensions
from pdf2zh.scanned.utils.ocr_text import (
    log_toc_hints,
    join_raw_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StageAParser:
    """Parser for Stage A of the scanned PDF pipeline.

    Pipeline per batch:
    1. ``load_from_file`` renders pages at Surya's native DPIs.
    2. ``layout_predictor`` detects layout regions on all standard-DPI images.
    3. Every non-BYPASS region is cropped from both standard and highres images.
    4. **All crops across all pages** in the batch are pooled into one
       ``recognition_predictor`` call for optimal GPU throughput.
    5. TABLE regions additionally run ``table_predictor`` for cell structure.
    """

    def __init__(
        self,
        device: str = "auto",
        batch_size: int | None = None,
        ocr_batch_size: int | None = None,
    ) -> None:
        """Initialize the Stage A parser and set up hardware.

        Args:
            device: Target device for Surya models — one of ``"auto"``,
                ``"cuda"``, ``"mps"``, or ``"cpu"``.
                ``"auto"`` detects CUDA → MPS → CPU in that priority order.
            batch_size: Number of pages to process per batch.  When ``None``
                the default for the resolved device is used (12 for CUDA,
                4 for MPS, 2 for CPU).
            ocr_batch_size: Number of OCR crops to process per batch.  When ``None``
                the default for the resolved device is used (12 for CUDA, 4 for MPS, 2 for CPU).
        """
        self.profile = resolve_hardware(device, batch_size, ocr_batch_size)
        set_torch_device_env(self.profile.device)

        self.predictors = SuryaPredictors()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_pdf(
        self,
        pdf_path: str | Path,
        cache_path: str | Path | None = None,
        pages: list[int] | None = None,
    ) -> ParsedDocument:
        """Parse a PDF file and produce a ParsedDocument.

        Args:
            pdf_path: Path to the PDF file
            cache_path: Optional path to save/load JSON cache
            pages: Optional list of 0-based page indices to process

        Returns:
            ParsedDocument with all Stage A fields populated
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info(f"Loading cached result from {cache_path}")
                return ParsedDocument.load(cache_path)

        self.predictors.preload_predictors()
        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF is empty")
            if pages is None:
                page_indices = list(range(len(doc)))
            else:
                page_indices = [i for i in pages if 0 <= i < len(doc)]
            page_dims = {idx: get_page_dimensions(doc[idx]) for idx in page_indices}
        finally:
            doc.close()

        parsed_pages = self._process_pages_batch(pdf_path, page_indices, page_dims)

        result = ParsedDocument(
            pdf_path=str(pdf_path),
            pages=parsed_pages,
            chapters=[],
            glossary={},
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(cache_path)
            logger.info(f"Saved result to {cache_path}")

        return result

    # ------------------------------------------------------------------
    # Internal: batch processing across pages
    # ------------------------------------------------------------------

    def _process_pages_batch(
        self,
        pdf_path: Path,
        page_indices: list[int],
        page_dims: dict[int, tuple[float, float]],
    ) -> list[PageData]:
        """Process pages from a PDF file in batches, running the full Surya pipeline."""
        batch_size, ocr_batch_size = resolve_batch_sizes(self.profile)
        all_page_data: list[PageData] = []

        for batch_start in range(0, len(page_indices), batch_size):
            batch_indices = page_indices[batch_start:batch_start + batch_size]
            logger.info("Processing pages %s", batch_indices)

            images, highres_images = load_batch_images(pdf_path, batch_indices)
            layout_results = self.predictors.layout_predictor(images)

            pages_block_infos, all_ocr_std, all_ocr_hr, all_ocr_targets = (
                collect_batch_block_infos(
                    batch_indices,
                    page_dims,
                    images,
                    highres_images,
                    layout_results,
                )
            )

            if all_ocr_std:
                run_batch_ocr(
                    self.predictors.recognition_predictor,
                    self.predictors.detection_predictor,
                    all_ocr_std,
                    all_ocr_hr,
                    all_ocr_targets,
                    ocr_batch_size,
                    self.profile.device,
                )

            process_table_batch(
                self.predictors.table_predictor,
                self.predictors.recognition_predictor,
                self.predictors.detection_predictor,
                batch_indices,
                page_dims,
                images,
                highres_images,
                pages_block_infos,
            )

            for i, idx in enumerate(batch_indices):
                pw, ph = page_dims[idx]
                elements = [info.to_element() for info in pages_block_infos[i]]
                elements.sort(key=lambda e: e.bbox_pdf[1])

                log_toc_hints(elements, idx)
                raw_text = join_raw_text(elements)

                all_page_data.append(PageData(
                    page_index=idx,
                    page_width=pw,
                    page_height=ph,
                    elements=elements,
                    raw_text=raw_text,
                    chapter_id="",
                ))

            cleanup_batch(
                images,
                highres_images,
                layout_results,
                pages_block_infos,
                all_ocr_std,
                all_ocr_hr,
                all_ocr_targets,
                device=self.profile.device,
            )

        return all_page_data

