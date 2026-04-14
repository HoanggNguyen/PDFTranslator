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

import gc
import logging
from pathlib import Path

import fitz  # PyMuPDF
import torch
from PIL import Image

from pdf2zh.scanned.enums import (
    DEFAULT_CATEGORY,
    SURYA_LABEL_MAP,
    ElementCategory,
)
from pdf2zh.scanned.models import (
    CellData,
    ElementData,
    PageData,
    ParsedDocument,
)
from pdf2zh.scanned.utils.bbox import (
    clamp_bbox,
    convert_bbox,
    image_bbox_to_pdf,
    is_degenerate,
    offset_bbox,
)
from pdf2zh.scanned.utils.hardware import (
    resolve_hardware,
    set_torch_device_env,
)
from pdf2zh.scanned.utils.image import (
    crop_image_to_bbox,
    get_page_dimensions,
)
from pdf2zh.scanned.utils.ocr_text import (
    collect_ocr_text,
    extract_text_for_region,
    join_raw_text,
    log_toc_hints,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data carrier used during per-page assembly (before ElementData)
# ---------------------------------------------------------------------------

class _BlockInfo:
    """Mutable scratch pad for one layout block while we collect OCR crops."""
    __slots__ = ("label", "category", "pdf_bbox", "source_text", "latex",
                 "cells", "page_seq", "needs_ocr")

    def __init__(self, label: str, category: ElementCategory,
                 pdf_bbox: list[float], page_seq: int) -> None:
        """Initialize a scratch-pad block info carrier.

        Args:
            label: Raw Surya layout label (e.g. "Text", "Table", "Picture")
            category: Resolved ElementCategory for downstream handling
            pdf_bbox: [x0, y0, x1, y1] bounding box in PDF points
            page_seq: Index of this page within the current batch (0-based)
        """
        self.label = label
        self.category = category
        self.pdf_bbox = pdf_bbox
        self.source_text = ""
        self.latex = ""
        self.cells: list[CellData] = []
        self.page_seq = page_seq      # index within batch
        self.needs_ocr = False

    def to_element(self) -> ElementData:
        """Convert this scratch-pad entry into a final ElementData instance.

        Returns:
            ElementData populated from the accumulated fields of this block info.
        """
        return ElementData(
            label=self.label,
            category=self.category,
            bbox_pdf=self.pdf_bbox,
            source_text=self.source_text,
            translated_text="",
            latex=self.latex,
            cells=self.cells,
        )


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

        self._foundation_predictor = None
        self._layout_foundation_predictor = None
        self._detection_predictor = None
        self._layout_predictor = None
        self._recognition_predictor = None
        self._table_predictor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def foundation_predictor(self):
        """FoundationPredictor — default (OCR) checkpoint."""
        if self._foundation_predictor is None:
            from surya.foundation import FoundationPredictor
            self._foundation_predictor = FoundationPredictor()
            logger.info("Loaded FoundationPredictor (OCR)")
        return self._foundation_predictor

    @property
    def layout_foundation_predictor(self):
        """FoundationPredictor — layout checkpoint (different weights)."""
        if self._layout_foundation_predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.settings import settings
            self._layout_foundation_predictor = FoundationPredictor(
                checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
            )
            logger.info("Loaded FoundationPredictor (layout)")
        return self._layout_foundation_predictor

    @property
    def detection_predictor(self):
        """DetectionPredictor — text region detection model (lazy loaded)."""
        if self._detection_predictor is None:
            from surya.detection import DetectionPredictor
            self._detection_predictor = DetectionPredictor()
            logger.info("Loaded DetectionPredictor")
        return self._detection_predictor

    @property
    def layout_predictor(self):
        """LayoutPredictor — page layout analysis model (lazy loaded)."""
        if self._layout_predictor is None:
            from surya.layout import LayoutPredictor
            self._layout_predictor = LayoutPredictor(self.layout_foundation_predictor)
            logger.info("Loaded LayoutPredictor")
        return self._layout_predictor

    @property
    def recognition_predictor(self):
        """RecognitionPredictor — OCR text recognition model (lazy loaded)."""
        if self._recognition_predictor is None:
            from surya.recognition import RecognitionPredictor
            self._recognition_predictor = RecognitionPredictor(self.foundation_predictor)
            logger.info("Loaded RecognitionPredictor")
        return self._recognition_predictor

    @property
    def table_predictor(self):
        """TableRecPredictor — table structure recognition model (lazy loaded)."""
        if self._table_predictor is None:
            from surya.table_rec import TableRecPredictor
            self._table_predictor = TableRecPredictor()
            logger.info("Loaded TableRecPredictor")
        return self._table_predictor

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
        """Process pages from a PDF file in batches, running the full Surya pipeline.

        Pages are grouped into batches of ``self.profile.batch_size`` and each
        batch goes through four phases:

        1. Load standard-DPI and highres-DPI images via Surya's ``load_from_file``.
        2. Detect layout regions on all standard-DPI images in parallel.
        3. Collect per-region OCR crops across the whole batch into **one**
           ``recognition_predictor`` call for maximum GPU utilisation.
        4. Run table structure recognition for TABLE regions.
        5. Assemble :class:`~pdf2zh.scanned.models.PageData` objects.

        Args:
            pdf_path: Resolved path to the PDF file.
            page_indices: Ordered list of 0-based page numbers to process.
            page_dims: Mapping from page index to ``(width, height)`` in PDF points.

        Returns:
            List of :class:`~pdf2zh.scanned.models.PageData` in the same order
            as *page_indices*.
        """
        from surya.input.load import load_from_file
        from surya.settings import settings

        batch_size = self.profile.batch_size
        ocr_batch_size = self.profile.ocr_batch_size
        all_page_data: list[PageData] = []

        for batch_start in range(0, len(page_indices), batch_size):
            batch_indices = page_indices[batch_start:batch_start + batch_size]
            logger.info(f"Processing pages {batch_indices}")

            # Load at Surya's native DPIs
            images, _ = load_from_file(str(pdf_path), page_range=batch_indices)
            highres_images, _ = load_from_file(
                str(pdf_path),
                dpi=settings.IMAGE_DPI_HIGHRES,
                page_range=batch_indices,
            )

            # Layout detection on standard-DPI images (batch parallel)
            layout_results = self.layout_predictor(images)

            # ----------------------------------------------------------
            # Phase 1 — collect block infos + OCR crops across ALL pages
            # ----------------------------------------------------------
            # Per-page list of _BlockInfo
            pages_block_infos: list[list[_BlockInfo]] = []
            # Pooled crops from every page (for one giant OCR call)
            all_ocr_std: list[Image.Image] = []
            all_ocr_hr: list[Image.Image] = []
            # Maps each crop back to its _BlockInfo
            all_ocr_targets: list[_BlockInfo] = []

            for i, idx in enumerate(batch_indices):
                pw, ph = page_dims[idx]
                layout_image_bbox = layout_results[i].image_bbox

                block_infos: list[_BlockInfo] = []
                for block in layout_results[i].bboxes:
                    label = block.label
                    category = SURYA_LABEL_MAP.get(label, DEFAULT_CATEGORY)

                    pdf_bbox = image_bbox_to_pdf(
                        block.bbox, layout_image_bbox, pw, ph,
                    )
                    pdf_bbox = clamp_bbox(pdf_bbox, pw, ph)

                    if is_degenerate(pdf_bbox):
                        logger.debug(f"Skipping degenerate bbox for {label}")
                        continue

                    info = _BlockInfo(label, category, pdf_bbox, page_seq=i)
                    block_infos.append(info)

                    if category == ElementCategory.BYPASS:
                        # Pure image — no OCR
                        pass
                    elif category == ElementCategory.TABLE:
                        # Handled in Phase 3
                        pass
                    else:
                        # FLOWING_TEXT, IN_PLACE, EQUATION → pool for batch OCR
                        info.needs_ocr = True
                        all_ocr_std.append(
                            crop_image_to_bbox(images[i], pdf_bbox, pw, ph)
                        )
                        all_ocr_hr.append(
                            crop_image_to_bbox(highres_images[i], pdf_bbox, pw, ph)
                        )
                        all_ocr_targets.append(info)

                pages_block_infos.append(block_infos)

            # ----------------------------------------------------------
            # Phase 2 — single batch OCR for ALL crops across all pages
            # ----------------------------------------------------------
            if all_ocr_std:
                logger.info(
                    f"Batch OCR: {len(all_ocr_std)} crops from "
                    f"{len(batch_indices)} pages"
                )
                ocr_results = []
                for i in range(0, len(all_ocr_std), ocr_batch_size):
                    sub_std = all_ocr_std[i:i + ocr_batch_size]
                    sub_hr = all_ocr_hr[i:i + ocr_batch_size]

                    sub_results = self.recognition_predictor(
                        sub_std,
                        det_predictor=self.detection_predictor,
                        highres_images=sub_hr,
                    )
                    ocr_results.extend(sub_results)

                    # Dọn dẹp ngay lập tức sau mỗi sub-batch
                    torch.cuda.empty_cache()

                for crop_i, info in enumerate(all_ocr_targets):
                    text = collect_ocr_text(ocr_results[crop_i])
                    info.source_text = text
                    if info.category == ElementCategory.EQUATION:
                        info.latex = "[EQUATION_PLACEHOLDER]"

            # ----------------------------------------------------------
            # Phase 3 — TABLE regions (table structure + per-cell OCR)
            # ----------------------------------------------------------
            for i, idx in enumerate(batch_indices):
                pw, ph = page_dims[idx]
                for info in pages_block_infos[i]:
                    if info.category != ElementCategory.TABLE:
                        continue
                    source_text, cells = self._process_table(
                        image=images[i],
                        highres_image=highres_images[i],
                        pdf_bbox=info.pdf_bbox,
                        page_width=pw,
                        page_height=ph,
                    )
                    info.source_text = source_text
                    info.cells = cells

            # ----------------------------------------------------------
            # Phase 4 — assemble PageData for each page
            # ----------------------------------------------------------
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

            del images, highres_images, layout_results
            if 'all_ocr_std' in locals():
                del all_ocr_std, all_ocr_hr, ocr_results

            gc.collect()
            torch.cuda.empty_cache() # Xóa cache của PyTorch trên GPU

        return all_page_data

    # ------------------------------------------------------------------
    # Internal: table processing
    # ------------------------------------------------------------------

    def _process_table(
        self,
        image: Image.Image,
        highres_image: Image.Image,
        pdf_bbox: list[float],
        page_width: float,
        page_height: float,
    ) -> tuple[str, list[CellData]]:
        """Run table structure recognition and per-cell OCR for a single table region.

        Processing steps:

        1. Crop the table region from the standard-DPI page image.
        2. Run ``table_predictor`` to detect row/column/cell structure.
        3. Re-crop the highres image and run ``recognition_predictor`` on the
           table crop to produce full-page OCR output for the region.
        4. For each detected cell, extract OCR text using bounding-box overlap
           with the recognition output.
        5. Convert cell bounding boxes from table-image pixel space to absolute
           PDF point coordinates.

        Args:
            image: Standard-DPI PIL Image of the full page.
            highres_image: Highres PIL Image of the full page (for OCR quality).
            pdf_bbox: [x0, y0, x1, y1] of the table in PDF points (page-absolute).
            page_width: Page width in PDF points (used for coordinate conversion).
            page_height: Page height in PDF points (used for coordinate conversion).

        Returns:
            A 2-tuple ``(source_text, cells)`` where:
            - *source_text* is all cell texts joined by " | " (for full-table search).
            - *cells* is a list of :class:`~pdf2zh.scanned.models.CellData`
              with PDF-point bboxes and OCR text.
        """
        table_crop = crop_image_to_bbox(image, pdf_bbox, page_width, page_height)

        try:
            table_results = self.table_predictor([table_crop])
            if not table_results or not table_results[0].cells:
                logger.warning("Table recognition returned no cells")
                return "", []
        except Exception as e:
            logger.error(f"Table recognition failed: {e}")
            return "", []

        table_result = table_results[0]
        crop_w, crop_h = table_crop.size
        table_x0, table_y0 = pdf_bbox[0], pdf_bbox[1]
        table_pdf_w = pdf_bbox[2] - pdf_bbox[0]
        table_pdf_h = pdf_bbox[3] - pdf_bbox[1]

        # OCR on the table crop
        try:
            hr_crop = crop_image_to_bbox(highres_image, pdf_bbox, page_width, page_height)
            ocr_results = self.recognition_predictor(
                [table_crop],
                det_predictor=self.detection_predictor,
                highres_images=[hr_crop],
            )
            ocr_result = ocr_results[0] if ocr_results else None

            del hr_crop, ocr_results
        except Exception as e:
            logger.error(f"Table OCR failed: {e}")
            ocr_result = None

        cells = []
        text_parts = []

        for cell in table_result.cells:
            cell_bbox_in_table = convert_bbox(
                cell.bbox, crop_w, crop_h, table_pdf_w, table_pdf_h,
            )
            cell_pdf_bbox = offset_bbox(cell_bbox_in_table, table_x0, table_y0)
            cell_pdf_bbox = clamp_bbox(cell_pdf_bbox, page_width, page_height)

            cell_text = ""
            if ocr_result is not None:
                cell_text = extract_text_for_region(
                    ocr_result, cell.bbox, crop_w, crop_h,
                )

            cells.append(CellData(
                bbox_pdf=cell_pdf_bbox,
                row_id=cell.row_id if hasattr(cell, "row_id") else 0,
                col_id=cell.col_id if hasattr(cell, "col_id") else 0,
                source_text=cell_text,
                translated_text="",
            ))

            if cell_text:
                text_parts.append(cell_text)

        return " | ".join(text_parts), cells
