"""Stage A parser with phase-based Surya workflow for scanned PDFs."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import fitz  # PyMuPDF
import torch
from PIL import Image

from pdf2zh.scanned.ai_models import (
    PaddleCellTableModule,
    SuryaLayoutModel,
    SuryaOCRModel,
)
from pdf2zh.scanned.enums import (
    DEFAULT_CATEGORY,
    SURYA_LABEL_MAP,
    ElementCategory,
)
from pdf2zh.scanned.models import (
    CellData,
    ElementData,
    EquationBlockResult,
    EquationParseResult,
    LayoutBlockResult,
    LayoutPageResult,
    LayoutParseResult,
    OCRPageResult,
    OCRParseResult,
    PageData,
    ParsedDocument,
    TableBlockResult,
    TableParseResult,
    _DocumentContext,
    _TableJob,
)
from pdf2zh.scanned.utils.bbox import (
    clamp_bbox,
    convert_bbox,
    image_bbox_to_pdf,
    is_degenerate,
    offset_bbox,
    polygon_to_bbox,
)
from pdf2zh.scanned.utils.hardware import configure_settings
from pdf2zh.scanned.utils.image import crop_image_to_bbox, get_page_dimensions
from pdf2zh.scanned.utils.ocr_text import (
    collect_ocr_text,
    extract_text_for_region,
    join_raw_text,
)

logger = logging.getLogger(__name__)


class StageAParser:
    """Phase-based Stage A parser for scanned PDFs."""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int | None = None,
        page_batch_size: int | None = None,
        layout_batch_size: int | None = None,
        detection_batch_size: int | None = None,
        ocr_batch_size: int | None = None,
        table_batch_size: int | None = None,
        equation_batch_size: int | None = None,
        enable_latex: bool = False,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        """Configure settings and initialize predictors."""

        self.layout_model = SuryaLayoutModel()
        self.ocr_model = SuryaOCRModel()
        # self.table_model = SuryaTableModel(self.hardware)
        self.table_model = PaddleCellTableModule()

        self.hardware = configure_settings(
            device=device,
            batch_size=batch_size,
            page_batch_size=page_batch_size,
            layout_batch_size=layout_batch_size,
            detection_batch_size=detection_batch_size,
            ocr_batch_size=ocr_batch_size,
            table_batch_size=table_batch_size,
            equation_batch_size=equation_batch_size,
            enable_latex=enable_latex,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def parse_layout(
        self,
        context: _DocumentContext,
    ) -> LayoutParseResult:
        """Run the layout phase only."""

        parsed_pages: list[LayoutPageResult] = []

        for batch_indices in self._chunked(
            context.page_indices, self.hardware.layout_batch_size
        ):
            images, _ = self._load_page_images(
                context.pdf_path, batch_indices, include_highres=False
            )
            parsed_pages.extend(
                self._parse_layout_batch(batch_indices, context.page_dims, images)
            )

        return LayoutParseResult(pdf_path=str(context.pdf_path), pages=parsed_pages)

    def parse_ocr(
        self,
        context: _DocumentContext,
    ) -> OCRParseResult:
        """Run the full-page OCR phase only."""

        parsed_pages: list[OCRPageResult] = []

        for batch_indices in self._chunked(
            context.page_indices, self.hardware.detection_batch_size
        ):
            images, highres_images = self._load_page_images(
                context.pdf_path,
                batch_indices,
                include_highres=True,
            )
            parsed_pages.extend(
                self._parse_ocr_batch(batch_indices, images, highres_images)
            )

        return OCRParseResult(pdf_path=str(context.pdf_path), pages=parsed_pages)

    def parse_tables(
        self,
        context: _DocumentContext,
        layout_result: LayoutParseResult,
    ) -> TableParseResult:
        """Run table structure recognition and merge cell text from full-page OCR."""

        if Path(layout_result.pdf_path) != context.pdf_path:
            raise ValueError("layout_result does not belong to the requested PDF")

        tables: dict[str, TableBlockResult] = {}

        for page_batch in self._chunked(
            layout_result.pages, self.hardware.table_batch_size
        ):
            batch_indices = [page.page_index for page in page_batch]
            images, _ = self._load_page_images(
                context.pdf_path, batch_indices, include_highres=False
            )

            batch_tables = self._parse_tables_batch(
                page_batch,
                images,
            )
            tables.update(batch_tables.tables)

        return TableParseResult(pdf_path=str(context.pdf_path), tables=tables)

    def parse_equations(
        self,
        context: _DocumentContext,
        layout_result: LayoutParseResult,
        enable_latex: bool = False,
    ) -> EquationParseResult:
        """Run equation crop OCR for LaTeX when enabled."""

        if Path(layout_result.pdf_path) != context.pdf_path:
            raise ValueError("layout_result does not belong to the requested PDF")

        if not enable_latex:
            return EquationParseResult(
                pdf_path=str(context.pdf_path),
                equations={
                    block.block_id: EquationBlockResult(
                        block_id=block.block_id,
                        latex="[EQUATION_PLACEHOLDER]",
                    )
                    for page in layout_result.pages
                    for block in page.blocks
                    if block.category == ElementCategory.EQUATION
                },
            )

        equations: dict[str, EquationBlockResult] = {}

        for page_batch in self._chunked(
            layout_result.pages, self.hardware.equation_batch_size
        ):
            batch_indices = [page.page_index for page in page_batch]
            _, highres_images = self._load_page_images(
                context.pdf_path, batch_indices, include_highres=True
            )
            batch_equations = self._parse_equations_batch(
                page_batch, highres_images, enable_latex=True
            )
            equations.update(batch_equations.equations)

        return EquationParseResult(pdf_path=str(context.pdf_path), equations=equations)

    def merge_results(
        self,
        pdf_path: str | Path,
        layout_result: LayoutParseResult,
        ocr_result: OCRParseResult,
        table_result: TableParseResult | None = None,
        equation_result: EquationParseResult | None = None,
    ) -> ParsedDocument:
        """Merge phase outputs into the final ParsedDocument."""

        pdf_path = self._resolve_pdf_path(pdf_path)
        if Path(layout_result.pdf_path) != pdf_path:
            raise ValueError("layout_result does not belong to the requested PDF")
        if Path(ocr_result.pdf_path) != pdf_path:
            raise ValueError("ocr_result does not belong to the requested PDF")

        table_map = table_result.tables if table_result else {}
        equation_map = equation_result.equations if equation_result else {}
        ocr_page_map = ocr_result.page_map()

        pages: list[PageData] = []
        for layout_page in layout_result.pages:
            page_ocr = ocr_page_map.get(layout_page.page_index)
            elements: list[ElementData] = []

            for block in layout_page.blocks:
                source_text = ""
                latex = ""
                cells: list[CellData] = []

                if block.category == ElementCategory.BYPASS:
                    font_size = 0.0
                    pass
                elif block.category == ElementCategory.TABLE:
                    table_block = table_map.get(block.block_id)
                    crop_w, crop_h = table_block.crop_size
                    block_image_w = block.bbox_image[2] - block.bbox_image[0]
                    block_image_h = block.bbox_image[3] - block.bbox_image[1]
                    block_pdf_w = block.bbox_pdf[2] - block.bbox_pdf[0]
                    block_pdf_h = block.bbox_pdf[3] - block.bbox_pdf[1]

                    source_parts: list[str] = []
                    cells: list[CellData] = []

                    for cell_bbox in table_block.cells_bbox:
                        cell_bbox_image = offset_bbox(
                            convert_bbox(
                                cell_bbox, crop_w, crop_h, block_image_w, block_image_h
                            ),
                            block.bbox_image[0],
                            block.bbox_image[1],
                        )
                        cell_bbox_pdf = clamp_bbox(
                            offset_bbox(
                                convert_bbox(
                                    cell_bbox, crop_w, crop_h, block_pdf_w, block_pdf_h
                                ),
                                block.bbox_pdf[0],
                                block.bbox_pdf[1],
                            ),
                            layout_page.page_width,
                            layout_page.page_height,
                        )
                        cell_text, cell_font_size = extract_text_for_region(
                            page_ocr.ocr_result, cell_bbox_image
                        )
                        cells.append(
                            CellData(
                                bbox_pdf=cell_bbox_pdf,
                                source_text=cell_text,
                                translated_text="",
                                cell_font_size=cell_font_size,
                            )
                        )
                        source_parts.append(cell_text)

                    source_text = " | ".join(source_parts)
                    font_size = (
                        median(
                            [c.cell_font_size for c in cells if c.cell_font_size > 0]
                        )
                        if any(c.cell_font_size > 0 for c in cells)
                        else 0.0
                    )
                    if not cells:
                        source_text, font_size = extract_text_for_region(
                            page_ocr.ocr_result, block.bbox_image
                        )
                else:
                    source_text, font_size = extract_text_for_region(
                        page_ocr.ocr_result, block.bbox_image
                    )
                    if block.category == ElementCategory.EQUATION:
                        equation_block = equation_map.get(block.block_id)
                        latex = (
                            equation_block.latex
                            if equation_block is not None
                            else "[EQUATION_PLACEHOLDER]"
                        )

                elements.append(
                    ElementData(
                        label=block.label,
                        category=block.category,
                        bbox_pdf=block.bbox_pdf,
                        source_text=source_text,
                        translated_text="",
                        latex=latex,
                        cells=cells,
                        font_size=font_size,
                    )
                )

            # Don't need sort because surya-ocr return layout with reading order
            # elements.sort(key=lambda element: element.bbox_pdf[1])
            # log_toc_hints(elements, layout_page.page_index)
            pages.append(
                PageData(
                    page_index=layout_page.page_index,
                    page_width=layout_page.page_width,
                    page_height=layout_page.page_height,
                    elements=elements,
                    raw_text=join_raw_text(elements),
                    chapter_id="",
                )
            )

        return ParsedDocument(
            pdf_path=str(pdf_path),
            pages=pages,
            chapters=[],
            glossary={},
        )

    def parse_pdf(
        self,
        pdf_path: str | Path,
        cache_path: str | Path | None = None,
        pages: list[int] | None = None,
        enable_latex: bool | None = None,
    ) -> ParsedDocument:
        """Backward-compatible wrapper that executes the phase pipeline."""

        pdf_path = self._resolve_pdf_path(pdf_path)
        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading cached Stage A result from %s", cache_path)
                return ParsedDocument.load(cache_path)

        context = self._prepare_document_context(pdf_path, pages)
        enable_latex = (
            self.hardware.enable_latex if enable_latex is None else enable_latex
        )

        layout_pages: list[LayoutPageResult] = []
        ocr_pages: list[OCRPageResult] = []
        tables: dict[str, TableBlockResult] = {}
        equations: dict[str, EquationBlockResult] = {}

        for batch_indices in self._chunked(
            context.page_indices, self.hardware.page_batch_size
        ):
            images, highres_images = self._load_page_images(
                context.pdf_path,
                batch_indices,
                include_highres=True,
            )
            batch_layout_pages = self._parse_layout_batch(
                batch_indices,
                context.page_dims,
                images,
            )

            batch_tables = self._parse_tables_batch(
                batch_layout_pages,
                images,
            )
            batch_equations = self._parse_equations_batch(
                batch_layout_pages,
                highres_images,
                enable_latex=enable_latex,
            )
            batch_ocr_pages = self._parse_ocr_batch(
                batch_indices, images, highres_images
            )

            layout_pages.extend(batch_layout_pages)
            ocr_pages.extend(batch_ocr_pages)
            tables.update(batch_tables.tables)
            equations.update(batch_equations.equations)
            self._release_batch(images, highres_images)

        parsed_doc = self.merge_results(
            context.pdf_path,
            LayoutParseResult(pdf_path=str(context.pdf_path), pages=layout_pages),
            OCRParseResult(pdf_path=str(context.pdf_path), pages=ocr_pages),
            table_result=TableParseResult(
                pdf_path=str(context.pdf_path), tables=tables
            ),
            equation_result=EquationParseResult(
                pdf_path=str(context.pdf_path), equations=equations
            ),
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_doc.save(cache_path)
            logger.info("Saved Stage A result to %s", cache_path)

        return parsed_doc

    def _prepare_document_context(
        self,
        pdf_path: str | Path,
        pages: list[int] | None,
    ) -> _DocumentContext:
        pdf_path = self._resolve_pdf_path(pdf_path)
        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF is empty")
            if pages is None:
                page_indices = list(range(len(doc)))
            else:
                page_indices = [index for index in pages if 0 <= index < len(doc)]
            page_dims = {
                index: get_page_dimensions(doc[index]) for index in page_indices
            }
        finally:
            doc.close()

        return _DocumentContext(
            pdf_path=pdf_path,
            page_indices=page_indices,
            page_dims=page_dims,
        )

    def _load_page_images(
        self,
        pdf_path: Path,
        page_indices: list[int],
        include_highres: bool,
    ) -> tuple[list[Image.Image], list[Image.Image] | None]:
        from surya.input.load import load_from_file
        from surya.settings import settings

        images, _ = load_from_file(str(pdf_path), page_range=page_indices)
        if not include_highres:
            return images, None

        highres_images, _ = load_from_file(
            str(pdf_path),
            dpi=settings.IMAGE_DPI_HIGHRES,
            page_range=page_indices,
        )
        return images, highres_images

    def _parse_layout_batch(
        self,
        batch_indices: list[int],
        page_dims: dict[int, tuple[float, float]],
        images: list[Image.Image],
    ) -> list[LayoutPageResult]:
        layout_predictions = self.layout_model.predict(
            images, batch_size=self.hardware.layout_batch_size
        )

        layout_pages: list[LayoutPageResult] = []

        for seq, page_index in enumerate(batch_indices):
            page_width, page_height = page_dims[page_index]
            image_bbox = [0, 0, images[seq].size[0], images[seq].size[1]]
            layout_image_bbox = list(layout_predictions[seq].image_bbox)
            blocks: list[LayoutBlockResult] = []

            for position, block in enumerate(layout_predictions[seq].bboxes):
                raw_bbox = list(getattr(block, "bbox", polygon_to_bbox(block.polygon)))
                label = block.label
                category = SURYA_LABEL_MAP.get(label, DEFAULT_CATEGORY)
                bbox_pdf = clamp_bbox(
                    image_bbox_to_pdf(
                        raw_bbox, layout_image_bbox, page_width, page_height
                    ),
                    page_width,
                    page_height,
                )
                bbox_image = clamp_bbox(
                    convert_bbox(
                        raw_bbox,
                        layout_image_bbox[2],
                        layout_image_bbox[3],
                        image_bbox[2],
                        image_bbox[3],
                    ),
                    image_bbox[2],
                    image_bbox[3],
                )

                if is_degenerate(bbox_pdf) or is_degenerate(bbox_image):
                    logger.debug(
                        "Skipping degenerate layout bbox on page %s", page_index
                    )
                    continue

                blocks.append(
                    LayoutBlockResult(
                        block_id=f"{page_index}:{getattr(block, 'position', position)}",
                        page_index=page_index,
                        position=getattr(block, "position", position),
                        label=label,
                        category=category,
                        bbox_layout=raw_bbox,
                        bbox_image=bbox_image,
                        bbox_pdf=bbox_pdf,
                    )
                )

            layout_pages.append(
                LayoutPageResult(
                    page_index=page_index,
                    page_width=page_width,
                    page_height=page_height,
                    layout_image_bbox=layout_image_bbox,
                    image_bbox=image_bbox,
                    blocks=blocks,
                )
            )

        return layout_pages

    def _parse_ocr_batch(
        self,
        batch_indices: list[int],
        images: list[Image.Image],
        highres_images: list[Image.Image] | None,
    ) -> list[OCRPageResult]:
        ocr_predictions = self.ocr_model.predict(
            images,
            highres_images=highres_images,
            math_mode=False,
            detection_batch_size=self.hardware.detection_batch_size,
            ocr_batch_size=self.hardware.ocr_batch_size,
        )

        return [
            OCRPageResult(
                page_index=page_index,
                image_bbox=list(
                    getattr(
                        prediction,
                        "image_bbox",
                        [0, 0, images[seq].size[0], images[seq].size[1]],
                    )
                ),
                ocr_result=prediction,
            )
            for seq, (page_index, prediction) in enumerate(
                zip(batch_indices, ocr_predictions)
            )
        ]

    def _parse_tables_batch(
        self,
        layout_pages: list[LayoutPageResult],
        images: list[Image.Image],
    ) -> TableParseResult:

        table_jobs: list[_TableJob] = []
        table_crops: list[Image.Image] = []

        for seq, page in enumerate(layout_pages):
            for block in page.blocks:
                if block.category != ElementCategory.TABLE:
                    continue
                table_crop = crop_image_to_bbox(
                    images[seq],
                    block.bbox_pdf,
                    page.page_width,
                    page.page_height,
                )
                table_jobs.append(
                    _TableJob(
                        block=block,
                        page_width=page.page_width,
                        page_height=page.page_height,
                        table_crop=table_crop,
                    )
                )
                table_crops.append(table_crop)

        if not table_jobs:
            return TableParseResult(pdf_path="", tables={})

        table_predictions = self.table_model.predict(
            table_crops, batch_size=self.hardware.table_batch_size
        )

        tables: dict[str, TableBlockResult] = {}

        for job, prediction in zip(table_jobs, table_predictions):
            table_result = TableBlockResult(
                block_id=job.block.block_id,
                cells_bbox=prediction,
                crop_size=job.table_crop.size,
            )
            tables[job.block.block_id] = table_result

        return TableParseResult(pdf_path="", tables=tables)

    def _parse_equations_batch(
        self,
        layout_pages: list[LayoutPageResult],
        highres_images: list[Image.Image] | None,
        enable_latex: bool,
    ) -> EquationParseResult:
        equations: dict[str, EquationBlockResult] = {}

        equation_blocks = [
            (seq, page, block)
            for seq, page in enumerate(layout_pages)
            for block in page.blocks
            if block.category == ElementCategory.EQUATION
        ]

        if not equation_blocks:
            return EquationParseResult(pdf_path="", equations={})

        if not enable_latex:
            for _seq, _page, block in equation_blocks:
                equations[block.block_id] = EquationBlockResult(
                    block_id=block.block_id,
                    latex="[EQUATION_PLACEHOLDER]",
                )
            return EquationParseResult(pdf_path="", equations=equations)

        if highres_images is None:
            raise ValueError("highres images are required for equation OCR")

        from surya.common.surya.schema import TaskNames

        equation_crops: list[Image.Image] = []
        task_names: list[str] = []
        block_ids: list[str] = []

        for seq, page, block in equation_blocks:
            crop = crop_image_to_bbox(
                highres_images[seq],
                block.bbox_pdf,
                page.page_width,
                page.page_height,
            )
            equation_crops.append(crop)
            task_names.append(TaskNames.block_without_boxes)
            block_ids.append(block.block_id)

        predictions = self.ocr_model.predict(
            equation_crops,
            task_names=task_names,
            bboxes=[[[0, 0, crop.size[0], crop.size[1]]] for crop in equation_crops],
            math_mode=True,
            ocr_batch_size=self.hardware.equation_batch_size,
        )

        for block_id, prediction in zip(block_ids, predictions):
            latex_content = collect_ocr_text(prediction)
            latex = latex_content if latex_content else "[EQUATION_PLACEHOLDER]"
            equations[block_id] = EquationBlockResult(block_id=block_id, latex=latex)

        return EquationParseResult(pdf_path="", equations=equations)

    def _resolve_pdf_path(self, pdf_path: str | Path) -> Path:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return pdf_path

    def _release_batch(self, *objects: Any) -> None:
        for obj in objects:
            if obj is None:
                continue
            del obj

        gc.collect()
        if self.hardware.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _chunked(self, items: list[Any], size: int) -> Iterable[list[Any]]:
        for start in range(0, len(items), size):
            yield items[start : start + size]
