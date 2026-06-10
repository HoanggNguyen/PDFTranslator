"""Stage A parser with phase-based Surya workflow for scanned PDFs."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import torch
from PIL import Image

from pdf2zh.parser.ai_models import (
    PaddleCellTableModule,
    SuryaLayoutModel,
    SuryaOCRModel,
)
from pdf2zh.parser.enums import (
    DEFAULT_CATEGORY,
    SURYA_LABEL_MAP,
    ElementCategory,
)
from pdf2zh.parser.models import (
    CellData,
    ElementData,
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
from pdf2zh.parser.utils.bbox import (
    bbox_area,
    bbox_intersection,
    bbox_union_area,
    clamp_bbox,
    convert_bbox,
    image_bbox_to_pdf,
    is_degenerate,
    offset_bbox,
    polygon_to_bbox,
)
from pdf2zh.parser.utils.block import (
    get_line_bbox,
    is_sparse_text_block,
)
from pdf2zh.parser.utils.hardware import configure_settings
from pdf2zh.parser.utils.image import crop_image_to_bbox, get_page_dimensions
from pdf2zh.parser.utils.ocr_text import (
    clean_ocr_text,
    extract_text_for_region,
    join_raw_text,
    smart_join_text_lines,
    sort_text_lines,
)

logger = logging.getLogger(__name__)


class StageAParser:
    """Phase-based Stage A parser for scanned PDFs."""

    def __init__(
        self,
        device: str = "auto",
        page_batch_size: int | None = None,
        layout_batch_size: int | None = None,
        detection_batch_size: int | None = None,
        ocr_batch_size: int | None = None,
        table_batch_size: int | None = None,
        detector_blank_threshold: float | None = None,
        detector_text_threshold: float | None = None,
    ) -> None:
        """Configure settings and initialize predictors."""

        self.hardware = configure_settings(
            device=device,
            page_batch_size=page_batch_size,
            layout_batch_size=layout_batch_size,
            detection_batch_size=detection_batch_size,
            ocr_batch_size=ocr_batch_size,
            table_batch_size=table_batch_size,
        )
        self.layout_model = SuryaLayoutModel()
        self.ocr_model = SuryaOCRModel(
            detector_blank_threshold=detector_blank_threshold,
            detector_text_threshold=detector_text_threshold,
        )
        # self.table_model = SuryaTableModel(self.hardware)
        self.table_model = PaddleCellTableModule()

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
                context.pdf_path,
                batch_indices,
                include_highres=False,
            )
            parsed_pages.extend(
                self._parse_layout_batch(
                    batch_indices,
                    context.page_dims,
                    images,
                    ocr_pages=None,
                )
            )
            self._release_batch(images)

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
                context.pdf_path, batch_indices, include_highres=True
            )
            parsed_pages.extend(
                self._parse_ocr_batch(batch_indices, images, highres_images)
            )
            self._release_batch(images, highres_images)

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
            self._release_batch(images)

        return TableParseResult(pdf_path=str(context.pdf_path), tables=tables)

    def parse_pdf(
        self,
        pdf_path: str | Path,
        cache_path: str | Path | None = None,
        pages: list[int] | None = None,
    ) -> ParsedDocument:
        """Backward-compatible wrapper that executes the phase pipeline."""

        pdf_path = self._resolve_pdf_path(pdf_path)
        if cache_path:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading cached Stage A result from %s", cache_path)
                return ParsedDocument.load(cache_path)

        context = self._prepare_document_context(pdf_path, pages)

        layout_pages: list[LayoutPageResult] = []
        ocr_pages: list[OCRPageResult] = []
        tables: dict[str, TableBlockResult] = {}

        for batch_indices in self._chunked(
            context.page_indices, self.hardware.page_batch_size
        ):
            images, highres_images = self._load_page_images(
                context.pdf_path,
                batch_indices,
                include_highres=True,
            )

            batch_ocr_pages = self._parse_ocr_batch(
                batch_indices, images, highres_images
            )

            batch_layout_pages = self._parse_layout_batch(
                batch_indices,
                context.page_dims,
                images,
                ocr_pages=batch_ocr_pages,
            )

            batch_tables = self._parse_tables_batch(
                batch_layout_pages,
                images,
            )

            layout_pages.extend(batch_layout_pages)
            ocr_pages.extend(batch_ocr_pages)
            tables.update(batch_tables.tables)

            self._release_batch(images, highres_images)

        parsed_doc = self.merge_results(
            context.pdf_path,
            LayoutParseResult(pdf_path=str(context.pdf_path), pages=layout_pages),
            OCRParseResult(pdf_path=str(context.pdf_path), pages=ocr_pages),
            table_result=TableParseResult(
                pdf_path=str(context.pdf_path), tables=tables
            ),
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_doc.save(cache_path)
            logger.info("Saved Stage A result to %s", cache_path)

        return parsed_doc

    def merge_results(
        self,
        pdf_path: str | Path,
        layout_result: LayoutParseResult,
        ocr_result: OCRParseResult,
        table_result: TableParseResult | None = None,
    ) -> ParsedDocument:
        """Merge phase outputs into the final ParsedDocument."""

        pdf_path = self._resolve_pdf_path(pdf_path)
        if Path(layout_result.pdf_path) != pdf_path:
            raise ValueError("layout_result does not belong to the requested PDF")
        if Path(ocr_result.pdf_path) != pdf_path:
            raise ValueError("ocr_result does not belong to the requested PDF")

        table_map = table_result.tables if table_result else {}
        ocr_page_map = ocr_result.page_map()

        pages: list[PageData] = []
        for layout_page in layout_result.pages:
            page_ocr = ocr_page_map.get(layout_page.page_index)
            if page_ocr is None:
                raise ValueError(f"ocr_result is missing page {layout_page.page_index}")

            elements: list[ElementData] = []

            for block in layout_page.blocks:
                source_text = ""
                cells: list[CellData] = []

                if block.category == ElementCategory.BYPASS:
                    pass
                elif block.category == ElementCategory.TABLE:
                    table_block = table_map.get(block.block_id)
                    if table_block is None:
                        matching_lines = extract_text_for_region(
                            page_ocr.ocr_result, block.bbox_image
                        )
                        source_text = " ".join(line.text for line in matching_lines)
                    else:
                        crop_w, crop_h = table_block.crop_size
                        block_image_w = block.bbox_image[2] - block.bbox_image[0]
                        block_image_h = block.bbox_image[3] - block.bbox_image[1]
                        block_pdf_w = block.bbox_pdf[2] - block.bbox_pdf[0]
                        block_pdf_h = block.bbox_pdf[3] - block.bbox_pdf[1]

                        source_parts: list[str] = []
                        cells = []
                        cell_boxes_image: list[list[float]] = []

                        for cell_bbox in table_block.cells_bbox:
                            cell_bbox_image = offset_bbox(
                                convert_bbox(
                                    cell_bbox,
                                    crop_w,
                                    crop_h,
                                    block_image_w,
                                    block_image_h,
                                    pad_right=0,
                                    pad_bottom=0,
                                ),
                                block.bbox_image[0],
                                block.bbox_image[1],
                            )
                            cell_boxes_image.append(cell_bbox_image)
                            cell_bbox_pdf = clamp_bbox(
                                offset_bbox(
                                    convert_bbox(
                                        cell_bbox,
                                        crop_w,
                                        crop_h,
                                        block_pdf_w,
                                        block_pdf_h,
                                        pad_right=0,
                                        pad_bottom=0,
                                    ),
                                    block.bbox_pdf[0],
                                    block.bbox_pdf[1],
                                ),
                                layout_page.page_width,
                                layout_page.page_height,
                            )
                            matching_cell_lines = extract_text_for_region(
                                page_ocr.ocr_result, cell_bbox_image
                            )
                            cell_text = smart_join_text_lines(matching_cell_lines)
                            cells.append(
                                CellData(
                                    bbox_pdf=cell_bbox_pdf,
                                    source_text=cell_text,
                                    translated_text="",
                                )
                            )
                            source_parts.append(cell_text)

                        for orphan_line in self._collect_orphan_table_lines(
                            page_ocr.ocr_result,
                            block.bbox_image,
                            cell_boxes_image,
                        ):
                            orphan_text = clean_ocr_text(
                                getattr(orphan_line, "text", "")
                            )
                            if not orphan_text:
                                continue

                            orphan_bbox_line = get_line_bbox(orphan_line)

                            if orphan_bbox_line is None or is_degenerate(
                                orphan_bbox_line
                            ):
                                continue

                            orphan_bbox_pdf = clamp_bbox(
                                image_bbox_to_pdf(
                                    orphan_bbox_line,
                                    page_ocr.image_bbox,
                                    layout_page.page_width,
                                    layout_page.page_height,
                                    pad_right=0,
                                    pad_bottom=0,
                                ),
                                layout_page.page_width,
                                layout_page.page_height,
                            )
                            cells.append(
                                CellData(
                                    bbox_pdf=orphan_bbox_pdf,
                                    source_text=orphan_text,
                                    translated_text="",
                                )
                            )
                            source_parts.append(orphan_text)

                        source_text = " | ".join(source_parts)

                        if not cells:
                            matching_lines = extract_text_for_region(
                                page_ocr.ocr_result, block.bbox_image
                            )
                            source_text = smart_join_text_lines(matching_lines)
                else:
                    matching_lines = extract_text_for_region(
                        page_ocr.ocr_result, block.bbox_image
                    )
                    source_text = smart_join_text_lines(matching_lines)

                elements.append(
                    ElementData(
                        label=block.label,
                        category=block.category,
                        bbox_pdf=block.bbox_pdf,
                        source_text=source_text,
                        translated_text="",
                        cells=cells,
                    )
                )

            orphan_elements = self._collect_orphan_ocr_data(
                layout_page,
                page_ocr,
            )
            elements = self._insert_orphan_elements(elements, orphan_elements)
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
        ocr_pages: list[OCRPageResult] | None = None,
    ) -> list[LayoutPageResult]:
        layout_predictions = self.layout_model(
            images, batch_size=self.hardware.layout_batch_size, auto_unload=False
        )

        layout_pages: list[LayoutPageResult] = []
        ocr_page_map = (
            {page.page_index: page for page in ocr_pages}
            if ocr_pages is not None
            else {}
        )

        for seq, page_index in enumerate(batch_indices):
            page_width, page_height = page_dims[page_index]
            image_bbox = [0.0, 0.0, images[seq].size[0], images[seq].size[1]]
            layout_image_bbox = list(layout_predictions[seq].image_bbox)
            blocks: list[LayoutBlockResult] = []
            page_ocr = ocr_page_map.get(page_index)

            for position, block in enumerate(layout_predictions[seq].bboxes):
                block_bbox = getattr(block, "bbox", None)
                raw_bbox = list(
                    block_bbox
                    if block_bbox is not None
                    else polygon_to_bbox(block.polygon)
                )
                label = block.label
                category = SURYA_LABEL_MAP.get(label, DEFAULT_CATEGORY)
                bbox_pdf = clamp_bbox(
                    image_bbox_to_pdf(
                        raw_bbox,
                        layout_image_bbox,
                        page_width,
                        page_height,
                        pad_right=1.0,
                        pad_bottom=1.0,
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
                        pad_right=1.0,
                        pad_bottom=1.0,
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

            if page_ocr is not None:
                blocks = self._expand_layout_blocks(
                    blocks,
                    page_ocr,
                    image_bbox,
                    page_width,
                    page_height,
                )

                blocks = self._prune_overlapping_layout_blocks(blocks)

                blocks = self._refine_sparse_text_blocks(
                    blocks,
                    page_ocr,
                    image_bbox,
                    layout_image_bbox,
                    page_width,
                    page_height,
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
        ocr_predictions = self.ocr_model(
            images,
            highres_images=highres_images,
            math_mode=False,
            detection_batch_size=self.hardware.detection_batch_size,
            ocr_batch_size=self.hardware.ocr_batch_size,
            auto_unload=False,
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

        table_predictions = self.table_model(
            table_crops, batch_size=self.hardware.table_batch_size, auto_unload=False
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

    def _expand_layout_blocks(
        self,
        blocks: list[LayoutBlockResult],
        page_ocr: OCRPageResult,
        image_bbox: list[float],
        page_width: float,
        page_height: float,
        overlap_threshold: float = 0.3,
    ) -> list[LayoutBlockResult]:
        text_lines = getattr(page_ocr.ocr_result, "text_lines", None) or []
        if not text_lines:
            return blocks

        expanded_blocks: list[LayoutBlockResult] = []
        for block in blocks:
            if block.category == ElementCategory.BYPASS:
                expanded_blocks.append(block)
                continue

            matched_boxes: list[list[float]] = [block.bbox_image]
            for line in text_lines:
                line_bbox = get_line_bbox(line)
                if line_bbox is None or is_degenerate(line_bbox):
                    continue

                intersection = bbox_intersection(line_bbox, block.bbox_image)
                if intersection is None:
                    continue

                overlap_ratio = bbox_area(intersection) / max(1.0, bbox_area(line_bbox))

                if overlap_ratio >= overlap_threshold:
                    matched_boxes.append(line_bbox)

            merged_bbox = self._merge_bboxes(matched_boxes)
            if merged_bbox is None:
                expanded_blocks.append(block)
                continue

            bbox_image = clamp_bbox(merged_bbox, image_bbox[2], image_bbox[3])
            bbox_pdf = clamp_bbox(
                image_bbox_to_pdf(
                    bbox_image,
                    image_bbox,
                    page_width,
                    page_height,
                    pad_right=1.0,
                    pad_bottom=1.0,
                ),
                page_width,
                page_height,
            )
            expanded_blocks.append(
                LayoutBlockResult(
                    block_id=block.block_id,
                    page_index=block.page_index,
                    position=block.position,
                    label=block.label,
                    category=block.category,
                    bbox_layout=block.bbox_layout,
                    bbox_image=bbox_image,
                    bbox_pdf=bbox_pdf,
                )
            )

        return expanded_blocks

    def _prune_overlapping_layout_blocks(
        self,
        blocks: list[LayoutBlockResult],
        overlap_threshold: float = 0.7,
        containment_threshold: float = 0.9,
    ) -> list[LayoutBlockResult]:
        if len(blocks) < 2:
            return blocks

        kept_blocks: list[LayoutBlockResult] = []
        for block in sorted(
            blocks,
            key=lambda item: (-bbox_area(item.bbox_image), item.position),
        ):
            block_area = max(1.0, bbox_area(block.bbox_image))
            should_drop = False

            for kept in kept_blocks:
                intersection = bbox_intersection(block.bbox_image, kept.bbox_image)
                if intersection is None:
                    continue

                overlap_ratio = bbox_area(intersection) / block_area
                kept_area = bbox_area(kept.bbox_image)
                if overlap_ratio >= overlap_threshold and kept_area >= block_area:
                    should_drop = True
                    break

            if not should_drop:
                kept_blocks.append(block)

        filtered_blocks: list[LayoutBlockResult] = []
        for block in kept_blocks:
            block_area = max(1.0, bbox_area(block.bbox_image))
            covered_by_larger = False
            for other in kept_blocks:
                if other.block_id == block.block_id:
                    continue

                other_area = bbox_area(other.bbox_image)
                if other_area < block_area:
                    continue

                intersection = bbox_intersection(block.bbox_image, other.bbox_image)
                if intersection is None:
                    continue

                overlap_ratio = bbox_area(intersection) / block_area
                if overlap_ratio >= containment_threshold:
                    covered_by_larger = True
                    break

            if not covered_by_larger:
                filtered_blocks.append(block)

        return sorted(filtered_blocks, key=lambda item: item.position)

    def _refine_sparse_text_blocks(
        self,
        blocks: list[LayoutBlockResult],
        page_ocr: OCRPageResult,
        image_bbox: list[float],
        layout_image_bbox: list[float],
        page_width: float,
        page_height: float,
    ) -> list[LayoutBlockResult]:
        refined_blocks: list[LayoutBlockResult] = []

        for block in blocks:
            if block.category not in [
                ElementCategory.FLOWING_TEXT,
                ElementCategory.EQUATION,
            ]:
                refined_blocks.append(block)
                continue

            split_label = block.label
            split_category = block.category

            is_equation = True if block.category == ElementCategory.EQUATION else False

            is_sparse, text_lines = is_sparse_text_block(
                page_ocr.ocr_result, block.bbox_image, is_equation
            )

            if not is_sparse:
                refined_blocks.append(block)
                continue

            line_blocks = self._make_line_layout_blocks(
                block,
                text_lines,
                split_label,
                split_category,
                image_bbox,
                layout_image_bbox,
                page_width,
                page_height,
            )
            refined_blocks.extend(line_blocks or [block])

        return refined_blocks

    def _make_line_layout_blocks(
        self,
        block: LayoutBlockResult,
        text_lines: list[Any],
        label: str,
        category: ElementCategory,
        image_bbox: list[float],
        layout_image_bbox: list[float],
        page_width: float,
        page_height: float,
    ) -> list[LayoutBlockResult]:
        line_blocks: list[LayoutBlockResult] = []

        for index, line in enumerate(text_lines):
            line_bbox = get_line_bbox(line)
            if line_bbox is None or is_degenerate(line_bbox):
                continue

            bbox_image = clamp_bbox(line_bbox, image_bbox[2], image_bbox[3])
            bbox_pdf = clamp_bbox(
                image_bbox_to_pdf(
                    bbox_image,
                    image_bbox,
                    page_width,
                    page_height,
                    pad_right=2.5,
                    pad_bottom=1.5,
                ),
                page_width,
                page_height,
            )
            bbox_layout = clamp_bbox(
                convert_bbox(
                    bbox_image,
                    image_bbox[2],
                    image_bbox[3],
                    layout_image_bbox[2],
                    layout_image_bbox[3],
                    pad_right=2.5,
                    pad_bottom=1.5,
                ),
                layout_image_bbox[2],
                layout_image_bbox[3],
            )
            line_blocks.append(
                LayoutBlockResult(
                    block_id=f"{block.block_id}:line:{index}",
                    page_index=block.page_index,
                    position=block.position * 1000 + index,
                    label=label,
                    category=category,
                    bbox_layout=bbox_layout,
                    bbox_image=bbox_image,
                    bbox_pdf=bbox_pdf,
                )
            )

        return line_blocks

    def _create_orphan_element_from_line(
        self,
        line: Any,
        line_bbox: list[float],
        page_ocr: OCRPageResult,
        layout_page: LayoutPageResult,
    ) -> ElementData | None:
        line_text = clean_ocr_text(getattr(line, "text", ""))
        if not line_text or is_degenerate(line_bbox):
            return None

        orphan_bbox_pdf = clamp_bbox(
            image_bbox_to_pdf(
                line_bbox,
                page_ocr.image_bbox,
                layout_page.page_width,
                layout_page.page_height,
                pad_right=2.5,
                pad_bottom=1.5,
            ),
            layout_page.page_width,
            layout_page.page_height,
        )
        return ElementData(
            label="Text",
            category=DEFAULT_CATEGORY,
            bbox_pdf=orphan_bbox_pdf,
            source_text=line_text,
            translated_text="",
        )

    def _collect_orphan_table_lines(
        self,
        ocr_result: Any,
        table_bbox_image: list[float],
        cell_bboxes_image: list[list[float]],
        table_overlap_threshold: float = 0.5,
        cell_overlap_threshold: float = 0.3,
    ) -> list[Any]:
        orphan_lines: list[Any] = []
        for line in getattr(ocr_result, "text_lines", None) or []:
            line_bbox = get_line_bbox(line)
            if line_bbox is None or is_degenerate(line_bbox):
                continue

            intersection = bbox_intersection(line_bbox, table_bbox_image)
            if intersection is None:
                continue

            if (
                bbox_area(intersection) / max(1.0, bbox_area(line_bbox))
                < table_overlap_threshold
            ):
                continue

            overlaps_cell = False
            for cell_bbox in cell_bboxes_image:
                cell_intersection = bbox_intersection(line_bbox, cell_bbox)
                if cell_intersection is None:
                    continue
                if (
                    bbox_area(cell_intersection) / max(1.0, bbox_area(line_bbox))
                    >= cell_overlap_threshold
                ):
                    overlaps_cell = True
                    break

            if not overlaps_cell:
                orphan_lines.append(line)

        orphan_lines = sort_text_lines(orphan_lines)
        return orphan_lines

    def _collect_orphan_ocr_data(
        self,
        layout_page: LayoutPageResult,
        page_ocr: OCRPageResult,
        overlap_threshold: float = 0.5,
    ) -> list[ElementData]:
        text_lines = getattr(page_ocr.ocr_result, "text_lines", None)
        if not text_lines:
            return []

        orphan_elements: list[ElementData] = []
        layout_bboxes = [block.bbox_image for block in layout_page.blocks]

        for line in text_lines:
            line_bbox = get_line_bbox(line)
            if line_bbox is None or is_degenerate(line_bbox):
                continue

            line_area = bbox_area(line_bbox)
            if line_area <= 0:
                continue

            covered_regions: list[list[float]] = []
            for layout_bbox in layout_bboxes:
                intersection = bbox_intersection(line_bbox, layout_bbox)
                if intersection is not None:
                    covered_regions.append(intersection)

            covered_ratio = bbox_union_area(covered_regions) / line_area
            # This condition ensures that lines can't duplicate with function extract_text_from_region
            if covered_ratio >= overlap_threshold:
                continue

            orphan = self._create_orphan_element_from_line(
                line,
                line_bbox,
                page_ocr,
                layout_page,
            )

            if orphan is not None:
                orphan_elements.append(orphan)

        return orphan_elements

    def _insert_orphan_elements(
        self,
        elements: list[ElementData],
        orphan_elements: list[ElementData],
    ) -> list[ElementData]:
        """Insert orphan OCR elements without disturbing layout block order."""

        if not orphan_elements:
            return elements

        merged_elements = list(elements)
        for orphan in orphan_elements:
            insert_at = len(merged_elements)
            for index, element in enumerate(merged_elements):
                if self._bbox_precedes_in_reading_order(
                    orphan.bbox_pdf,
                    element.bbox_pdf,
                ):
                    insert_at = index
                    break
            merged_elements.insert(insert_at, orphan)

        return merged_elements

    def _bbox_precedes_in_reading_order(
        self,
        first_bbox: list[float],
        second_bbox: list[float],
        row_overlap_ratio: float = 0.35,
    ) -> bool:
        """Return True when the first bbox should be read before the second."""

        first_height = max(1.0, first_bbox[3] - first_bbox[1])
        second_height = max(1.0, second_bbox[3] - second_bbox[1])
        row_overlap = max(
            0.0,
            min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
        )

        same_row = row_overlap >= min(first_height, second_height) * row_overlap_ratio
        if same_row:
            return first_bbox[0] < second_bbox[0]

        first_center_y = (first_bbox[1] + first_bbox[3]) / 2.0
        second_center_y = (second_bbox[1] + second_bbox[3]) / 2.0
        return first_center_y < second_center_y

    def _merge_bboxes(self, boxes: list[list[float]]) -> list[float] | None:
        if not boxes:
            return None

        return [
            min(bbox[0] for bbox in boxes),
            min(bbox[1] for bbox in boxes),
            max(bbox[2] for bbox in boxes),
            max(bbox[3] for bbox in boxes),
        ]

    def _release_batch(self, *objects: Any) -> None:
        for obj in objects:
            if obj is None:
                continue
            del obj

        gc.collect()
        if self.hardware.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_pdf_path(self, pdf_path: str | Path) -> Path:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return pdf_path

    def _chunked(self, items: list[Any], size: int) -> Iterable[list[Any]]:
        for start in range(0, len(items), size):
            yield items[start : start + size]
