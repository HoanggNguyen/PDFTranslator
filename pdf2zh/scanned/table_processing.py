from __future__ import annotations

import logging
from PIL import Image

from pdf2zh.scanned.enums import ElementCategory
from pdf2zh.scanned.models import BlockInfo, CellData
from pdf2zh.scanned.utils.bbox import clamp_bbox, convert_bbox, offset_bbox
from pdf2zh.scanned.utils.image import crop_image_to_bbox
from pdf2zh.scanned.utils.ocr_text import extract_text_for_region

logger = logging.getLogger(__name__)


def process_table_batch(
    table_predictor,
    recognition_predictor,
    detection_predictor,
    batch_indices: list[int],
    page_dims: dict[int, tuple[float, float]],
    images: list[Image.Image],
    highres_images: list[Image.Image],
    pages_block_infos: list[list[BlockInfo]],
) -> None:
    for i, idx in enumerate(batch_indices):
        pw, ph = page_dims[idx]
        for info in pages_block_infos[i]:
            if info.category != ElementCategory.TABLE:
                continue

            source_text, cells = process_table(
                table_predictor=table_predictor,
                recognition_predictor=recognition_predictor,
                detection_predictor=detection_predictor,
                image=images[i],
                highres_image=highres_images[i],
                pdf_bbox=info.pdf_bbox,
                page_width=pw,
                page_height=ph,
            )
            info.source_text = source_text
            info.cells = cells


def process_table(
    table_predictor,
    recognition_predictor,
    detection_predictor,
    image: Image.Image,
    highres_image: Image.Image,
    pdf_bbox: list[float],
    page_width: float,
    page_height: float,
) -> tuple[str, list[CellData]]:
    table_crop = crop_image_to_bbox(image, pdf_bbox, page_width, page_height)

    try:
        table_results = table_predictor([table_crop])
        if not table_results or not getattr(table_results[0], "cells", None):
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

    try:
        hr_crop = crop_image_to_bbox(highres_image, pdf_bbox, page_width, page_height)
        ocr_results = recognition_predictor(
            [table_crop],
            det_predictor=detection_predictor,
            highres_images=[hr_crop],
        )
        ocr_result = ocr_results[0] if ocr_results else None
        del hr_crop, ocr_results
    except Exception as e:
        logger.error(f"Table OCR failed: {e}")
        ocr_result = None

    cells: list[CellData] = []
    text_parts: list[str] = []

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
            row_id=getattr(cell, "row_id", 0),
            col_id=getattr(cell, "col_id", 0),
            source_text=cell_text,
            translated_text="",
        ))

        if cell_text:
            text_parts.append(cell_text)

    return " | ".join(text_parts), cells
