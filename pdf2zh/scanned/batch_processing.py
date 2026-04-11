from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path

import torch
import gc
from PIL import Image

from pdf2zh.scanned.enums import ElementCategory, DEFAULT_CATEGORY, SURYA_LABEL_MAP
from pdf2zh.scanned.models import BlockInfo
from pdf2zh.scanned.utils.bbox import clamp_bbox, image_bbox_to_pdf, is_degenerate
from pdf2zh.scanned.utils.hardware import get_gpu_memory_mb
from pdf2zh.scanned.utils.image import crop_image_to_bbox
from pdf2zh.scanned.utils.ocr_text import collect_ocr_text

logger = logging.getLogger(__name__)


def resolve_batch_sizes(profile) -> tuple[int, int]:
    batch_size = profile.batch_size
    ocr_batch_size = profile.ocr_batch_size
    free_mb = get_gpu_memory_mb()

    if profile.device == "cuda" and free_mb is not None:
        if free_mb < 6000:
            batch_size = min(batch_size, 8)
            ocr_batch_size = min(ocr_batch_size, 8)
        elif free_mb < 10000:
            batch_size = min(batch_size, 12)
            ocr_batch_size = min(ocr_batch_size, 16)

    return batch_size, ocr_batch_size


def load_batch_images(
    pdf_path: Path,
    batch_indices: list[int],
) -> tuple[list[Image.Image], list[Image.Image]]:
    from surya.input.load import load_from_file
    from surya.settings import settings

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        std_future = executor.submit(
            load_from_file,
            str(pdf_path),
            page_range=batch_indices,
        )
        hr_future = executor.submit(
            load_from_file,
            str(pdf_path),
            dpi=settings.IMAGE_DPI_HIGHRES,
            page_range=batch_indices,
        )

        (images, _), (highres_images, _) = std_future.result(), hr_future.result()

    return images, highres_images


def collect_batch_block_infos(
    batch_indices: list[int],
    page_dims: dict[int, tuple[float, float]],
    images: list[Image.Image],
    highres_images: list[Image.Image],
    layout_results: list,
) -> tuple[list[list[BlockInfo]], list[Image.Image], list[Image.Image], list[BlockInfo]]:
    pages_block_infos: list[list[BlockInfo]] = []
    all_ocr_std: list[Image.Image] = []
    all_ocr_hr: list[Image.Image] = []
    all_ocr_targets: list[BlockInfo] = []

    for i, idx in enumerate(batch_indices):
        pw, ph = page_dims[idx]
        layout_image_bbox = layout_results[i].image_bbox
        page_block_infos: list[BlockInfo] = []

        for block in layout_results[i].bboxes:
            label = block.label
            category = SURYA_LABEL_MAP.get(label, DEFAULT_CATEGORY)

            pdf_bbox = image_bbox_to_pdf(
                block.bbox,
                layout_image_bbox,
                pw,
                ph,
            )
            pdf_bbox = clamp_bbox(pdf_bbox, pw, ph)

            if is_degenerate(pdf_bbox):
                logger.debug("Skipping degenerate bbox for %s", label)
                continue

            info = BlockInfo(label, category, pdf_bbox, page_seq=i)
            page_block_infos.append(info)

            if category not in (ElementCategory.BYPASS, ElementCategory.TABLE):
                info.needs_ocr = True
                all_ocr_std.append(crop_image_to_bbox(images[i], pdf_bbox, pw, ph))
                all_ocr_hr.append(crop_image_to_bbox(highres_images[i], pdf_bbox, pw, ph))
                all_ocr_targets.append(info)

        pages_block_infos.append(page_block_infos)

    return pages_block_infos, all_ocr_std, all_ocr_hr, all_ocr_targets


def run_batch_ocr(
    recognition_predictor,
    detection_predictor,
    all_ocr_std: list[Image.Image],
    all_ocr_hr: list[Image.Image],
    all_ocr_targets: list[BlockInfo],
    ocr_batch_size: int,
    device: str,
) -> None:
    logger.info("Batch OCR: %d crops total", len(all_ocr_std))

    for start in range(0, len(all_ocr_std), ocr_batch_size):
        sub_std = all_ocr_std[start:start + ocr_batch_size]
        sub_hr = all_ocr_hr[start:start + ocr_batch_size]
        sub_targets = all_ocr_targets[start:start + len(sub_std)]

        sub_results = recognition_predictor(
            sub_std,
            det_predictor=detection_predictor,
            highres_images=sub_hr,
        )
        for result, target in zip(sub_results, sub_targets):
            target.source_text = collect_ocr_text(result)

        if device == "cuda":
            torch.cuda.empty_cache()


def cleanup_batch(
    images: list[Image.Image],
    highres_images: list[Image.Image],
    layout_results: list,
    pages_block_infos: list[list[BlockInfo]],
    all_ocr_std: list[Image.Image] | None = None,
    all_ocr_hr: list[Image.Image] | None = None,
    all_ocr_targets: list[BlockInfo] | None = None,
    device: str = "cpu",
) -> None:
    del images, highres_images, layout_results, pages_block_infos
    if all_ocr_std is not None:
        del all_ocr_std
    if all_ocr_hr is not None:
        del all_ocr_hr
    if all_ocr_targets is not None:
        del all_ocr_targets

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
