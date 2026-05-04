"""Table models: table structure and cell recognition."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from pdf2zh.scanned.ai_models.base import BaseImageToTextModel

logger = logging.getLogger(__name__)


class SuryaTableModel(BaseImageToTextModel):
    """
    Wraps Surya's TableRecPredictor.

    Identifies row/column structure and cell bounding boxes within a cropped
    table image. Text extraction is handled separately.
    """

    model_name = "SuryaTable"

    def __init__(self, hardware: Any) -> None:
        """Initialize hardware config and load Surya model immediately."""
        super().__init__(hardware)
        logger.info("Initializing %s...", self.model_name)

        from surya.table_rec import TableRecPredictor

        self.model = TableRecPredictor()
        logger.info("Loaded TableRecPredictor")

    def prepare(self, images: list[Image.Image]) -> list[Image.Image]:
        """Preprocess a batch of cropped table images."""
        # Surya models accept raw PIL images directly
        return images

    def predict(self, images: list[Image.Image]) -> list[Any]:
        """
        Recognize table structure for a batch of prepared table images.
        """
        try:
            # self.model is the callable TableRecPredictor instantiated in __init__
            raw_results = self.model(
                images,
                batch_size=self.hardware.table_batch_size,
            )
            return self.postprocess(raw_results)
        except Exception:
            logger.exception(
                "Table recognition failed for batch of %d crops — returning nulls.",
                len(images),
            )
            return [None] * len(images)

    def postprocess(self, raw_results: list[Any]) -> list[list[list[float]]]:
        """Convert objects into a simple list of bounding boxes."""
        batch_boxes = []
        for result in raw_results:
            if result is None:
                batch_boxes.append([])
                continue

            # Extract only bboxes and ensure float type
            boxes = [
                [float(x) for x in cell.bbox] for cell in getattr(result, "cells", [])
            ]
            batch_boxes.append(boxes)
        return batch_boxes


class PaddleCellTableModule(BaseImageToTextModel):
    """
    Wraps Paddle's Table Cell Detection Module.
    """

    model_name = "PaddleCellTableModule"

    def __init__(self, hardware: Any) -> None:
        """Initialize hardware config and load Paddle model immediately."""
        super().__init__(hardware)
        logger.info("Initializing %s...", self.model_name)

        from paddleocr import TableCellsDetection

        self.model = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det")
        logger.info("Loaded TableCellsDetection")

    def prepare(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Convert PIL images to numpy arrays to satisfy PaddleOCR requirements.
        """
        # Chuyển đổi từng ảnh PIL trong list sang định dạng NumPy ndarray
        return [np.array(img.convert("RGB")) for img in images]

    def predict(self, images: list[Image.Image], threshold: float = 0.3) -> list[Any]:
        """
        Recognize cell detection for a batch of prepared table images.
        """
        try:
            images = self.prepare(images)

            raw_results = self.model.predict(
                images,
                threshold=threshold,
                batch_size=self.hardware.table_batch_size,
            )

            return self.postprocess(raw_results)
        except Exception:
            logger.exception(
                "Paddle table cell detection failed for batch of %d crops — returning nulls.",
                len(images),
            )
            return [None] * len(images)

    def postprocess(self, raw_results: list[Any]) -> list[list[list[float]]]:
        """Normalize Paddle output into simple bbox lists."""
        batch_boxes = []
        for result in raw_results:
            if result is None:
                batch_boxes.append([])
                continue

            # Check both 'boxes' and 'coordinate' attributes
            raw_cells = result["boxes"]

            boxes = []
            for cell in raw_cells:
                coords = cell["coordinate"]
                if coords:
                    boxes.append([float(x) for x in coords])

            batch_boxes.append(boxes)
        return batch_boxes
