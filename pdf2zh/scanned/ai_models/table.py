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
    Models are loaded lazily upon first inference call.
    """

    model_name = "SuryaTable"

    def __init__(self) -> None:
        """Initialize empty state to defer model loading."""
        super().__init__()

    def load_model(self) -> None:
        """Load Surya model into VRAM."""
        logger.info("Initializing %s...", self.model_name)

        from surya.table_rec import TableRecPredictor

        self.model = TableRecPredictor()
        logger.info("Loaded TableRecPredictor")

    def prepare(
        self, images: list[Image.Image], *args: Any, **kwargs: Any
    ) -> list[Image.Image]:
        """Preprocess a batch of cropped table images."""
        # Surya models accept raw PIL images directly
        return images

    def predict(
        self,
        prepared_inputs: list[Image.Image],
        batch_size: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Recognize table structure for a batch of prepared table images.
        """
        try:
            # self.model is guaranteed to be loaded by the Base class
            raw_results = self.model(
                prepared_inputs,
                batch_size=batch_size,
            )
            return raw_results
        except Exception:
            logger.exception(
                "Table recognition failed for batch of %d crops — returning nulls.",
                len(prepared_inputs),
            )
            return [None] * len(prepared_inputs)

    def postprocess(
        self, raw_results: list[Any], *args: Any, **kwargs: Any
    ) -> list[list[list[float]]]:
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
    Models are loaded lazily upon first inference call.
    """

    model_name = "PaddleCellTableModule"

    def __init__(self) -> None:
        """Initialize empty state to defer model loading."""
        super().__init__()

    def load_model(self) -> None:
        """Load Paddle model into memory/VRAM."""
        logger.info("Initializing %s...", self.model_name)

        from paddleocr import TableCellsDetection

        self.model = TableCellsDetection(model_name="RT-DETR-L_wireless_table_cell_det")
        logger.info("Loaded TableCellsDetection")

    def prepare(
        self, images: list[Image.Image], *args: Any, **kwargs: Any
    ) -> list[np.ndarray]:
        """
        Convert PIL images to numpy arrays to satisfy PaddleOCR requirements.
        """
        return [np.array(img.convert("RGB")) for img in images]

    def predict(
        self,
        prepared_inputs: list[np.ndarray],
        batch_size: int | None = None,
        threshold: float = 0.3,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Recognize cell detection for a batch of prepared table images.
        """
        try:
            raw_results = self.model.predict(
                prepared_inputs,
                threshold=threshold,
                batch_size=batch_size,
            )
            return raw_results
        except Exception:
            logger.exception(
                "Paddle table cell detection failed for batch of %d crops — returning nulls.",
                len(prepared_inputs),
            )
            return [None] * len(prepared_inputs)

    def postprocess(
        self, raw_results: list[Any], *args: Any, **kwargs: Any
    ) -> list[list[list[float]]]:
        """Normalize Paddle output into simple bbox lists."""
        batch_boxes = []
        for result in raw_results:
            if result is None:
                batch_boxes.append([])
                continue

            # Check both 'boxes' and 'coordinate' attributes
            raw_cells = result.get("boxes", [])

            boxes = []
            for cell in raw_cells:
                coords = cell.get("coordinate")
                if coords:
                    boxes.append([float(x) for x in coords])

            batch_boxes.append(boxes)
        return batch_boxes
