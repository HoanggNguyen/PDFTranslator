"""Table models: table structure and cell recognition."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from pdf2zh.parser.ai_models.base import BaseImageToTextModel
from pdf2zh.parser.utils.bbox import bbox_area, bbox_intersection

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

            batch_boxes.append(self._prune_nested_cell_boxes(boxes))
        return batch_boxes

    def _prune_nested_cell_boxes(
        self,
        boxes: list[list[float]],
        containment_threshold: float = 0.8,
    ) -> list[list[float]]:
        if len(boxes) < 2:
            return boxes

        kept_boxes: list[list[float]] = []
        sorted_boxes = sorted(boxes, key=bbox_area)

        for box in sorted_boxes:
            box_area = max(1.0, bbox_area(box))
            is_duplicate = False
            for kept in kept_boxes:
                intersection = bbox_intersection(box, kept)
                if intersection is None:
                    continue

                overlap_ratio = bbox_area(intersection) / box_area
                if overlap_ratio >= containment_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_boxes.append(box)

        filtered_boxes: list[list[float]] = []
        for box in kept_boxes:
            box_area = max(1.0, bbox_area(box))
            contains_smaller_box = False
            for other in kept_boxes:
                if other is box:
                    continue

                other_area = bbox_area(other)
                if other_area >= box_area:
                    continue

                intersection = bbox_intersection(box, other)
                if intersection is None:
                    continue

                overlap_ratio = bbox_area(intersection) / max(1.0, other_area)
                if overlap_ratio >= containment_threshold:
                    contains_smaller_box = True
                    break

            if not contains_smaller_box:
                filtered_boxes.append(box)

        return filtered_boxes
