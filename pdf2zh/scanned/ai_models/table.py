"""Table model: table structure recognition."""

from __future__ import annotations

import logging
from typing import Any

from base import BaseModel

logger = logging.getLogger(__name__)


class SuryaTableModel(BaseModel):
    """
    Wraps Surya's ``TableRecPredictor``.

    Identifies row/column structure and cell bounding boxes within a cropped
    table image.  Text extraction from cells is handled separately (by the
    OCR model) and is not the responsibility of this class.
    """

    model_name = "SuryaTable"

    def _load(self) -> Any:
        from surya.table_rec import TableRecPredictor

        predictor = TableRecPredictor()
        logger.info("Loaded TableRecPredictor")
        return predictor

    def predict(self, table_crops) -> list[Any]:
        """
        Recognise table structure for a batch of cropped table images.

        Args:
            table_crops: List of PIL images, each containing exactly one table.

        Returns:
            List of Surya table result objects (one per crop), or a list of
            ``None`` values when the batch fails (caller handles fallback).
        """
        try:
            return self.predictor(
                table_crops,
                batch_size=self.hardware.table_batch_size,
            )
        except Exception:
            logger.exception(
                "Table recognition failed for batch of %d crops — returning nulls.",
                len(table_crops),
            )
            return [None] * len(table_crops)
