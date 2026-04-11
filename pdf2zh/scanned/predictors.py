from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SuryaPredictors:
    """Lazy Surya predictor loader with eager preload support."""

    def __init__(self) -> None:
        self._foundation_predictor = None
        self._layout_foundation_predictor = None
        self._detection_predictor = None
        self._layout_predictor = None
        self._recognition_predictor = None
        self._table_predictor = None

    @property
    def foundation_predictor(self):
        if self._foundation_predictor is None:
            from surya.foundation import FoundationPredictor

            self._foundation_predictor = FoundationPredictor()
            logger.info("Loaded FoundationPredictor (OCR)")
        return self._foundation_predictor

    @property
    def layout_foundation_predictor(self):
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
        if self._detection_predictor is None:
            from surya.detection import DetectionPredictor

            self._detection_predictor = DetectionPredictor()
            logger.info("Loaded DetectionPredictor")
        return self._detection_predictor

    @property
    def layout_predictor(self):
        if self._layout_predictor is None:
            from surya.layout import LayoutPredictor

            self._layout_predictor = LayoutPredictor(self.layout_foundation_predictor)
            logger.info("Loaded LayoutPredictor")
        return self._layout_predictor

    @property
    def recognition_predictor(self):
        if self._recognition_predictor is None:
            from surya.recognition import RecognitionPredictor

            self._recognition_predictor = RecognitionPredictor(self.foundation_predictor)
            logger.info("Loaded RecognitionPredictor")
        return self._recognition_predictor

    @property
    def table_predictor(self):
        if self._table_predictor is None:
            from surya.table_rec import TableRecPredictor

            self._table_predictor = TableRecPredictor()
            logger.info("Loaded TableRecPredictor")
        return self._table_predictor

    def preload_predictors(self) -> None:
        logger.info("Preloading Surya predictors before parsing")
        _ = self.foundation_predictor
        _ = self.layout_foundation_predictor
        _ = self.detection_predictor
        _ = self.layout_predictor
        _ = self.recognition_predictor
        _ = self.table_predictor
