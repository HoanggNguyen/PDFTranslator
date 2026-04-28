"""OCR model: text detection + recognition"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseModel

logger = logging.getLogger(__name__)


class SuryaOCRModel(BaseModel):
    """
    Wraps Surya's ``DetectionPredictor`` + ``RecognitionPredictor``.

    Both predictors share the same ``FoundationPredictor`` backbone which is
    loaded once and reused, mirroring the original lazy-init pattern.
    """

    model_name = "SuryaOCR"

    def __init__(self, hardware: Any) -> None:
        super().__init__(hardware)
        self._foundation_predictor: Any = None
        self._detection_predictor: Any = None
        self._recognition_predictor: Any = None

    @property
    def _foundation(self) -> Any:
        if self._foundation_predictor is None:
            from surya.foundation import FoundationPredictor

            self._foundation_predictor = FoundationPredictor()
            logger.info("Loaded FoundationPredictor (OCR backbone)")
        return self._foundation_predictor

    @property
    def _detection(self) -> Any:
        if self._detection_predictor is None:
            from surya.detection import DetectionPredictor

            self._detection_predictor = DetectionPredictor()
            logger.info("Loaded DetectionPredictor")
        return self._detection_predictor

    @property
    def _recognition(self) -> Any:
        if self._recognition_predictor is None:
            from surya.recognition import RecognitionPredictor

            self._recognition_predictor = RecognitionPredictor(self._foundation)
            logger.info("Loaded RecognitionPredictor")
        return self._recognition_predictor

    def _load(self) -> Any:
        # Trigger lazy init of all three sub-predictors.
        _ = self._foundation
        _ = self._detection
        _ = self._recognition
        return self  # predictor IS this instance

    @property
    def predictor(self) -> "SuryaOCRModel":
        """Returns self after ensuring all sub-predictors are loaded."""
        if not self.is_loaded:
            self._load()
            self._predictor = True  # sentinel — sub-predictors are the real handles
        return self

    @property
    def is_loaded(self) -> bool:
        return self._recognition_predictor is not None

    def predict(
        self,
        images,
        *,
        highres_images=None,
        math_mode: bool = False,
        task_names=None,
        bboxes=None,
    ) -> list[Any]:
        """
        Run full-page OCR (detection → recognition).

        Args:
            images: List of PIL images at standard DPI.
            highres_images: Optional list of the same pages at high DPI.
            math_mode: Enable math/LaTeX recognition mode.
            sort_lines: Sort detected lines by reading order.
            task_names: Optional per-image task overrides (Surya TaskNames).
            bboxes: Optional pre-computed bounding boxes per image.

        Returns:
            List of Surya OCR result objects, one per input image.
        """
        if not math_mode:
            logger.info("Running OCR with detection + recognition")
            kwargs: dict = dict(
                det_predictor=self._detection,
                detection_batch_size=self.hardware.detection_batch_size,
                recognition_batch_size=self.hardware.ocr_batch_size,
                highres_images=highres_images,
                math_mode=math_mode,
            )
        else:  # Set math_mode to True, use Latex
            logger.info("Running OCR in math mode (LaTex recognition)")
            kwargs: dict = dict(
                recognition_batch_size=self.hardware.equation_batch_size,
                math_mode=math_mode,
            )
        if task_names is not None:
            kwargs["task_names"] = task_names
        if bboxes is not None:
            kwargs["bboxes"] = bboxes

        return self._recognition(images, **kwargs)
