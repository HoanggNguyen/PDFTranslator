"""Base class for all Surya AI models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Shared interface and common logic for all Surya-backed AI models.

    Subclasses must implement:
        - ``model_name``  – human-readable name used in log messages.
        - ``_load()``     – instantiate and return the raw Surya predictor.
        - ``predict()``   – run inference and return structured results.

    The predictor is loaded lazily on first access via :py:attr:`predictor`
    and kept alive for the lifetime of the instance (no GPU swap-out).
    """

    def __init__(self, hardware: Any) -> None:
        """
        Args:
            hardware: A ``HardwareConfig``-like object that exposes at least
                      ``device`` (str) and batch-size attributes consumed by
                      each subclass.
        """
        self.hardware = hardware
        self._predictor: Any = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name, e.g. ``'OCR'`` or ``'Layout'``."""

    @abstractmethod
    def _load(self) -> Any:
        """Instantiate and return the raw Surya predictor object."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run inference.

        Args and return type are defined by each concrete subclass.
        """

    @property
    def predictor(self) -> Any:
        """Return the predictor, loading it on first access."""
        if self._predictor is None:
            logger.info("Loading %s model…", self.model_name)
            self._predictor = self._load()
            logger.info("%s model loaded.", self.model_name)
        return self._predictor

    @property
    def is_loaded(self) -> bool:
        """``True`` after the predictor has been instantiated."""
        return self._predictor is not None
