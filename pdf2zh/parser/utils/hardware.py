"""Hardware-awaresettings configuration."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu", "auto"]

_DEFAULT_BATCHES = {
    "cuda": {
        "layout": 32,
        "detection": 32,
        "recognition": 128,
        "table": 256,
    },
    "mps": {
        "layout": 4,
        "detection": 8,
        "recognition": 64,
        "table": 64,
    },
    "cpu": {
        "layout": 4,
        "detection": 8,
        "recognition": 32,
        "table": 32,
    },
}


@dataclass(slots=True)
class HardwareConfig:
    """Resolved hardware configuration used to drive Surya settings."""

    device: str
    page_batch_size: int
    layout_batch_size: int
    detection_batch_size: int
    ocr_batch_size: int
    table_batch_size: int


# Backward-compatible alias for older imports.
HardwareProfile = HardwareConfig


def _detect_device() -> str:
    """Detect the best torch device available for Surya."""

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA device detected")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.info("MPS device detected")
            return "mps"
    except ImportError:
        logger.warning("PyTorch is unavailable, falling back to CPU")

    logger.info("Using CPU device")
    return "cpu"


def set_torch_device_env(device: str) -> None:
    """Set the torch device for downstream Surya imports."""

    os.environ["TORCH_DEVICE"] = device


def configure_settings(
    device: DeviceType = "auto",
    page_batch_size: int | None = None,
    layout_batch_size: int | None = None,
    detection_batch_size: int | None = None,
    ocr_batch_size: int | None = None,
    table_batch_size: int | None = None,
) -> HardwareConfig:
    """Resolve and apply settings using local hardware heuristics."""

    resolved_device = _detect_device() if device == "auto" else device

    resolved_layout_batch = (
        layout_batch_size
        if layout_batch_size
        else _DEFAULT_BATCHES[resolved_device]["layout"]
    )
    resolved_detection_batch = (
        detection_batch_size
        if detection_batch_size
        else _DEFAULT_BATCHES[resolved_device]["detection"]
    )
    resolved_table_batch = (
        table_batch_size
        if table_batch_size
        else _DEFAULT_BATCHES[resolved_device]["table"]
    )
    resolved_ocr_batch = (
        ocr_batch_size
        if ocr_batch_size
        else _DEFAULT_BATCHES[resolved_device]["recognition"]
    )

    resolved_page_batch = (
        page_batch_size
        if page_batch_size
        else min(resolved_layout_batch, resolved_detection_batch)
    )

    config = HardwareConfig(
        device=resolved_device,
        page_batch_size=resolved_page_batch,
        layout_batch_size=resolved_layout_batch,
        detection_batch_size=resolved_detection_batch,
        ocr_batch_size=resolved_ocr_batch,
        table_batch_size=resolved_table_batch,
    )

    logger.info(
        "Configured settings: device=%s page=%s layout=%s detection=%s "
        "ocr=%s table=%s",
        config.device,
        config.page_batch_size,
        config.layout_batch_size,
        config.detection_batch_size,
        config.ocr_batch_size,
        config.table_batch_size,
    )
    return config


def resolve_hardware(
    device: DeviceType = "auto",
    ocr_batch_size: int | None = None,
    **kwargs,
) -> HardwareConfig:
    """Backward-compatible wrapper around ``configure_settings``."""

    return configure_settings(
        device=device,
        ocr_batch_size=ocr_batch_size,
        **kwargs,
    )


def main() -> None:
    """Print a resolved Surya hardware config for local tuning."""

    parser = argparse.ArgumentParser(description="Inspect resolved settings")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--page-batch-size", type=int, default=None)
    parser.add_argument("--layout-batch-size", type=int, default=None)
    parser.add_argument("--detection-batch-size", type=int, default=None)
    parser.add_argument("--ocr-batch-size", type=int, default=None)
    parser.add_argument("--table-batch-size", type=int, default=None)
    args = parser.parse_args()

    config = configure_settings(
        device=args.device,
        page_batch_size=args.page_batch_size,
        layout_batch_size=args.layout_batch_size,
        detection_batch_size=args.detection_batch_size,
        ocr_batch_size=args.ocr_batch_size,
        table_batch_size=args.table_batch_size,
    )
    print(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    main()
