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

_PHASE_MEMORY_MB = {
    "layout": 220,
    "detection": 440,
    "recognition": 40,
    "table": 150,
    "equation": 40,
}

_DEFAULT_BATCHES = {
    "cuda": {
        "layout": 32,
        "detection": 32,
        "recognition": 128,
        "table": 32,
        "equation": 256,
    },
    "mps": {
        "layout": 4,
        "detection": 8,
        "recognition": 64,
        "table": 8,
        "equation": 64,
    },
    "cpu": {
        "layout": 4,
        "detection": 8,
        "recognition": 32,
        "table": 8,
        "equation": 32,
    },
}


@dataclass(slots=True)
class HardwareConfig:
    """Resolved hardware configuration used to drive Surya settings."""

    device: str
    free_vram_mb: int | None
    usable_vram_mb: int | None
    page_batch_size: int
    layout_batch_size: int
    detection_batch_size: int
    ocr_batch_size: int
    table_batch_size: int
    gpu_memory_utilization: float


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


def get_gpu_memory_mb(device: str | None = None) -> tuple[int, int] | tuple[None, None]:
    """Return available GPU memory in MB when it can be detected."""

    try:
        import torch

        device = device or _detect_device()
        if device == "cuda" and torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes / (1024 * 1024)), int(_total_bytes / (1024 * 1024))
        if device == "mps" and torch.backends.mps.is_available():
            return None, None
    except Exception:
        logger.debug("Could not query GPU memory", exc_info=True)

    return None, None


def _estimate_phase_batch(
    device: str,
    phase: str,
    usable_vram_mb: int | None,
    override: int | None,
) -> int:
    """Estimate a batch size for one phase."""

    if override is not None:
        return max(1, override)

    default_value = _DEFAULT_BATCHES[device][phase]
    if device != "cuda" or usable_vram_mb is None:
        return default_value

    memory_per_item = _PHASE_MEMORY_MB[phase]
    estimated = max(1, usable_vram_mb // memory_per_item)
    return estimated


def configure_settings(
    device: DeviceType = "auto",
    batch_size: int | None = None,
    page_batch_size: int | None = None,
    layout_batch_size: int | None = None,
    detection_batch_size: int | None = None,
    ocr_batch_size: int | None = None,
    table_batch_size: int | None = None,
    gpu_memory_utilization: float = 0.9,
) -> HardwareConfig:
    """Resolve and apply settings using local hardware heuristics."""

    resolved_device = _detect_device() if device == "auto" else device
    free_vram_mb, total_vram_mb = get_gpu_memory_mb(resolved_device)
    usable_vram_mb = (
        int(free_vram_mb * gpu_memory_utilization) if free_vram_mb is not None else None
    )

    resolved_layout_batch = min(
        _estimate_phase_batch(
            resolved_device, "layout", usable_vram_mb, layout_batch_size
        ),
        _DEFAULT_BATCHES[resolved_device]["layout"],
    )

    resolved_detection_batch = min(
        _estimate_phase_batch(
            resolved_device, "detection", usable_vram_mb, detection_batch_size
        ),
        _DEFAULT_BATCHES[resolved_device]["detection"],
    )

    resolved_table_batch = min(
        _estimate_phase_batch(
            resolved_device, "table", usable_vram_mb, table_batch_size
        ),
        _DEFAULT_BATCHES[resolved_device]["table"],
    )

    THRESHOLD_20GB_MB = 20 * 1024

    resolved_ocr_batch = _estimate_phase_batch(
        resolved_device, "recognition", usable_vram_mb, ocr_batch_size
    )

    if total_vram_mb:
        if total_vram_mb <= THRESHOLD_20GB_MB:
            resolved_ocr_batch = min(
                resolved_ocr_batch, _DEFAULT_BATCHES[resolved_device]["recognition"]
            )
        else:
            resolved_ocr_batch = 512

    resolved_page_batch = page_batch_size
    if resolved_page_batch is None:
        resolved_page_batch = batch_size
    if resolved_page_batch is None:
        resolved_page_batch = min(resolved_layout_batch, resolved_detection_batch)
    resolved_page_batch = max(1, resolved_page_batch)

    config = HardwareConfig(
        device=resolved_device,
        free_vram_mb=free_vram_mb,
        usable_vram_mb=usable_vram_mb,
        page_batch_size=resolved_page_batch,
        layout_batch_size=resolved_layout_batch,
        detection_batch_size=resolved_detection_batch,
        ocr_batch_size=resolved_ocr_batch,
        table_batch_size=resolved_table_batch,
        gpu_memory_utilization=gpu_memory_utilization,
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
    batch_size: int | None = None,
    ocr_batch_size: int | None = None,
    **kwargs,
) -> HardwareConfig:
    """Backward-compatible wrapper around ``configure_settings``."""

    return configure_settings(
        device=device,
        batch_size=batch_size,
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    config = configure_settings(
        device=args.device,
        page_batch_size=args.page_batch_size,
        layout_batch_size=args.layout_batch_size,
        detection_batch_size=args.detection_batch_size,
        ocr_batch_size=args.ocr_batch_size,
        table_batch_size=args.table_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    main()
