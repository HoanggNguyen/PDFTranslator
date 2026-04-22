"""Hardware-aware Surya settings configuration.

This module configures Surya by updating ``surya.settings.settings`` directly,
instead of maintaining a parallel hardware profile layer in ``pdf2zh``.
"""

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
        "detection": 36,
        "recognition": 256,
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
class SuryaHardwareConfig:
    """Resolved hardware configuration used to drive Surya settings."""

    device: str
    free_vram_mb: int | None
    usable_vram_mb: int | None
    page_batch_size: int
    layout_batch_size: int
    detection_batch_size: int
    ocr_batch_size: int
    table_batch_size: int
    equation_batch_size: int
    allow_parallel_phases: bool
    parallel_workers: int
    enable_latex: bool = False


# Backward-compatible alias for older imports.
HardwareProfile = SuryaHardwareConfig


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


def get_gpu_memory_mb(device: str | None = None) -> int | None:
    """Return available GPU memory in MB when it can be detected."""

    try:
        import torch

        device = device or _detect_device()
        if device == "cuda" and torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes / (1024 * 1024))
        if device == "mps" and torch.backends.mps.is_available():
            return None
    except Exception:
        logger.debug("Could not query GPU memory", exc_info=True)

    return None


def _estimate_phase_batch(
    device: str,
    phase: str,
    usable_vram_mb: int | None,
    override: int | None,
) -> int:
    """Estimate a Surya batch size for one phase."""

    if override is not None:
        return max(1, override)

    default_value = _DEFAULT_BATCHES[device][phase]
    if device != "cuda" or usable_vram_mb is None:
        return default_value

    memory_per_item = _PHASE_MEMORY_MB[phase]
    estimated = max(1, usable_vram_mb // memory_per_item)
    return min(default_value, estimated)


def _patch_predictor_batch_sizes(config: SuryaHardwareConfig) -> None:
    """Update class-level batch size defaults if Surya modules are imported."""

    try:
        from surya.detection import DetectionPredictor

        DetectionPredictor.batch_size = config.detection_batch_size
    except Exception:
        pass

    try:
        from surya.layout import LayoutPredictor

        LayoutPredictor.batch_size = config.layout_batch_size
    except Exception:
        pass

    try:
        from surya.recognition import RecognitionPredictor

        RecognitionPredictor.batch_size = config.ocr_batch_size
    except Exception:
        pass

    try:
        from surya.table_rec import TableRecPredictor

        TableRecPredictor.batch_size = config.table_batch_size
    except Exception:
        pass


def apply_surya_settings(config: SuryaHardwareConfig) -> SuryaHardwareConfig:
    """Apply the resolved configuration to Surya's runtime settings."""

    from surya.settings import settings

    set_torch_device_env(config.device)
    settings.TORCH_DEVICE = config.device
    settings.LAYOUT_BATCH_SIZE = config.layout_batch_size
    settings.DETECTOR_BATCH_SIZE = config.detection_batch_size
    settings.RECOGNITION_BATCH_SIZE = config.ocr_batch_size
    settings.TABLE_REC_BATCH_SIZE = config.table_batch_size

    _patch_predictor_batch_sizes(config)
    return config


def configure_surya_settings(
    device: DeviceType = "auto",
    batch_size: int | None = None,
    page_batch_size: int | None = None,
    layout_batch_size: int | None = None,
    detection_batch_size: int | None = None,
    ocr_batch_size: int | None = None,
    table_batch_size: int | None = None,
    equation_batch_size: int | None = None,
    enable_latex: bool = False,
    allow_parallel_phases: bool | None = None,
    parallel_workers: int | None = None,
) -> SuryaHardwareConfig:
    """Resolve and apply Surya settings using local hardware heuristics."""

    resolved_device = _detect_device() if device == "auto" else device
    free_vram_mb = get_gpu_memory_mb(resolved_device)
    usable_vram_mb = int(free_vram_mb * 0.8) if free_vram_mb is not None else None

    resolved_layout_batch = _estimate_phase_batch(
        resolved_device, "layout", usable_vram_mb, layout_batch_size
    )
    resolved_detection_batch = _estimate_phase_batch(
        resolved_device, "detection", usable_vram_mb, detection_batch_size
    )
    resolved_ocr_batch = _estimate_phase_batch(
        resolved_device, "recognition", usable_vram_mb, ocr_batch_size
    )
    resolved_table_batch = _estimate_phase_batch(
        resolved_device, "table", usable_vram_mb, table_batch_size
    )
    resolved_equation_batch = _estimate_phase_batch(
        resolved_device, "equation", usable_vram_mb, equation_batch_size
    )

    resolved_page_batch = page_batch_size
    if resolved_page_batch is None:
        resolved_page_batch = batch_size
    if resolved_page_batch is None:
        resolved_page_batch = min(resolved_layout_batch, resolved_detection_batch)
    resolved_page_batch = max(1, resolved_page_batch)

    if allow_parallel_phases is None:
        parallel_budget_mb = None
        if usable_vram_mb is not None:
            parallel_budget_mb = int(
                resolved_page_batch
                * (
                    _PHASE_MEMORY_MB["layout"]
                    + _PHASE_MEMORY_MB["detection"]
                    + _PHASE_MEMORY_MB["recognition"]
                )
                * 1.2
            )
        allow_parallel_phases = bool(
            resolved_device == "cuda"
            and parallel_budget_mb is not None
            and usable_vram_mb is not None
            and usable_vram_mb >= parallel_budget_mb
        )

    resolved_parallel_workers = parallel_workers
    if resolved_parallel_workers is None:
        resolved_parallel_workers = 2 if allow_parallel_phases else 1

    config = SuryaHardwareConfig(
        device=resolved_device,
        free_vram_mb=free_vram_mb,
        usable_vram_mb=usable_vram_mb,
        page_batch_size=resolved_page_batch,
        layout_batch_size=resolved_layout_batch,
        detection_batch_size=resolved_detection_batch,
        ocr_batch_size=resolved_ocr_batch,
        table_batch_size=resolved_table_batch,
        equation_batch_size=resolved_equation_batch,
        allow_parallel_phases=allow_parallel_phases,
        parallel_workers=max(1, resolved_parallel_workers),
        enable_latex=enable_latex,
    )

    logger.info(
        "Configured Surya settings: device=%s page=%s layout=%s detection=%s "
        "ocr=%s table=%s equation=%s parallel=%s",
        config.device,
        config.page_batch_size,
        config.layout_batch_size,
        config.detection_batch_size,
        config.ocr_batch_size,
        config.table_batch_size,
        config.equation_batch_size,
        config.allow_parallel_phases,
    )
    return apply_surya_settings(config)


def resolve_hardware(
    device: DeviceType = "auto",
    batch_size: int | None = None,
    ocr_batch_size: int | None = None,
    **kwargs,
) -> SuryaHardwareConfig:
    """Backward-compatible wrapper around ``configure_surya_settings``."""

    return configure_surya_settings(
        device=device,
        batch_size=batch_size,
        ocr_batch_size=ocr_batch_size,
        **kwargs,
    )


def main() -> None:
    """Print a resolved Surya hardware config for local tuning."""

    parser = argparse.ArgumentParser(description="Inspect resolved Surya settings")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--page-batch-size", type=int, default=None)
    parser.add_argument("--layout-batch-size", type=int, default=None)
    parser.add_argument("--detection-batch-size", type=int, default=None)
    parser.add_argument("--ocr-batch-size", type=int, default=None)
    parser.add_argument("--table-batch-size", type=int, default=None)
    parser.add_argument("--equation-batch-size", type=int, default=None)
    parser.add_argument("--enable-latex", action="store_true")
    parser.add_argument("--allow-parallel-phases", action="store_true")
    args = parser.parse_args()

    config = configure_surya_settings(
        device=args.device,
        page_batch_size=args.page_batch_size,
        layout_batch_size=args.layout_batch_size,
        detection_batch_size=args.detection_batch_size,
        ocr_batch_size=args.ocr_batch_size,
        table_batch_size=args.table_batch_size,
        equation_batch_size=args.equation_batch_size,
        enable_latex=args.enable_latex,
        allow_parallel_phases=args.allow_parallel_phases,
    )
    print(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    main()
