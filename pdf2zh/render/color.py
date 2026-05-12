from __future__ import annotations

import fitz
import numpy as np


def pixmap_to_array(pm: fitz.Pixmap) -> np.ndarray:
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    return arr[:, :, :3] if pm.n == 4 else arr


def bbox_to_pixels(
    bbox_pdf: list[float],
    page_width: float,
    page_height: float,
    pm: fitz.Pixmap,
) -> tuple[int, int, int, int]:
    sx = pm.width / page_width
    sy = pm.height / page_height
    x0, y0, x1, y1 = bbox_pdf
    return (
        max(0, int(x0 * sx)),
        max(0, int(y0 * sy)),
        min(pm.width, int(x1 * sx + 0.5)),
        min(pm.height, int(y1 * sy + 0.5)),
    )


def _mode_rgb(pixels: np.ndarray, qstep: int) -> tuple[int, int, int] | None:
    if pixels.size == 0:
        return None
    q = (pixels // qstep) * qstep + qstep // 2
    keys = (
        q[:, 0].astype(np.int32) * 65536
        + q[:, 1].astype(np.int32) * 256
        + q[:, 2].astype(np.int32)
    )
    vals, counts = np.unique(keys, return_counts=True)
    w = vals[counts.argmax()]
    return (int((w >> 16) & 0xFF), int((w >> 8) & 0xFF), int(w & 0xFF))


def detect_bg_color(
    arr: np.ndarray,
    bbox_px: tuple[int, int, int, int],
    edge_band_px: int = 2,
    qstep: int = 16,
    fallback: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    px0, py0, px1, py1 = bbox_px
    if px1 - px0 < 2 * edge_band_px + 1 or py1 - py0 < 2 * edge_band_px + 1:
        return fallback
    top = arr[py0 : py0 + edge_band_px, px0:px1].reshape(-1, 3)
    bot = arr[py1 - edge_band_px : py1, px0:px1].reshape(-1, 3)
    left = arr[
        py0 + edge_band_px : py1 - edge_band_px, px0 : px0 + edge_band_px
    ].reshape(-1, 3)
    right = arr[
        py0 + edge_band_px : py1 - edge_band_px, px1 - edge_band_px : px1
    ].reshape(-1, 3)
    band = np.concatenate([top, bot, left, right])
    return _mode_rgb(band, qstep) or fallback


def detect_text_color(
    arr: np.ndarray,
    bbox_px: tuple[int, int, int, int],
    bg: tuple[int, int, int],
    edge_band_px: int = 2,
    qstep: int = 16,
    dist_threshold: int = 32,
    min_ratio: float = 0.05,
    fallback: tuple[int, int, int] = (0, 0, 0),
) -> tuple[int, int, int]:
    px0, py0, px1, py1 = bbox_px
    inner = arr[
        py0 + edge_band_px : py1 - edge_band_px, px0 + edge_band_px : px1 - edge_band_px
    ]
    if inner.size == 0:
        return fallback
    flat = inner.reshape(-1, 3).astype(np.int32)
    bg_arr = np.array(bg, dtype=np.int32)
    dist = np.sqrt(((flat - bg_arr) ** 2).sum(axis=1))
    keep = flat[dist > dist_threshold]
    if keep.size == 0 or len(keep) / max(1, len(flat)) < min_ratio:
        return fallback
    return _mode_rgb(keep.astype(np.uint8), qstep) or fallback
