from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import fitz

from .config import CompressConfig

logger = logging.getLogger(__name__)


def finalize_save(
    doc: fitz.Document,
    output_path: Path,
    cfg: CompressConfig,
) -> None:
    """Subset fonts, apply deflate/garbage optimisation, optionally re-encode images."""
    if cfg.subset_fonts:
        doc.subset_fonts()
        logger.debug("subset_fonts done")

    save_kwargs: dict = {"garbage": 4, "clean": True}
    if cfg.deflate:
        save_kwargs.update(
            deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=1
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path), **save_kwargs)
    logger.debug("Saved %s", output_path)

    if cfg.pikepdf_image_recompress:
        _recompress_images(output_path, cfg)


def _recompress_images(path: Path, cfg: CompressConfig) -> None:
    """Re-encode images in PDF via pikepdf (optional, aggressive compression)."""
    from importlib.util import find_spec

    if find_spec("pikepdf") is None or find_spec("PIL") is None:
        logger.warning("pikepdf or Pillow not installed; skipping image recompression")
        return
    import pikepdf

    original_size = path.stat().st_size

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pikepdf.open(str(path)) as pdf:
            for page in pdf.pages:
                for xobj_name in list(page.Resources.get("/XObject", {}).keys()):
                    xobj = page.Resources.XObject[xobj_name]
                    if xobj.get("/Subtype") != "/Image":
                        continue
                    _recompress_one(pdf, xobj, cfg)

            pdf.save(str(tmp_path), compress_streams=True, recompress_flate=True)

        new_size = tmp_path.stat().st_size
        if new_size < original_size:
            shutil.move(str(tmp_path), str(path))
            logger.info(
                "Image recompression: %d → %d bytes (%.1f%%)",
                original_size,
                new_size,
                100 * new_size / max(1, original_size),
            )
        else:
            tmp_path.unlink(missing_ok=True)
            logger.debug("Image recompression skipped: not smaller")
    except Exception as exc:
        logger.warning("Image recompression failed: %s", exc)
        tmp_path.unlink(missing_ok=True)


def _recompress_one(pdf, xobj, cfg: CompressConfig) -> None:
    import io

    import pikepdf
    from PIL import Image

    try:
        # Skip non-standard images
        cs = xobj.get("/ColorSpace")
        if cs in ("/DeviceCMYK",):
            return
        bpc = xobj.get("/BitsPerComponent", 8)
        if int(bpc) != 8:
            return
        if "/Mask" in xobj or "/SMask" in xobj:
            return

        w = int(xobj["/Width"])
        h = int(xobj["/Height"])
        if w * h == 0:
            return

        raw = xobj.read_raw_bytes()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Resize to target DPI if image is very large
        # (we don't know display DPI here so skip resize — just recompress)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.jpeg_quality, optimize=True)
        encoded = buf.getvalue()

        if len(encoded) < len(raw):
            xobj.write(encoded, filter=pikepdf.Name("/DCTDecode"))
    except Exception:
        pass  # Leave image unchanged on any error
