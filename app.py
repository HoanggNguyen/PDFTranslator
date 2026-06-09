"""Gradio app entry point: end-to-end PDF translation (OCR -> Translate -> Render).

Single entry point for the Hugging Face Space (Docker SDK). UI, styling, and the
pipeline runner live in ``pdf2zh.webapp``; this module only warms up the heavy
models and launches the server.
"""

from __future__ import annotations

import logging
import tempfile

from pdf2zh.e2e import warmup
from pdf2zh.webapp.ui import build_ui

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load the heavy Phase-1 models at boot so the first request isn't penalized.
try:
    warmup()
except Exception as exc:  # noqa: BLE001 — log but still start the UI
    logger.warning("warmup failed (models will load on first request): %s", exc)

demo = build_ui()

if __name__ == "__main__":
    # The rendered PDF lives under the system temp dir (see runner.py); allow
    # Gradio to serve it so the preview/download components can load it.
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[tempfile.gettempdir()],
    )
