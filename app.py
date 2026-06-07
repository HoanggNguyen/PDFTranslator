"""Gradio app: end-to-end PDF translation (OCR -> Translate -> Render).

Single entry point for the Hugging Face Space (Docker SDK). Mirrors the
PDFMathTranslate layout: a left sidebar to pick a translation provider and enter
credentials, a main area to upload a PDF and preview/download the result.
"""

from __future__ import annotations

import logging
import tempfile
import traceback
import uuid
from pathlib import Path

import gradio as gr
from gradio_pdf import PDF

from pdf2zh.e2e import (
    BUNDLED_FONTS,
    DEFAULT_FONT,
    SUPPORTED_LANGUAGES,
    run_pipeline,
    warmup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# UI label -> Phase-2 provider key.
PROVIDER_KEY = {"OpenRouter": "openrouter", "Gemini": "gemini", "OpenAI": "openai"}
# Default model placeholders (mirror translation/config.py PROVIDERS).
PROVIDER_DEFAULT_MODEL = {
    "OpenRouter": "google/gemini-2.5-flash-lite",
    "Gemini": "gemini-2.5-flash-lite",
    "OpenAI": "gpt-4o-mini",
}
PROVIDER_CHOICES = list(PROVIDER_KEY)

# Page selection presets. "CUSTOM" -> use the "First N" number input.
PAGE_MAP: dict[str, object] = {
    "All": None,
    "First page": [0],
    "First 5 pages": list(range(5)),
    "First N…": "CUSTOM",
}

MAX_CUSTOM_PAGES = 50  # guardrail against OOM on a single T4


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #
def on_provider_change(provider: str):
    """Update the model placeholder and clear the key when provider changes."""
    return (
        gr.update(placeholder=PROVIDER_DEFAULT_MODEL.get(provider, "")),
        gr.update(value=""),
    )


def on_page_choice(choice: str):
    return gr.update(visible=(choice == "First N…"))


def _resolve_pages(page_choice: str, page_n) -> list[int] | None:
    sel = PAGE_MAP[page_choice]
    if sel == "CUSTOM":
        n = max(1, min(int(page_n or 1), MAX_CUSTOM_PAGES))
        return list(range(n))
    return sel  # None or a list


def handle_translate(
    pdf_path,
    provider,
    api_key,
    model,
    lang_from,
    lang_to,
    font,
    page_choice,
    page_n,
    progress=gr.Progress(),
):
    """Run the full pipeline. Plain ``def`` (not ``async``) so Gradio runs it in a
    worker thread — the ``asyncio.run`` inside Phase 2 then works correctly."""
    pages = _resolve_pages(page_choice, page_n)
    work_dir = Path(tempfile.gettempdir()) / f"pdf2zh_{uuid.uuid4().hex}"

    def cb(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    try:
        out = run_pipeline(
            pdf_path=pdf_path,
            src_lang=lang_from,
            tgt_lang=lang_to,
            provider=PROVIDER_KEY[provider],
            api_key=api_key,
            model=model,
            pages=pages,
            font=font,
            work_dir=work_dir,
            progress=cb,
        )
        return (
            gr.update(value=out, visible=True),
            gr.update(value=out, visible=True),
            "✅ Dịch xong.",
        )
    except UnicodeError as exc:  # encoding bug, not user input — show where it failed
        logger.exception("pipeline failed (unicode)")
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            f"❌ {type(exc).__name__}: {exc}\n```\n{tail}\n```",
        )
    except ValueError as exc:  # user-facing input errors
        return gr.update(visible=False), gr.update(visible=False), f"⚠️ {exc}"
    except Exception as exc:  # noqa: BLE001 — surface anything else to the UI
        logger.exception("pipeline failed")
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            f"❌ Lỗi: {type(exc).__name__}: {exc}\n```\n{tail}\n```",
        )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PDF Translator", theme=gr.themes.Default()) as demo:
        gr.Markdown("# PDF Translator\nDịch PDF end-to-end: OCR → Dịch → Dựng lại PDF.")
        with gr.Row():
            with gr.Column(scale=1):
                provider = gr.Dropdown(
                    PROVIDER_CHOICES, value="OpenRouter", label="Provider"
                )
                api_key = gr.Textbox(label="API Key", type="password", value="")
                model = gr.Textbox(
                    label="Model (tuỳ chọn)",
                    placeholder=PROVIDER_DEFAULT_MODEL["OpenRouter"],
                )
                lang_from = gr.Dropdown(
                    SUPPORTED_LANGUAGES, value="English", label="Dịch từ"
                )
                lang_to = gr.Dropdown(
                    SUPPORTED_LANGUAGES, value="Vietnamese", label="Dịch sang"
                )
                font = gr.Dropdown(
                    BUNDLED_FONTS, value=DEFAULT_FONT, label="Font đầu ra"
                )
                page_choice = gr.Radio(list(PAGE_MAP), value="All", label="Số trang")
                page_n = gr.Number(
                    value=10, precision=0, label="First N pages", visible=False
                )
                translate_btn = gr.Button("Dịch", variant="primary")
            with gr.Column(scale=2):
                pdf_in = PDF(label="Tải lên PDF", height=600)
                pdf_out = PDF(label="Bản dịch (xem trước)", height=600, visible=False)
                download = gr.File(label="Tải PDF bản dịch", visible=False)
                status = gr.Markdown("")

        provider.change(on_provider_change, inputs=provider, outputs=[model, api_key])
        page_choice.change(on_page_choice, inputs=page_choice, outputs=page_n)
        translate_btn.click(
            handle_translate,
            inputs=[
                pdf_in,
                provider,
                api_key,
                model,
                lang_from,
                lang_to,
                font,
                page_choice,
                page_n,
            ],
            outputs=[pdf_out, download, status],
        )
    return demo


# Load the heavy Phase-1 models at boot so the first request isn't penalized.
try:
    warmup()
except Exception as exc:  # noqa: BLE001 — log but still start the UI
    logger.warning("warmup failed (models will load on first request): %s", exc)

demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860)
