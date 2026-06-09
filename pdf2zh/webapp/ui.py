"""Gradio UI: layout, styling, and event wiring for the translation app."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from gradio_pdf import PDF

from pdf2zh.e2e import BUNDLED_FONTS, DEFAULT_FONT, SUPPORTED_LANGUAGES
from pdf2zh.webapp.config import (
    CUSTOM_LABEL,
    PAGE_PRESETS,
    PROVIDER_CHOICES,
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_KEY,
    resolve_pages,
)
from pdf2zh.webapp.runner import (
    Progress,
    Result,
    TranslationRequest,
    stream_translation,
    validate,
)

MODAL_CSS = Path(__file__).with_name("modal.css").read_text(encoding="utf-8")

# Static header of the "translating" modal (spinner + title); the phase message
# below it is updated live.
MODAL_HEADER = '<div class="spinner"></div><div class="modal-title">Đang dịch…</div>'


def _msg_html(msg: str) -> str:
    return f'<div class="modal-msg">{msg}</div>'


# Output order of the translation event: [modal, modal_msg, pdf_out, download, status]
def _progress_outputs(msg: str):
    """Keep the modal open and refresh its phase message."""
    return (gr.update(visible=True), _msg_html(msg), gr.update(), gr.update(), "")


def _hide(status_msg: str):
    """Hide the modal + preview/download and show a status message (error cases)."""
    return (
        gr.update(visible=False),  # modal
        gr.update(),  # modal_msg
        gr.update(visible=False),  # pdf_out
        gr.update(visible=False),  # download
        status_msg,  # status
    )


def _result_outputs(result: Result):
    if result.status == "ok":
        # Separate update dicts per component — don't share one mutable dict.
        return (
            gr.update(visible=False),  # modal
            gr.update(),  # modal_msg
            gr.update(value=result.out_path, visible=True),  # pdf_out (preview)
            gr.update(value=result.out_path, visible=True),  # download
            "✅ Dịch xong.",  # status
        )
    if result.status == "invalid":
        return _hide(f"⚠️ {result.detail}")
    return _hide(f"❌ Lỗi: {result.detail}")


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
    return gr.update(visible=(choice == CUSTOM_LABEL))


def handle_translate(
    pdf_path, provider, api_key, model, lang_from, lang_to, font, page_choice, page_n
):
    """Show the blocking modal, stream per-phase progress, then show the result."""
    req = TranslationRequest(
        pdf_path=pdf_path,
        provider=PROVIDER_KEY[provider],
        api_key=api_key,
        model=model or None,
        src_lang=lang_from,
        tgt_lang=lang_to,
        font=font,
        pages=resolve_pages(page_choice, page_n),
    )
    invalid = validate(req)
    if invalid:  # don't flash the modal when there's nothing to do
        yield _hide(f"⚠️ {invalid}")
        return

    # First yield opens the modal (its overlay covers the page, so nothing else is
    # clickable); subsequent yields stream phase messages until the run finishes.
    yield _progress_outputs("Đang chuẩn bị…")
    for event in stream_translation(req):
        if isinstance(event, Progress):
            yield _progress_outputs(event.msg)
        else:
            yield _result_outputs(event)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="PDF Translator", theme=gr.themes.Default(), css=MODAL_CSS
    ) as demo:
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
                page_choice = gr.Radio(
                    list(PAGE_PRESETS), value="All", label="Số trang"
                )
                page_n = gr.Number(
                    value=10, precision=0, label="First N pages", visible=False
                )
                translate_btn = gr.Button("Dịch", variant="primary")
            with gr.Column(scale=2):
                pdf_in = PDF(label="Tải lên PDF", height=600)
                pdf_out = PDF(label="Bản dịch (xem trước)", height=600, visible=False)
                download = gr.File(label="Tải PDF bản dịch", visible=False)
                status = gr.Markdown("")

        # Translating modal — hidden until a run starts. Its fixed-position overlay
        # covers the page, so no control is clickable until the run completes.
        with gr.Column(elem_id="modal-overlay", visible=False) as modal:
            with gr.Column(elem_id="modal-box"):
                gr.HTML(MODAL_HEADER)
                modal_msg = gr.HTML(_msg_html("Đang chuẩn bị…"))

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
            outputs=[modal, modal_msg, pdf_out, download, status],
        )
    return demo
