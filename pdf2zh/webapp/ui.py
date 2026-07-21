"""Gradio UI: stepped, human-in-the-loop PDF translation.

Flow: Config → Step 1 (Phase 1: review/edit extraction) → [translate + render]
→ Step 2 (Phase 3: review render, edit translations, re-render) → download.

Both review steps let the user click a page preview to select an element and
edit it (with a #-dropdown as an explicit fallback selector). The pure logic
lives in ``review.py``; heavy phases run in worker threads via ``runner.py``.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import gradio as gr
from gradio_pdf import PDF

from pdf2zh.e2e import BUNDLED_FONTS, DEFAULT_FONT, SUPPORTED_LANGUAGES
from pdf2zh.webapp.config import (
    PAGE_MODE_ALL,
    PAGE_MODE_RANGE,
    PAGE_MODES,
    PROVIDER_CHOICES,
    PROVIDER_KEY,
    resolve_pages,
)
from pdf2zh.webapp.review import (
    LABEL_CHOICES,
    add_element,
    apply_phase1_edit,
    apply_phase2_edit,
    hit_test,
    normalize_click,
    output_page_position,
    render_page_with_boxes,
)
from pdf2zh.webapp.runner import (
    Progress,
    list_models,
    stream_parse,
    stream_render,
    stream_translate_render,
)

REVIEW_DPI = 150
_SCALE = REVIEW_DPI / 72.0


# --------------------------------------------------------------------------- #
# Small helpers (pure-ish, operate on the session dict)
# --------------------------------------------------------------------------- #
def _page_choices(doc: dict) -> list[tuple[str, int]]:
    """Dropdown choices for pages: label = 1-based document page number."""
    return [
        (f"Trang {p.get('page_index', i) + 1}", i)
        for i, p in enumerate(doc.get("pages", []))
    ]


def _elem_choices(page: dict) -> list[tuple[str, int]]:
    out = []
    for i, e in enumerate(page.get("elements", [])):
        cat = e.get("category", "")
        label = e.get("label", "")
        out.append((f"#{i} · {label} · {cat}", i))
    return out


def _render_p1(session: dict, highlight: int | None = None):
    """Render the current Phase-1 page (original PDF) with element boxes."""
    parsed = session["parsed"]
    page_i = session["p1_page"]
    page = parsed["pages"][page_i]
    img, boxes, _ = render_page_with_boxes(
        session["pdf_path"],
        page.get("page_index", page_i),
        page.get("elements", []),
        REVIEW_DPI,
        highlight,
    )
    session["p1_boxes"] = boxes
    return img


def _render_p3(session: dict, highlight: int | None = None):
    """Render the current Phase-3 page (rendered output PDF) with element boxes."""
    translated = session["translated"]
    page_i = session["p3_page"]
    page = translated["pages"][page_i]
    out_pos = output_page_position(session["pages"], page.get("page_index", page_i))
    if out_pos is None:
        return None
    img, boxes, _ = render_page_with_boxes(
        session["out_path"],
        out_pos,
        page.get("elements", []),
        REVIEW_DPI,
        highlight,
    )
    session["p3_boxes"] = boxes
    return img


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PDF Translator", theme=gr.themes.Default()) as demo:
        session = gr.State({})
        gr.Markdown(
            "# PDF Translator\n"
            "Dịch PDF end-to-end với 2 bước kiểm tra: **Phase 1** (sửa trích xuất) "
            "→ dịch & dựng → **Phase 3** (soát bản dịch trên trang đã render)."
        )

        # ---- Config ---------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                provider = gr.Dropdown(
                    PROVIDER_CHOICES, value="OpenRouter", label="Provider"
                )
                api_key = gr.Textbox(label="API Key", type="password", value="")
                model = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Model (trống = mặc định)",
                    allow_custom_value=True,
                    info="Nhập API key để tự tải danh sách, hoặc gõ tên model.",
                )
                load_models_btn = gr.Button("🔄 Tải danh sách model", size="sm")
                lang_from = gr.Dropdown(
                    SUPPORTED_LANGUAGES, value="English", label="Dịch từ"
                )
                lang_to = gr.Dropdown(
                    SUPPORTED_LANGUAGES, value="Vietnamese", label="Dịch sang"
                )
                font = gr.Dropdown(
                    BUNDLED_FONTS, value=DEFAULT_FONT, label="Font đầu ra"
                )
                page_mode = gr.Radio(
                    PAGE_MODES, value=PAGE_MODE_ALL, label="Trang dịch"
                )
                with gr.Row():
                    page_from = gr.Number(
                        value=1, precision=0, label="Từ trang", visible=False, minimum=1
                    )
                    page_to = gr.Number(
                        value=1,
                        precision=0,
                        label="Đến trang",
                        visible=False,
                        minimum=1,
                    )
                parse_btn = gr.Button("① Trích xuất (Phase 1)", variant="primary")
            with gr.Column(scale=2):
                pdf_in = PDF(label="Tải lên PDF", height=520)
                status = gr.Markdown("")

        # ---- Step 1: Phase-1 review ----------------------------------------
        with gr.Column(visible=False) as p1_group:
            gr.Markdown(
                "## ① Kiểm tra trích xuất\n"
                "Click vào ô trên trang để chọn element (hoặc chọn theo số). "
                "Sửa nhãn/nội dung, bỏ qua element thừa, hoặc thêm box cho vùng bị sót."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    p1_page_dd = gr.Dropdown(label="Trang", choices=[], value=None)
                    p1_img = gr.Image(
                        label="Trang gốc (click để chọn element)",
                        interactive=False,
                        height=560,
                    )
                with gr.Column(scale=1):
                    p1_elem_dd = gr.Dropdown(
                        label="Element đang chọn (# theo số trên ô)",
                        choices=[],
                        value=None,
                    )
                    p1_label = gr.Dropdown(
                        LABEL_CHOICES, label="Nhãn (label)", value=None
                    )
                    p1_source = gr.Textbox(label="Văn bản gốc (source_text)", lines=4)
                    p1_bypass = gr.Checkbox(label="Bỏ qua element này (không dịch)")
                    p1_save_btn = gr.Button("💾 Lưu element", variant="secondary")

                    gr.Markdown("**Thêm box cho vùng bị sót**")
                    p1_draw_mode = gr.Checkbox(
                        label="Chế độ vẽ box (click 2 góc trên ảnh)"
                    )
                    with gr.Row():
                        p1_x0 = gr.Number(label="x0", value=0)
                        p1_y0 = gr.Number(label="y0", value=0)
                        p1_x1 = gr.Number(label="x1", value=0)
                        p1_y1 = gr.Number(label="y1", value=0)
                    p1_new_label = gr.Dropdown(
                        LABEL_CHOICES, label="Nhãn box mới", value="Text"
                    )
                    p1_new_source = gr.Textbox(label="Văn bản box mới", lines=3)
                    p1_add_btn = gr.Button("➕ Thêm box")
            confirm_btn = gr.Button("② Xác nhận → Dịch & Render", variant="primary")

        # ---- Step 2: Phase-3 review ----------------------------------------
        with gr.Column(visible=False) as p3_group:
            gr.Markdown(
                "## ② Soát bản dịch (trên trang đã render)\n"
                "Click vào element để sửa bản dịch, rồi **Render lại**. "
                "Màu nền/chữ, font, cỡ chữ được giữ nguyên."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    p3_pdf = PDF(label="Bản dịch (xem trước)", height=560)
                    download = gr.File(label="Tải PDF bản dịch")
                with gr.Column(scale=1):
                    p3_page_dd = gr.Dropdown(label="Trang", choices=[], value=None)
                    p3_img = gr.Image(
                        label="Trang đã render (click để chọn element)",
                        interactive=False,
                        height=560,
                    )
                    p3_elem_dd = gr.Dropdown(
                        label="Element đang chọn", choices=[], value=None
                    )
                    p3_source = gr.Textbox(
                        label="Văn bản gốc", lines=3, interactive=False
                    )
                    p3_translated = gr.Textbox(label="Bản dịch (sửa được)", lines=5)
                    p3_save_btn = gr.Button("💾 Lưu bản dịch", variant="secondary")
                    rerender_btn = gr.Button("🔁 Render lại", variant="primary")

        # ================================================================== #
        # Handlers
        # ================================================================== #
        def on_provider_change(_provider):
            return (gr.update(choices=[], value=None), gr.update(value=""))

        def on_load_models(prov, key):
            models = list_models(PROVIDER_KEY[prov], key)
            if not models:
                gr.Warning("Không tải được model — kiểm tra key/provider, hoặc gõ tay.")
                return gr.update()
            gr.Info(f"Đã tải {len(models)} model.")
            return gr.update(choices=models)

        def on_page_mode(mode):
            vis = mode == PAGE_MODE_RANGE
            return gr.update(visible=vis), gr.update(visible=vis)

        # ---- Step 1: parse -------------------------------------------------
        def do_parse(session, pdf, prov, key, lfrom, lto, mode, frm, to):
            if not pdf:
                yield {status: "⚠️ Vui lòng tải lên một file PDF."}
                return
            if not key or not key.strip():
                yield {status: "⚠️ Thiếu API key."}
                return

            work_dir = Path(tempfile.gettempdir()) / f"pdf2zh_{uuid.uuid4().hex}"
            work_dir.mkdir(parents=True, exist_ok=True)
            # Copy the uploaded PDF so later steps survive Gradio cache cleanup.
            local_pdf = work_dir / "source.pdf"
            shutil.copyfile(pdf, local_pdf)
            pages = resolve_pages(mode, frm, to)

            session["pdf_path"] = str(local_pdf)
            session["work_dir"] = str(work_dir)
            session["pages"] = pages
            session["provider"] = PROVIDER_KEY[prov]
            session["src_lang"] = lfrom
            session["tgt_lang"] = lto

            yield {status: "⏳ Phase 1 — OCR & phân tích bố cục…"}
            result = None
            for ev in stream_parse(session["pdf_path"], pages, session["work_dir"]):
                if isinstance(ev, Progress):
                    yield {status: f"⏳ {ev.msg}"}
                else:
                    result = ev
            if result is None or result.status != "ok":
                yield {status: f"❌ {result.detail if result else 'Không rõ lỗi.'}"}
                return

            if not result.data.get("pages"):
                yield {
                    status: "⚠️ Không có trang nào để dịch (khoảng trang nằm ngoài tài liệu?)."
                }
                return

            session["parsed"] = result.data
            session["p1_page"] = 0
            session["p1_sel"] = None
            page_choices = _page_choices(session["parsed"])
            img = _render_p1(session)
            first_page = session["parsed"]["pages"][0]
            yield {
                session: session,
                status: "✅ Phase 1 xong — kiểm tra & sửa rồi bấm Xác nhận.",
                p1_group: gr.update(visible=True),
                p1_page_dd: gr.update(choices=page_choices, value=0),
                p1_img: img,
                p1_elem_dd: gr.update(choices=_elem_choices(first_page), value=None),
            }

        def on_p1_page(session, page_i):
            if page_i is None:
                return {}
            session["p1_page"] = int(page_i)
            session["p1_sel"] = None
            page = session["parsed"]["pages"][int(page_i)]
            return {
                session: session,
                p1_img: _render_p1(session),
                p1_elem_dd: gr.update(choices=_elem_choices(page), value=None),
                p1_label: gr.update(value=None),
                p1_source: gr.update(value=""),
                p1_bypass: gr.update(value=False),
            }

        def _p1_select(session, elem_i):
            """Populate the edit panel for the selected element + highlight it."""
            session["p1_sel"] = elem_i
            page = session["parsed"]["pages"][session["p1_page"]]
            elem = page["elements"][elem_i]
            return {
                session: session,
                p1_img: _render_p1(session, highlight=elem_i),
                p1_elem_dd: gr.update(value=elem_i),
                p1_label: gr.update(value=elem.get("label")),
                p1_source: gr.update(value=elem.get("source_text", "")),
                p1_bypass: gr.update(value=elem.get("category") == "BYPASS"),
            }

        def on_p1_img_click(session, draw_mode, evt: gr.SelectData):
            pt = normalize_click(evt.index)
            if pt is None:
                return {}
            x_px, y_px = pt
            if draw_mode:
                # Capture a corner (converted to PDF points) into the box inputs.
                pts = session.get("draw_pts", [])
                pts.append((x_px / _SCALE, y_px / _SCALE))
                session["draw_pts"] = pts
                if len(pts) == 1:
                    return {session: session, p1_x0: pts[0][0], p1_y0: pts[0][1]}
                # Second click: normalize the rectangle, then reset.
                (ax, ay), (bx, by) = pts[0], pts[1]
                session["draw_pts"] = []
                return {
                    session: session,
                    p1_x0: min(ax, bx),
                    p1_y0: min(ay, by),
                    p1_x1: max(ax, bx),
                    p1_y1: max(ay, by),
                }
            elem_i = hit_test(session.get("p1_boxes", []), x_px, y_px)
            if elem_i is None:
                return {}
            return _p1_select(session, elem_i)

        def on_p1_pick(session, elem_i):
            if elem_i is None:
                return {}
            return _p1_select(session, int(elem_i))

        def do_p1_save(session, label, source, bypass):
            sel = session.get("p1_sel")
            if sel is None:
                return {status: "⚠️ Chưa chọn element nào."}
            msg = apply_phase1_edit(
                session["parsed"], session["p1_page"], sel, label, source, bypass
            )
            page = session["parsed"]["pages"][session["p1_page"]]
            return {
                session: session,
                p1_img: _render_p1(session, highlight=sel),
                p1_elem_dd: gr.update(choices=_elem_choices(page), value=sel),
                status: f"⚠️ {msg}" if msg else "✅ Đã lưu element.",
            }

        def do_p1_add(session, x0, y0, x1, y1, label, source):
            if x1 <= x0 or y1 <= y0:
                return {status: "⚠️ Box không hợp lệ (cần x1>x0, y1>y0)."}
            page_i = session["p1_page"]
            new_idx = add_element(
                session["parsed"], page_i, [x0, y0, x1, y1], label, source
            )
            page = session["parsed"]["pages"][page_i]
            session["p1_sel"] = new_idx
            return {
                session: session,
                p1_img: _render_p1(session, highlight=new_idx),
                p1_elem_dd: gr.update(choices=_elem_choices(page), value=new_idx),
                p1_new_source: gr.update(value=""),
                status: f"✅ Đã thêm box #{new_idx}.",
            }

        # ---- Confirm → translate + render ----------------------------------
        def do_confirm(session, prov, key, mdl, lfrom, lto, fnt):
            if not key or not key.strip():
                yield {status: "⚠️ Thiếu API key."}
                return
            session["src_lang"], session["tgt_lang"] = lfrom, lto
            yield {status: "⏳ Phase 2 — đang dịch…"}
            result = None
            for ev in stream_translate_render(
                session["pdf_path"],
                session["parsed"],
                lfrom,
                lto,
                PROVIDER_KEY[prov],
                key,
                mdl or None,
                session["pages"],
                fnt,
                session["work_dir"],
            ):
                if isinstance(ev, Progress):
                    yield {status: f"⏳ {ev.msg}"}
                else:
                    result = ev
            if result is None or result.status != "ok":
                yield {status: f"❌ {result.detail if result else 'Không rõ lỗi.'}"}
                return

            session["translated"] = result.data["translated"]
            session["out_path"] = result.data["out_path"]
            session["p3_page"] = 0
            session["p3_sel"] = None
            page_choices = _page_choices(session["translated"])
            img = _render_p3(session)
            first_page = session["translated"]["pages"][0]
            yield {
                session: session,
                status: "✅ Đã dịch & dựng. Soát bản dịch rồi tải về.",
                p3_group: gr.update(visible=True),
                p3_pdf: session["out_path"],
                download: session["out_path"],
                p3_page_dd: gr.update(choices=page_choices, value=0),
                p3_img: img,
                p3_elem_dd: gr.update(choices=_elem_choices(first_page), value=None),
            }

        def on_p3_page(session, page_i):
            if page_i is None:
                return {}
            session["p3_page"] = int(page_i)
            session["p3_sel"] = None
            page = session["translated"]["pages"][int(page_i)]
            return {
                session: session,
                p3_img: _render_p3(session),
                p3_elem_dd: gr.update(choices=_elem_choices(page), value=None),
                p3_source: gr.update(value=""),
                p3_translated: gr.update(value=""),
            }

        def _p3_select(session, elem_i):
            session["p3_sel"] = elem_i
            page = session["translated"]["pages"][session["p3_page"]]
            elem = page["elements"][elem_i]
            return {
                session: session,
                p3_img: _render_p3(session, highlight=elem_i),
                p3_elem_dd: gr.update(value=elem_i),
                p3_source: gr.update(value=elem.get("source_text", "")),
                p3_translated: gr.update(value=elem.get("translated_text", "")),
            }

        def on_p3_img_click(session, evt: gr.SelectData):
            pt = normalize_click(evt.index)
            if pt is None:
                return {}
            elem_i = hit_test(session.get("p3_boxes", []), pt[0], pt[1])
            if elem_i is None:
                return {}
            return _p3_select(session, elem_i)

        def on_p3_pick(session, elem_i):
            if elem_i is None:
                return {}
            return _p3_select(session, int(elem_i))

        def do_p3_save(session, translated_text):
            sel = session.get("p3_sel")
            if sel is None:
                return {status: "⚠️ Chưa chọn element nào."}
            apply_phase2_edit(
                session["translated"], session["p3_page"], sel, translated_text
            )
            return {
                session: session,
                status: "✅ Đã lưu bản dịch (bấm Render lại để cập nhật).",
            }

        def do_rerender(session, fnt):
            yield {status: "⏳ Đang render lại…"}
            result = None
            for ev in stream_render(
                session["pdf_path"],
                session["translated"],
                session["pages"],
                fnt,
                session["work_dir"],
            ):
                if isinstance(ev, Progress):
                    yield {status: f"⏳ {ev.msg}"}
                else:
                    result = ev
            if result is None or result.status != "ok":
                yield {status: f"❌ {result.detail if result else 'Không rõ lỗi.'}"}
                return
            session["out_path"] = result.data["out_path"]
            yield {
                session: session,
                status: "✅ Đã render lại.",
                p3_pdf: session["out_path"],
                download: session["out_path"],
                p3_img: _render_p3(session, highlight=session.get("p3_sel")),
            }

        # ================================================================== #
        # Wiring
        # ================================================================== #
        provider.change(on_provider_change, provider, [model, api_key])
        api_key.blur(on_load_models, [provider, api_key], model)
        load_models_btn.click(on_load_models, [provider, api_key], model)
        page_mode.change(on_page_mode, page_mode, [page_from, page_to])

        parse_out = [
            session,
            status,
            p1_group,
            p1_page_dd,
            p1_img,
            p1_elem_dd,
            p1_label,
            p1_source,
            p1_bypass,
        ]
        parse_btn.click(
            do_parse,
            [
                session,
                pdf_in,
                provider,
                api_key,
                lang_from,
                lang_to,
                page_mode,
                page_from,
                page_to,
            ],
            parse_out,
        )

        p1_edit_out = [
            session,
            p1_img,
            p1_elem_dd,
            p1_label,
            p1_source,
            p1_bypass,
            p1_x0,
            p1_y0,
            p1_x1,
            p1_y1,
            status,
        ]
        p1_page_dd.change(on_p1_page, [session, p1_page_dd], p1_edit_out)
        p1_img.select(on_p1_img_click, [session, p1_draw_mode], p1_edit_out)
        p1_elem_dd.select(on_p1_pick, [session, p1_elem_dd], p1_edit_out)
        p1_save_btn.click(
            do_p1_save, [session, p1_label, p1_source, p1_bypass], p1_edit_out
        )
        p1_add_btn.click(
            do_p1_add,
            [session, p1_x0, p1_y0, p1_x1, p1_y1, p1_new_label, p1_new_source],
            p1_edit_out + [p1_new_source],
        )

        confirm_out = [
            session,
            status,
            p3_group,
            p3_pdf,
            download,
            p3_page_dd,
            p3_img,
            p3_elem_dd,
        ]
        confirm_btn.click(
            do_confirm,
            [session, provider, api_key, model, lang_from, lang_to, font],
            confirm_out,
        )

        p3_edit_out = [
            session,
            p3_img,
            p3_elem_dd,
            p3_source,
            p3_translated,
            p3_pdf,
            download,
            status,
        ]
        p3_page_dd.change(on_p3_page, [session, p3_page_dd], p3_edit_out)
        p3_img.select(on_p3_img_click, [session], p3_edit_out)
        p3_elem_dd.select(on_p3_pick, [session, p3_elem_dd], p3_edit_out)
        p3_save_btn.click(do_p3_save, [session, p3_translated], p3_edit_out)
        rerender_btn.click(do_rerender, [session, font], p3_edit_out)

    return demo
