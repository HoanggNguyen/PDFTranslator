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
import time
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
    apply_phase1_cell_edit,
    apply_phase1_edit,
    apply_phase2_cell_edit,
    apply_phase2_edit,
    hex_to_rgb,
    hit_test,
    normalize_click,
    output_page_position,
    overlay_svg,
    render_page_plain,
)
from pdf2zh.webapp.runner import (
    Progress,
    TranslationRequest,
    list_models,
    stream_parse,
    stream_render,
    stream_translate_render,
    stream_translation,
    validate,
)

REVIEW_DPI = 150
_SCALE = REVIEW_DPI / 72.0
_P1_SOURCE_LABEL = "Văn bản gốc (source_text)"

# Static base image + live SVG overlay: the base <img> reloads only on page change
# / re-render, the overlay (a pure innerHTML swap) updates on every edit action, so
# selecting/saving/adding no longer flickers the preview.
REVIEW_CSS = """
#p1-stage, #p3-stage { position:relative; padding:0 !important; gap:0 !important; }
#p1-stage .image-frame img, #p1-stage .image-container img,
#p3-stage .image-frame img, #p3-stage .image-container img {
    width:100% !important; height:auto !important; object-fit:fill !important;
    display:block; }
#p1-stage .image-container, #p1-stage .image-frame,
#p3-stage .image-container, #p3-stage .image-frame { height:auto !important; }
#p1-stage .image-container, #p1-stage .image-frame, #p1-stage button,
#p3-stage .image-container, #p3-stage .image-frame, #p3-stage button {
    padding:0 !important; border:0 !important; margin:0 !important; }
#p1-stage .icon-button-wrapper, #p3-stage .icon-button-wrapper,
#p1-stage button[aria-label*="ullscreen"],
#p3-stage button[aria-label*="ullscreen"] { display:none !important; }
#p1-overlay, #p3-overlay { position:absolute; top:0; left:0; width:100%;
    padding:0 !important; margin:0 !important; pointer-events:none; }
#p1-overlay svg, #p3-overlay svg { display:block; width:100%; height:auto; }
svg.draw-rubber { position:absolute; pointer-events:none; z-index:6; }
"""

# Client-side box drawing: in draw mode the user clicks two corners. Click 1 drops
# a marker; moving the mouse shows a live dashed preview rectangle to the cursor;
# click 2 locks it and writes the PDF-point coords into the hidden x0/y0/x1/y1
# inputs. No server round-trip happens until "Thêm box", so drawing never flickers.
# Coords go display-px → natural-px (naturalWidth/rect) → PDF points (÷ _SCALE),
# matching ``on_p1_img_click``'s old math. __SCALE__ is filled from REVIEW_DPI below.
_DRAW_JS_TMPL = """
() => {
  const SCALE = __SCALE__;
  const CFG = { stage:'p1-stage', chk:'p1-draw-mode',
                x0:'p1-x0', y0:'p1-y0', x1:'p1-x1', y1:'p1-y1' };
  const q = (s) => document.querySelector(s);
  const drawOn = () => { const c = q('#'+CFG.chk+' input[type=checkbox]');
                         return !!(c && c.checked); };

  function setNum(id, val) {
    const inp = q('#'+id+' input');
    if (!inp) return;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value').set;
    setter.call(inp, String(Math.round(val * 100) / 100));
    inp.dispatchEvent(new Event('input', { bubbles: true }));
    inp.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function layer(stageEl, r) {
    const sr = stageEl.getBoundingClientRect();
    let svg = stageEl.querySelector('svg.draw-rubber');
    if (!svg) {
      svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('class', 'draw-rubber');
      stageEl.appendChild(svg);
    }
    svg.style.left = (r.left - sr.left) + 'px';
    svg.style.top = (r.top - sr.top) + 'px';
    svg.style.width = r.width + 'px';
    svg.style.height = r.height + 'px';
    return svg;
  }

  function paint(stageEl, r, ax, ay, bx, by, dashed) {
    const rx = Math.min(ax, bx), ry = Math.min(ay, by);
    layer(stageEl, r).innerHTML =
      '<rect x="'+rx+'" y="'+ry+'" width="'+Math.abs(bx-ax)+'" height="'
        +Math.abs(by-ay)+'" fill="red" fill-opacity="0.12" stroke="red" '
        +'stroke-width="2" '+(dashed ? 'stroke-dasharray="6 4" ' : '')+'/>'
      +'<circle cx="'+ax+'" cy="'+ay+'" r="4" fill="red"/>';
  }

  const at = (r, e) => ({
    x: Math.max(0, Math.min(r.width, e.clientX - r.left)),
    y: Math.max(0, Math.min(r.height, e.clientY - r.top)),
  });

  let P = null;  // first corner: {stageEl, img, r, x, y} once click 1 lands
  const reset = () => { P = null; };

  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('#p1-add-btn')) { clearAll(); return; }
    if (!drawOn()) return;
    const stageEl = e.target.closest && e.target.closest('#'+CFG.stage);
    if (!stageEl) return;
    const img = stageEl.querySelector('img');
    if (!img) return;
    const r = img.getBoundingClientRect();
    const p = at(r, e);
    if (!P) {
      // Click 1: mark the first corner.
      P = { stageEl, img, r, x: p.x, y: p.y };
      layer(stageEl, r).innerHTML =
        '<circle cx="'+p.x+'" cy="'+p.y+'" r="5" fill="red" fill-opacity="0.9"/>';
      return;
    }
    // Click 2: finalize.
    const x0 = Math.min(P.x, p.x), y0 = Math.min(P.y, p.y);
    const x1 = Math.max(P.x, p.x), y1 = Math.max(P.y, p.y);
    const sx = (img.naturalWidth || r.width) / r.width;
    const sy = (img.naturalHeight || r.height) / r.height;
    if (x1 - x0 >= 3 && y1 - y0 >= 3) {
      setNum(CFG.x0, x0 * sx / SCALE);
      setNum(CFG.y0, y0 * sy / SCALE);
      setNum(CFG.x1, x1 * sx / SCALE);
      setNum(CFG.y1, y1 * sy / SCALE);
      paint(P.stageEl, r, P.x, P.y, p.x, p.y, false);
    }
    reset();
  }, true);

  document.addEventListener('mousemove', (e) => {
    if (!P || !drawOn()) return;
    const p = at(P.r, e);
    paint(P.stageEl, P.r, P.x, P.y, p.x, p.y, true);
  }, true);

  const clearAll = () => {
    const svg = q('#'+CFG.stage+' svg.draw-rubber');
    if (svg) svg.innerHTML = '';
    reset();
  };
  // Clear the preview when leaving draw mode and whenever the base image
  // (re)loads — page change, new file, re-render — so no stale rectangle lingers.
  document.addEventListener('change', (e) => {
    if (e.target.closest && e.target.closest('#'+CFG.chk) && !drawOn()) clearAll();
  }, true);
  document.addEventListener('load', (e) => {
    if (e.target.tagName === 'IMG' && e.target.closest
        && e.target.closest('#'+CFG.stage)) clearAll();
  }, true);
}
"""
DRAW_JS = _DRAW_JS_TMPL.replace("__SCALE__", repr(_SCALE))


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


def _base_p1(session: dict):
    """Rasterize the current Phase-1 page (no boxes). Reloads the base <img>."""
    page_i = session["p1_page"]
    page = session["parsed"]["pages"][page_i]
    img, size = render_page_plain(
        session["pdf_path"], page.get("page_index", page_i), REVIEW_DPI
    )
    session["p1_size"] = size
    return img


def _overlay_p1(session: dict) -> str:
    """Build the Phase-1 SVG overlay + refresh session box list for hit_test.

    Reads the current selection straight from ``session["p1_sel"]`` — a
    ``(elem_i, cell_i)`` tuple (``cell_i`` None for a whole-element selection)
    or None — so callers just set that key and never pass highlight state
    separately (one less thing to keep in sync).
    """
    page = session["parsed"]["pages"][session["p1_page"]]
    w, h = session["p1_size"]
    sel = session.get("p1_sel")
    highlight_idx = sel[0] if sel and sel[1] is None else None
    highlight_cell = sel if sel and sel[1] is not None else None
    svg, boxes = overlay_svg(
        page.get("elements", []), _SCALE, w, h, highlight_idx, highlight_cell
    )
    session["p1_boxes"] = boxes
    return svg


def _base_p3(session: dict):
    """Rasterize the current Phase-3 page (no boxes), or None if not rendered."""
    page_i = session["p3_page"]
    page = session["translated"]["pages"][page_i]
    out_pos = output_page_position(session["pages"], page.get("page_index", page_i))
    if out_pos is None:
        return None
    img, size = render_page_plain(session["out_path"], out_pos, REVIEW_DPI)
    session["p3_size"] = size
    return img


def _overlay_p3(session: dict) -> str:
    """Build the Phase-3 SVG overlay + refresh session box list for hit_test.

    Reads the current selection straight from ``session["p3_sel"]``, same
    convention as ``_overlay_p1``.
    """
    size = session.get("p3_size")
    if size is None:
        return ""
    page = session["translated"]["pages"][session["p3_page"]]
    sel = session.get("p3_sel")
    highlight_idx = sel[0] if sel and sel[1] is None else None
    highlight_cell = sel if sel and sel[1] is not None else None
    svg, boxes = overlay_svg(
        page.get("elements", []),
        _SCALE,
        size[0],
        size[1],
        highlight_idx,
        highlight_cell,
    )
    session["p3_boxes"] = boxes
    return svg


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="PDF Translator", theme=gr.themes.Default(), css=REVIEW_CSS
    ) as demo:
        sess_state = gr.State({})
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
                e2e_btn = gr.Button(
                    "⚡ Chạy end-to-end (bỏ qua sửa)", variant="secondary"
                )
            with gr.Column(scale=2):
                pdf_in = PDF(label="Tải lên PDF", height=520)
                status = gr.Markdown("")
                e2e_pdf = PDF(label="Bản dịch (end-to-end)", height=520, visible=False)
                e2e_download = gr.File(label="Tải PDF (end-to-end)", visible=False)

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
                    with gr.Column(elem_id="p1-stage"):
                        p1_img = gr.Image(
                            interactive=False,
                            show_label=False,
                            container=False,
                            elem_id="p1-img",
                        )
                        p1_overlay = gr.HTML(
                            elem_id="p1-overlay", container=False, padding=False
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
                    p1_source = gr.Textbox(label=_P1_SOURCE_LABEL, lines=4)
                    p1_bypass = gr.Checkbox(label="Bỏ qua element này (không dịch)")
                    p1_save_btn = gr.Button("💾 Lưu element", variant="secondary")

                    gr.Markdown("**Thêm box cho vùng bị sót**")
                    p1_draw_mode = gr.Checkbox(
                        label="Chế độ vẽ box (nhấn 2 điểm trên ảnh)",
                        elem_id="p1-draw-mode",
                    )
                    with gr.Row():
                        p1_x0 = gr.Number(label="x0", value=0, elem_id="p1-x0")
                        p1_y0 = gr.Number(label="y0", value=0, elem_id="p1-y0")
                        p1_x1 = gr.Number(label="x1", value=0, elem_id="p1-x1")
                        p1_y1 = gr.Number(label="y1", value=0, elem_id="p1-y1")
                    p1_new_label = gr.Dropdown(
                        LABEL_CHOICES, label="Nhãn box mới", value="Text"
                    )
                    p1_new_source = gr.Textbox(label="Văn bản box mới", lines=3)
                    p1_new_auto_color = gr.Checkbox(
                        label="Tự động lấy màu nền/chữ từ trang", value=True
                    )
                    with gr.Row():
                        p1_new_bg = gr.ColorPicker(label="Màu nền", value="#ffffff")
                        p1_new_text = gr.ColorPicker(label="Màu chữ", value="#000000")
                    p1_add_btn = gr.Button("➕ Thêm box", elem_id="p1-add-btn")
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
                    with gr.Column(elem_id="p3-stage"):
                        p3_img = gr.Image(
                            interactive=False,
                            show_label=False,
                            container=False,
                            elem_id="p3-img",
                        )
                        p3_overlay = gr.HTML(
                            elem_id="p3-overlay", container=False, padding=False
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

        def on_new_file(session):
            """Reset all derived state when the uploaded PDF changes/clears.

            Drops the previous file's work dir + session keys and hides every
            review panel / download so a new file never shows the old one's
            outputs.
            """
            old = session.get("work_dir")
            if old:
                shutil.rmtree(old, ignore_errors=True)
            for k in (
                "pdf_path",
                "work_dir",
                "pages",
                "parsed",
                "p1_page",
                "p1_sel",
                "p1_size",
                "p1_boxes",
                "translated",
                "out_path",
                "p3_page",
                "p3_sel",
                "p3_size",
                "p3_boxes",
            ):
                session.pop(k, None)
            return {
                sess_state: session,
                status: "",
                p1_group: gr.update(visible=False),
                p3_group: gr.update(visible=False),
                p3_pdf: gr.update(value=None),
                download: gr.update(value=None),
                e2e_pdf: gr.update(value=None, visible=False),
                e2e_download: gr.update(value=None, visible=False),
            }

        # ---- End-to-end (skip the per-phase review) ------------------------
        def do_e2e(pdf, prov, key, mdl, lfrom, lto, fnt, mode, frm, to):
            req = TranslationRequest(
                pdf_path=pdf,
                provider=PROVIDER_KEY[prov],
                api_key=key or "",
                model=mdl or None,
                src_lang=lfrom,
                tgt_lang=lto,
                font=fnt,
                pages=resolve_pages(mode, frm, to),
            )
            # Hide any PDF/download left over from a previous successful run so a
            # failure never displays a stale result as if it were the new output.
            hide = {
                e2e_pdf: gr.update(visible=False),
                e2e_download: gr.update(visible=False),
            }
            err = validate(req)
            if err:
                yield {status: f"⚠️ {err}", **hide}
                return
            t0 = time.perf_counter()
            yield {status: "⏳ Đang chạy end-to-end (Phase 1 → 2 → 3)…", **hide}
            result = None
            for ev in stream_translation(req):
                if isinstance(ev, Progress):
                    yield {status: f"⏳ {ev.msg}"}
                else:
                    result = ev
            if result is None or result.status != "ok":
                yield {
                    status: f"❌ {result.detail if result else 'Không rõ lỗi.'}",
                    **hide,
                }
                return
            elapsed = time.perf_counter() - t0
            yield {
                status: f"✅ Xong end-to-end trong {elapsed:.1f}s (xem log để biết chi tiết từng phase).",
                e2e_pdf: gr.update(value=result.out_path, visible=True),
                e2e_download: gr.update(value=result.out_path, visible=True),
            }

        # ---- Step 1: parse -------------------------------------------------
        def do_parse(session, pdf, prov, key, lfrom, lto, mode, frm, to):
            if not pdf:
                yield {status: "⚠️ Vui lòng tải lên một file PDF."}
                return
            if not key or not key.strip():
                yield {status: "⚠️ Thiếu API key."}
                return

            prev = session.get("work_dir")
            if prev:
                shutil.rmtree(prev, ignore_errors=True)
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
            img = _base_p1(session)
            first_page = session["parsed"]["pages"][0]
            yield {
                sess_state: session,
                status: "✅ Phase 1 xong — kiểm tra & sửa rồi bấm Xác nhận.",
                p1_group: gr.update(visible=True),
                p1_page_dd: gr.update(choices=page_choices, value=0),
                p1_img: img,
                p1_overlay: _overlay_p1(session),
                p1_elem_dd: gr.update(choices=_elem_choices(first_page), value=None),
                # Re-show label/bypass in case a table cell was selected in a
                # previous file (which hides them) before this parse ran.
                p1_label: gr.update(value=None, visible=True),
                p1_source: gr.update(value="", label=_P1_SOURCE_LABEL),
                p1_bypass: gr.update(value=False, visible=True),
            }

        def on_p1_page(session, page_i):
            if page_i is None:
                return {sess_state: session}
            session["p1_page"] = int(page_i)
            session["p1_sel"] = None
            page = session["parsed"]["pages"][int(page_i)]
            return {
                sess_state: session,
                p1_img: _base_p1(session),
                p1_overlay: _overlay_p1(session),
                p1_elem_dd: gr.update(choices=_elem_choices(page), value=None),
                p1_label: gr.update(value=None, visible=True),
                p1_source: gr.update(value="", label=_P1_SOURCE_LABEL),
                p1_bypass: gr.update(value=False, visible=True),
            }

        def _p1_select_elem(session, elem_i):
            """Populate the edit panel for a whole element + highlight it."""
            session["p1_sel"] = (elem_i, None)
            page = session["parsed"]["pages"][session["p1_page"]]
            elem = page["elements"][elem_i]
            return {
                sess_state: session,
                p1_overlay: _overlay_p1(session),
                p1_elem_dd: gr.update(value=elem_i),
                p1_label: gr.update(value=elem.get("label"), visible=True),
                p1_source: gr.update(
                    value=elem.get("source_text", ""), label=_P1_SOURCE_LABEL
                ),
                p1_bypass: gr.update(
                    value=elem.get("category") == "BYPASS", visible=True
                ),
                status: "",
            }

        def _p1_select_cell(session, elem_i, cell_i):
            """Populate the edit panel for one TABLE cell + highlight it.

            Cells have no label/category of their own (translation runs per
            cell), so the label/bypass controls are hidden — only the cell's
            OCR text is editable.
            """
            session["p1_sel"] = (elem_i, cell_i)
            page = session["parsed"]["pages"][session["p1_page"]]
            cell = page["elements"][elem_i]["cells"][cell_i]
            return {
                sess_state: session,
                p1_overlay: _overlay_p1(session),
                p1_elem_dd: gr.update(value=elem_i),
                p1_label: gr.update(visible=False),
                p1_source: gr.update(
                    value=cell.get("source_text", ""),
                    label=f"Văn bản gốc (Table #{elem_i}, cell #{cell_i})",
                ),
                p1_bypass: gr.update(visible=False),
                status: "",
            }

        def on_p1_img_click(session, draw_mode, evt: gr.SelectData):
            # In draw mode the box is drawn client-side (see DRAW_JS), so a stray
            # click here must not also select an element.
            if draw_mode:
                return {sess_state: session}
            pt = normalize_click(evt.index)
            if pt is None:
                return {sess_state: session}
            hit = hit_test(session.get("p1_boxes", []), pt[0], pt[1])
            if hit is None:
                return {sess_state: session}
            elem_i, cell_i = hit
            if cell_i is None:
                return _p1_select_elem(session, elem_i)
            return _p1_select_cell(session, elem_i, cell_i)

        def on_p1_pick(session, elem_i):
            # Dropdown selects whole elements only — table cells are picked by
            # clicking them directly on the image.
            if elem_i is None:
                return {sess_state: session}
            return _p1_select_elem(session, int(elem_i))

        def do_p1_save(session, label, source, bypass):
            sel = session.get("p1_sel")
            if sel is None:
                gr.Warning("Chưa chọn element nào.")
                return {sess_state: session, status: "⚠️ Chưa chọn element nào."}
            elem_i, cell_i = sel
            if cell_i is not None:
                msg = apply_phase1_cell_edit(
                    session["parsed"], session["p1_page"], elem_i, cell_i, source
                )
                if msg:
                    gr.Warning(msg)
                else:
                    gr.Info(f"Đã lưu cell #{cell_i} (Table #{elem_i}).")
                return {
                    sess_state: session,
                    p1_overlay: _overlay_p1(session),
                    status: (
                        f"⚠️ {msg}"
                        if msg
                        else f"✅ Đã lưu cell #{cell_i} (Table #{elem_i})."
                    ),
                }
            msg = apply_phase1_edit(
                session["parsed"], session["p1_page"], elem_i, label, source, bypass
            )
            page = session["parsed"]["pages"][session["p1_page"]]
            if msg:
                gr.Warning(msg)
            else:
                gr.Info(f"Đã lưu element #{elem_i}.")
            return {
                sess_state: session,
                p1_overlay: _overlay_p1(session),
                p1_elem_dd: gr.update(choices=_elem_choices(page), value=elem_i),
                status: f"⚠️ {msg}" if msg else f"✅ Đã lưu element #{elem_i}.",
            }

        def do_p1_add(
            session, x0, y0, x1, y1, label, source, auto_color, bg_hex, text_hex
        ):
            if x1 <= x0 or y1 <= y0:
                gr.Warning("Box không hợp lệ (cần x1>x0, y1>y0).")
                return {
                    sess_state: session,
                    status: "⚠️ Box không hợp lệ (cần x1>x0, y1>y0).",
                }
            page_i = session["p1_page"]
            # auto_color → let the renderer sample bg/text from the page (old
            # default); unchecked → pin the user-picked colors.
            bg = None if auto_color else hex_to_rgb(bg_hex)
            text = None if auto_color else hex_to_rgb(text_hex)
            new_idx = add_element(
                session["parsed"],
                page_i,
                [x0, y0, x1, y1],
                label,
                source,
                bg_color=bg,
                text_color=text,
            )
            page = session["parsed"]["pages"][page_i]
            gr.Info(f"Đã thêm box #{new_idx}.")
            # Populate the panel exactly as if the new element had been clicked
            # (also re-shows label/bypass in case a table cell was selected
            # beforehand, which hides them), then layer on the add-specific
            # resets — including the dropdown's choices, which _p1_select_elem
            # doesn't refresh since it doesn't know an element was just added.
            out = _p1_select_elem(session, new_idx)
            out.update(
                {
                    p1_elem_dd: gr.update(choices=_elem_choices(page), value=new_idx),
                    p1_new_source: gr.update(value=""),
                    status: f"✅ Đã thêm box #{new_idx}.",
                    # Turn draw mode back off: leaving it checked would make the
                    # next click on the image be treated as drawing instead of
                    # selecting an element, silently blocking further edits.
                    p1_draw_mode: gr.update(value=False),
                }
            )
            return out

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
            img = _base_p3(session)
            first_page = session["translated"]["pages"][0]
            yield {
                sess_state: session,
                status: "✅ Đã dịch & dựng. Soát bản dịch rồi tải về.",
                p3_group: gr.update(visible=True),
                p3_pdf: session["out_path"],
                download: session["out_path"],
                p3_page_dd: gr.update(choices=page_choices, value=0),
                p3_img: img,
                p3_overlay: _overlay_p3(session),
                p3_elem_dd: gr.update(choices=_elem_choices(first_page), value=None),
            }

        def on_p3_page(session, page_i):
            if page_i is None:
                return {sess_state: session}
            session["p3_page"] = int(page_i)
            session["p3_sel"] = None
            page = session["translated"]["pages"][int(page_i)]
            return {
                sess_state: session,
                p3_img: _base_p3(session),
                p3_overlay: _overlay_p3(session),
                p3_elem_dd: gr.update(choices=_elem_choices(page), value=None),
                p3_source: gr.update(value=""),
                p3_translated: gr.update(value=""),
            }

        def _p3_select_elem(session, elem_i):
            session["p3_sel"] = (elem_i, None)
            page = session["translated"]["pages"][session["p3_page"]]
            elem = page["elements"][elem_i]
            return {
                sess_state: session,
                p3_overlay: _overlay_p3(session),
                p3_elem_dd: gr.update(value=elem_i),
                p3_source: gr.update(value=elem.get("source_text", "")),
                p3_translated: gr.update(value=elem.get("translated_text", "")),
                status: "",
            }

        def _p3_select_cell(session, elem_i, cell_i):
            session["p3_sel"] = (elem_i, cell_i)
            page = session["translated"]["pages"][session["p3_page"]]
            cell = page["elements"][elem_i]["cells"][cell_i]
            return {
                sess_state: session,
                p3_overlay: _overlay_p3(session),
                p3_elem_dd: gr.update(value=elem_i),
                p3_source: gr.update(value=cell.get("source_text", "")),
                p3_translated: gr.update(value=cell.get("translated_text", "")),
                status: "",
            }

        def on_p3_img_click(session, evt: gr.SelectData):
            pt = normalize_click(evt.index)
            if pt is None:
                return {sess_state: session}
            hit = hit_test(session.get("p3_boxes", []), pt[0], pt[1])
            if hit is None:
                return {sess_state: session}
            elem_i, cell_i = hit
            if cell_i is None:
                return _p3_select_elem(session, elem_i)
            return _p3_select_cell(session, elem_i, cell_i)

        def on_p3_pick(session, elem_i):
            # Dropdown selects whole elements only — table cells are picked by
            # clicking them directly on the image.
            if elem_i is None:
                return {sess_state: session}
            return _p3_select_elem(session, int(elem_i))

        def do_p3_save(session, translated_text):
            sel = session.get("p3_sel")
            if sel is None:
                gr.Warning("Chưa chọn element nào.")
                return {sess_state: session, status: "⚠️ Chưa chọn element nào."}
            elem_i, cell_i = sel
            if cell_i is not None:
                apply_phase2_cell_edit(
                    session["translated"],
                    session["p3_page"],
                    elem_i,
                    cell_i,
                    translated_text,
                )
                gr.Info(
                    f"Đã lưu bản dịch cell #{cell_i} (Table #{elem_i}, "
                    "bấm Render lại để cập nhật)."
                )
            else:
                apply_phase2_edit(
                    session["translated"], session["p3_page"], elem_i, translated_text
                )
                gr.Info(f"Đã lưu bản dịch #{elem_i} (bấm Render lại để cập nhật).")
            return {
                sess_state: session,
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
                sess_state: session,
                status: "✅ Đã render lại.",
                p3_pdf: session["out_path"],
                download: session["out_path"],
                p3_img: _base_p3(session),
                p3_overlay: _overlay_p3(session),
            }

        # ================================================================== #
        # Wiring
        # ================================================================== #
        provider.change(on_provider_change, provider, [model, api_key])
        api_key.blur(on_load_models, [provider, api_key], model)
        load_models_btn.click(on_load_models, [provider, api_key], model)
        page_mode.change(on_page_mode, page_mode, [page_from, page_to])
        pdf_in.change(
            on_new_file,
            sess_state,
            [
                sess_state,
                status,
                p1_group,
                p3_group,
                p3_pdf,
                download,
                e2e_pdf,
                e2e_download,
            ],
        )

        parse_out = [
            sess_state,
            status,
            p1_group,
            p1_page_dd,
            p1_img,
            p1_elem_dd,
            p1_label,
            p1_source,
            p1_bypass,
            p1_overlay,
        ]
        parse_btn.click(
            do_parse,
            [
                sess_state,
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
        e2e_btn.click(
            do_e2e,
            [
                pdf_in,
                provider,
                api_key,
                model,
                lang_from,
                lang_to,
                font,
                page_mode,
                page_from,
                page_to,
            ],
            [status, e2e_pdf, e2e_download],
        )

        # Page change reloads the base <img>; edit events must NOT touch p1_img
        # (its presence in outputs makes Gradio flash a loading state over the
        # image on every click). show_progress="hidden" suppresses the same
        # pulse on the other edited components.
        p1_page_out = [
            sess_state,
            p1_img,
            p1_overlay,
            p1_elem_dd,
            p1_label,
            p1_source,
            p1_bypass,
        ]
        p1_edit_out = [
            sess_state,
            p1_overlay,
            p1_elem_dd,
            p1_label,
            p1_source,
            p1_bypass,
            p1_new_source,
            status,
        ]
        p1_page_dd.change(on_p1_page, [sess_state, p1_page_dd], p1_page_out)
        p1_img.select(
            on_p1_img_click,
            [sess_state, p1_draw_mode],
            p1_edit_out,
            show_progress="hidden",
        )
        p1_elem_dd.select(
            on_p1_pick, [sess_state, p1_elem_dd], p1_edit_out, show_progress="hidden"
        )
        p1_save_btn.click(
            do_p1_save,
            [sess_state, p1_label, p1_source, p1_bypass],
            p1_edit_out,
            show_progress="hidden",
        )
        p1_add_btn.click(
            do_p1_add,
            [
                sess_state,
                p1_x0,
                p1_y0,
                p1_x1,
                p1_y1,
                p1_new_label,
                p1_new_source,
                p1_new_auto_color,
                p1_new_bg,
                p1_new_text,
            ],
            p1_edit_out + [p1_draw_mode],
            show_progress="hidden",
        )

        confirm_out = [
            sess_state,
            status,
            p3_group,
            p3_pdf,
            download,
            p3_page_dd,
            p3_img,
            p3_elem_dd,
            p3_overlay,
        ]
        confirm_btn.click(
            do_confirm,
            [sess_state, provider, api_key, model, lang_from, lang_to, font],
            confirm_out,
        )

        # Same split as Phase 1: only page-change / re-render reload p3_img.
        p3_page_out = [
            sess_state,
            p3_img,
            p3_overlay,
            p3_elem_dd,
            p3_source,
            p3_translated,
        ]
        p3_edit_out = [
            sess_state,
            p3_overlay,
            p3_elem_dd,
            p3_source,
            p3_translated,
            status,
        ]
        p3_rerender_out = [
            sess_state,
            status,
            p3_pdf,
            download,
            p3_img,
            p3_overlay,
        ]
        p3_page_dd.change(on_p3_page, [sess_state, p3_page_dd], p3_page_out)
        p3_img.select(
            on_p3_img_click, [sess_state], p3_edit_out, show_progress="hidden"
        )
        p3_elem_dd.select(
            on_p3_pick, [sess_state, p3_elem_dd], p3_edit_out, show_progress="hidden"
        )
        p3_save_btn.click(
            do_p3_save, [sess_state, p3_translated], p3_edit_out, show_progress="hidden"
        )
        rerender_btn.click(do_rerender, [sess_state, font], p3_rerender_out)

        # fn must be explicit: Blocks.load()'s fn defaults to the sentinel string
        # "decorator" (for @demo.load()-style usage), not None — passing js= alone
        # silently never registers the trigger, so the script never runs.
        demo.load(fn=None, js=DRAW_JS)

    return demo
