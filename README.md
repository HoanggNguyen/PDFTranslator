---
title: PDF Translator
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# PDF Translator — End-to-End (OCR → Translate → Render)

A single Gradio app that runs the full document-translation pipeline and rebuilds
a layout-faithful PDF in the target language. It merges three phases into one
end-to-end flow:

| Phase | Package | What it does |
|-------|---------|--------------|
| **1 · OCR** | [`pdf2zh/scanned`](pdf2zh/scanned) | Layout detection + OCR (Surya) and table cells (PaddleOCR) → a structured `ParsedDocument` (JSON). |
| **2 · Translate** | [`pdf2zh/translation`](pdf2zh/translation) | Async, chunked LLM translation with glossary, math-fix and TOC-fix passes. |
| **3 · Render** | [`pdf2zh/render`](pdf2zh/render) | Rebuilds the PDF with Typst, overlaying translated text on the original layout. |

**Entry point:** [`app.py`](app.py) · **Orchestration:** [`pdf2zh/e2e.py`](pdf2zh/e2e.py)
(`run_pipeline()` chains Phase 1 → 2 → 3).

The UI mirrors PDFMathTranslate: a left sidebar to pick a translation provider and
enter credentials, a main area to upload a PDF and preview/download the result.

---

## How it works

```
app.py  (Gradio UI, sync handler + gr.Progress + demo.queue)
   │
   ▼
pdf2zh/e2e.py :: run_pipeline(pdf, src, tgt, provider, api_key, model, pages, font, work_dir)
   ├─ Phase 1  get_parser().parse_pdf(...)         # StageAParser, models loaded once (singleton)
   ├─ Phase 2  translate_document(parsed, TranslatorConfig)
   └─ Phase 3  render_document(pdf, translated, out.pdf, RenderConfig)
```

- **Model singleton** — `StageAParser` loads ~3–5 GB of OCR weights exactly once
  (`warmup()` runs at startup), not per request.
- **Providers** — OpenRouter, Gemini, OpenAI. The user supplies their **own API key**
  in the UI (nothing is stored server-side).
- **Fonts** — selectable in the UI. The chosen font heads a multilingual fallback
  chain (Noto Sans / Noto Serif / Noto CJK) so missing glyphs degrade gracefully.

---

## Requirements

- **GPU strongly recommended.** OCR (Surya + PaddleOCR + Torch) is slow on CPU.
- **`typst` binary** on `PATH` (installed in the Docker image).
- Fonts with the needed coverage (bundled in the image; Noto fonts cover Vietnamese).
- Python 3.10–3.12. Deps in [`requirements.txt`](requirements.txt).

---

## Deploy to a Hugging Face Space (one time)

1. Create a new Space → **SDK = Docker** → push this repo (the `Dockerfile` and the
   YAML front-matter above configure the Space automatically).
2. **Settings → Hardware → T4 small (GPU)**. The build takes a few minutes (the OCR
   models are *not* baked in — they download on first use, see below).
3. **Settings → Persistent Storage** → enable a small tier. The container’s
   entrypoint points the model caches (`MODEL_CACHE_DIR`, `PADDLE_PDX_CACHE_HOME`,
   `HF_HOME`) at `/data`, so the ~3–5 GB of OCR weights download **once** and survive
   sleep/restart. Without it, the weights re-download on every cold start.
4. Open the Space — the app is ready. **No extra configuration**; users paste their
   own API key in the sidebar.
5. To control cost: set **auto-sleep** (Settings → Sleep time) so the GPU is released
   when idle, or **Pause** the Space manually. Billing stops while sleeping/paused.

---

## Using the app

Drop in a PDF → in the sidebar pick a **Provider** (OpenRouter / Gemini / OpenAI),
paste your **API key**, choose **languages**, **output font**, and **page range** →
click **Translate** → preview and download the translated PDF.

---

## Run locally

```bash
# Docker (with GPU) — closest to the Space environment
cp .env.example .env
docker build -t pdf2zh .
docker run --gpus all -p 7860:7860 pdf2zh        # open http://localhost:7860

# Or run directly (needs typst on PATH + a GPU/CPU torch install)
pip install -r requirements.txt
cp .env.example .env
python app.py
```

---

## Testing

```bash
# 1. Cheap import check (does NOT load models)
python -c "import pdf2zh.e2e; print('e2e import OK')"

# 2. End-to-end, headless, single page (exercises all 3 phases)
python - <<'PY'
from pdf2zh.e2e import run_pipeline
out = run_pipeline(
    pdf_path="test/file/translate.cli.plain.text.pdf",
    src_lang="English", tgt_lang="Vietnamese",
    provider="openrouter", api_key="<YOUR_KEY>",
    model=None, pages=[0], font="Noto Sans", work_dir="/tmp/e2e_test",
    progress=lambda f, m: print(f"{f:.0%} {m}"),
)
print("OUTPUT:", out)
PY

# 3. Per-phase isolation (when E2E fails)
python test/verify_translate.py --input test_local/output_math.json \
       --provider openrouter --api-key "$OPENROUTER_API_KEY" --src English --tgt Vietnamese
python test/verify_render.py --input <source.pdf> \
       --parsed test_local/output_math.translated.json --output /tmp/render_test.pdf
```

Tips: the first run downloads the OCR models (~3–5 GB); use a single page while
iterating to keep API cost and latency low.

---

## Customizing

- **Add a font** — drop a `.ttf/.otf` into `/app/fonts` (see [`Dockerfile`](Dockerfile))
  and add its family name to `BUNDLED_FONTS` in [`pdf2zh/e2e.py`](pdf2zh/e2e.py).
- **Add a provider** — add an entry to `PROVIDERS` in
  [`pdf2zh/translation/config.py`](pdf2zh/translation/config.py) and to `PROVIDER_KEY`
  in [`app.py`](app.py).
- **Change page limit / default language / default font** — edit the constants at the
  top of [`app.py`](app.py) and [`pdf2zh/e2e.py`](pdf2zh/e2e.py).

---

## Known limitations

- Equation elements without `equation_words` are rendered as-is (not translated) —
  acceptable for the demo.
- A single T4 (16 GB) can OOM on large PDFs; the UI caps custom page counts and
  serializes requests via `demo.queue()`. Reduce the page range if needed.

---

## License

This project builds on [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
(AGPL-3.0). See [`LICENSE`](LICENSE).
