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

A single Gradio app that runs a full document-translation pipeline and rebuilds a
**layout-faithful PDF** in the target language. It chains three phases into one
end-to-end flow:

| Phase | Package | What it does |
|-------|---------|--------------|
| **1 · Parse / OCR** | [`pdf2zh/parser`](pdf2zh/parser) | Layout detection + OCR (Surya) and table cells (PaddleOCR) → a structured `ParsedDocument` (JSON). |
| **2 · Translate** | [`pdf2zh/translation`](pdf2zh/translation) | Async, chunked LLM translation with glossary, math-fix, TOC-fix and vision passes. |
| **3 · Render** | [`pdf2zh/render`](pdf2zh/render) | Rebuilds the PDF with **Typst**, overlaying translated text on the original layout. |

- **Entry point:** [`app.py`](app.py) → warms up models, then launches the Gradio UI ([`pdf2zh/webapp/ui.py`](pdf2zh/webapp/ui.py)).
- **Orchestration:** [`pdf2zh/e2e.py`](pdf2zh/e2e.py) — `run_pipeline()` chains Phase 1 → 2 → 3.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Prerequisites (exact versions)](#prerequisites-exact-versions)
3. [Quick start — Docker (recommended, closest to production)](#quick-start--docker-recommended-closest-to-production)
4. [Run locally without Docker (personal GPU / laptop)](#run-locally-without-docker-personal-gpu--laptop)
5. [Configuration reference (`.env`)](#configuration-reference-env)
6. [Model downloads & caching](#model-downloads--caching)
7. [Using the web app](#using-the-web-app)
8. [Command-line & programmatic use](#command-line--programmatic-use)
9. [Testing](#testing)
10. [Deploy to a Hugging Face Space](#deploy-to-a-hugging-face-space)
11. [Troubleshooting](#troubleshooting)
12. [Customizing](#customizing)
13. [Known limitations](#known-limitations)
14. [License](#license)

---

## How it works

```
app.py  (Gradio UI, warmup() on boot, demo.queue serializes requests)
   │
   ▼
pdf2zh/e2e.py :: run_pipeline(pdf_path, src_lang, tgt_lang, provider,
                              api_key, model, pages, font, work_dir, progress)
   ├─ Phase 1  get_parser().parse_pdf(...)         # StageAParser — models loaded ONCE (singleton)
   │              └─ writes work_dir/phase1_parsed.json
   ├─ Phase 2  translate_document(parsed_dict, TranslatorConfig)
   │              └─ writes work_dir/phase2_translated.json
   └─ Phase 3  render_document(pdf_path, translated_dict, out.pdf, RenderConfig)
                  └─ shells out to the `typst` binary → translated_<hash>.pdf
```

- **Model singleton** — `StageAParser` loads ~3–5 GB of OCR weights exactly once
  (`warmup()` at startup), not per request. See [`pdf2zh/e2e.py`](pdf2zh/e2e.py).
- **Providers** — OpenRouter, Gemini, OpenAI, DeepSeek, MiniMax, Anthropic, LiteLLM.
  The user supplies their **own API key** in the UI; nothing is stored server-side.
- **Fonts** — the chosen font heads a multilingual fallback chain
  (Noto Sans / Noto Serif / Noto CJK / Be Vietnam Pro) so missing glyphs degrade
  gracefully. The default Helvetica lacks Vietnamese glyphs and is always overridden.

---

## Prerequisites (exact versions)

Reproducing this project reliably means matching the following stack. Deviating
(especially on the OCR/GPU pins) is the most common cause of a broken build.

| Component | Version / constraint | Notes |
|-----------|----------------------|-------|
| **Python** | `>=3.10, <3.13` | 3.10 / 3.11 / 3.12 only. Set in [`pyproject.toml`](pyproject.toml). |
| **Typst** | `v0.14.2` binary on `PATH` | Phase 3 shells out to it. Later 0.x may work but is untested. |
| **CUDA (GPU path)** | **13.x** runtime + driver | Docker base = `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04`. |
| **surya-ocr** | `==0.17.1` (pinned) | 0.18+ dropped the `settings.*_BATCH_SIZE` API used by `hardware.py`. |
| **transformers** | `==4.56.1` (pinned) | Matches surya-ocr 0.17.1. |
| **paddleocr** | `==3.6.0` (pinned) | Table-cell recognition. |
| **paddlepaddle-gpu** | `==3.3.1` (cu130) | Installed from the `cu130` extra index (see `requirements.txt`). |
| **torch / torchvision / numpy** | unpinned | Left to pip so it resolves a CUDA stack compatible with paddle. |
| **Fonts** | Noto Sans, Noto Serif, Noto CJK, Be Vietnam Pro | Must be visible to Typst (bundled in the Docker image). |

> **GPU is strongly recommended.** Phase 1 (Surya + PaddleOCR + Torch) is slow on CPU.
> A single 16 GB T4 works for small page ranges; an A100 (80 GB) handles the full
> batch-size settings in `.env.example`. On Apple Silicon the parser auto-selects
> the `mps` device with reduced batch sizes.

---

## Quick start — Docker (recommended, closest to production)

Docker is the only path that pins **every** system dependency (CUDA, Typst, fonts,
locale). Use it for the most reproducible result.

```bash
# 1. Clone
git clone https://github.com/HoanggNguyen/PDFTranslator.git
cd PDFTranslator

# 2. Seed the environment file (the container also does this, but do it locally too)
cp .env.example .env

# 3. Build the image (installs CUDA deps, Typst v0.14.2, fonts, Python deps)
docker build -t pdf2zh .

# 4a. Run WITH a GPU (requires the NVIDIA Container Toolkit on the host)
docker run --gpus all -p 7860:7860 pdf2zh

# 4b. Run WITHOUT a GPU (CPU only — much slower, fine for testing wiring)
docker run -p 7860:7860 -e DEVICE=cpu pdf2zh

# 5. Open the app
#    http://localhost:7860
```

**Persisting model weights across restarts** (avoid re-downloading ~3–5 GB):

```bash
docker run --gpus all -p 7860:7860 \
  -v "$PWD/.model-cache:/data" \
  pdf2zh
```

The container entrypoint prefers `/data` for all model caches when it is writable
(see the `entrypoint.sh` block in the [`Dockerfile`](Dockerfile)); mounting a host
volume there makes the weights survive container restarts.

> **NVIDIA Container Toolkit** is required for `--gpus all`. Install it on the host
> first: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>.
> Verify with `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi`.

---

## Run locally without Docker (personal GPU / laptop)

Use this when you want to develop against the code directly. You are responsible
for three system dependencies that Docker would otherwise provide: **Python 3.10–3.12**,
the **Typst binary**, and **fonts**.

### 1. Clone and create an isolated environment

```bash
git clone https://github.com/HoanggNguyen/PDFTranslator.git
cd PDFTranslator

python3.12 -m venv .venv          # any 3.10–3.12 interpreter
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install the Typst binary (v0.14.2)

Phase 3 calls the `typst` executable. It must be on `PATH` (or point `TYPST_BIN`
at it).

```bash
# macOS (Homebrew)
brew install typst          # then verify the version is 0.14.x
typst --version

# Linux (x86_64) — pinned release, matches the Docker image
wget -qO /tmp/typst.tar.xz \
  "https://github.com/typst/typst/releases/download/v0.14.2/typst-x86_64-unknown-linux-musl.tar.xz"
tar -xJf /tmp/typst.tar.xz -C /tmp
sudo install -m 0755 /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst
typst --version

# Any platform (Cargo)
cargo install typst-cli --locked
```

### 3. Install fonts (Vietnamese + CJK coverage)

Typst renders with whatever fonts it can find. Install Noto (covers Vietnamese and
CJK) and optionally Be Vietnam Pro, then confirm Typst sees them:

- **Linux:** `sudo apt-get install -y fonts-noto-core fonts-noto-cjk && fc-cache -f`
- **macOS:** install the Noto families (e.g. via Homebrew casks or Google Fonts).
- Alternatively, drop `.ttf/.otf` files into a directory and set `PDF2ZH_FONT_DIR`
  to it; the app passes that directory to Typst's `--font-path`.

```bash
typst fonts | grep -i noto     # should list Noto families
```

### 4. Install Python dependencies

```bash
# GPU stack (Linux + CUDA 13) — installs paddlepaddle-gpu (cu130) via the extra index
pip install -r requirements.txt

# CPU / macOS: requirements.txt targets a CUDA GPU. On a machine without a
# CUDA GPU, edit requirements.txt to drop the `paddlepaddle-gpu` line and the
# cu130 --extra-index-url, and install the CPU wheel instead:
#   pip install paddlepaddle==3.3.1
# then: pip install -r requirements.txt
```

### 5. Configure and run

```bash
cp .env.example .env            # then edit as needed (see Configuration reference)
python app.py                   # warms up models, serves http://localhost:7860
```

The first launch downloads the OCR weights (~3–5 GB) — see the next section.

---

## Configuration reference (`.env`)

`.env` is **gitignored**; [`.env.example`](.env.example) is the tracked template —
always `cp .env.example .env` after cloning. Values are read by
[`pdf2zh/config.py`](pdf2zh/config.py) (`Settings`, via `pydantic-settings`) and
consumed by the Phase-1 parser.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` (→ CUDA if a GPU is present, else MPS, else CPU). |
| `PAGE_BATCH_SIZE` | *(unset)* | Pages processed per batch. Leave unset on small GPUs. |
| `LAYOUT_BATCH_SIZE` | *(unset)* | Surya layout batch. |
| `DETECTION_BATCH_SIZE` | *(unset)* | Surya text-detection batch. |
| `OCR_BATCH_SIZE` | *(unset)* | Surya recognition batch (heaviest on VRAM). |
| `TABLE_BATCH_SIZE` | *(unset)* | Paddle table-cell batch. |
| `DETECTOR_BLANK_THRESHOLD` | `0.5` | OCR accuracy tuning (not VRAM related). |
| `DETECTOR_TEXT_THRESHOLD` | `0.6` | Must be **>** the blank threshold. |

**Batch-size guidance** (from `.env.example`):
- **Large GPU (A100 80 GB):** the values in `.env.example` (OCR 512, layout/detection 64,
  page 32, table 512) peak around 45–55 GB VRAM.
- **Small GPU (T4 16 GB):** leave every batch size **unset (empty)** so Surya picks
  safe defaults and avoids OOM.
- Unset values fall back to per-device defaults in
  [`pdf2zh/parser/utils/hardware.py`](pdf2zh/parser/utils/hardware.py).

### Provider / API keys

You do **not** put translation API keys in `.env` for normal use — they are entered
in the web UI per request and never stored. For **headless/CLI** runs you may set the
provider's env var instead of passing `--api-key`:

| Provider (UI label) | Key (config) | Env var | Default model |
|---------------------|--------------|---------|---------------|
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `google/gemini-3.1-flash-lite` |
| Gemini | `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash-lite` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` | `MiniMax-Text-01` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` |
| LiteLLM | `litellm` | `LITELLM_API_KEY` (+ `LITELLM_BASE_URL`) | proxy-routed |

Defined in [`pdf2zh/translation/config.py`](pdf2zh/translation/config.py) (`PROVIDERS`).

---

## Model downloads & caching

The OCR weights are **not** bundled — they download on the **first request** and are
then cached. Point these env vars at a persistent, writable directory to download
them only once:

| Env var | What it caches |
|---------|----------------|
| `MODEL_CACHE_DIR` | Surya layout / detection / recognition models (Datalab). |
| `PADDLE_PDX_CACHE_HOME` | Paddle table-cell model (PaddleX). |
| `HF_HOME` / `TRANSFORMERS_CACHE` | Hugging Face / transformers assets. |

In Docker these default to `/app/.cache/*` and, when Hugging Face Persistent Storage
(or a mounted `-v host:/data`) is available, to `/data/*` (handled by the entrypoint).
Locally they default to the standard per-tool locations unless you export them, e.g.:

```bash
export MODEL_CACHE_DIR="$HOME/.cache/pdf2zh/datalab"
export PADDLE_PDX_CACHE_HOME="$HOME/.cache/pdf2zh/paddlex"
export HF_HOME="$HOME/.cache/pdf2zh/huggingface"
```

> First run also warms the Typst package cache (`cmarker`, `mitex`). In Docker this
> is pre-baked; locally Typst fetches them once from `@preview` (needs network on
> first render).

---

## Using the web app

1. Upload a PDF in the main panel.
2. In the sidebar pick a **Provider**, paste your **API key** (optionally click
   *Load models* to fetch the model list, or type a model name).
3. Choose **source / target language**, **output font**, and **page range**
   (All / First page / First 5 / First N — capped at 50 pages per request to guard
   against OOM).
4. Click **Translate**. A modal streams per-phase progress.
5. Preview and download the translated PDF.

---

## Command-line & programmatic use

Useful for automation, batch jobs, and debugging a single phase in isolation.

### End-to-end (all three phases)

```python
from pdf2zh.e2e import run_pipeline

out = run_pipeline(
    pdf_path="test/file/translate.cli.plain.text.pdf",
    src_lang="English", tgt_lang="Vietnamese",
    provider="openrouter", api_key="<YOUR_KEY>",
    model=None,                 # None → provider default
    pages=[0],                  # 0-based list, or None for all pages
    font="Noto Sans",
    work_dir="/tmp/e2e_test",   # phase1/phase2 JSON + final PDF land here
    progress=lambda f, m: print(f"{f:.0%} {m}"),
)
print("OUTPUT:", out)
```

### Phase 2 only — translate an existing parsed JSON

```bash
python test/verify_translate.py \
  --input test_local/output_math.json \
  --provider openrouter --api-key "$OPENROUTER_API_KEY" \
  --src English --tgt Vietnamese
```

(The underlying CLI is [`pdf2zh/translation/cli.py`](pdf2zh/translation/cli.py):
`--provider`, `--model`, `--api-key`, `--concurrent`, `--chunk-bytes`,
`--no-glossary`, `--no-math-fix`, `--no-toc-fix`, …)

### Phase 3 only — render translated JSON back onto the original PDF

```bash
python -m pdf2zh.render \
  --pdf <source.pdf> \
  --parsed test_local/output_math.translated.json \
  --output /tmp/render_test.pdf \
  --font-family "Noto Sans" \
  --typst-bin typst
```

See [`pdf2zh/render/cli.py`](pdf2zh/render/cli.py) for all flags
(`--pages`, `--min-font`, `--no-redact`, `--keep-typst-source`, `--aggressive-compress`, …).

---

## Testing

```bash
# 0. Cheap import check (does NOT load models)
python -c "import pdf2zh.e2e; print('e2e import OK')"

# 1. Unit / integration tests (do not require a GPU or API key for most cases)
pytest -q

# 2. Phase-2 smoke (needs an API key)
python test/verify_translate.py --input test_local/output_math.json \
       --provider openrouter --api-key "$OPENROUTER_API_KEY" --src English --tgt Vietnamese

# 3. Phase-3 render feasibility / smoke
python test/verify_render.py --input <source.pdf> \
       --parsed test_local/output_math.translated.json --output /tmp/render_test.pdf
```

Tips: the first run downloads the OCR models (~3–5 GB); use a **single page** while
iterating to keep API cost and latency low.

---

## Customizing

- **Add a font** — drop a `.ttf/.otf` into the font directory (`PDF2ZH_FONT_DIR`,
  `/app/fonts` in Docker; see [`Dockerfile`](Dockerfile)) and add its family name to
  `BUNDLED_FONTS` in [`pdf2zh/e2e.py`](pdf2zh/e2e.py).
- **Add a provider** — add an entry to `PROVIDERS` in
  [`pdf2zh/translation/config.py`](pdf2zh/translation/config.py) and to `PROVIDER_KEY`
  in [`pdf2zh/webapp/config.py`](pdf2zh/webapp/config.py).
- **Change page limit / default language / default font** — edit the constants in
  [`pdf2zh/webapp/config.py`](pdf2zh/webapp/config.py) (`MAX_CUSTOM_PAGES`,
  `PAGE_PRESETS`) and [`pdf2zh/e2e.py`](pdf2zh/e2e.py) (`SUPPORTED_LANGUAGES`,
  `DEFAULT_FONT`).
- **Tune OCR batch sizes / device** — edit `.env` (see Configuration reference).

---

## Known limitations

- Equation elements without `equation_words` are rendered as-is (not translated).
- A single T4 (16 GB) can OOM on large PDFs; the UI caps custom page counts at 50 and
  serializes requests via `demo.queue()`. Reduce the page range if needed.
- Phase 3 depends on the external `typst` binary; a version mismatch can change layout.

---

## License

This project builds on [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
(AGPL-3.0). See [`LICENSE`](LICENSE).
