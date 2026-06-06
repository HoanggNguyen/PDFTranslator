# syntax=docker/dockerfile:1
# E2E PDF translator (OCR -> Translate -> Render) for a GPU Hugging Face Space.
# Base: CUDA 13 runtime + Ubuntu 22.04 (Python 3.10) — the config that builds cleanly.
# Only surya-ocr/paddleocr are pinned (in requirements.txt); torch/paddle/numpy stay
# unpinned so pip resolves a mutually compatible CUDA stack. T4 = Turing sm_75 (OK on CUDA 13).
# NOTE: if this tag 404s at build, pick an existing one from
#   https://hub.docker.com/r/nvidia/cuda/tags  (e.g. 13.0.1-cudnn-runtime-ubuntu22.04).
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_DEVICE=cuda \
    TYPST_BIN=typst \
    PDF2ZH_FONT_DIR=/app/fonts \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TYPST_PACKAGE_CACHE_PATH=/app/.cache/typst \
    MODEL_CACHE_DIR=/app/.cache/datalab/models \
    PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex

WORKDIR /app
EXPOSE 7860

# ── System deps ───────────────────────────────────────────────────────────────
#  - python 3.10 + pip (Ubuntu 22.04 default)
#  - OpenCV / PyMuPDF runtime libs (libgl1, libglib2.0-0, ...)
#  - fonts: Noto Sans/Serif + Noto CJK (covers Vietnamese + CJK), fontconfig
#  - wget/xz to fetch the typst binary
RUN apt-get update && apt-get install --no-install-recommends -y \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
        fontconfig fonts-noto-core fonts-noto-cjk \
        wget xz-utils ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ── Typst binary ────────────────────────────────────────────────────────────────
ARG TYPST_VERSION=v0.14.2
RUN wget -qO /tmp/typst.tar.xz \
        "https://github.com/typst/typst/releases/download/${TYPST_VERSION}/typst-x86_64-unknown-linux-musl.tar.xz" && \
    tar -xJf /tmp/typst.tar.xz -C /tmp && \
    install -m 0755 /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst && \
    rm -rf /tmp/typst* && typst --version

# ── Extra fonts (Be Vietnam Pro — open-source Google Font) ───────────────────────
RUN mkdir -p /app/fonts && \
    for w in Regular Bold Italic; do \
        wget -qO "/app/fonts/BeVietnamPro-${w}.ttf" \
          "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-${w}.ttf" || true; \
    done && \
    # also surface the system Noto fonts to the typst --font-path dir
    cp -n /usr/share/fonts/truetype/noto/*.ttf /app/fonts/ 2>/dev/null || true && \
    cp -n /usr/share/fonts/opentype/noto/*.otf /app/fonts/ 2>/dev/null || true && \
    fc-cache -f

# ── Python deps ──────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────────
# Running from /app puts the pdf2zh package on sys.path, so no editable install is
# needed (and it avoids pulling pyproject's heavier optional deps like babeldoc).
COPY . .

# ── Pre-cache typst packages (cmarker + mitex) so runtime needs no network ─────────
RUN mkdir -p /app/.cache/typst && \
    printf '#import "@preview/cmarker:0.1.8"\n#import "@preview/mitex:0.2.6": *\n#cmarker.render("ok")\n' \
        > /tmp/warm.typ && \
    typst compile /tmp/warm.typ /tmp/warm.pdf || echo "typst package pre-cache skipped"

# OCR models (~3-5GB) are NOT baked in — they download on the first request:
#   - Surya (layout/detection/recognition) -> Datalab's servers, cached in MODEL_CACHE_DIR
#   - Paddle table-cell model              -> PaddleX model server, cached in PADDLE_PDX_CACHE_HOME
# Make caches writable in case the Space runs the container as a non-root user.
RUN mkdir -p /app/.cache/datalab/models /app/.cache/paddlex /app/.cache/huggingface \
        /app/.cache/typst && chmod -R 777 /app/.cache

# Entrypoint: prefer HF Persistent Storage (/data) for the model caches so they
# survive sleep/restart and download only once. Falls back to /app/.cache (ephemeral)
# when persistent storage is not enabled.
RUN cat > /usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -e
if mkdir -p /data 2>/dev/null && [ -w /data ]; then
  CACHE_ROOT=/data
else
  CACHE_ROOT=/app/.cache
fi
export MODEL_CACHE_DIR="$CACHE_ROOT/datalab/models"
export PADDLE_PDX_CACHE_HOME="$CACHE_ROOT/paddlex"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface"
mkdir -p "$MODEL_CACHE_DIR" "$PADDLE_PDX_CACHE_HOME" "$HF_HOME"
echo "[entrypoint] model cache root = $CACHE_ROOT"
exec python3 app.py
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

CMD ["/usr/local/bin/entrypoint.sh"]
