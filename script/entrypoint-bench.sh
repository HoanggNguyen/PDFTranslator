#!/usr/bin/env bash
# =============================================================================
# ENTRYPOINT của image benchmark. Đặt thư mục cache rồi exec lệnh được truyền vào.
#
# VÌ SAO phải là ENTRYPOINT chứ không phải CMD như Dockerfile của Space demo:
# `hf jobs run <image> <command>` **thay CMD**. Dockerfile demo nhét toàn bộ logic
# chọn cache root vào CMD, nên mọi job đều chạy mà không có logic đó — MODEL_CACHE_DIR
# rơi về /app/.cache thay vì /data, và MỖI job tải lại 3–5 GB Surya/Paddle/RT-DETR
# trong khi bạn đang bị tính tiền theo phút. ENTRYPOINT thì không bị thay, nên cấu
# hình luôn được áp, dù job chạy lệnh gì.
# =============================================================================
set -e

if mkdir -p /data 2>/dev/null && [ -w /data ]; then
  CACHE_ROOT=/data                 # volume HF Jobs mount vào -> sống qua nhiều job
else
  CACHE_ROOT=/app/.cache           # không mount -> ephemeral, tải lại mỗi lần
  echo "[bench] CẢNH BÁO: /data không ghi được — model sẽ tải lại mỗi job."
fi

export MODEL_CACHE_DIR="$CACHE_ROOT/datalab/models"     # Surya
export PADDLE_PDX_CACHE_HOME="$CACHE_ROOT/paddlex"      # PaddleOCR table
export HF_HOME="$CACHE_ROOT/huggingface"                # Docling RT-DETR, CometKiwi
export TRANSFORMERS_CACHE="$HF_HOME"
export TYPST_PACKAGE_CACHE_PATH="$CACHE_ROOT/typst"
# BabelDOC ghim cache vào $HOME/.cache/babeldoc (const.py, không có env override),
# nên phải dời cả HOME thì asset ONNX + font của nó mới nằm trên volume.
export HOME="$CACHE_ROOT/home"

mkdir -p "$MODEL_CACHE_DIR" "$PADDLE_PDX_CACHE_HOME" "$HF_HOME" \
         "$TYPST_PACKAGE_CACHE_PATH" "$HOME"

echo "[bench] cache root = $CACHE_ROOT | accelerator = ${ACCELERATOR:-?} | job = ${JOB_ID:-local}"
exec "$@"
