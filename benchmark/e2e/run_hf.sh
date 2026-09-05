#!/usr/bin/env bash
# =============================================================================
# Driver cho lượt chạy CHÍNH THỨC trên Hugging Face Jobs.
#
# Khác run_all.sh thế nào: run_all.sh là để debug ở máy. File này là lượt lấy số
# cho luận văn — ép ba hệ mã nguồn mở chạy TUẦN TỰ trên CÙNG một flavor, để cột
# giây/trang có nghĩa, rồi chấm điểm trong một job riêng.
#
# Vì sao tuần tự: hai job cùng lúc là hai job giành GPU và băng thông của nhau,
# và cột thời gian thành rác. Chậm hơn ~1 giờ, đổi lại là số đọc được.
#
# DeepL KHÔNG có ở đây — nó là API thuần, chạy ở máy (xem bước 4 bên dưới), rồi
# push lên cùng dataset repo trước khi phát job chấm điểm.
#
# YÊU CẦU: .env ở gốc repo đã điền theo benchmark/e2e/.env.bench.example, và
#          Space $HF_BENCH_SPACE đã build xanh từ Dockerfile.bench.
#
#   bash benchmark/e2e/run_hf.sh warm      # nung cache model vào /data, 1 lần
#   bash benchmark/e2e/run_hf.sh check     # kiểm image có đủ 3 hệ + dep chấm điểm
#   bash benchmark/e2e/run_hf.sh run       # 3 hệ, tuần tự
#   bash benchmark/e2e/run_hf.sh score     # detector + mọi metric + report
#   bash benchmark/e2e/run_hf.sh pull      # kéo kết quả về máy
# =============================================================================
set -u
cd "$(dirname "$0")/../.."   # repo root

[ -f .env ] && { set -a; . ./.env; set +a; }

ACTION=${1:-help}
TIERS=${TIERS:-T1}
LANGS=${LANGS:-vi}
FLAVOR=${FLAVOR:-t4-medium}          # MỘT flavor cho mọi hệ — đó là cả điểm của file này
FLAVOR_WARM=${FLAVOR_WARM:-t4-small}
TIMEOUT=${TIMEOUT:-3h}
SYSTEMS=${SYSTEMS:-"pdftranslator babeldoc pdfmathtranslate"}
QE_MODEL=${QE_MODEL:-Unbabel/wmt22-cometkiwi-da}
DETECTOR=${DETECTOR:-docling}

need() { [ -n "${!1:-}" ] || { echo "!! thiếu \$$1 — xem benchmark/e2e/.env.bench.example"; exit 1; }; }
need HF_TOKEN; need HF_EVAL_REPO; need HF_BENCH_SPACE
command -v hf >/dev/null 2>&1 || { echo "!! không thấy CLI 'hf' (pip install huggingface_hub)"; exit 1; }

IMG="hf.co/spaces/$HF_BENCH_SPACE"
CACHE_MOUNT=""
[ -n "${HF_CACHE_BUCKET:-}" ] && CACHE_MOUNT="-v hf-bucket://$HF_CACHE_BUCKET:/data:rw"
[ -n "${HF_CACHE_BUCKET:-}" ] || echo "!! CẢNH BÁO: chưa có HF_CACHE_BUCKET — mỗi job sẽ tải lại 3-5 GB model và bạn trả tiền cho thời gian đó."

# --secrets cho thứ bí mật, --env cho thứ không. Cả hai đều KHÔNG bake vào image.
COMMON_ENV=(
  --secrets "HF_TOKEN=$HF_TOKEN"
  --secrets "LITELLM_API_KEY=${LITELLM_API_KEY:-}"
  --env "LITELLM_BASE_URL=${LITELLM_BASE_URL:-}"
  --env "HF_EVAL_REPO=$HF_EVAL_REPO"
  --env "BENCH_MODEL=${BENCH_MODEL:-}"
  --env "KEY_ALIAS_PREFIX=${KEY_ALIAS_PREFIX:-}"
)

case "$ACTION" in

check)
  # Cửa kiểm image: ba dòng --version chạy sạch nghĩa là image đúng. Chạy cái này
  # TRƯỚC khi tiêu bất kỳ token LLM nào.
  echo ">>> kiểm image $IMG"
  hf jobs run --name eval-check --flavor cpu-basic --timeout 20m "${COMMON_ENV[@]}" \
    $CACHE_MOUNT "$IMG" \
    bash -lc '
      set -e
      echo "--- PDFTranslator ---"; python3 -c "import pdf2zh, fitz; print(fitz.__doc__)"
      echo "--- BabelDOC ---";      "$BABELDOC_BIN" --version
      echo "--- PDFMathTranslate ---"; "$PDFMATHTRANSLATE_BIN" --version
      echo "--- tầng chấm điểm ---"
      python3 -c "import docling_ibm_models, fasttext, skimage, comet; print(\"scoring deps OK\")"
      echo "--- cache ---"; echo "HOME=$HOME  HF_HOME=$HF_HOME"'
  ;;

warm)
  echo ">>> nung cache model vào /data (chạy 1 lần, ~15 phút)"
  hf jobs run --name eval-warm --flavor "$FLAVOR_WARM" --timeout 1h "${COMMON_ENV[@]}" \
    $CACHE_MOUNT "$IMG" \
    bash -lc '
      set -e
      python3 -m benchmark.e2e.runners.pdftranslator --warmup-only
      "$BABELDOC_BIN" --warmup
      python3 -m benchmark.e2e.parse.run_detectors --warmup-only --detectors docling
      python3 -c "from benchmark.e2e.metrics.langid import LangID; print(LangID().backend)"'
  ;;

push-corpus)
  echo ">>> đẩy corpus lên $HF_EVAL_REPO"
  python -m benchmark.e2e.sync init --private
  python -m benchmark.e2e.sync push --only corpus
  ;;

run)
  [ -n "${BENCH_MODEL:-}" ] || { echo "!! BENCH_MODEL rỗng — baseline sẽ rơi về gpt-4o-mini và bảng mất nghĩa."; exit 1; }
  for S in $SYSTEMS; do
    echo ""
    echo "=================================================================="
    echo ">>> $S   flavor=$FLAVOR   model=$BENCH_MODEL"
    echo "=================================================================="
    JOB=$(hf jobs run -d --name "eval-$S" --flavor "$FLAVOR" --timeout "$TIMEOUT" \
        "${COMMON_ENV[@]}" $CACHE_MOUNT "$IMG" \
        bash -lc "
          set -e
          python3 -m benchmark.e2e.sync pull --only corpus
          python3 -m benchmark.e2e.manifest write --out benchmark/e2e/out \
              --corpus benchmark/e2e/datasets/corpus --tiers $TIERS --langs $LANGS \
              --model \$BENCH_MODEL
          python3 -m benchmark.e2e.runners.$S \
              --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
              --tiers $TIERS --langs $LANGS --model \$BENCH_MODEL
          python3 -m benchmark.e2e.sync push --only out/$S --only out/_run
        " 2>&1 | tail -1)
    echo "    job: $JOB"
    # Tuần tự: chờ xong mới phát cái tiếp theo.
    hf jobs wait "$JOB" || { echo "!! $S FAILED — log:"; hf jobs logs "$JOB" | tail -60; exit 1; }
  done
  echo ""
  echo "XONG 3 hệ. Bước tiếp: chạy DeepL ở máy rồi 'score'."
  ;;

score)
  # CHẠY SAU KHI ĐỦ 4 HỆ. Job này không phân biệt hệ nào — nó quét mọi thư mục
  # dưới out/ và chấm y hệt nhau, nên DeepL phải được push lên TRƯỚC.
  echo ">>> chấm điểm: detector $DETECTOR + layout + visual + text + QE + report"
  hf jobs run --name eval-score --flavor "$FLAVOR" --timeout "$TIMEOUT" \
    "${COMMON_ENV[@]}" $CACHE_MOUNT "$IMG" \
    bash -lc "
      set -e
      python3 -m benchmark.e2e.sync pull --only corpus --only out
      python3 -m benchmark.e2e.runners.identity \
          --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
          --tiers $TIERS --langs $LANGS
      python3 -m benchmark.e2e.parse.render_pages \
          --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
          --tiers $TIERS --langs $LANGS
      python3 -m benchmark.e2e.parse.run_detectors \
          --out benchmark/e2e/out --detectors $DETECTOR --langs $LANGS
      python3 -m benchmark.e2e.metrics.eval_preserve \
          --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
          --tiers $TIERS --langs $LANGS --detector $DETECTOR
      python3 -m benchmark.e2e.metrics.eval_visual \
          --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
          --tiers $TIERS --langs $LANGS
      python3 -m benchmark.e2e.metrics.eval_text \
          --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
          --tiers $TIERS --langs $LANGS
      python3 -m benchmark.e2e.align.extract_pairs \
          --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
          --tiers $TIERS --langs $LANGS
      python3 -m benchmark.e2e.metrics.eval_qe \
          --out benchmark/e2e/out --langs $LANGS --model $QE_MODEL
      python3 -m benchmark.e2e.metrics.aggregate \
          --out benchmark/e2e/out --langs $LANGS --detector $DETECTOR
      python3 -m benchmark.e2e.sync push --only out
    "
  ;;

pull)
  echo ">>> kéo kết quả về máy"
  python -m benchmark.e2e.sync pull --only out/report
  echo ""
  echo "Báo cáo: benchmark/e2e/out/report/report.md"
  echo "Muốn cả artifact trung gian (~500 MB):  python -m benchmark.e2e.sync pull"
  ;;

*)
  sed -n '2,25p' "$0"
  ;;
esac
