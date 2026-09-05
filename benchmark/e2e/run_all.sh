#!/usr/bin/env bash
# =============================================================================
# Driver cho benchmark E2E: chạy từng system lên cùng một corpus, cùng một LLM.
# Thiết kế đầy đủ: docs/EVALUATION_PLAN.md
#
# Hai chế độ chạy, CÙNG một script:
#   RUNTIME=local   -> python -m benchmark.e2e.runners.<system>   (debug trên máy)
#   RUNTIME=hfjobs  -> hf jobs run ...                            (lượt chính thức)
#
# Thứ tự bắt buộc: verify_corpus (cửa chặn) -> DeepL dry-run (xem trước tiền) ->
# các runner. Không bao giờ phát job khi corpus chưa qua cửa chặn: 3 lỗi nó bắt
# (sàn 50k ký tự của DeepL, dấu chấm trong stem, tầng text-layer sai) đều làm
# hỏng kết quả hoặc đốt tiền — xem §7.4 / §1.4.
#
# YÊU CẦU: env có pdf2zh + deepl + pymupdf (vd conda `thesis`); .env ở gốc repo
#          có LITELLM_BASE_URL + LITELLM_API_KEY; DEEPL_AUTH_KEY nếu chạy deepl.
#
# Override qua biến môi trường, ví dụ:
#   TIERS=T1 LANGS=vi bash benchmark/e2e/run_all.sh
#   SYSTEMS="pdftranslator" LANGS="vi zh" bash benchmark/e2e/run_all.sh
#   DRY_RUN=1 bash benchmark/e2e/run_all.sh              # chỉ verify + forecast, không dịch
#   RUNTIME=hfjobs HF_SPACE=user/space bash benchmark/e2e/run_all.sh
# =============================================================================
set -u
cd "$(dirname "$0")/../.."   # repo root

PY=${PYTHON:-python}
RUNTIME=${RUNTIME:-local}
CORPUS=${CORPUS:-benchmark/e2e/datasets/corpus}
OUT=${OUT:-benchmark/e2e/out}
TIERS=${TIERS:-T1}
LANGS=${LANGS:-vi}                            # cách nhau bởi khoảng trắng
# Bốn system đều đã có runner. Default để 2 cái chạy được không cần venv ngoài;
# lượt so sánh chính thức: SYSTEMS="pdftranslator babeldoc pdfmathtranslate deepl".
SYSTEMS=${SYSTEMS:-"pdftranslator deepl"}
# Baseline chạy trong venv RIÊNG (dep vênh: pymupdf<1.25.3 vs >=1.26.7, và cả hai
# repo đều đặt tên package là pdf2zh). Trỏ tường minh vào console script của venv đó.
BABELDOC_BIN=${BABELDOC_BIN:-}
PDFMATHTRANSLATE_BIN=${PDFMATHTRANSLATE_BIN:-}
# Mỗi system một virtual key ở LiteLLM để quy token/tiền qua /spend/logs (§7).
KEY_ALIAS_PREFIX=${KEY_ALIAS_PREFIX:-}
PROVIDER=${PROVIDER:-litellm}
MODEL=${MODEL:-}                              # rỗng = dùng default của provider
# Hạn mức ký tự DeepL. Gói Developer = 1000000; để 950000 chừa biên (§7.4b).
DEEPL_CHAR_BUDGET=${DEEPL_CHAR_BUDGET:-950000}
RESUME=${RESUME:-1}                           # 0 = chạy lại cả những doc đã có artifact
STRICT=${STRICT:-0}                           # 1 = verify_corpus coi warning là lỗi
DRY_RUN=${DRY_RUN:-0}                         # 1 = dừng sau verify + forecast
# Chỉ dùng khi RUNTIME=hfjobs
HF_SPACE=${HF_SPACE:-}
HF_CACHE_BUCKET=${HF_CACHE_BUCKET:-}
FLAVOR_GPU=${FLAVOR_GPU:-t4-medium}
FLAVOR_CPU=${FLAVOR_CPU:-cpu-upgrade}
TIMEOUT=${TIMEOUT:-3h}

LANGS_CSV=$(echo "$LANGS" | tr ' ' ',')

# --- Preflight ---------------------------------------------------------------
echo ">>> Preflight..."
for S in $SYSTEMS; do
  case "$S" in
    pdftranslator|deepl|identity) ;;
    babeldoc)
      [ -n "$BABELDOC_BIN" ] || command -v babeldoc >/dev/null 2>&1 \
        || { echo "!! babeldoc: đặt BABELDOC_BIN=<venv>/bin/babeldoc (venv RIÊNG — dep vênh với PDFTranslator)."; exit 1; }
      [ -n "$MODEL" ] || { echo "!! babeldoc cần MODEL=<id> — để mặc định nó dùng gpt-4o-mini, bảng so sánh vô nghĩa."; exit 1; } ;;
    pdfmathtranslate)
      [ -n "$PDFMATHTRANSLATE_BIN" ] \
        || { echo "!! pdfmathtranslate: đặt PDFMATHTRANSLATE_BIN=<venv>/bin/pdf2zh. KHÔNG dựa vào PATH: 'pdf2zh' trong PATH gần như chắc chắn là của PDFTranslator (trùng tên package)."; exit 1; }
      [ -n "$MODEL" ] || { echo "!! pdfmathtranslate cần MODEL=<id> — mặc định là gpt-4o-mini."; exit 1; } ;;
    *) echo "!! system lạ: '$S'"; exit 1 ;;
  esac
done

$PY -c "import fitz, deepl" 2>/dev/null \
  || { echo "!! Thiếu deps (pymupdf, deepl). Chạy trong env có pdf2zh."; exit 1; }

if echo "$SYSTEMS" | grep -qw pdftranslator; then
  $PY -c "
from dotenv import load_dotenv; load_dotenv()
import os, sys
from pdf2zh.translation.config import PROVIDERS
p = PROVIDERS.get('$PROVIDER')
if p is None:
    print('!! provider lạ: $PROVIDER'); sys.exit(1)
if not os.environ.get(p['env_var']):
    print(f\"!! Thiếu key: \$PROVIDER -> {p['env_var']} (đặt trong .env)\"); sys.exit(1)
" || exit 1
fi

if echo "$SYSTEMS" | grep -qw deepl && [ "$DRY_RUN" != "1" ]; then
  $PY -c "
from dotenv import load_dotenv; load_dotenv()
import os, sys
if not os.environ.get('DEEPL_AUTH_KEY'): print('!! Thiếu DEEPL_AUTH_KEY'); sys.exit(1)
" || exit 1
fi

if [ "$RUNTIME" = "hfjobs" ]; then
  command -v hf >/dev/null 2>&1 || { echo "!! không thấy CLI 'hf' (pip install huggingface_hub)"; exit 1; }
  [ -n "$HF_SPACE" ] || { echo "!! RUNTIME=hfjobs cần HF_SPACE=<user>/<space>"; exit 1; }
fi
echo "    OK — runtime: $RUNTIME | systems: $SYSTEMS | tiers: $TIERS | langs: $LANGS"

# --- Cửa chặn: corpus phải sạch trước khi tiêu tiền -------------------------
echo ""
echo ">>> [gate] verify_corpus"
strict_flag=""; [ "$STRICT" = "1" ] && strict_flag="--strict"
$PY -m benchmark.e2e.datasets.verify_corpus --corpus "$CORPUS" --tiers "$TIERS" $strict_flag \
  || { echo "!! corpus KHÔNG sạch — dừng. Sửa corpus rồi chạy lại."; exit 1; }

# --- Manifest: chụp lại corpus + model + phiên bản lib của lượt này -----------
# 4 hệ chạy cách nhau hàng tuần (dep xung đột, DeepL còn bị hạn mức theo tháng).
# Không có mốc này thì corpus dựng lại hoặc model đổi giữa chừng sẽ không để lại
# dấu vết nào. Xem benchmark/e2e/manifest.py.
echo ""
echo ">>> [manifest] ghi mốc lượt chạy"
$PY -m benchmark.e2e.manifest write --out "$OUT" --corpus "$CORPUS" \
    --tiers "$TIERS" --langs "$LANGS_CSV" --systems "$(echo $SYSTEMS | tr ' ' ',')" \
    --provider "$PROVIDER" --model "$MODEL" || exit 1

# --- Xem trước chi phí DeepL ------------------------------------------------
if echo "$SYSTEMS" | grep -qw deepl; then
  echo ""
  echo ">>> [forecast] ký tự DeepL sẽ bị tính (đã gồm sàn 50k/file)"
  $PY -m benchmark.e2e.runners.deepl_doc --corpus "$CORPUS" --out "$OUT" \
      --tiers "$TIERS" --langs "$LANGS_CSV" --dry-run || exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo ">>> DRY_RUN=1 — dừng sau cửa chặn + forecast. Không dịch, không tốn tiền."
  exit 0
fi

resume_flag=""; [ "$RESUME" != "1" ] && resume_flag="--no-resume"
model_flag=""; [ -n "$MODEL" ] && model_flag="--model $MODEL"

# --- Chạy từng system -------------------------------------------------------
for S in $SYSTEMS; do
  echo ""
  echo "=================================================================="
  echo ">>> SYSTEM: $S   ->   $OUT/$S"
  echo "=================================================================="

  if [ "$RUNTIME" = "local" ]; then
    case "$S" in
      identity)
        # Hàng chuẩn kiểm chính harness. Không API, không tiền — chạy trước mọi lượt.
        $PY -m benchmark.e2e.runners.identity \
            --corpus "$CORPUS" --out "$OUT" --tiers "$TIERS" --langs "$LANGS_CSV" \
            $resume_flag || echo "!! $S FAILED" ;;
      pdftranslator)
        $PY -m benchmark.e2e.runners.pdftranslator \
            --corpus "$CORPUS" --out "$OUT" --tiers "$TIERS" --langs "$LANGS_CSV" \
            --provider "$PROVIDER" $model_flag $resume_flag \
          || echo "!! $S FAILED — xem meta.json để biết doc nào lỗi" ;;
      babeldoc)
        LITELLM_KEY_ALIAS="${KEY_ALIAS_PREFIX}babeldoc" \
        BABELDOC_BIN="$BABELDOC_BIN" \
        $PY -m benchmark.e2e.runners.babeldoc \
            --corpus "$CORPUS" --out "$OUT" --tiers "$TIERS" --langs "$LANGS_CSV" \
            --model "$MODEL" $resume_flag \
          || echo "!! $S FAILED — xem meta.json + run.log của doc lỗi" ;;
      pdfmathtranslate)
        LITELLM_KEY_ALIAS="${KEY_ALIAS_PREFIX}pdfmathtranslate" \
        PDFMATHTRANSLATE_BIN="$PDFMATHTRANSLATE_BIN" \
        $PY -m benchmark.e2e.runners.pdfmathtranslate \
            --corpus "$CORPUS" --out "$OUT" --tiers "$TIERS" --langs "$LANGS_CSV" \
            --model "$MODEL" $resume_flag \
          || echo "!! $S FAILED — xem meta.json + run.log của doc lỗi" ;;
      deepl)
        $PY -m benchmark.e2e.runners.deepl_doc \
            --corpus "$CORPUS" --out "$OUT" --tiers "$TIERS" --langs "$LANGS_CSV" \
            --char-budget "$DEEPL_CHAR_BUDGET" $resume_flag
        rc=$?
        [ $rc -eq 2 ] && { echo "!! DeepL dừng vì hết hạn mức — mua Growth hoặc nâng budget."; }
        ;;
    esac
  else
    # HF Jobs: mount cache volume cho model (entrypoint của Dockerfile tự dùng /data
    # khi ghi được), corpus read-only, out read-write. Timeout mặc định của HF là
    # 30 phút nên PHẢI truyền --timeout.
    cache_mount=""
    [ -n "$HF_CACHE_BUCKET" ] && cache_mount="-v hf-bucket://$HF_CACHE_BUCKET:/data:rw"
    case "$S" in
      pdftranslator)
        hf jobs run --name "eval-$S" --flavor "$FLAVOR_GPU" --timeout "$TIMEOUT" \
            --secrets "LITELLM_API_KEY=${LITELLM_API_KEY:-}" \
            --env "LITELLM_BASE_URL=${LITELLM_BASE_URL:-}" \
            $cache_mount -v "$CORPUS:/corpus:ro" -v "$OUT:/out:rw" \
            "hf.co/spaces/$HF_SPACE" \
            python3 -m benchmark.e2e.runners.pdftranslator \
              --corpus /corpus --out /out --tiers "$TIERS" --langs "$LANGS_CSV" \
              --provider "$PROVIDER" $model_flag $resume_flag \
          || echo "!! $S job FAILED" ;;
      babeldoc|pdfmathtranslate)
        # Space hiện tại chỉ cài env của PDFTranslator. Baseline cần image riêng
        # (dep vênh) — chạy local, đừng phát job rồi nhận số của hệ thống khác.
        echo "!! RUNTIME=hfjobs chưa hỗ trợ '$S' — chạy RUNTIME=local."; continue ;;
      deepl)
        # Không cần GPU: DeepL là API thuần, trả tiền GPU để ngồi chờ HTTP là vô nghĩa.
        hf jobs run --name "eval-$S" --flavor "$FLAVOR_CPU" --timeout "$TIMEOUT" \
            --secrets "DEEPL_AUTH_KEY=${DEEPL_AUTH_KEY:-}" \
            -v "$CORPUS:/corpus:ro" -v "$OUT:/out:rw" \
            "hf.co/spaces/$HF_SPACE" \
            python3 -m benchmark.e2e.runners.deepl_doc \
              --corpus /corpus --out /out --tiers "$TIERS" --langs "$LANGS_CSV" \
              --char-budget "$DEEPL_CHAR_BUDGET" $resume_flag \
          || echo "!! $S job FAILED" ;;
    esac
  fi
done

echo ""
echo "XONG. Artifact:"
for S in $SYSTEMS; do
  for L in $LANGS; do
    n=$(find "$OUT/$S/$L" -name output.pdf 2>/dev/null | wc -l | tr -d ' ')
    echo "  - $S / $L : $n output.pdf  ($OUT/$S/$L/<doc_id>/)"
  done
done
# --- Metric không cần detector: chạy luôn, chỉ đọc artifact ------------------
echo ""
echo ">>> [metric] khối không cần detector (page inflation, UTB, number recall,"
echo "             sec/page, success rate) — chạy cho CẢ 4 hệ kể cả DeepL"
$PY -m benchmark.e2e.metrics.eval_text \
    --corpus "$CORPUS" --out "$OUT" --tiers "$TIERS" --langs "$LANGS_CSV" \
  || echo "!! eval_text lỗi — artifact vẫn còn nguyên, chạy lại riêng nó được"

echo ""
echo "Bước tiếp: detector chung rồi tính metric layout/visual —"
echo "  benchmark/e2e/parse/          (chưa dựng)  -> mIoU, Anchor-IoU, Masked-SSIM"
echo "  benchmark/e2e/metrics/eval_preserve.py, eval_visual.py, eval_qe.py (chưa dựng)"
