#!/usr/bin/env bash
# =============================================================================
# Chạy pipeline eval qua nhiều system (mặc định provider = litellm):
#   mặc định: qwen3.6-35b-a3b, gemini-3.1-flash-lite  (đổi qua biến MODELS)
#
# Mỗi system: dịch WMT24++ -> đo latency -> chấm COMET-DA + chrF++ + các metric.
# Output:
#   benchmark/translation/out/<provider-model>/  hypotheses.jsonl, latency.jsonl, comet_scores.json, report.md
#   benchmark/translation/out/_all/              gộp tất cả -> comet_scores.json + report.md (bảng SO SÁNH)
#
# YÊU CẦU: chạy trong env có pdf2zh + unbabel-comet + sacrebleu (vd conda `thesis`),
#          và .env (gốc repo) có LITELLM_BASE_URL + LITELLM_API_KEY (harness tự load).
#
# Override qua biến môi trường, ví dụ:
#   PAIRS="vi_VN de_DE" LIMIT_DOCS=30 REPEATS=1 bash benchmark/translation/run_all.sh
#   LIMIT_DOCS="" REPEATS=5 bash benchmark/translation/run_all.sh          # full pair, đo latency N=5
#   PAIRS=ALL LIMIT_DOCS="" bash benchmark/translation/run_all.sh          # 55 cặp, tất cả docs
#   # đổi provider/model để reuse (mỗi entry là "model" hoặc "provider:model"):
#   MODELS="openai:gpt-4o-mini anthropic:claude-haiku-4-5 gemini:gemini-2.5-flash" bash benchmark/translation/run_all.sh
#   # tắt thinking cho model reasoning qua OpenRouter (Qwen, DeepSeek R1, ...):
#   NO_REASONING=1 MODELS="openrouter:qwen/qwen3.5-flash-02-23" bash benchmark/translation/run_all.sh
# =============================================================================
set -u
cd "$(dirname "$0")/../.."   # repo root

PY=${PYTHON:-python}
# Provider MẶC ĐỊNH cho entry không ghi tiền tố "provider:". Đổi cả loạt: PROVIDER=openai ...
PROVIDER=${PROVIDER:-litellm}
# MODELS: mỗi entry là "model" (dùng PROVIDER) HOẶC "provider:model" (tự chỉ định).
#   litellm:   MODELS="qwen3.6-35b-a3b gemini-3.1-flash-lite"
#   trực tiếp:  MODELS="openai:gpt-4o-mini anthropic:claude-haiku-4-5"
read -ra ENTRIES <<< "${MODELS:-gemini-3.1-flash-lite}"
# Resolve mỗi entry -> "provider|model"
RESOLVED=()
for e in "${ENTRIES[@]}"; do
  if [[ "$e" == *:* ]]; then RESOLVED+=("${e%%:*}|${e#*:}"); else RESOLVED+=("$PROVIDER|$e"); fi
done
PROVS=$(printf '%s\n' "${RESOLVED[@]}" | cut -d'|' -f1 | sort -u | tr '\n' ' ')
PAIRS=${PAIRS:-"vi_VN"}                       # danh sách locale, cách nhau bởi khoảng trắng
LIMIT_DOCS=${LIMIT_DOCS-20}                   # số doc/pair; LIMIT_DOCS="" -> FULL (~170 doc); unset -> 20
REPEATS=${REPEATS:-1}                         # >1 = latency-measure ổn định (tuần tự, chậm)
DOC_WORKERS=${DOC_WORKERS:-1}                 # >1 = quality-gen: dịch nhiều doc song song (nhanh; latency KHÔNG faithful)
SCORE=${SCORE:-1}                             # 0 = CHỈ DỊCH (bỏ hết COMET) -> chấm sau trên GPU box
SCORE_PER_MODEL=${SCORE_PER_MODEL:-1}         # 0 = bỏ chấm COMET per-model, chỉ chấm _all (nhanh hơn nhiều khi nhiều model)
RUN_AGGREGATE=${RUN_AGGREGATE:-1}             # 0 = bỏ bước report.md (chỉ ra comet_scores.json + latency.jsonl)
RESUME=${RESUME:-0}                           # 1 = KHÔNG xóa dữ liệu cũ; bỏ qua cặp đã xong (chạy tiếp sau khi stuck)
NO_REASONING=${NO_REASONING:-0}               # 1 = tắt thinking (OpenRouter: reasoning={enabled:false}; no-op provider khác)
OUT_ROOT=${OUT_ROOT:-benchmark/translation/out}
COMET_MODEL=${COMET_MODEL:-Unbabel/wmt22-comet-da}

limit_flag=""; [ -n "$LIMIT_DOCS" ] && limit_flag="--limit-docs $LIMIT_DOCS"

# --- Preflight: kiểm tra deps + key trước khi tốn thời gian ------------------
echo ">>> Preflight..."
$PY -c "import httpx, json_repair, dotenv, comet, sacrebleu" 2>/dev/null \
  || { echo "!! Thiếu deps. Chạy trong env có pdf2zh + unbabel-comet + sacrebleu."; exit 1; }
# Kiểm key cho ĐÚNG các provider được dùng (litellm/openai/anthropic/gemini...).
$PY -c "
from dotenv import load_dotenv; load_dotenv()
import os, sys
from pdf2zh.translation.config import PROVIDERS
provs = '$PROVS'.split()
miss = [p for p in provs if p in PROVIDERS and not os.environ.get(PROVIDERS[p]['env_var'])]
if miss:
    print('!! Thiếu key trong .env: ' + ', '.join(f\"{p} -> {PROVIDERS[p]['env_var']}\" for p in miss))
    sys.exit(1)
" || exit 1
# PAIRS=ALL -> bung ra toàn bộ 55 locale (ưu tiên list từ HF, fallback LOCALE_NAME).
if [ "$PAIRS" = "ALL" ] || [ "$PAIRS" = "all" ]; then
  PAIRS=$($PY -c "
try:
    from benchmark.translation.wmt24pp_adapter import list_all_pairs
    print(' '.join(sorted(list_all_pairs())))
except Exception:
    from benchmark.translation.wmt24pp_adapter import LOCALE_NAME
    print(' '.join(sorted(LOCALE_NAME)))
") || { echo '!! Không lấy được danh sách ngôn ngữ.'; exit 1; }
  echo "    PAIRS=ALL -> $(echo $PAIRS | wc -w | tr -d ' ') ngôn ngữ"
fi
echo "    OK — systems: $(printf '%s ' "${RESOLVED[@]}" | tr '|' '/')| pairs: $(echo $PAIRS | wc -w | tr -d ' ') | limit_docs: ${LIMIT_DOCS:-ALL} | repeats: $REPEATS"

ALL_DIR="$OUT_ROOT/_all"
mkdir -p "$ALL_DIR"
: > "$ALL_DIR/hypotheses.jsonl"; : > "$ALL_DIR/latency.jsonl"   # reset combined

# --- Vòng lặp system (provider|model) ---------------------------------------
for RES in "${RESOLVED[@]}"; do
  PROV="${RES%%|*}"; MODEL="${RES#*|}"
  SLUG=$(echo "${PROV}-${MODEL}" | sed 's/[^a-zA-Z0-9]/-/g')   # gồm provider -> không đụng nhau
  DIR="$OUT_ROOT/$SLUG"
  mkdir -p "$DIR"
  if [ "$RESUME" != "1" ]; then
    : > "$DIR/hypotheses.jsonl"; : > "$DIR/latency.jsonl"      # reset (run_translate append)
  fi
  echo ""
  echo "=================================================================="
  echo ">>> SYSTEM: $PROV / $MODEL   ->   $DIR  ${RESUME:+(RESUME=$RESUME)}"
  echo "=================================================================="

  resume_flag=""; [ "$RESUME" = "1" ] && resume_flag="--resume"
  no_reasoning_flag=""; [ "$NO_REASONING" = "1" ] && no_reasoning_flag="--no-reasoning"
  for PAIR in $PAIRS; do
    # RESUME: run_translate tự lọc theo TỪNG DOC (bỏ doc đã OK, dịch lại doc
    # lỗi/thiếu) dựa vào latency.jsonl hiện có — không bỏ qua cả cặp một cách thô.
    echo "--- [dịch] en-$PAIR ${resume_flag:+(resume)} ${no_reasoning_flag:+(no-reasoning)} ---"
    $PY -m benchmark.translation.run_translate --provider "$PROV" --model "$MODEL" \
        --pair "$PAIR" $limit_flag --repeats "$REPEATS" --doc-workers "$DOC_WORKERS" \
        --out "$DIR" $resume_flag $no_reasoning_flag \
      || echo "!! translate FAILED: $PROV/$MODEL / en-$PAIR — bỏ qua cặp này"
  done

  if [ -s "$DIR/hypotheses.jsonl" ]; then
    if [ "$SCORE" != "0" ] && [ "$SCORE_PER_MODEL" != "0" ]; then
      echo "--- [chấm] COMET-DA + chrF++ cho $PROV/$MODEL ---"
      if $PY -m benchmark.translation.score_comet --hyp "$DIR/hypotheses.jsonl" \
             --out "$DIR/comet_scores.json" --model "$COMET_MODEL"; then
        [ "$RUN_AGGREGATE" != "0" ] && $PY -m benchmark.translation.aggregate --out-dir "$DIR"
      else
        echo "!! scoring FAILED: $PROV/$MODEL"
      fi
    fi
    # gộp vào combined để so sánh các system
    cat "$DIR/hypotheses.jsonl" >> "$ALL_DIR/hypotheses.jsonl"
    cat "$DIR/latency.jsonl"    >> "$ALL_DIR/latency.jsonl"
  else
    echo "!! $PROV/$MODEL không có hypothesis nào — bỏ qua chấm điểm."
  fi
done

# --- Tổng hợp so sánh 4 model -----------------------------------------------
echo ""
echo "=================================================================="
echo ">>> TỔNG HỢP SO SÁNH (benchmark/translation/out/_all)"
echo "=================================================================="
if [ "$SCORE" != "0" ] && [ -s "$ALL_DIR/hypotheses.jsonl" ]; then
  $PY -m benchmark.translation.score_comet --hyp "$ALL_DIR/hypotheses.jsonl" \
      --out "$ALL_DIR/comet_scores.json" --model "$COMET_MODEL"
  [ "$RUN_AGGREGATE" != "0" ] && $PY -m benchmark.translation.aggregate --out-dir "$ALL_DIR"
elif [ "$SCORE" = "0" ]; then
  echo ">>> SCORE=0: bỏ COMET. Chấm sau: python -m benchmark.translation.score_comet --hyp $ALL_DIR/hypotheses.jsonl --out $ALL_DIR/comet_scores.json [--gpus 1]"
fi

echo ""
echo "XONG. Output:"
for RES in "${RESOLVED[@]}"; do
  PROV="${RES%%|*}"; MODEL="${RES#*|}"
  SLUG=$(echo "${PROV}-${MODEL}" | sed 's/[^a-zA-Z0-9]/-/g')
  echo "  - $PROV/$MODEL : $OUT_ROOT/$SLUG/{hypotheses,latency}.jsonl · comet_scores.json · report.md"
done
echo "  - SO SÁNH các system         : $ALL_DIR/report.md  (+ comet_scores.json keyed theo system)"
