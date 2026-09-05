#!/usr/bin/env bash
# =============================================================================
# Dep cho tầng METRIC (không phải cho runner — runner dùng env sẵn có của mỗi hệ).
#
# Chạy trong env `thesis`:
#     conda activate thesis && bash benchmark/e2e/install_deps.sh
#
# Đã có sẵn trong `thesis` (kiểm 2026-09-05): pymupdf 1.25.2 · numpy 1.26.4 ·
# scipy 1.17.1 · scikit-image 0.26.0 · unbabel-comet 2.2.7 · surya-ocr 0.17.1.
# Script này chỉ cài phần còn thiếu, và in lại đủ danh sách để manifest ghi được
# phiên bản (benchmark/e2e/manifest.py: TRACKED_LIBS).
#
# YÊU CẦU MẠNG: pypi.org, huggingface.co, dl.fbaipublicfiles.com. Tại thời điểm
# viết, cả ba đều timeout từ máy này trong khi google.com vẫn thông — nghĩa là
# mạng đang chặn, không phải hỏng. Cần VPN/proxy trước khi chạy.
# =============================================================================
set -u
PY=${PYTHON:-python}

echo ">>> Kiểm mạng trước khi cài (tránh treo 10 phút rồi mới biết)"
for h in pypi.org huggingface.co dl.fbaipublicfiles.com; do
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 "https://$h" 2>/dev/null)
  if [ "${code:-000}" = "000" ]; then
    echo "  !! $h KHÔNG tới được — bật VPN rồi chạy lại."; exit 1
  fi
  echo "  ok  $h -> $code"
done

echo ""
echo ">>> Cài phần còn thiếu"
# docling: detector chấm điểm chung (§3). Cố ý chọn model KHÔNG hệ nào dưới bài
# kiểm dùng, nên nó không thiên vị ai.
$PY -m pip install "docling" || exit 1

# fasttext cho UTB. Bản source hay gãy khi build trên macOS arm64 + Python 3.12,
# nên thử wheel trước. Thiếu nó thì langid.py vẫn chạy bằng backend heuristic và
# tự đánh dấu — chạy được để debug, KHÔNG dùng cho số liệu luận văn.
$PY -m pip install "fasttext-wheel" || $PY -m pip install "fasttext" \
  || echo "  !! fasttext không cài được — UTB sẽ chạy backend heuristic"

echo ""
echo ">>> Tải model LID một lần (917 KB, không phải bản .bin 126 MB)"
$PY - <<'EOF'
from benchmark.e2e.metrics.langid import LangID
lid = LangID()
print(f"  lid_backend = {lid.backend}")
EOF

echo ""
echo ">>> Phiên bản manifest sẽ ghi lại"
$PY -c "
from benchmark.e2e.manifest import _lib_versions
for k, v in _lib_versions().items(): print(f'  {k:16} {v}')
"
