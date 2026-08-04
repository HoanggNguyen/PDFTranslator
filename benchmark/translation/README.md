# Harness đánh giá Phase-2 (dịch) — hướng dẫn chạy

Đánh giá **đúng core dịch** (`translate_document`) trên WMT24++ mà **không sửa** một dòng nào
trong `pdf2zh/translation/`. Thiết kế đầy đủ: [../../docs/EVALUATION_PLAN.md](../../docs/EVALUATION_PLAN.md).

```
wmt24pp_adapter.py  WMT24++ jsonl -> doc dict + alignment (55 cặp)
instrument.py       httpx hook read-only: latency + token/request
run_translate.py    Bước A: dịch + đo latency  -> hypotheses.jsonl, latency.jsonl
score_comet.py      Bước B: COMET-DA + chrF++ (local CPU/MPS)  -> comet_scores.json
aggregate.py        gộp -> report.md (bảng §10)
```

## 0. Chuẩn bị

Đặt key/base_url vào file **`.env` ở gốc repo** (harness tự `load_dotenv()`, khỏi cần `export`):

```dotenv
# .env  (gốc repo)
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
# provider litellm:
LITELLM_API_KEY=sk-local
LITELLM_BASE_URL=http://localhost:4000/v1
```
(Bước dịch chỉ cần httpx — đã có sẵn trong pdf2zh; `datasets` là tùy chọn.)

# Bước chấm COMET cần torch + unbabel-comet (ƯA Python 3.10–3.12).
# Nếu env chính là 3.13 → tạo venv riêng CHỈ cho chấm điểm:
python3.11 -m venv .venv-score && source .venv-score/bin/activate
pip install -r benchmark/translation/requirements.txt
deactivate
```

> Chạy mọi lệnh từ **thư mục gốc repo**. COMET-DA tải model công khai (~2.3GB) lần đầu.

---

## TEST TRƯỚC (1 doc) — làm đủ 2 bước này trước khi chạy full

### T1. Dịch thử 1 document (kiểm tra key + luồng + alignment)
```bash
python -m benchmark.translation.run_translate --pair vi_VN --provider gemini --limit-docs 1 --repeats 1
```
Xem `benchmark/translation/out/hypotheses.jsonl` (source/hypothesis/reference thẳng hàng,
`is_fallback`) và `benchmark/translation/out/latency.jsonl` (wall_s, n_req, n_retry, tok_out).
Nếu `is_fallback` nhiều → key/model lỗi hoặc rate-limit.

### T2. Chấm thử COMET trên đúng 1 doc đó
```bash
# trong .venv-score nếu tách env:
python -m benchmark.translation.score_comet \
    --hyp benchmark/translation/out/hypotheses.jsonl --out benchmark/translation/out/comet_test.json
```
Ra điểm COMET-DA + chrF++. Nếu chạy được → sẵn sàng full.

> Reset trước khi chạy full: `rm -f benchmark/translation/out/*.jsonl` (run_translate **append**).

---

## CHẠY FULL

### Tier A — quét đa ngôn ngữ (55 cặp, 1–2 hệ) — dữ liệu chính RQ1
Chỉ cần **hypotheses** → chạy **quality-gen** (song song cho nhanh; latency KHÔNG faithful):
```bash
for L in vi_VN de_DE zh_CN ja_JP ru_RU es_MX fr_FR hi_IN ar_SA ...; do   # đủ 55 locale
  python -m benchmark.translation.run_translate --pair $L --provider gemini \
      --repeats 1 --doc-workers 6
done
```

### Tier B — so sánh hệ (3–5 provider × ~9 cặp đại diện, có vi_VN)
Cần **latency trung thực** → chạy **latency-measure** (tuần tự, concurrency=8, N≥5):
```bash
for P in "gemini:" "openai:" "deepseek:" "anthropic:claude-haiku-4-5"; do
  prov=${P%%:*}; model=${P#*:}
  for L in vi_VN de_DE zh_CN ja_JP hi_IN ru_RU fr_FR ar_SA th_TH; do
    python -m benchmark.translation.run_translate --pair $L --provider $prov \
        ${model:+--model $model} --repeats 5           # doc-workers=1 mặc định
  done
done
```

### Chấm điểm + gộp báo cáo
```bash
python -m benchmark.translation.score_comet --hyp benchmark/translation/out/hypotheses.jsonl \
       --out benchmark/translation/out/comet_scores.json
python -m benchmark.translation.aggregate   --out-dir benchmark/translation/out   # -> report.md
```

---

## Lưu ý quan trọng

- **quality-gen (`--doc-workers>1`) vs latency-measure (mặc định):** số `s/doc` chỉ trung thực ở
  latency-measure. `aggregate.py` chỉ lấy bản ghi `mode="latency-measure"` cho bảng latency.
- **Nút thắt = rate-limit provider**, không phải Mac. Free tier → nhiều 429 (`n_retry` cao) →
  chậm. Dùng key trả phí cho lượt 55 cặp. Ước lượng thời gian: [§6.2 của plan](../../docs/EVALUATION_PLAN.md).
- **RAM 16GB:** dùng `wmt22-comet-da` (mặc định), **không** XCOMET. `--gpus 0` (CPU) an toàn;
  thử MPS thủ công nếu muốn nhanh hơn.
- **Bất biến:** sau cùng `git diff pdf2zh/translation/` phải **rỗng** — harness chỉ đọc & gọi.
- `run_translate.py` **append** vào jsonl → xóa `benchmark/translation/out/*.jsonl` khi muốn chạy lại từ đầu.
