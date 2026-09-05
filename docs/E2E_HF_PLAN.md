# Plan: đưa benchmark E2E lên Hugging Face, cùng phần cứng, đủ metric

Mục tiêu: **một bảng so sánh 4 kiến trúc mà mọi con số đều bảo vệ được** — cùng bài
kiểm, cùng LLM, cùng phần cứng, cùng thước đo, và mọi bước trung gian đều tải về máy
được để kiểm lại.

Liên quan: thiết kế metric ở [EVALUATION_PLAN.md](EVALUATION_PLAN.md), thao tác hằng
ngày ở [E2E_RUNBOOK.md](E2E_RUNBOOK.md). File này là **kế hoạch thi công**.

---

## 0. Bảy quyết định chốt

| # | Quyết định | Lý do |
|---|---|---|
| 1 | **Một image riêng cho benchmark** (`Dockerfile.bench` → Space `*-bench`), không sửa Space demo | 3 hệ = 3 venv ≈ +4 GB. Nhét vào Space demo là làm chậm build và phình runtime của bản demo đang phục vụ người dùng |
| 2 | **Ba venv trong cùng một image** | Xung đột dep là chuyện của môi trường Python, không phải của máy. Nhốt mỗi hệ một venv là hết, mà vẫn "cùng phần cứng" |
| 3 | **Mọi job dùng đúng một flavor `t4-medium`, chạy tuần tự** | Cột giây/trang chỉ so được khi cùng máy và không giành tài nguyên nhau |
| 4 | **DeepL chạy ở máy, không lên HF** | Nó là API thuần; trả tiền GPU để ngồi chờ HTTP là vô nghĩa. Nhưng output của nó **bắt buộc** đi qua đúng bộ metric như 3 hệ kia |
| 5 | **Đồng bộ bằng HF dataset repo riêng**, không dựa vào cơ chế mount bucket | Push/pull tường minh, có version, resume được, và là cái link vĩnh viễn để luận văn trích dẫn |
| 6 | **Detector chấm điểm = Docling layout (RT-DETR)** | Không hệ nào dưới bài kiểm dùng nó ⇒ không thiên vị ai. Nó train trên DocLayNet *train*, corpus lấy từ *test* ⇒ trần đo cao, không rò rỉ |
| 7 | **Chấm điểm là một job riêng, chạy sau khi đủ 4 hệ** | Detector + SSIM + QE đều là GPU-bound và chỉ đọc PDF. Tách ra thì chấm lại bao nhiêu lần cũng được mà không phải dịch lại |

---

## 1. Kiến trúc chạy

```
   MÁY (macOS)                    HF DATASET REPO                  HF JOBS (t4-medium)
   ───────────                    ───────────────                  ───────────────────
   corpus T1  ──── push ────────►  corpus/                ──pull──►  job A  pdftranslator
   DeepL runner ── push ────────►  out/deepl-document/    ──pull──►  job B  babeldoc
   (API, chạy tại chỗ)                                              job C  pdfmathtranslate
                                                                    job D  identity + ceiling
   report/    ◄─── pull ─────────  out/<system>/...       ◄─push───  (mỗi job push xong artifact)
   figures/   ◄─── pull ─────────  out/_metrics/          ◄─push───  job E  chấm điểm (detector
                                   out/_render/                              + layout + visual + QE)
```

Nguyên tắc: **HF dataset repo là nguồn sự thật duy nhất.** Máy và job đều chỉ push/pull
với nó. Không có đường nào máy nói chuyện trực tiếp với job.

Repo cần tạo: `<user>/pdftranslator-eval` (private lúc làm, public khi nộp).

---

## 2. Image benchmark — `Dockerfile.bench`

Kế thừa image hiện tại rồi thêm hai venv. Điểm khác biệt bắt buộc so với
`Dockerfile` đang có:

### 2.1 Cái bẫy phải sửa: `hf jobs run` xoá `CMD`

`Dockerfile` hiện tại đặt toàn bộ logic chọn thư mục cache vào
`CMD ["/usr/local/bin/entrypoint.sh"]` và **không có `ENTRYPOINT`**. `hf jobs run`
truyền command riêng ⇒ **`CMD` bị thay, `entrypoint.sh` không bao giờ chạy**, nên
`MODEL_CACHE_DIR` / `PADDLE_PDX_CACHE_HOME` / `HF_HOME` rơi về `/app/.cache` chứ
không phải `/data` — tức là **mount volume cache mà vẫn tải lại 3–5 GB model mỗi
job, và trả tiền cho thời gian tải đó.**

Sửa: tách phần export cache ra một `ENTRYPOINT` bọc, `CMD` giữ nguyên cho Space.

```dockerfile
# entrypoint-bench.sh: set cache root rồi exec cái gì được truyền vào
ENTRYPOINT ["/usr/local/bin/entrypoint-bench.sh"]
CMD ["python3", "app.py"]
```

```bash
#!/usr/bin/env bash
set -e
if mkdir -p /data 2>/dev/null && [ -w /data ]; then CACHE_ROOT=/data; else CACHE_ROOT=/app/.cache; fi
export MODEL_CACHE_DIR="$CACHE_ROOT/datalab/models"
export PADDLE_PDX_CACHE_HOME="$CACHE_ROOT/paddlex"
export HF_HOME="$CACHE_ROOT/huggingface"
export BABELDOC_CACHE_ROOT="$CACHE_ROOT/babeldoc"   # BabelDOC cache theo $HOME
export HOME="${HOME:-/app}"
mkdir -p "$MODEL_CACHE_DIR" "$PADDLE_PDX_CACHE_HOME" "$HF_HOME"
echo "[bench] cache root = $CACHE_ROOT"
exec "$@"
```

### 2.2 Hai venv baseline

Image nền là Ubuntu 22.04 ⇒ Python hệ thống là **3.10**. BabelDOC nhận 3.10 nhưng
**PDFMathTranslate đòi >=3.11,<3.13** ⇒ phải có interpreter khác. `uv` tự tải bản
CPython standalone nên không cần đụng python hệ thống.

```dockerfile
# uv (một binary tĩnh, ~30 MB)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hai checkout baseline, pin theo commit để tái lập được
ARG BABELDOC_REF=v0.6.4
ARG PDFMATH_REF=v1.9.11
RUN git clone --depth 1 --branch $BABELDOC_REF https://github.com/funstory-ai/BabelDOC /opt/BabelDOC && \
    git clone --depth 1 --branch $PDFMATH_REF https://github.com/Byaidu/PDFMathTranslate /opt/PDFMathTranslate

RUN uv venv /opt/venv-babeldoc --python 3.12 && \
    uv pip install --python /opt/venv-babeldoc -e /opt/BabelDOC && \
    uv venv /opt/venv-pdfmath  --python 3.12 && \
    uv pip install --python /opt/venv-pdfmath  -e /opt/PDFMathTranslate

ENV BABELDOC_BIN=/opt/venv-babeldoc/bin/babeldoc \
    PDFMATHTRANSLATE_BIN=/opt/venv-pdfmath/bin/pdf2zh
```

### 2.3 Dep cho tầng chấm điểm

```dockerfile
RUN python3 -m pip install docling docling-ibm-models fasttext-wheel \
        scikit-image opencv-python-headless unbabel-comet
```

`docling` kéo theo model RT-DETR khi chạy lần đầu → nằm trong `HF_HOME` ⇒ đã được
`/data` cache. `fasttext` phải có, thiếu nó thì UTB rơi về heuristic và **không dùng
được cho luận văn**.

### 2.4 Nung asset trước, đừng nung vào đồng hồ đo

Một job `warmup` chạy trước mọi thứ, tải hết vào `/data`:

```bash
python3 -m benchmark.e2e.runners.pdftranslator --warmup-only   # Surya + Paddle
/opt/venv-babeldoc/bin/babeldoc --warmup                       # DocLayout-YOLO ONNX + font
python3 -m benchmark.e2e.parse.run_detectors --warmup-only     # Docling RT-DETR
python3 -c "from benchmark.e2e.metrics.langid import LangID; LangID()"   # lid.176
```

---

## 3. Chính sách phần cứng

| Job | Flavor | Vì sao |
|---|---|---|
| warmup | `t4-small` | chỉ tải file |
| A pdftranslator | **`t4-medium`** | Surya + Paddle cần GPU |
| B babeldoc | **`t4-medium`** | *không* cần GPU, nhưng phải **cùng máy** thì cột giây/trang mới so được |
| C pdfmathtranslate | **`t4-medium`** | như trên |
| D identity + ceiling | `t4-medium` | dùng detector |
| E chấm điểm | `t4-medium` | detector + SSIM; CometKiwi-XL thì cần `l4x1` (≥24 GB) |

**Chạy tuần tự**, chain bằng `hf jobs wait <id> && ...`. Chạy song song là hai job
giành GPU/băng thông của nhau và cột thời gian thành rác.

Phải nói rõ trong luận văn: cùng máy **không** có nghĩa là cùng được tăng tốc.
BabelDOC và PDFMathTranslate dùng ONNX layout model rất nhỏ, thời gian thật của
chúng là **chờ mạng gọi LLM**. Cùng flavor ở đây là để **loại biến phần cứng**, không
phải để cho chúng lợi thế GPU.

DeepL không có hàng phần cứng — nó là dịch vụ đám mây, ghi `accelerator: "(deepl cloud)"`
và **loại khỏi mọi so sánh tốc độ**, chỉ giữ ở các trục layout / nội dung / chất lượng.

---

## 4. DeepL: chạy ở máy, chấm chung

Ràng buộc thứ tự — **DeepL phải xong và đã push trước khi phát job E**:

```
1. máy:  DEEPL_AUTH_KEY=... python -m benchmark.e2e.runners.deepl_doc --dry-run   # xem tiền
2. máy:  ... (bỏ --dry-run) chạy thật
3. máy:  python -m benchmark.e2e.sync push --only out/deepl-document
4. HF:   job E chấm điểm — quét mọi thư mục dưới out/, DeepL nằm trong đó
```

Job E **không phân biệt hệ nào**: nó nhận `out/<system>/<lang>/<doc_id>/output.pdf`
và chấm y hệt nhau. Đó chính là lý do bắt mọi runner ghi cùng một hình dạng artifact.

Ba điều riêng của DeepL phải giữ trong bảng:
- `chars_billed` (không hệ nào khác có) — để đối chiếu sổ ngân sách
- không có `parse_s`/`translate_s`/`render_s` — hộp đen, cột để trống chứ không đoán
- không tham gia ablation pseudo-translator §4.6 và ablation glossary — không điều khiển được

---

## 5. Hợp đồng artifact — lưu đủ để không phải chạy lại

### 5.1 Cây thư mục sau khi xong

```
corpus/T1/                          # đầu vào, bất biến
  T1_<domain>.pdf  gt.json  mapping.json

out/_run/<run_id>.json              # git rev, model, sha256 corpus, version lib, flavor
out/<system>/<lang>/<doc_id>/
  output.pdf                        # bản MONO — thứ được chấm
  meta.json                         # timing từng pha, token, trang vào/ra, lỗi, phần cứng
  run.log                           # baseline: stdout+stderr tiến trình con
  raw/                              # baseline: NGUYÊN VĂN mọi file nó đẻ ra (kể cả dual)
  phase1_parsed.json                # PDFTranslator: kết quả đọc PDF   ← MỚI: copy từ _parse
  phase2_translated.json            # PDFTranslator: kết quả dịch
  debug/                            # BabelDOC --debug: layout_generator / typsetting /
                                    #   translate_tracking json        ← MỚI, tuỳ chọn
out/pdftranslator/_parse/<doc_id>/phase1_parsed.json    # cache dùng chung mọi ngôn ngữ

out/_render/<system>/<lang>/<doc_id>/p<NNN>.png         ← MỚI: ảnh 150 DPI từng trang
out/_render/_source/<doc_id>/p<NNN>.png                 ← MỚI: ảnh trang nguồn
out/_layout/<detector>/<system>/<lang>/<doc_id>/p<NNN>.json   ← MỚI: box đã chuẩn hoá
out/_layout/<detector>/_source/<doc_id>/p<NNN>.json           ← MỚI: cho hàng ceiling

out/_pairs/<system>.<lang>.jsonl    ← MỚI: cặp {src, mt} đã align, input cho CometKiwi
out/_metrics/text/<system>.<lang>.json
out/_metrics/layout/<system>.<lang>.json     ← MỚI
out/_metrics/visual/<system>.<lang>.json     ← MỚI
out/_metrics/qe/<system>.<lang>.json         ← MỚI
out/_metrics/summary.json
out/report/report.md  tables/*.csv  figures/*.png        ← MỚI
```

### 5.2 Ba thứ đang thiếu, phải bổ sung

| Thiếu | Vì sao cần | Chi phí |
|---|---|---|
| **Ảnh render 150 DPI** từng trang, nguồn + mọi output | Masked-SSIM cần; detector cũng chạy trên ảnh này ⇒ render một lần dùng hai chỗ | ~250 MB cho 120 trang × 5 (nguồn + 4 hệ) |
| **`phase1_parsed.json` copy vào từng thư mục doc** | Hiện chỉ nằm ở cache `_parse/`; ai đọc artifact của một doc không thấy được đầu vào của pha dịch | vài MB |
| **`debug/` của BabelDOC** | Nó dump sẵn `translate_tracking.json` đã align theo đoạn — đối chiếu chéo với module align dùng chung | **một lượt LLM nữa** ⇒ để cờ `--debug-pass`, mặc định tắt |

### 5.3 Nguyên tắc lưu

- **Không xoá gì.** Đĩa rẻ, chạy lại thì đắt (tiền LLM, tiền GPU, hạn mức DeepL).
- **Mọi file sinh ra đều ghi kèm `run_id`** để biết nó thuộc lượt nào.
- Thư mục `_render/`, `_layout/`, `_metrics/`, `report/` **sinh lại được** ⇒ xoá thoải
  mái. Thư mục `<system>/` thì **không** ⇒ có cửa chặn resume.

---

## 6. Đồng bộ HF ↔ máy

Viết một module nhỏ `benchmark/e2e/sync.py` bọc `huggingface_hub`:

```bash
# máy -> HF (trước khi phát job)
python -m benchmark.e2e.sync push --repo <user>/pdftranslator-eval --only corpus
python -m benchmark.e2e.sync push --repo <user>/pdftranslator-eval --only out/deepl-document

# HF -> máy (sau khi job xong)
python -m benchmark.e2e.sync pull --repo <user>/pdftranslator-eval --only out
python -m benchmark.e2e.sync pull --repo <user>/pdftranslator-eval --only out/report   # nhẹ

# xem có gì trên đó
python -m benchmark.e2e.sync ls --repo <user>/pdftranslator-eval
```

Bên trong dùng `upload_folder` / `snapshot_download` với `allow_patterns`, `repo_type="dataset"`.
Job HF cũng gọi đúng module này: `pull` corpus lúc bắt đầu, `push` artifact lúc kết thúc.
Nhờ vậy **một đường code duy nhất**, không có hai cơ chế đồng bộ song song.

Vì sao dataset repo chứ không phải mount bucket: có version (`revision`), có
`allow_patterns` để kéo riêng phần nhẹ, không phụ thuộc ngữ nghĩa mount, và khi nộp
luận văn thì đó là **link tái lập vĩnh viễn**.

Cỡ dữ liệu ước tính cho T1 (120 trang, 4 hệ, 1 ngôn ngữ):

| Phần | Dung lượng |
|---|---|
| corpus | ~10 MB |
| output.pdf + raw/ (4 hệ) | ~150 MB |
| json trung gian | ~50 MB |
| `_render/` ảnh 150 DPI | ~250 MB |
| `_layout/` | ~20 MB |
| **tổng** | **~500 MB** |

Kéo riêng `out/report/` thì chỉ vài MB.

---

## 7. Code — 8 file, **đã viết xong**

> Trạng thái 2026-09-05: cả 8 file đã có trong repo và đã chạy được đầu-cuối ở máy
> trên hàng chuẩn `identity`. Kiểm chứng đã làm:
>
> * đưa chính GT trở lại làm "đầu ra detector" ⇒ `mIoU = 1.000`, `Anchor-IoU = 1.000`
>   (351/351 box anchor), `F1@[.5:.95] = 1.000`, `containment = 1.000`, `τ = 1.000`
>   — nghĩa là phần hình học không có lỗi dấu, lỗi trục, hay lỗi chuẩn hoá;
> * `identity` (copy PDF nguồn làm đầu ra) ⇒ `Masked-SSIM = 1.0000`,
>   `ink-distance = 0.0000`, 70.6% pixel được chấm (29.4% là chữ, bị che);
> * `extract_pairs` trên `identity` ⇒ **1703/1703 cặp trùng khít** — bộ ghép không
>   lệch hàng;
> * `aggregate` ⇒ hàng `identity` trùng khít hàng `source_ceiling`.
>
> Chưa chạy được ở máy này: bản thân detector Docling và CometKiwi (mạng chặn
> pypi + huggingface), và ba hệ thật (chưa có venv/image). Đó là việc môi trường,
> không phải việc code.

## 7. Danh mục file

| # | File | Vào | Ra | Ghi chú |
|---|---|---|---|---|
| 1 | `e2e/sync.py` | thư mục local | dataset repo | Làm trước, vì mọi job đều cần |
| 2 | `e2e/parse/render_pages.py` | mọi `output.pdf` + PDF nguồn | `_render/**.png` 150 DPI | PyMuPDF. Cùng DPI, cùng khổ, memo hoá theo `sha256` |
| 3 | `e2e/parse/run_detectors.py` | `_render/**.png` | `_layout/docling/**.json` | Dùng lớp thấp `docling_ibm_models` `LayoutPredictor` (kiểm tên hàm sau khi cài), **không** dùng `DocumentConverter` — ta chỉ cần box, không cần convert cả tài liệu. Chuẩn hoá về `{page, class, bbox_norm[4], reading_order}`. Map nhãn về bộ rút gọn 8 lớp |
| 4 | `e2e/metrics/eval_preserve.py` | `_layout/` + `gt.json` | `_metrics/layout/` | **Trục chính.** Hungarian matching cost `1−IoU`, cùng class. Ra: mIoU, F1@0.5, mF1@[.5:.95], **Anchor-IoU**, text containment, element retention, **collision rate**, **margin violation**, reading-order τ. Tái dùng lõi hình học của `benchmark/parser/evaluation/eval_layout.py`. Sinh luôn **hàng `Source ceiling`** = detector chạy trên PDF nguồn chấm với GT người vẽ |
| 5 | `e2e/metrics/eval_visual.py` | `_render/` + `gt.json` | `_metrics/visual/` | **Masked-SSIM** (che vùng text theo GT, dilate 2 px, SSIM phần còn lại), ink-profile 1D Wasserstein, và full-page SSIM *chỉ để chứng minh nó là metric tồi ở đây* |
| 6 | `e2e/align/extract_pairs.py` | `output.pdf` 4 hệ + PDF nguồn | `_pairs/*.jsonl` | Trích text theo block + align nguồn↔đích theo bbox và thứ tự đọc. **Bắt buộc dùng chung cho cả 4 hệ** — không được ưu ái PDFTranslator bằng `phase2_translated.json` của chính nó, dù có sẵn |
| 7 | `e2e/metrics/eval_qe.py` | `_pairs/*.jsonl` | `_metrics/qe/` | CometKiwi QE (`wmt23-cometkiwi-da-xl`, fallback `wmt22`). Thêm nhánh hiệu chuẩn: tương quan QE ↔ COMET-DA trên WMT24++ `vi_VN` để chứng minh dùng QE xếp hạng là hợp lệ |
| 8 | `e2e/metrics/aggregate.py` | mọi `_metrics/**` | `report/` | **Bootstrap CI 95%** + paired test PDFTranslator vs từng baseline + effect size. Bảng CSV + biểu đồ. Mẫu: `benchmark/translation/aggregate.py` |

Mở rộng thêm cho `metrics/eval_text.py` đã có: content loss rate, terminology
consistency, và đọc `render_stats` (`elements_fallback`, `elements_skipped`) thành
tín hiệu lỗi render — cái này **miễn phí**, dict đã nằm sẵn trong `meta.json`.

### Thứ tự thi công đề xuất

```
sync.py ──► Dockerfile.bench ──► warmup job ──► chạy 4 hệ
                                                    │
              render_pages ──► run_detectors ──► eval_preserve ──┐
                       └──────────────────────► eval_visual ─────┤
              extract_pairs ─────────────────► eval_qe ──────────┼──► aggregate ──► report
              eval_text (mở rộng) ────────────────────────────────┘
```

`eval_preserve` là mốc quan trọng nhất: xong nó là **có bảng layout đầu tiên**, tức
là phần lõi của luận văn.

---

## 8. File môi trường cần chuẩn bị

### 8.1 `.env` ở gốc repo (chỉ dùng ở máy, **không** bake vào image)

```dotenv
# ── LLM: cả 3 hệ mã nguồn mở đi chung một proxy ───────────────────────────────
LITELLM_BASE_URL=https://litellm.internal.cake.vn/v1
LITELLM_API_KEY=sk-...

# ── DeepL (chạy ở máy) ────────────────────────────────────────────────────────
DEEPL_AUTH_KEY=...                     # gói Developer miễn phí, 1M ký tự/tháng

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_TOKEN=hf_...                        # quyền write, để push dataset + tạo job
HF_USER=<user>
HF_EVAL_REPO=<user>/pdftranslator-eval # dataset repo chứa artifact
HF_BENCH_SPACE=<user>/pdftranslator-bench   # Space dùng làm image cho job

# ── Chốt cho cả lượt chạy — đổi là bảng mất so sánh ───────────────────────────
BENCH_MODEL=google/gemini-3.1-flash-lite
KEY_ALIAS_PREFIX=thesis-               # -> thesis-pdftranslator, thesis-babeldoc, ...
```

### 8.2 Cấu hình phía LiteLLM proxy — **việc phải làm ngoài repo**

`pdf2zh/translation/gateway.py` hardcode `temperature` 0.7/0.2, không có nút chỉnh,
và harness không được sửa `pdf2zh/`. Hai baseline thì gửi `temperature=0`. Nên
**phải ép `temperature=0` tại proxy** cho model dùng trong benchmark — nếu không,
hệ của bạn chạy ở nhiệt độ khác hai đối thủ và bảng không bảo vệ được.

Đồng thời tạo **4 virtual key** (`thesis-pdftranslator`, `thesis-babeldoc`,
`thesis-pdfmathtranslate`, và một key cho ablation) để quy token/USD riêng từng hệ
qua `/spend/logs`. PDFMathTranslate v1 **không đếm token ở đâu cả** — đây là đường
duy nhất lấy được số của nó.

### 8.3 Secret truyền vào job HF (không bake vào image)

```bash
--secrets LITELLM_API_KEY=$LITELLM_API_KEY
--secrets HF_TOKEN=$HF_TOKEN
--env LITELLM_BASE_URL=$LITELLM_BASE_URL
--env HF_EVAL_REPO=$HF_EVAL_REPO
--env BENCH_MODEL=$BENCH_MODEL
--env KEY_ALIAS_PREFIX=$KEY_ALIAS_PREFIX
```

`DEEPL_AUTH_KEY` **không lên HF** — DeepL chỉ chạy ở máy.

### 8.4 Biến đã nằm sẵn trong image

`BABELDOC_BIN`, `PDFMATHTRANSLATE_BIN`, `MODEL_CACHE_DIR`, `PADDLE_PDX_CACHE_HOME`,
`HF_HOME`, `TYPST_BIN`, `PDF2ZH_FONT_DIR`. Không cần truyền lại.

---

## 9. Trình tự chạy trên HF

> Phần này là **bản rút gọn để hiểu kiến trúc**. Hướng dẫn thao tác đầy đủ — chuẩn
> bị token, dựng Space, cửa kiểm sau từng bước, sự cố hay gặp — ở
> [HF_GUIDE.md](HF_GUIDE.md).

```bash
set -a && source .env && set +a
BUCKET=$HF_USER/pdftranslator-eval-cache
IMG=hf.co/spaces/$HF_BENCH_SPACE

# ── 0. Một lần: tạo dataset repo + đẩy corpus lên ────────────────────────────
python -m benchmark.e2e.sync init --repo $HF_EVAL_REPO --private
python -m benchmark.e2e.sync push --repo $HF_EVAL_REPO --only corpus

# ── 1. Một lần: build image ──────────────────────────────────────────────────
#    Space $HF_BENCH_SPACE trỏ vào Dockerfile.bench. Đợi build xanh rồi mới chạy job.
#    ⚠ Image là ẢNH CHỤP CỦA SPACE. Sửa benchmark/ mà chưa push Space thì job chạy code cũ.

# ── 2. Warm cache (một lần, ~15 phút) ────────────────────────────────────────
hf jobs run --name eval-warm --flavor t4-small --timeout 1h \
  -v hf-bucket://$BUCKET:/data:rw $IMG \
  bash -lc "python3 -m benchmark.e2e.runners.pdftranslator --warmup-only && \
            \$BABELDOC_BIN --warmup && \
            python3 -m benchmark.e2e.parse.run_detectors --warmup-only"

# ── 3. Ba hệ mã nguồn mở, TUẦN TỰ, cùng flavor ───────────────────────────────
for S in pdftranslator babeldoc pdfmathtranslate; do
  JOB=$(hf jobs run -d --name eval-$S --flavor t4-medium --timeout 3h \
    --secrets LITELLM_API_KEY=$LITELLM_API_KEY --secrets HF_TOKEN=$HF_TOKEN \
    --env LITELLM_BASE_URL=$LITELLM_BASE_URL --env HF_EVAL_REPO=$HF_EVAL_REPO \
    --env BENCH_MODEL=$BENCH_MODEL --env KEY_ALIAS_PREFIX=$KEY_ALIAS_PREFIX \
    -v hf-bucket://$BUCKET:/data:rw $IMG \
    bash -lc "python3 -m benchmark.e2e.sync pull --only corpus && \
              python3 -m benchmark.e2e.runners.$S --corpus corpus --out out \
                --tiers T1 --langs vi --model \$BENCH_MODEL && \
              python3 -m benchmark.e2e.sync push --only out/$S")
  hf jobs wait $JOB || { echo "!! $S FAILED"; hf jobs logs $JOB | tail -50; break; }
done

# ── 4. DeepL ở máy, rồi đẩy lên ──────────────────────────────────────────────
python -m benchmark.e2e.runners.deepl_doc --corpus benchmark/e2e/datasets/corpus \
    --out benchmark/e2e/out --tiers T1 --langs vi --dry-run     # xem tiền TRƯỚC
python -m benchmark.e2e.runners.deepl_doc --corpus benchmark/e2e/datasets/corpus \
    --out benchmark/e2e/out --tiers T1 --langs vi --char-budget 950000
python -m benchmark.e2e.sync push --only out/deepl-document

# ── 5. Chấm điểm — MỘT job, chạy sau khi đủ 4 hệ ─────────────────────────────
hf jobs run --name eval-score --flavor t4-medium --timeout 3h \
  --secrets HF_TOKEN=$HF_TOKEN --env HF_EVAL_REPO=$HF_EVAL_REPO \
  -v hf-bucket://$BUCKET:/data:rw $IMG \
  bash -lc "python3 -m benchmark.e2e.sync pull --only corpus --only out && \
            python3 -m benchmark.e2e.runners.identity --corpus corpus --out out --langs vi && \
            python3 -m benchmark.e2e.parse.render_pages   --corpus corpus --out out && \
            python3 -m benchmark.e2e.parse.run_detectors  --out out --detectors docling && \
            python3 -m benchmark.e2e.metrics.eval_preserve --corpus corpus --out out && \
            python3 -m benchmark.e2e.metrics.eval_visual   --corpus corpus --out out && \
            python3 -m benchmark.e2e.metrics.eval_text     --corpus corpus --out out && \
            python3 -m benchmark.e2e.align.extract_pairs   --corpus corpus --out out && \
            python3 -m benchmark.e2e.metrics.aggregate     --out out && \
            python3 -m benchmark.e2e.sync push --only out"

# ── 6. Kéo kết quả về ────────────────────────────────────────────────────────
python -m benchmark.e2e.sync pull --only out/report      # nhẹ, vài MB
python -m benchmark.e2e.sync pull --only out             # đầy đủ, ~500 MB
```

CometKiwi-XL cần ≥24 GB VRAM ⇒ **không chạy trên T4 16 GB**. Tách thành job riêng
`--flavor l4x1`, hoặc dùng bản `wmt22-cometkiwi-da` nhỏ hơn và ghi rõ trong luận văn.

---

## 10. Kiểm chứng từng bước

Không bước nào được tính là xong nếu chưa qua cửa của nó:

| Bước | Xong khi |
|---|---|
| Image | `hf jobs run ... $IMG bash -lc "$BABELDOC_BIN --version && $PDFMATHTRANSLATE_BIN --version && python3 -c 'import docling, fasttext'"` chạy sạch |
| Cache | Job thứ hai khởi động nhanh hơn job đầu **≥5 phút** (bằng chứng `/data` có tác dụng) |
| Chạy 4 hệ | `manifest verify` không lỗi: cùng `sha256` corpus, cùng `model`, `key_alias` khác nhau |
| Identity | mọi metric ra giá trị lý tưởng — `page_inflation=1.000`, `mIoU≈1.0`, `Masked-SSIM≈1.0` |
| Ceiling | detector trên PDF nguồn chấm với GT người vẽ ra **mIoU ≥ 0.8** — dưới ngưỡng đó thì detector quá yếu, mọi so sánh phía sau vô nghĩa |
| Layout | bảng có đủ 4 hệ + 2 hàng chuẩn, và **thứ hạng không đổi** khi đổi sang detector thứ hai |
| Đồng bộ | xoá sạch `out/` ở máy rồi `sync pull` dựng lại được toàn bộ bảng mà không chạy lại job nào |

Hai hàng `Identity` và `Source ceiling` chính là chỗ paper BabelDOC bỏ trống — bảng
nào của luận văn cũng phải có chúng.

---

## 11. Chi phí & thời gian — ước lượng có neo vào số đo thật

### 11.1 Neo thời gian: PDFTranslator chạy nhanh cỡ nào

Lấy từ `test_local/make_benchmark_charts.py` — **207 trang đã đo thật**:

| | giây | % | giây/trang |
|---|---:|---:|---:|
| parse (Surya + Paddle, GPU) | 327 | 65% | 1.58 |
| translate (chờ mạng LLM) | 137 | 27% | 0.66 |
| render (typst) | 36 | 7% | 0.18 |
| **tổng** | **500** | | **2.42** |

⇒ **120 trang T1 ≈ 5 phút compute thật.** Chưa rõ số này đo trên phần cứng nào, nên
nhân hệ số an toàn 1.5–2.5× cho T4 ⇒ **7–10 phút**.

**Kết luận quan trọng nhất về chi phí: công việc thật quá nhỏ so với chi phí cố định.**
Hoá đơn không do 120 trang quyết định, mà do khởi động container + nạp model + đồng
bộ artifact. Tối ưu code để chạy nhanh hơn là vô ích; thứ thật sự quyết định tiền là
**có mount cache hay không** và **có phải chạy lại hay không**.

### 11.2 GPU — một lượt EN→VI

Giá flavor theo `huggingface.co/docs/hub/jobs-pricing` (kiểm lại trước khi chạy).

| Job | Flavor | $/h | lạc quan | thực tế |
|---|---|---:|---|---|
| `check` kiểm image | `cpu-basic` | 0.00 | 4 ph · $0.00 | 8 ph · $0.00 |
| `warm` tải model vào `/data` | `t4-small` | 0.40 | 15 ph · $0.10 | 30 ph · $0.20 |
| A pdftranslator | `t4-medium` | 0.60 | 18 ph · $0.18 | 30 ph · $0.30 |
| B babeldoc | `t4-medium` | 0.60 | 14 ph · $0.14 | 25 ph · $0.25 |
| C pdfmathtranslate | `t4-medium` | 0.60 | 14 ph · $0.14 | 25 ph · $0.25 |
| E chấm điểm (detector + SSIM + QE) | `t4-medium` | 0.60 | 30 ph · $0.30 | 55 ph · $0.55 |
| **một lượt sạch** | | | **$0.86** | **$1.55** |
| \+ lượt EN→ZH (parse đã cache, bỏ warm/check) | | | $0.47 | $0.85 |
| **× 2.5 khi debug** | | | **$2.15** | **$3.88** |

CometKiwi bản XL cần ≥24 GB ⇒ `l4x1` $0.80/h × ~40 ph = **$0.53** — gần như không
đắt hơn bản nhỏ trên T4, nên nếu muốn dùng XL thì cứ dùng.

**Rủi ro tiền lớn nhất, và nó không nằm ở giá flavor:** quên mount `/data` (hoặc
dính bẫy `CMD` ở §2.1) ⇒ mỗi job tải lại 3–5 GB Surya/Paddle/RT-DETR trong lúc đang
bị tính tiền. 4 job × 10–20 phút × $0.60/h = **+$0.40–0.80, tức đội ~50% hoá đơn**,
mà không đổi lấy một con số nào.

### 11.3 DeepL — **$0**, nhưng biên rất hẹp

Sổ ký tự T1 EN→VI (đo bằng `verify_corpus`, DeepL tính **tối thiểu 50.000/file**):

| file | ký tự thật | bị tính | |
|---|---:|---:|---|
| T1_manuals | 38,498 | 50,000 | ⚠️ sàn, overpay 1.3× |
| T1_laws_and_regulations | 41,034 | 50,000 | ⚠️ sàn, overpay 1.2× |
| T1_government_tenders | 50,439 | 50,439 | |
| T1_scientific_articles | 52,878 | 52,878 | |
| T1_financial_reports | 57,500 | 57,500 | |
| T1_patents | 92,844 | 92,844 | |
| **tổng** | **333,193** | **353,661** | lãng phí do sàn: 20,468 (5.8%) |

Gói Developer miễn phí **1.000.000 ký tự/tháng**:

| kịch bản | bị tính | |
|---|---:|---|
| 1 lượt vi | 353,661 | ✅ |
| 1 vi + 1 zh | 707,322 | ✅ còn dư 292,678 |
| 2 vi + 1 zh (chạy lại vi **một** lần) | 1,060,983 | ❌ vượt 60,983 |
| 2 vi + 2 zh | 1,414,644 | ❌ vượt 414,644 |

⇒ **Được đúng một lần chạy sạch cho mỗi ngôn ngữ. Không có chỗ cho một lần chạy lại.**

Hai kỷ luật rút ra:

1. **Chạy DeepL SAU CÙNG**, sau khi 3 hệ mã nguồn mở đã ra kết quả và luồng chấm
   điểm đã chạy trơn. Debug trên DeepL là đốt hạn mức của cả tháng.
2. **Một document thất bại vẫn bị tính tiền** — DeepL đã nhận file và đã xử lý.
   Nên `--dry-run` trước, và `--char-budget 950000` để dừng có kiểm soát.

Nếu buộc phải vượt: phải mua gói trả tiền, **kiểm giá hiện hành của DeepL trước khi
cam kết** — đừng lấy con số cũ trong tài liệu này.

Nếu sau này thêm corpus document-mode 126 trang (421,582 ký tự, **0 file bị sàn**):
chỉ document-mode vi = 421,582 ✅ · cả hai chế độ vi = 775,243 ✅ · cả hai chế độ
vi+zh = 1,550,486 ❌.

### 11.4 LLM — hạng mục thứ ba, và nó phụ thuộc model

333k ký tự nguồn ≈ 100k token vào, ~120k token ra, **× 3 hệ** (thêm ~20% cho lượt
trích glossary của PDFTranslator và BabelDOC):

* model rẻ (~$0.10 / $0.40 mỗi triệu token) ⇒ **~$0.20**
* model tầm trung (~$1 / $5) ⇒ **~$2**

Chọn model rẻ cho lượt debug, model chốt cho lượt lấy số.

### 11.5 Tổng

| | lạc quan | thực tế | có debug |
|---|---:|---:|---:|
| GPU (HF Jobs) | $0.86 | $1.55 | $3.88 |
| DeepL | $0 | $0 | $0 (nhưng hết hạn mức) |
| LLM | $0.20 | $0.50 | $2–5 |
| **tổng** | **~$1** | **~$2** | **~$6–9** |

Thời gian tường: **~2 giờ** cho một lượt EN→VI đầy đủ, trong đó phần chạy thật chỉ
~10 phút — phần còn lại là khởi động, nạp model, đồng bộ.

## 12. Rủi ro đã biết

| Rủi ro | Xử lý |
|---|---|
| **`hf jobs run` xoá `CMD` ⇒ mất cấu hình cache** | Đổi sang `ENTRYPOINT` bọc (§2.1). Nếu quên: mỗi job tải lại 3–5 GB và bạn trả tiền cho nó |
| **Image là ảnh chụp của Space** | Push Space trước, chạy job sau. Ghi `git_rev` vào `_run/<run_id>.json` để hậu kiểm |
| **Timeout mặc định 30 phút** | Luôn truyền `--timeout` |
| **API `docling` chưa xác minh** | Chưa cài được vì mạng đang chặn. Chốt tên hàm ngay bước đầu, có phương án hai là `DocumentConverter` |
| **CometKiwi-XL không vừa T4** | Job riêng `l4x1`, hoặc dùng bản `wmt22` và ghi rõ |
| **`--lang-out zh` của BabelDOC rơi về họ font Latin** | `get_font_family` khớp `"CN"` chứ không khớp `"ZH"`. Lát EN→ZH phải truyền `zh-CN`. Không ảnh hưởng EN→VI |
| **PDFMathTranslate không đếm token** | Lấy qua virtual key ở LiteLLM `/spend/logs` |
| **Dựng lại corpus giữa chừng** | `manifest verify` chặn bằng `sha256`; mọi artifact cũ hết giá trị |
| **DeepL hết hạn mức giữa batch** | Runner kiểm `/v2/usage` trước mỗi lần gọi, chạm trần thì exit code 2 và resume được |

---

## 13. Việc đầu tiên — chỉ còn phần môi trường

Code đã xong. Sáu việc còn lại đều là cấu hình, làm theo đúng thứ tự này:

1. Điền `.env` theo [`../benchmark/e2e/.env.bench.example`](../benchmark/e2e/.env.bench.example)
2. Tạo `HF_TOKEN` (quyền **write**) + dataset repo + bucket cache, rồi
   `bash benchmark/e2e/run_hf.sh push-corpus`
3. Tạo Space `$HF_BENCH_SPACE`, đẩy `Dockerfile.bench` lên đó dưới tên `Dockerfile`
4. `bash benchmark/e2e/run_hf.sh check` — ba dòng `--version` chạy sạch là image đúng
5. Ép `temperature=0` + tạo 4 virtual key ở LiteLLM proxy
6. Chốt `BENCH_MODEL`, ghi vào `.env`, **không đổi nữa**

Rồi: `warm` → `run` → DeepL ở máy → `score` → `pull`.
