# Runbook — chạy benchmark E2E so 4 kiến trúc

Hướng dẫn thao tác. Thiết kế và lý do chọn metric nằm ở
[EVALUATION_PLAN.md](EVALUATION_PLAN.md); tóm tắt kiến trúc package ở
[../benchmark/e2e/README.md](../benchmark/e2e/README.md). Tài liệu này trả lời đúng ba
câu: **chạy được chưa**, **chạy thế nào ở máy**, **chạy thế nào trên Hugging Face**.

Mọi lệnh chạy từ **gốc repo** `PDFTranslator/`.

---

## 0. Trạng thái tính đến 2026-09-05

### Chạy được ngay

| | Bằng chứng |
|---|---|
| Corpus T1 + cửa chặn | `verify_corpus` → `OK — 1 tier(s), 2 warning(s)` |
| Manifest + cửa chặn drift | `manifest list` → 1 lượt `20260905-083811` |
| Hàng chuẩn Identity | 6/6 doc, `page_inflation=1.000`, `number_recall=1.000` |
| Khối metric không cần detector | `eval_text` chạy xong, ghi `out/_metrics/summary.json` |
| 4 runner (code) | `pdftranslator`, `babeldoc`, `pdfmathtranslate`, `deepl_doc` |

Identity đi qua sạch nghĩa là **harness tự kiểm đúng**: copy nguyên PDF nguồn làm
"bản dịch" thì mọi metric phải ra giá trị lý tưởng, và nó ra đúng vậy.

### Chưa chạy được — 6 việc chặn

| # | Việc | Chặn cái gì | Cách gỡ |
|---|---|---|---|
| 1 | **Mạng chặn `pypi.org` + `huggingface.co`** (đo: cả hai trả `000`, `google.com` trả `301`) | cài `docling`/`fasttext`, dựng venv baseline, mọi thứ liên quan HF | bật VPN |
| 2 | **Venv baseline chưa dựng** (`~/venvs/` không tồn tại) | BabelDOC + PDFMathTranslate | §1.2 |
| 3 | **`DEEPL_AUTH_KEY` không có trong `.env`** | DeepL | xin key Developer (miễn phí, 1M ký tự/tháng) |
| 4 | **Chưa chốt `MODEL`** (manifest ghi `model=null`) | cả 3 hệ dùng LLM — `run_all.sh` chặn baseline nếu `MODEL` rỗng | chọn 1 model có trên LiteLLM proxy |
| 5 | **`fasttext` chưa cài** → `lid_backend=heuristic` | UTB **không dùng được cho luận văn** | §1.1 |
| 6 | ~~Tầng detector + metric layout/visual/QE~~ | — | **đã viết xong**, xem [E2E_HF_PLAN.md §7](E2E_HF_PLAN.md); chỉ còn chờ cài `docling` (việc 1) |

Nói gọn: **toàn bộ code đã xong và đã tự kiểm bằng hàng chuẩn; còn lại thuần là
việc môi trường** — mở mạng, dựng image, chạy thật. Lượt chính thức chạy bằng
`run_hf.sh` (xem [E2E_HF_PLAN.md](E2E_HF_PLAN.md)), không phải `run_all.sh`.

Sau khi gỡ 1–4, thứ ra được ngay là bảng §4.3/§4.5: success rate, page inflation,
sec/page, number recall, UTB (UTB cần thêm việc 5).

---

## 1. Chuẩn bị môi trường — ba interpreter tách rời

Ba môi trường, **không được gộp**. PDFMathTranslate đặt tên package là `pdf2zh`
trùng y hệt PDFTranslator; chung interpreter thì cái nào vào `sys.modules` trước sẽ
che cái kia — benchmark vẫn chạy trơn tru và đo nhầm hệ thống. Thêm nữa
PDFMathTranslate ghim `pymupdf<1.25.3` còn BabelDOC cần `pymupdf>=1.26.7`.

### 1.1 Env harness (`thesis`)

Chạy toàn bộ runner của PDFTranslator, mọi metric, mọi cửa chặn.

```bash
conda activate thesis          # Python 3.12.12
```

Đã có: `pymupdf 1.25.2` · `deepl 1.28.0` · `surya-ocr 0.17.1` · `unbabel-comet 2.2.7`
· `numpy` · `scipy` · `scikit-image` · `torch`.

Còn thiếu — cài bằng script có sẵn (script tự kiểm mạng trước, không treo):

```bash
bash benchmark/e2e/install_deps.sh
```

Nó cài `docling` (detector chấm điểm) + `fasttext` (LID cho UTB) rồi tải model LID
917 KB. **Không có `fasttext` thì `langid.py` rơi về backend heuristic**, `eval_text`
in cảnh báo và ghi `lid_backend: "heuristic"` vào kết quả — chạy được để debug,
không được đưa vào luận văn.

Nếu không dùng `conda activate`, truyền interpreter tường minh:

```bash
export PYTHON=~/miniconda3/envs/thesis/bin/python
```

> `run_all.sh` mặc định gọi `python`. Máy này **không có `python` trần trong PATH**
> khi chưa activate conda — hoặc activate, hoặc đặt `PYTHON=`.

### 1.2 Hai venv baseline

Checkout đã có sẵn ở `../BabelDOC` và `../PDFMathTranslate`:

```bash
uv venv ~/venvs/babeldoc --python 3.12
uv pip install --python ~/venvs/babeldoc -e ../BabelDOC

uv venv ~/venvs/pdfmath  --python 3.12
uv pip install --python ~/venvs/pdfmath  -e ../PDFMathTranslate

# tải asset ONNX/font TRƯỚC, đừng để nó rơi vào đồng hồ đo
~/venvs/babeldoc/bin/babeldoc --warmup
```

Rồi khai báo cho harness:

```bash
export BABELDOC_BIN=~/venvs/babeldoc/bin/babeldoc
export PDFMATHTRANSLATE_BIN=~/venvs/pdfmath/bin/pdf2zh
```

`--bin` nhận **cả console script lẫn interpreter**: trỏ vào `.../bin/python` thì
runner tự gọi `python -c "<module_expr>"`. Với PDFMathTranslate, runner **cố tình
không** rơi về PATH — `pdf2zh` trong PATH gần như chắc chắn là của PDFTranslator.

### 1.3 Khoá API

`.env` ở gốc repo, harness tự `load_dotenv()`:

```dotenv
LITELLM_BASE_URL=https://litellm.internal.cake.vn/v1
LITELLM_API_KEY=sk-...
DEEPL_AUTH_KEY=...          # ← hiện CHƯA có, DeepL không chạy được
```

Cả ba hệ mã nguồn mở đi qua **cùng một LiteLLM proxy**: PDFTranslator dùng provider
`litellm`; BabelDOC và PDFMathTranslate nhận cùng base-url/key qua
`--openai-base-url` / `OPENAI_BASE_URL`. Đây là chỗ duy nhất ép được `temperature=0`
cho cả ba, vì `pdf2zh/translation/gateway.py` hardcode 0.7/0.2 và harness không
được sửa `pdf2zh/`.

Nên tách virtual key mỗi hệ để quy token/tiền riêng ở `/spend/logs`:

```bash
export KEY_ALIAS_PREFIX=thesis-      # -> thesis-pdftranslator, thesis-babeldoc, ...
```

---

## 2. Dataset — đang có bao nhiêu

### Đã dựng: T1, 120 trang

| doc_id | trang | ký tự | tài liệu gốc | sàn DeepL 50k |
|---|---:|---:|---:|---|
| T1_patents | 20 | 92,844 | 20 | ✅ |
| T1_financial_reports | 20 | 57,500 | 20 | ✅ |
| T1_scientific_articles | 20 | 52,878 | 20 | ✅ |
| T1_government_tenders | 20 | 50,439 | 16 | ✅ |
| T1_laws_and_regulations | 20 | 41,034 | 20 | ⚠️ overpay 1.2× |
| T1_manuals | 20 | 38,498 | **7** | ⚠️ overpay 1.3× |
| **TỔNG** | **120** | **333,193** | 103 | |

Kèm **1,785 box ground-truth do người vẽ** (Text 785 · List-item 400 ·
Section-header 166 · Page-header 105 · Page-footer 87 · Picture 69 · Table 60 ·
Footnote 36 · Title 31 · Formula 30 · Caption 16). Nhóm *anchor* của §4.1
(Picture + Table + Formula + Page-header/footer) = **351 box**.

Nguồn: `docling-project/DocLayNet-v1.2`, split **`test`**, seed 42, 20 trang/domain,
`--min-chars 500`, mỗi trang ưu tiên một tài liệu gốc khác nhau.

`manuals` chỉ có 7 tài liệu gốc cho 20 trang — **giới hạn của dataset, phải công bố
trong luận văn**, không phải lựa chọn lấy mẫu.

### Chưa dựng

**T2** (tài liệu nhiều trang) và **T3** (scan) chưa có ⇒ đang ở **120/300 trang**
của plan. Không có T2 thì không đo được cross-page context và glossary; không có T3
thì không có "capability gap" với hai baseline không OCR được.

### Dựng lại corpus

```bash
python -m benchmark.e2e.datasets.build_doclaynet \
    --out benchmark/e2e/datasets/corpus/T1 --per-domain 20 --seed 42 \
    --scan-cache benchmark/e2e/datasets/.doclaynet_scan.json
```

**LUÔN truyền `--scan-cache`.** Quét lần đầu ~12 phút (bandwidth-bound, song song 6
shard chỉ nhanh 1.7× chứ không 6×); có cache thì lấy mẫu lại với seed khác là tức thì.
Cache đã nằm sẵn trong repo (4,999 dòng).

> **Dựng lại corpus = đổi bài kiểm.** Mọi artifact cũ mất giá trị so sánh.
> `manifest verify` bắt được qua `sha256` và chặn lại.

### Cửa chặn corpus

```bash
python -m benchmark.e2e.datasets.verify_corpus --corpus benchmark/e2e/datasets/corpus
```

Bắt: PDF không có text layer · dưới sàn 50k ký tự của DeepL · stem có dấu chấm
(surya lấy key trang bằng `basename.split(".")[0]`, hai file trùng stem bị trộn) ·
`sha256` lệch manifest. `--strict` coi warning là lỗi.

---

## 3. Chạy ở máy

### 3.1 Thứ tự bắt buộc

```
verify_corpus → manifest write → DeepL dry-run (xem tiền) → runner → eval_text
```

`run_all.sh` đã ràng đúng thứ tự này. Đừng phát job khi corpus chưa qua cửa chặn.

### 3.2 Nháp — không tốn một đồng

```bash
DRY_RUN=1 TIERS=T1 LANGS=vi bash benchmark/e2e/run_all.sh
```

Dừng sau cửa chặn + dự báo ký tự DeepL. Dùng để kiểm key, kiểm venv, kiểm corpus.

### 3.3 Hàng chuẩn Identity — chạy trước mọi thứ

```bash
SYSTEMS="identity" TIERS=T1 LANGS=vi bash benchmark/e2e/run_all.sh
```

Copy PDF nguồn làm "output". Mọi metric phải ra lý tưởng: `page_inflation=1.000`,
`number_recall=1.000`. Nếu không thì lỗi ở harness chứ không ở hệ nào. Rẻ, nhanh,
và là thứ paper BabelDOC không có.

### 3.4 Lượt so sánh chính thức

```bash
export BABELDOC_BIN=~/venvs/babeldoc/bin/babeldoc
export PDFMATHTRANSLATE_BIN=~/venvs/pdfmath/bin/pdf2zh
export KEY_ALIAS_PREFIX=thesis-

SYSTEMS="pdftranslator babeldoc pdfmathtranslate deepl" \
MODEL=google/gemini-3.1-flash-lite \
TIERS=T1 LANGS=vi \
bash benchmark/e2e/run_all.sh
```

`MODEL` là **bắt buộc** khi có baseline: để trống thì BabelDOC và PDFMathTranslate
đều rơi về `gpt-4o-mini`, khác model của PDFTranslator, và bảng mất nghĩa. Preflight
chặn thẳng trường hợp này.

Mức song song đã ép bằng nhau: `--qps 8` (BabelDOC) / `--thread 8` (PDFMathTranslate)
/ `concurrent=8` (PDFTranslator). Lệch concurrency thì cột sec/page chỉ đang đo
concurrency.

### 3.5 Chạy lẻ từng hệ

Bốn hệ **không chạy cùng lúc** — dep xung đột, và DeepL bị hạn mức theo tháng nên
lượt EN→ZH của nó thường rơi sang tháng sau. Runner resume theo `(doc, lang)`: đã có
`output.pdf` + `meta.json` thì bỏ qua, nên chạy bổ sung không đụng artifact cũ.

```bash
# PDFTranslator
python -m benchmark.e2e.runners.pdftranslator \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi --provider litellm --model <id>

# làm nóng cache model (Surya + Paddle), không dịch gì
python -m benchmark.e2e.runners.pdftranslator --warmup-only

# BabelDOC
BABELDOC_BIN=~/venvs/babeldoc/bin/babeldoc \
python -m benchmark.e2e.runners.babeldoc \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi --model <id>

# PDFMathTranslate
PDFMATHTRANSLATE_BIN=~/venvs/pdfmath/bin/pdf2zh \
python -m benchmark.e2e.runners.pdfmathtranslate \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi --model <id>

# DeepL — xem tiền trước
python -m benchmark.e2e.runners.deepl_doc \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi --dry-run
```

DeepL trả **exit code 2** khi chạm `--char-budget` — dừng có kiểm soát, không phải lỗi.

### 3.6 Chấm điểm

```bash
python -m benchmark.e2e.metrics.eval_text \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi
```

Chỉ đọc artifact, không gọi API, không tốn tiền — chạy lại bao nhiêu lần cũng được,
xoá cả `out/_metrics/` rồi chấm lại là an toàn. Ra: success rate · page inflation ·
số doc bị reflow · sec/page · UTB/trang · number recall.

Nó gọi `manifest.verify` trước và **dừng nếu có lỗi**. `--allow-drift` chỉ để debug.

### 3.7 Truy vết giữa các lượt

```bash
python -m benchmark.e2e.manifest list   --out benchmark/e2e/out
python -m benchmark.e2e.manifest verify --out benchmark/e2e/out
```

`verify` chặn ba thứ trôi được giữa các lượt mà PDF đầu ra không hề lộ ra:

| Kiểm | Mức | Vì sao chết người |
|---|---|---|
| cùng `doc_id` mà `sha256` khác nhau | **lỗi** | corpus đã dựng lại ⇒ hai hệ chấm trên hai bài khác nhau |
| `model` khác nhau giữa các hệ dùng LLM | **lỗi** | baseline để mặc định là rơi về `gpt-4o-mini` |
| `key_alias` trùng giữa hai hệ | cảnh báo | không quy được token/USD riêng từng hệ |
| phủ tài liệu lệch nhau | cảnh báo | bảng mất dòng mà không báo |

---

## 4. Đọc kết quả

```
out/_run/<run_id>.json          # mốc lượt chạy: git rev, model, corpus_sha256,
                                # sha256 từng file, phiên bản lib đo
out/<system>/<lang>/<doc_id>/
  output.pdf                    # bản SẠCH, luôn MONO
  meta.json                     # ts, sha256 nguồn, model, key_alias, wall/parse/
                                # translate/render_s, tokens, chars_billed,
                                # page_inflation, render_stats, accelerator, error
  phase2_translated.json        # chỉ PDFTranslator
  run.log                       # chỉ baseline: stdout+stderr tiến trình con
  raw/                          # chỉ baseline: nguyên văn output của nó
out/pdftranslator/_parse/<doc_id>/phase1_parsed.json   # cache Phase 1
out/_metrics/summary.json       # bảng gộp mọi (hệ, ngôn ngữ)
```

`output.pdf` **luôn là bản mono**. Bản dual xen trang gốc với trang dịch nên mọi
metric hình học sẽ vô nghĩa; bản dual vẫn giữ trong `raw/` để đối chiếu thủ công.

`_parse/` là cache **độc lập ngôn ngữ**: Phase 1 chạy một lần mỗi PDF rồi dùng lại
cho vi, zh và cả 5 mức ρ của ablation §4.6 — không có nó thì ρ-sweep đốt GPU 5×.

Hai điều phải nhớ khi đọc số:

- **`temperature` không đặt được trong hệ** (`gateway.py` hardcode 0.7/0.2, không có
  knob config) ⇒ ép ở LiteLLM proxy. Công bằng hơn: cả ba hệ nhận cùng một override
  tại cùng một điểm.
- **`peak_rss_mb` là trần của cả tiến trình**, đơn điệu tăng, không phải số của riêng
  một tài liệu.

---

## 5. Chạy trên Hugging Face

> **Toàn bộ phần này đã chuyển sang [HF_GUIDE.md](HF_GUIDE.md)** — hướng dẫn thao
> tác từng bước, có cửa kiểm sau mỗi bước và mục sự cố riêng của HF.

Ba điều cần nhớ ngay cả khi không mở guide:

1. **Lượt chính thức chạy bằng `benchmark/e2e/run_hf.sh`, không phải `run_all.sh`.**
   `run_all.sh` là để debug ở máy và nhánh `RUNTIME=hfjobs` của nó chỉ chạy được
   PDFTranslator + DeepL trên image Space demo. `run_hf.sh` dùng image
   `Dockerfile.bench` chứa **cả ba hệ**, nên chúng chạy trên **cùng một phần cứng**
   — điều kiện để cột giây/trang có nghĩa.

2. **DeepL không lên HF.** Nó là API thuần; chạy ở máy rồi
   `sync push --only out/deepl-document` **trước** khi phát job chấm điểm. Job chấm
   điểm quét mọi thư mục dưới `out/` và chấm y hệt nhau, nên DeepL vẫn có đủ metric
   như ba hệ kia.

3. **Bẫy tốn tiền nhất: `hf jobs run` thay `CMD`.** Logic chọn thư mục cache của
   `Dockerfile` (Space demo) nằm trong `CMD` nên job không bao giờ chạy nó — model
   cache rơi về `/app/.cache` và **mỗi job tải lại 3–5 GB trong lúc đang bị tính
   tiền theo phút**. `Dockerfile.bench` chuyển sang `ENTRYPOINT`, thứ mà
   `hf jobs run` không thay được.

Chi phí: một lượt EN→VI ≈ **$1.55**, kể cả debug ≈ $3.88 — chi tiết ở
[E2E_HF_PLAN.md §11](E2E_HF_PLAN.md).

## 6. Tra cứu biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PYTHON` | `python` | Interpreter của env harness |
| `RUNTIME` | `local` | `local` \| `hfjobs` |
| `SYSTEMS` | `"pdftranslator deepl"` | Cách nhau bởi khoảng trắng; thêm `babeldoc pdfmathtranslate identity` |
| `TIERS` | `T1` | Cách nhau bởi dấu phẩy |
| `LANGS` | `vi` | Cách nhau bởi khoảng trắng |
| `MODEL` | *(rỗng)* | **Bắt buộc khi có baseline** |
| `PROVIDER` | `litellm` | Provider của PDFTranslator |
| `BABELDOC_BIN` | *(rỗng)* | Console script hoặc python của venv BabelDOC |
| `PDFMATHTRANSLATE_BIN` | *(rỗng)* | Console script hoặc python của venv PDFMathTranslate |
| `KEY_ALIAS_PREFIX` | *(rỗng)* | Tiền tố virtual key LiteLLM mỗi hệ |
| `DEEPL_CHAR_BUDGET` | `950000` | Dừng trước khi vượt; Developer = 1M |
| `RESUME` | `1` | `0` = chạy lại cả doc đã có artifact |
| `STRICT` | `0` | `1` = `verify_corpus` coi warning là lỗi |
| `DRY_RUN` | `0` | `1` = dừng sau cửa chặn + dự báo |
| `CORPUS` / `OUT` | `benchmark/e2e/datasets/corpus` / `benchmark/e2e/out` | |
| `HF_SPACE` / `HF_CACHE_BUCKET` | *(rỗng)* | Chỉ khi `RUNTIME=hfjobs` |
| `FLAVOR_GPU` / `FLAVOR_CPU` | `t4-medium` / `cpu-upgrade` | |
| `TIMEOUT` | `3h` | Timeout job HF |

---

## 7. Sự cố hay gặp

**`python: command not found`** — chưa `conda activate thesis`, hoặc đặt `PYTHON=`.

**`ModuleNotFoundError: No module named 'benchmark'`** — chạy không đứng ở gốc repo.

**BabelDOC trả `rc=0` mà không có PDF nào** — hai nguyên nhân:
1. Đường dẫn tương đối. Baseline chạy với `cwd` là thư mục tạm nên mọi đường dẫn
   phải tuyệt đối; `_common.absolutize()` đã xử ở đầu runner.
2. BabelDOC xử lý sự kiện `error` bằng `logger.error` + `break` rồi **thoát bình
   thường**. Exit code không chứng minh gì — runner lấy *"có PDF hay không"* làm
   phán quyết và bới `run.log` tìm dòng lỗi.

**PDFMathTranslate nhanh bất thường ở ngôn ngữ thứ hai** — nó cache bản dịch trong
sqlite. Runner **luôn** truyền `--ignore-cache`; nếu bạn gọi tay thì nhớ thêm.

**PDFMathTranslate dùng sai model/base-url** — `~/.config/PDFMathTranslate/config.json`
sống dai: `set_envs` ghi `OPENAI_*` vào đó rồi đọc lại lần sau. Runner ép `--config`
sang file tạm riêng mỗi lượt.

**`lid_backend = heuristic`** — chưa cài `fasttext`. UTB vẫn ra số nhưng **không dùng
cho luận văn**.

**`manifest verify` báo `model không đồng nhất`** — có hệ chạy trước khi `MODEL` được
đặt. Xoá artifact của hệ đó rồi chạy lại; đừng dùng `--allow-drift` để lách.

**`manifest verify` báo `corpus đã trôi`** — corpus đã dựng lại giữa hai lượt. Mọi
artifact trước đó hết giá trị so sánh.

**DeepL exit code 2** — chạm `--char-budget`. Có kiểm soát, không phải lỗi.

**Job HF chết ở phút thứ 30** — quên `--timeout`.

**Job HF chạy code cũ** — quên push Space.

---

## 8. Cần gì nữa để ra bảng luận văn

Theo thứ tự phụ thuộc:

1. **Gỡ 6 việc chặn ở §0** → ra được bảng §4.3/§4.5 (success rate, page inflation,
   sec/page, number recall, UTB) cho cả 4 hệ.
2. **`parse/run_detectors.py`** — chạy Docling RT-DETR lên PDF nguồn và lên output
   của cả 4 hệ, ghi box đã chuẩn hoá về `out/layout/<detector>/<system>/...`. Đây là
   thứ chặn **trục chính**. Chọn Docling vì **không hệ nào dưới bài kiểm dùng nó**, và
   nó train trên DocLayNet train split nên trần đo rất cao — mọi sụt giảm quy được cho
   translator chứ không cho detector. Không rò rỉ vì corpus lấy từ split `test`.
3. **`metrics/eval_preserve.py`** — mIoU, F1@0.5, mF1@[.5:.95], **Anchor-IoU**, text
   containment, element retention, **collision rate**, **margin violation**,
   reading-order τ. Tái dùng lõi hình học của
   `benchmark/parser/evaluation/eval_layout.py`, chỉ đổi nguồn ground truth.
4. **Hàng `Source ceiling`** — chạy detector trên PDF nguồn chưa dịch rồi chấm với GT
   người vẽ. Mọi điểm phải đọc *tương đối* so với hàng này. Cùng với Identity, đây là
   hai hàng chuẩn mà paper BabelDOC không có.
5. **`metrics/eval_visual.py`** — Masked-SSIM, ink-profile distance.
6. **`metrics/eval_qe.py`** — CometKiwi QE + hiệu chuẩn bằng WMT24++ `vi_VN`.
7. **`metrics/aggregate.py`** — bootstrap CI 95% + paired test + `report.md`.

Việc 2–4 là phần lõi. Không có chúng thì chưa có luận điểm nào về layout preservation.
