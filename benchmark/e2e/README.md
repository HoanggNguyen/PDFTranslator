# Benchmark E2E — PDFTranslator vs BabelDOC / PDFMathTranslate / DeepL

Đo **PDF output** của từng hệ: layout preservation, visual fidelity, toàn vẹn nội
dung, chất lượng dịch, chi phí. Thiết kế đầy đủ: [../../docs/EVALUATION_PLAN.md](../../docs/EVALUATION_PLAN.md).
Hướng dẫn thao tác từ đầu đến bảng kết quả (kể cả HF Jobs): [../../docs/E2E_RUNBOOK.md](../../docs/E2E_RUNBOOK.md).
Chạy trên Hugging Face — hướng dẫn từng bước: [../../docs/HF_GUIDE.md](../../docs/HF_GUIDE.md) · kiến trúc và lý do: [../../docs/E2E_HF_PLAN.md](../../docs/E2E_HF_PLAN.md).

Vị trí trong bức tranh chung: `benchmark/parser/` đo Phase 1 (layout/OCR vs
OmniDocBench), `benchmark/translation/` đo Phase 2 (COMET/chrF++ vs WMT24++).
**Không cái nào đo Phase 3 hay mức tài liệu đầu-cuối** — đó là chỗ trống package này lấp.

Bất biến: harness **chỉ đọc và gọi**, không sửa `pdf2zh/`. Sau mọi thay đổi,
`git diff pdf2zh/` phải rỗng.

---

## Trạng thái

| | |
|---|---|
| ✅ T1 corpus (DocLayNet, 120 trang, GT người vẽ) | `datasets/build_doclaynet.py` |
| ✅ Cửa chặn corpus | `datasets/verify_corpus.py` |
| ✅ Runner PDFTranslator | `runners/pdftranslator.py` |
| ✅ Runner DeepL Document | `runners/deepl_doc.py` |
| ✅ Driver (local + HF Jobs) | `run_all.sh` |
| ✅ Runner BabelDOC | `runners/babeldoc.py` |
| ✅ Runner PDFMathTranslate | `runners/pdfmathtranslate.py` |
| ✅ Hàng chuẩn Identity | `runners/identity.py` — kiểm chính harness |
| ✅ Manifest + cửa chặn drift | `manifest.py` |
| ✅ Image benchmark (3 hệ, 1 phần cứng) | `Dockerfile.bench` + `script/entrypoint-bench.sh` |
| ✅ Đồng bộ HF ↔ máy | `sync.py` — dataset repo là nguồn sự thật duy nhất |
| ✅ Driver HF Jobs | `run_hf.sh` — warm / check / run / score / pull |
| ✅ Render trang 150 DPI | `parse/render_pages.py` — dùng chung cho detector và SSIM |
| ✅ Detector chấm điểm | `parse/run_detectors.py` — Docling RT-DETR (+ surya cho §P6) |
| ✅ Metric layout (nhóm A) | `metrics/eval_preserve.py` — mIoU, Anchor-IoU, collision, margin, τ |
| ✅ Metric visual (nhóm B) | `metrics/eval_visual.py` — Masked-SSIM, ink-profile |
| ✅ Metric không cần detector (nhóm C/E) | `metrics/eval_text.py` — page inflation, UTB, number recall, sec/page, success rate |
| ✅ Ghép cặp câu dùng chung 4 hệ | `align/extract_pairs.py` |
| ✅ Metric chất lượng dịch (nhóm D) | `metrics/eval_qe.py` — CometKiwi + hiệu chuẩn |
| ✅ Bootstrap CI + paired test + report | `metrics/aggregate.py` |
| ⬜ Venv baseline / image chưa dựng trên máy này | mạng đang chặn pypi + huggingface |
| ⬜ T2 (multi-page), T3 (scanned) | §2 |

---

## Chạy

```bash
# 0) Dựng corpus T1 (lần đầu ~12 phút cho scan; LUÔN dùng --scan-cache)
python -m benchmark.e2e.datasets.build_doclaynet \
    --out benchmark/e2e/datasets/corpus/T1 --per-domain 20 --seed 42 \
    --scan-cache benchmark/e2e/datasets/.doclaynet_scan.json

# 1) Cửa chặn — chạy TRƯỚC mọi job, fail thì dừng
python -m benchmark.e2e.datasets.verify_corpus --corpus benchmark/e2e/datasets/corpus

# 2) Xem trước tiền DeepL, không gọi API
DRY_RUN=1 TIERS=T1 LANGS=vi bash benchmark/e2e/run_all.sh

# 3) Chạy thật ở máy
TIERS=T1 LANGS=vi bash benchmark/e2e/run_all.sh

# 3b) Chấm điểm khối không cần detector (chỉ đọc artifact, chạy lại bao nhiêu lần cũng được)
python -m benchmark.e2e.metrics.eval_text \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out --tiers T1 --langs vi

# 4) Chấm điểm đầy đủ ở máy (cần docling + fasttext)
python -m benchmark.e2e.parse.render_pages    --corpus ... --out ... --tiers T1 --langs vi
python -m benchmark.e2e.parse.run_detectors   --out ... --detectors docling
python -m benchmark.e2e.metrics.eval_preserve --corpus ... --out ... --detector docling
python -m benchmark.e2e.metrics.eval_visual   --corpus ... --out ...
python -m benchmark.e2e.align.extract_pairs   --corpus ... --out ...
python -m benchmark.e2e.metrics.eval_qe       --out ... --langs vi
python -m benchmark.e2e.metrics.aggregate     --out ... --langs vi

# 5) Lượt CHÍNH THỨC: cả 3 hệ trên cùng một phần cứng ở HF Jobs
bash benchmark/e2e/run_hf.sh check       # image có đủ 3 hệ chưa
bash benchmark/e2e/run_hf.sh warm        # nung cache model vào /data, 1 lần
bash benchmark/e2e/run_hf.sh push-corpus
bash benchmark/e2e/run_hf.sh run         # 3 hệ, TUẦN TỰ, cùng flavor
#   ... rồi chạy DeepL ở máy + sync push --only out/deepl-document ...
bash benchmark/e2e/run_hf.sh score       # detector + mọi metric + report
bash benchmark/e2e/run_hf.sh pull
```

Biến môi trường và khoá cần chuẩn bị: [`.env.bench.example`](.env.bench.example).

Từng runner chạy độc lập được (mỗi hệ một tiến trình — xem phần venv bên dưới):

```bash
python -m benchmark.e2e.runners.pdftranslator --corpus ... --out ... --langs vi
python -m benchmark.e2e.runners.deepl_doc     --corpus ... --out ... --dry-run
python -m benchmark.e2e.runners.pdftranslator --warmup-only     # làm nóng cache model

# Baseline: chạy trong venv riêng, gọi qua subprocess
BABELDOC_BIN=~/venvs/babeldoc/bin/babeldoc \
python -m benchmark.e2e.runners.babeldoc --corpus ... --out ... --langs vi --model <id>
PDFMATHTRANSLATE_BIN=~/venvs/pdfmath/bin/pdf2zh \
python -m benchmark.e2e.runners.pdfmathtranslate --corpus ... --out ... --langs vi --model <id>
```

### Venv cho hai baseline

Bắt buộc tách venv, hai lý do độc lập:

* **Tên package trùng.** Repo PDFMathTranslate cũng đặt tên package là `pdf2zh`.
  Cùng interpreter thì cái vào `sys.modules` trước che cái sau — benchmark vẫn chạy
  bình thường, chỉ là đo sai hệ thống.
* **Dep vênh.** PDFMathTranslate ghim `pymupdf<1.25.3`, BabelDOC cần `pymupdf>=1.26.7`.

```bash
# checkout đã có sẵn ở ../BabelDOC và ../PDFMathTranslate
uv venv ~/venvs/babeldoc --python 3.12 && uv pip install --python ~/venvs/babeldoc -e ../BabelDOC
uv venv ~/venvs/pdfmath  --python 3.12 && uv pip install --python ~/venvs/pdfmath  -e ../PDFMathTranslate
~/venvs/babeldoc/bin/babeldoc --warmup     # tải asset onnx trước, đừng tính vào giờ đo
```

`--model` là **bắt buộc** ở cả hai runner: để mặc định thì BabelDOC và
PDFMathTranslate đều dùng `gpt-4o-mini`, khác model của PDFTranslator, và bảng so
sánh mất nghĩa ngay. Mức song song cũng ép bằng nhau (`--qps 8` / `--thread 8` /
`concurrent=8`) vì cột sec/page ở §4.5 mà lệch concurrency thì chỉ đang đo
concurrency.

Token: BabelDOC tự log tổng token nên runner bới lại từ `run.log`; PDFMathTranslate
v1 không đếm token ở đâu cả. Cách chung cho cả bốn hệ là mỗi hệ một virtual key ở
LiteLLM (`LITELLM_KEY_ALIAS`, runner ghi vào `meta.json`), rồi quy token/tiền từ
`/spend/logs` của proxy.

---

## Bảy cái bẫy đã xác minh là thật (đừng bỏ cửa chặn)

1. **DeepL tính tối thiểu 50.000 ký tự cho MỖI file PDF.** Corpus T1 gốc là PDF 1
   trang (~3k ký tự) ⇒ gửi lẻ thì đội **16.7×**. Đo thật: 6 PDF 1 trang × 2 ngôn ngữ
   = **600.000 ký tự bị tính** cho 40.000 ký tự thật. Sau khi gộp 20 trang/PDF:
   **353.661 ký tự** cho EN→VI, 4/6 file vượt sàn.
2. **`pdf_cells` của DocLayNet không phải bằng chứng "đọc được text".** Có trang là
   ảnh scan nhúng (0 font, 0 drawing) mà vẫn có `pdf_cells` đầy. `build_doclaynet`
   kiểm lại bằng `get_text()` thật rồi bù từ dự phòng (`--over`).
3. **Stem có dấu chấm làm surya gộp kết quả.** `surya/layout/label.py` lấy key trang
   bằng `basename.split(".")[0]` ⇒ `paper.v2.pdf` thành `paper`, hai file trùng stem
   bị trộn và đánh số lại.
4. **GT ở không gian COCO vuông 1025×1025, PDF thì không** (vd 612×792) ⇒ scale x và y
   khác nhau (1.675 vs 1.294). Chuẩn hoá **từng trục theo chiều của nó**; scale đều ra bbox sai.
5. **PDFMathTranslate cache bản dịch trong sqlite.** Lượt thứ hai (ngôn ngữ thứ hai,
   hay chạy lại sau timeout) ăn cache ⇒ latency và token tụt gần 0, cột hiệu năng
   thành số bịa. `runners/pdfmathtranslate.py` LUÔN truyền `--ignore-cache`.
6. **`~/.config/PDFMathTranslate/config.json` sống dai.** `set_envs` ghi `OPENAI_*`
   vào đó rồi đọc lại lần sau, nên chạy ở shell không có env var là nó âm thầm dùng
   base-url/model của lượt trước. Runner ép `--config` sang file tạm riêng mỗi run.
7. **Baseline chạy với `cwd` là thư mục tạm, nên mọi đường dẫn phải tuyệt đối.**
   Truyền `benchmark/e2e/datasets/corpus/...` tương đối thì BabelDOC trả `rc=0` mà
   không đẻ ra PDF nào — rất dễ đọc thành "baseline thất bại trên document này".
   `_common.absolutize()` xử ở đầu mỗi runner.

---

## Sơ đồ lưu trữ

Bốn pipeline **không chạy cùng lúc** (dep xung đột, và DeepL bị hạn mức theo tháng
nên lượt EN→ZH của nó rơi sang tháng sau). Bố cục dưới đây thiết kế để tích luỹ dần
qua nhiều lượt cách nhau hàng tuần mà vẫn truy được nguồn gốc từng con số.

### Đầu vào — bất biến sau khi dựng

```
datasets/corpus/T1/
  T1_<domain>.pdf     # 6 file × 20 trang, gộp để tránh sàn 50k ký tự của DeepL
  gt.json             # hộp NGƯỜI VẼ, xyxy chuẩn hoá [0,1] theo TỪNG TRỤC
  mapping.json        # ánh xạ ngược từng trang -> file gốc + page_no + page_hash
datasets/.doclaynet_scan.json      # cache quét dataset, KHÔNG phải dữ liệu đầu vào
```

Dựng lại corpus = **đổi bài kiểm**. Nếu buộc phải dựng lại thì mọi artifact cũ hết
giá trị so sánh; `manifest.py` phát hiện việc này qua `sha256` và chặn lại.

### Đầu ra của runner — mỗi lượt ghi thêm, không đè lượt cũ

```
out/_run/<run_id>.json          # mốc lượt chạy: git rev, model, corpus_sha256,
                                # sha256 từng file, phiên bản lib đo. Ghi ở đầu run.
out/<system>/<lang>/<doc_id>/
  output.pdf          # bản SẠCH, luôn là MONO (dual xen trang gốc ⇒ metric hình học vô nghĩa)
  meta.json           # ts, sha256 nguồn, model, system_version, key_alias,
                      # wall/parse/translate/render_s, tokens, chars_billed,
                      # page_inflation, render_stats, accelerator, error
  phase2_translated.json   # chỉ PDFTranslator
  run.log             # chỉ baseline: stdout+stderr của tiến trình con
  raw/                # chỉ baseline: nguyên văn output của nó (kể cả bản dual)
out/pdftranslator/_parse/<doc_id>/phase1_parsed.json    # cache Phase 1, dùng chung mọi ngôn ngữ
```

`_parse/` là cache **độc lập ngôn ngữ**: Phase 1 chạy một lần cho mỗi PDF rồi tái dùng
cho vi, zh và cả 5 mức ρ của ablation §4.6 — không có nó thì ρ-sweep đốt GPU 5×.

Runner **resume theo (doc, lang)**: đã có `output.pdf` + `meta.json` thì bỏ qua. Nên
chạy lại sau timeout không tốn tiền DeepL lần hai, và chạy bổ sung một hệ mới không
đụng gì tới artifact của hệ cũ.

### Kết quả đo — sinh lại được, xoá thoải mái

```
out/_metrics/_srctext/<doc_id>.json     # text trang nguồn, memo hoá dùng chung 4 hệ
out/_metrics/text/<system>.<lang>.json  # summary + record từng doc
out/_metrics/summary.json               # bảng gộp mọi (hệ, ngôn ngữ)
```

Toàn bộ `_metrics/` chỉ đọc artifact, không gọi API, không tốn tiền — chạy lại bao
nhiêu lần cũng được. Xoá cả thư mục rồi chấm lại là an toàn.

### Truy vết giữa các lượt

```bash
python -m benchmark.e2e.manifest list   --out benchmark/e2e/out   # đã chạy những lượt nào
python -m benchmark.e2e.manifest verify --out benchmark/e2e/out   # 4 hệ có so được không
```

`verify` là cửa chặn **thứ hai** (cửa thứ nhất là `verify_corpus`, chặn trước khi tiêu
tiền). Nó chặn ba thứ trôi được giữa các lượt mà PDF đầu ra không hề lộ ra:

| Kiểm | Mức | Vì sao chết người |
|---|---|---|
| cùng `doc_id` mà `sha256` khác nhau | **lỗi** | corpus đã dựng lại ⇒ 2 hệ chấm trên 2 bài khác nhau |
| `model` khác nhau giữa các hệ dùng LLM | **lỗi** | baseline để mặc định là rơi về `gpt-4o-mini` |
| `key_alias` trùng giữa 2 hệ | cảnh báo | không quy được token/USD riêng từng hệ ở LiteLLM |
| phủ tài liệu lệch nhau | cảnh báo | bảng mất dòng mà không báo |

`eval_text` gọi `verify` trước khi chấm và **dừng** nếu có lỗi (`--allow-drift` để bỏ
qua, chỉ dùng khi debug).

---

## Hai điều phải biết khi đọc số

- **`temperature` không đặt được trong hệ.** `pdf2zh/translation/gateway.py` hardcode
  0.7 (dòng 146) và 0.2 (dòng 246), không có knob config, và harness không được sửa
  `pdf2zh/`. Ép temperature ở **LiteLLM proxy** — công bằng hơn, vì cả 3 hệ nhận cùng
  một override tại một điểm.
- **`peak_rss_mb` là trần của cả tiến trình**, đơn điệu tăng, không phải số của riêng
  một tài liệu.
