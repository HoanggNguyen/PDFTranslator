# Plan: Benchmark E2E — PDFTranslator vs BabelDOC / PDFMathTranslate / DeepL

**Đã chốt:**
- **Ngôn ngữ:** EN→VI chính (cả 3 tầng) + EN→ZH phụ (chỉ T1)
- **Baseline:** BabelDOC + PDFMathTranslate + DeepL. Bỏ retain-pdf.
- **Quy mô:** **300 trang** — T1 = 120, T2 = 100, T3 = 80. Lộ trình đầy đủ P0–P6.
- **Chất lượng dịch:** CometKiwi QE + LLM-as-judge 2 model. Không human eval.
- **DeepL:** pilot bằng gói **Developer** (1M ký tự, miễn phí) cho track EN→VI, rồi mới quyết mua Growth.
- **Ngân sách:** ~**$10–50** tiền API (§7) + ~**$5–10** compute HF Jobs (§8) · ~8–10 h wall-clock. GPU thuê theo phút trên HF, tái dùng image Space đang demo.

---

## 1. Context

Mục tiêu: chứng minh PDFTranslator vượt trội trên 2 trục mà BabelDOC dùng trong paper ACL 2026 Demo (`arXiv:2605.10845`) — **layout preservation** và **translation quality** — theo cách chặt hơn paper đó.

### 1.1 Dataset của BabelDOC có dùng lại được không? → **KHÔNG**

Xác minh 3 chiều độc lập:

| Kiểm tra | Kết quả |
|---|---|
| Paper §4 | Benchmark 200 trang tự curate (80 arXiv physics/math, 60 technical docs, 60 patents). **Không release** — không URL, không HF, không danh sách nguồn |
| Repo `funstory-ai/BabelDOC` @ v0.6.4 | Không có thư mục benchmark/eval/metric. PDF duy nhất: `examples/ci/test.pdf` (4.4 KB smoke test) |
| **Toàn bộ git history** (`git log --all -S`) | `"DeepL"` → **0 commit**. `"arxiv"` → **0 commit**. `"benchmark"` trong README → **0 commit**. Bộ eval **chưa từng tồn tại** trong repo public |
| `PDFMathTranslate/test/` | Chỉ unit test với ONNX session mock + 3 PDF fixture |

→ Phải tự dựng. Đây là cơ hội: paper có 4 lỗ hổng phương pháp luận rõ ràng, vá được và viết thành một section đóng góp riêng.

### 1.2 Số liệu BabelDOC công bố (đối chiếu **định tính**, không so trực tiếp được)

| System | BIoU↑ | LF↑ | TP↑ | VA↑ | TC↑ | UTB↓ | sec/page |
|---|---|---|---|---|---|---|---|
| DeepL (Doc) | 19.8% | 4.20 | 4.19 | 4.24 | 4.38 | 2.03 | 1.88 |
| PDFMathTranslate | 48.7% | 2.55 | 2.78 | 2.54 | 3.02 | 5.55 | 1.47 |
| BabelDOC | 50.0% | 4.46 | 4.19 | 4.49 | 4.43 | 1.70 | 1.63 |

**Bốn lỗ hổng — và cách vá:**

| # | Lỗ hổng | Cách vá |
|---|---|---|
| 1 | **BIoU đo "layout *giống hệt*", không đo "layout *đẹp*".** DeepL BIoU 19.8% nhưng LF người chấm 4.20 > PDFMathTranslate (BIoU 48.7%, LF 2.55) — metric và cảm nhận **nghịch nhau** | Tách metric theo nhóm: *anchor* (hình/bảng/công thức phải đứng yên) vs *text* (được phép giãn), + đo trực tiếp lỗi nhìn thấy được (chồng chữ, tràn lề) |
| 2 | **Không có trần.** BIoU chạy cùng một parser lên trang gốc và trang dịch; parser có sai số nhưng không báo cáo baseline "gốc vs gốc" | GT do **người annotate** (DocLayNet) + hàng **Identity** và **Source ceiling** trong mọi bảng |
| 3 | **Không tách được layout khỏi dịch.** LLM backend không nêu. Layout kém có thể do text nở chứ không do engine typesetting | **Pseudo-translator** hệ số giãn ρ có kiểm soát (§4.6) + ép cùng LLM qua LiteLLM |
| 4 | Không kiểm định thống kê | Bootstrap CI 95% + paired test |

### 1.3 Phát hiện lớn: **PDFTranslator đã có sẵn ~70% hạ tầng eval** → mở rộng, không dựng mới

| Đã có | Ở đâu | Đo cái gì |
|---|---|---|
| `benchmark/parser/evaluation/eval_layout.py` (639 dòng) | PDFTranslator | IoU localization **P/R/F1@0.5, @0.75, mF1@[.5:.95]**, 2 matcher (COCO greedy 1-1 và connected-component), classification accuracy, OCR edit-distance/CER/WER, reading-order edit distance. Chuẩn hoá bbox [0,1]. Slice theo language/layout/subset/data_source |
| `evaluation/compare_matchers.py` | PDFTranslator | **Ablation 3 chiến lược matching** — vá đúng chỗ mù mờ *"matched by reading order and spatial proximity"* của paper |
| `evaluation/download_dataset.py` + `run_parser/build_pdfs.py` | PDFTranslator | Tải OmniDocBench, đóng ảnh trang thành PDF 32 trang + `mapping.json` — **hạ tầng T3 đã sẵn** |
| `run_parser/run_parser.py` | PDFTranslator | Batch runner: preload weights → warmup bỏ đi → vòng lặp có đo giờ → timing JSON. **Mẫu chuẩn cho runner mới** |
| `benchmark/translation/` (7 file) | PDFTranslator | WMT24++ adapter (**có `vi_VN`**), COMET-DA + chrF++, instrument latency/token/429 qua httpx, aggregate → `report.md` |
| `eval_formula.py`, `eval_formula_cdm.py`, `eval_table.py` | PDFTranslator | Edit-distance LaTeX, CDM, nội dung bảng |
| `detect_scanned_file.py:151-172` | BabelDOC | Hàm **SSIM so 2 ảnh trang** (`skimage.structural_similarity`) — lift được |
| `utils/raster_geometry.py` | BabelDOC | Render trang → ảnh, quản DPI/pixel budget |
| `layout_helper.py:calculate_iou_for_boxes` | BabelDOC | IoU primitive **đúng định nghĩa BabelDOC dùng** |

**Khoảng trống duy nhất — chính là deliverable:**
> Không có evaluation cho **Phase 3 (render)** và **không có evaluation ở mức tài liệu đầu-cuối**. Không gì đo: layout của PDF output so với PDF nguồn, độ giống ảnh, tràn chữ, sai lệch cỡ chữ, số trang, latency E2E theo corpus.

### 1.4 Ràng buộc kỹ thuật đã xác minh

**LLM backend — ép được cùng model:**
BabelDOC **chỉ nhận `--openai`** (`main.py:495` hard-error) · PDFMathTranslate `-s openai` với `OPENAI_BASE_URL` · PDFTranslator có provider `litellm` với `LITELLM_BASE_URL` overridable.
→ **Cả 3 hệ mã nguồn mở route được qua một LiteLLM proxy chung** ⇒ cùng model, `temperature=0`. `.env` của PDFTranslator **đã có `LITELLM_API_KEY` + `LITELLM_BASE_URL`** — dùng được ngay. DeepL là hộp đen ⇒ đánh dấu riêng trong mọi bảng.

**Thiên vị layout model — vấn đề quan trọng nhất:**

| Hệ | Layout model |
|---|---|
| BabelDOC | DocLayout-YOLO DocStructBench ONNX, imgsz 1024, sha3 `60be0612…` |
| PDFMathTranslate | **Cùng model đó** (import từ `babeldoc.assets`) |
| PDFTranslator (Pipeline B) | **Surya** layout + Surya OCR + PaddleOCR table |

→ Chấm bằng Surya thì thiên vị PDFTranslator; bằng DocLayout-YOLO thì thiên vị 2 hệ kia. Xử lý ở §3.

**PDFTranslator có 2 pipeline rời nhau — nhầm là hỏng cả benchmark:**

| | Pipeline A (legacy) | **Pipeline B (hệ thống luận văn)** |
|---|---|---|
| Entry | console script `pdf2zh` | `pdf2zh/e2e.py:run_pipeline` |
| Layout | DocLayout-YOLO + pdfminer | Surya + PyMuPDF + **Typst** |
| Output | `<stem>-mono.pdf`, `<stem>-dual.pdf` | `translated_<uuid8>.pdf` |
| Scanned PDF | **trả JSON, không ra PDF** (`high_level.py:360-374`) | xử lý được |

→ Harness gọi **`pdf2zh.e2e`**, tuyệt đối không dùng console script. `get_parser()` là singleton toàn tiến trình → `warmup()` một lần rồi loop. Chưa có batch runner nào cho Pipeline B.

**Gotcha khác đã xác minh:**
- BabelDOC `--debug` **vẽ hộp debug xanh lá đè lên PDF output** và đổi tên file (thêm infix `.debug.`) → chạy **2 lượt**: sạch (chấm visual) + debug (lấy JSON cấu trúc).
- BabelDOC dump sẵn `layout_generator.json`, `typsetting.json`, và **`translate_tracking.json` chứa cặp `{input, output}` đã align theo paragraph** → dùng thẳng cho COMET, không cần tự align.
- PDFMathTranslate CLI **không có flag `--key`** → API key chỉ qua env var/`config.json`. `--mode precise` không dùng được (submodule chưa init). Có `--dir` batch đệ quy.
- surya: **label set trong README đã cũ**; nguồn đúng là `surya/layout/label.py:LAYOUT_PRED_RELABEL` (14 nhãn CamelCase). Batch folder ghi **một** `results.json` chung, và `split(".")[0]` gộp file trùng stem → đổi tên input thành stem duy nhất không dấu chấm.

### 1.5 Hai việc xử lý ngoài lề (phát hiện khi khảo sát)

1. 🔴 **`test_local/test_pipeline.txt` chứa API key OpenRouter plaintext (`sk-or-v1-…`).** `test_local/` có trong `.gitignore` nên chưa lộ lên remote — nhưng nên **revoke key** và xoá khỏi file.
2. `docs/EVALUATION_PLAN.md` được `benchmark/translation/README.md:4,98` và `__init__.py:5` tham chiếu nhưng **không tồn tại, kể cả trong git history**. Plan này sẽ được ghi vào đó để vá tham chiếu chết.

---

## 2. Dataset — 3 tầng, 300 trang (T1 120 · T2 100 · T3 80)

### T1 — Born-digital, ground-truth layout người vẽ ⭐ *trục chính*

**DocLayNet** (IBM) — `CDLA-Permissive-1.0`.

- Bản HF **`ds4sd/DocLayNet-v1.2`** nhúng cột `pdf` binary → nhẹ hơn nhiều so với `DocLayNet_extra.zip` (7.5 GB) + `DocLayNet_core.zip` (28 GB).
- **PDF 1 trang gốc, born-digital, có text layer** (JSON sidecar chứa text cell + font + size) → cả 3 hệ đều nuốt được.
- 11 class: `Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title`.
- 6 domain: **scientific articles, manuals, patents**, financial reports, laws & regulations, government tenders.
- **120 trang** stratified từ split `test` (5,000 trang), seed cố định, **20 trang/domain × 6 domain**.
- **Gộp thành 6 PDF (1 PDF/domain, 20 trang ≈ 60k ký tự)** trước khi đưa vào mọi runner — bắt buộc để tránh bẫy 50k ký tự của DeepL (§7.4). Gộp **giống nhau cho cả 4 hệ**. Dùng lại `benchmark/parser/run_parser/build_pdfs.py` + `mapping.json` để giữ ánh xạ về từng trang gốc.

**Vì sao đúng:** 3 domain của BabelDOC (khoa học / tài liệu kỹ thuật / bằng sáng chế) **map gần trọn vẹn** vào scientific articles / manuals / patents → giữ so sánh định tính với paper, **mà có GT người vẽ nên xoá sạch sai số parser phía nguồn** (vá lỗ hổng #2).

**Hạn chế phải nói rõ:** PDF 1 trang → không đo được cross-page context / glossary xuyên trang. Đó là việc của T2.

**T1 chạy cả EN→VI và EN→ZH** — đây là lát duy nhất có ZH.

### T2 — Tài liệu nhiều trang, mirror đúng thành phần của BabelDOC

**100 trang**, tự thu thập, giữ tỉ lệ paper ~40/30/30: **~40 trang arXiv** (physics/math, qua arXiv API) + **~30 trang technical documentation** (RFC, datasheet, manual OSS) + **~30 trang patents** (Google Patents / USPTO, public domain). Tỉ lệ làm tròn theo ranh giới tài liệu.

⚠️ **Ràng buộc chọn tài liệu:** mỗi tài liệu phải **≥ 18 trang** (~54k ký tự) để không bị DeepL áp mức tối thiểu 50.000 ký tự/file — xem §7.4. ⇒ khoảng **5–6 tài liệu dài** (~18–20 trang mỗi cái), không phải 30–40 tài liệu ngắn. Điều này cũng *tốt hơn* cho việc đo cross-page context: tài liệu dài thì glossary và ngữ cảnh xuyên trang mới thực sự có tác dụng.

Không có GT layout → **BIoU kiểu BabelDOC** (detector chung chạy cả nguồn lẫn đích) + hàng ceiling + metric visual + metric text. Tầng này đo cross-page context, glossary consistency, page inflation, tốc độ trên tài liệu thật. Chỉ EN→VI.

### T3 — Scanned PDF: năng lực riêng của PDFTranslator

Hạ tầng **đã có sẵn**: `download_dataset.py` (tải OmniDocBench) + `build_pdfs.py` (đóng ảnh thành PDF). Chỉ cần nối vào E2E.

- **OmniDocBench** (CVPR 2025): 981 trang ảnh, 9 loại tài liệu, GT block-level (bbox + reading order) **và GT text/LaTeX/HTML**. License research-only → hợp lệ luận văn, ghi rõ. Lấy **80 trang**, đóng thành **3 PDF (~27 trang/PDF)** bằng `build_pdfs.py` — vừa đúng định dạng sẵn có, vừa vượt ngưỡng 50k ký tự của DeepL.
- **BabelDOC raise `ScannedPDFError` khi ≥80% trang là scan**; PDFMathTranslate không có đường OCR. DeepL có OCR riêng cho PDF — **phải kiểm chứng thực nghiệm ở P0**, không giả định.
- Đóng khung là **"capability gap"**, không phải "điểm số cao hơn". Báo cáo trung thực failure rate của từng baseline.

### Ghi chú về độ giãn ngôn ngữ — một phân tích có giá trị

EN→VI **giãn** (~1.15–1.3×), EN→ZH **co** (~0.6×). Cùng một engine typesetting bị stress theo hai hướng ngược nhau. Chạy T1 ở cả hai đích cho phép kết luận: *engine nào chỉ chịu được co, engine nào chịu được giãn* — thứ mà paper BabelDOC (chỉ EN→ZH) **không thể** phát hiện.

---

## 3. Xử lý thiên vị detector (quyết định nền)

- **Detector chấm chính:** **Docling layout (RT-DETR, huấn luyện trên DocLayNet train split)** — **không hệ nào dưới bài kiểm dùng nó**, và vì cùng phân bố nhãn với GT DocLayNet nên **trần đo rất cao** ⇒ mọi sụt giảm quy được cho translator chứ không cho detector. Không rò rỉ dữ liệu vì lấy mẫu từ split **test**.
- **Robustness (P6):** chạy lại toàn bộ với **DocLayout-YOLO** và **Surya**. Thứ hạng không đổi qua cả 3 ⇒ kết luận vững; đổi ⇒ phải nói ra. Chính là chỗ paper im lặng.
- **Trục thứ hai không cần detector:** metric visual (§4.2). Trục detector và trục pixel **đồng thuận** ⇒ claim rất mạnh.
- **Hai hàng chuẩn bắt buộc có trong MỌI bảng:**
  - `Identity` — copy PDF gốc làm "output" ⇒ mọi metric phải ≈ lý tưởng. Kiểm tra chính harness.
  - `Source ceiling` — detector chạy trên PDF gốc chưa dịch, chấm với GT ⇒ trần thực tế. Mọi điểm đọc *tương đối* so với hàng này.

---

## 4. Metric

Nguyên tắc: **tách nhóm câu hỏi, không gộp thành một con số.** Tái dùng `eval_layout.py` làm lõi hình học (IoU, matching, chuẩn hoá, P/R/F1, slicing, format report) — chỉ đổi *nguồn ground truth*.

> Phân biệt rõ: `eval_layout.py` hiện đo **"parser tìm box giỏi tới đâu"** (accuracy vs GT). Cái mới đo **"PDF output giữ layout trang nguồn tới đâu"** (preservation vs source). Hình học giống hệt, ngữ nghĩa khác nhau ⇒ module mới `eval_preserve.py` import helper từ `eval_layout`.

### 4.1 Nhóm A — Layout / hình học

Ghép hộp bằng **Hungarian**, cost `1 − IoU`, ràng buộc cùng class sau khi map taxonomy về bộ rút gọn `{Text, Title/Header, List, Table, Figure, Formula, Caption, Page-furniture}`. Toạ độ chuẩn hoá theo khổ trang. Map nhãn surya pin theo `surya/layout/label.py`, **không** theo README.

| Metric | Định nghĩa | Vì sao cần |
|---|---|---|
| **mIoU (≈ BIoU)** | IoU trung bình trên cặp đã ghép | Trục so sánh trực tiếp với paper |
| **F1@0.5 / mF1@[.5:.95]** | Đã có sẵn trong `eval_layout.py` | mIoU thô bỏ qua box không ghép được; F1/mF1 thì không |
| **Anchor-IoU** ⭐ | mIoU **chỉ trên** `Figure, Table, Formula, Page-furniture` | Các phần tử này **phải đứng yên tuyệt đối** — tín hiệu sạch nhất, không nhiễu bởi text nở/co |
| **Text containment** | `area(pred ∩ gt) / area(pred)` trên block text | Text tràn ngoài khung gốc ⇒ containment tụt. Ghép với IoU thành cặp precision/recall |
| **Element retention** | `count_out / count_gt` theo từng class | Bắt lỗi **mất hẳn hình/bảng** — mIoU giấu lỗi này |
| **Collision rate** ⭐ | Tỉ lệ block output chồng lên block khác (`intersection > ε`) | Chữ đè chữ là lỗi nhìn thấy được số 1. **BabelDOC không đo** |
| **Margin violation** ⭐ | Tỉ lệ box vượt content-bbox / khổ trang gốc | Tràn lề |
| **Page inflation** | `pages_out / pages_in` | DeepL reflow ⇒ phình trang. Một con số, tín hiệu rất mạnh |
| **Reading-order τ** | Kendall's tau thứ tự đọc nguồn vs đích trên cặp đã ghép | `eval_layout.py` đã có reading-order edit distance để tái dùng |

⭐ = mới so với BabelDOC → đóng góp của luận văn.

### 4.2 Nhóm B — Visual (không cần detector)

Render nguồn và đích cùng DPI (150), cùng khổ. Tái dùng `raster_geometry.py` + pattern SSIM ở `detect_scanned_file.py:151-172`. `scikit-image` + `opencv-python-headless` đã là dependency sẵn.

| Metric | Ghi chú |
|---|---|
| **Masked-SSIM** ⭐ | Che toàn bộ vùng text (GT box, dilate 2 px) rồi SSIM phần còn lại ⇒ đo hình/bảng/đường kẻ/logo có **đứng nguyên từng pixel** không. Tín hiệu rất cao, gần như không mơ hồ |
| **Ink-profile distance** | Chiếu mật độ mực lên trục ngang/dọc → 1D Wasserstein. Rẻ, không phụ thuộc hệ chữ viết |
| Full-page SSIM | **Chỉ báo cáo để chứng minh nó là metric tồi ở đây** — glyph khác ngôn ngữ làm SSIM sụp bất kể layout tốt hay xấu. Một đoạn phân tích ngắn, có ích cho luận văn |
| LPIPS / DreamSim | Tuỳ chọn. Nhiễu khi đổi hệ chữ viết ⇒ **không** làm metric chính |

### 4.3 Nhóm C — Toàn vẹn nội dung

| Metric | Cách đo |
|---|---|
| **UTB** (untranslated blocks/trang) | Language-ID (fasttext `lid.176`) trên text từng block output. Tái lập đúng metric của BabelDOC |
| **Content loss rate** | Align block nguồn↔đích theo bbox + reading order; % ký tự nguồn không có đích tương ứng. Bắt lỗi LLM nuốt cả đoạn |
| **Number/formula integrity** | Trích mọi số + token toán inline ở nguồn và đích, tính recall. Rẻ, bắt lỗi kinh điển của LLM |
| **Terminology consistency tự động** ⭐ | Mỗi thuật ngữ nguồn → đếm số bản dịch đích khác nhau trong cùng tài liệu ⇒ `1 − (distinct−1)/occurrences`. **Tự động hoá được TC mà paper phải thuê người chấm** |
| **Render-failure signals** | Free từ dict trả về của `render_document` (`renderer.py:77`): `elements_fallback > 0` = typst repair loop đã cháy; `elements_skipped` = nội dung bị rơi. Không cần instrument thêm |

### 4.4 Nhóm D — Chất lượng dịch

**(i) Xếp hạng cross-system ở mức PDF — không cần reference:**
- **CometKiwi QE**: `Unbabel/wmt23-cometkiwi-da-xl` (XLM-R XL 3.5B, cần ≥15 GB VRAM; fallback `wmt22-cometkiwi-da`). Hỗ trợ tiếng Việt. `score_comet.py` hiện dùng `wmt22-comet-da` (có tham chiếu) → **thêm nhánh QE**.
- **LLM-as-a-judge, 2 model**: tái lập **đúng rubric 4 chiều** của BabelDOC trên ảnh trang gốc + ảnh trang dịch:
  - *Layout Fidelity* — "maintenance of columns, margins, font hierarchies, positioning of figures/tables"
  - *Translation Precision* — "accuracy of academic meaning, scientific claims, data descriptions"
  - *Visual Aesthetics* — "professional typography, line spacing, absence of text overlaps or bleeding"
  - *Terminology Consistency* — "uniform use of domain-specific jargon, citations, figure labels"
  - Thang 1–5 + đếm UTB. Judge 1 = **Gemini-2.5-Flash** (để so trực tiếp với paper), judge 2 = model mạnh hơn. **Báo cáo độ đồng thuận giữa 2 judge** (Spearman + % agreement). Anonymize hệ thống + random hoá thứ tự trình bày.
- \+ UTB, terminology consistency, content loss ở §4.3.

**(ii) ⭐ Hiệu chuẩn metric QE bằng reference — đây là lập luận then chốt để (i) hợp lệ:**

Không có dataset PDF nào có reference translation. Nhưng ở **mức text** thì có, và harness đã chạy được:
- **WMT24++** `vi_VN` và `zh_CN` — `benchmark/translation/wmt24pp_adapter.py` đã hoạt động, `score_comet.py` đã cho COMET-DA (có tham chiếu) + chrF++.
- **DoTA** (`liangyupu/DoTA_dataset`) — 126K trang arXiv, test split 1,003, có reference **ZH/FR/DE**. Domain khoa học, sát T2 hơn WMT24++. Thêm 1 adapter theo đúng khuôn `wmt24pp_adapter.py`.

Dùng để chứng minh: **trên chính cặp ngôn ngữ và domain này, CometKiwi (không tham chiếu) tương quan cao với COMET-DA (có tham chiếu)** — báo cáo Pearson/Spearman ở mức segment và mức system. Có con số đó rồi thì việc dùng QE để xếp hạng ở mức PDF là hợp lệ, và luận văn có một mục "metric validation" mà paper BabelDOC không có.

*Rủi ro cần kiểm ở P0:* DoTA có expose arXiv ID để lấy lại PDF gốc không. Nếu **không**, DoTA chỉ dùng ở mức text (đúng mục đích hiệu chuẩn) — không ảnh hưởng plan.

Nguồn cặp câu để chấm: BabelDOC cho sẵn `translate_tracking.json` đã align theo paragraph; PDFTranslator có `phase1_parsed.json`/`phase2_translated.json` cùng schema; PDFMathTranslate **không dump gì** và DeepL là hộp đen → phải trích text từ PDF output và align theo bbox + reading order (module `align/` dùng chung).

### 4.5 Nhóm E — Hiệu năng

sec/page (tách `parse`/`translate`/`render` — `e2e.py:251` đã log sẵn), peak RSS, VRAM, token + chi phí USD, và **success rate** (số PDF chạy xong không crash — nhiều tool crash, con số này thật sự quan trọng và paper không báo cáo). Tái dùng `benchmark/translation/instrument.py` cho token/429/latency HTTP.

Neo sanity có sẵn: `test_local/make_benchmark_charts.py` chứa kết quả timing trước đây (parse chiếm ~60–75% wall time).

### 4.6 ⭐ Thiết kế then chốt: tách layout khỏi dịch bằng pseudo-translator

Chạy mỗi hệ **hai chế độ**:

**(a) Typesetting-only** — thay engine dịch bằng hàm tất định: mỗi segment nguồn → chuỗi tổng hợp có **hệ số giãn ρ ∈ {0.6, 0.8, 1.0, 1.2, 1.5}** (giữ ranh giới từ; ρ=0.6 mô phỏng ZH, ρ=1.2 mô phỏng VI). Khi đó **100% chênh lệch layout do engine typesetting**, không do LLM ngẫu nhiên. Vẽ **"đường cong bền vững layout"** = metric layout theo ρ.
→ Đo trực tiếp claim "adaptive typesetting engine". Paper BabelDOC **không** làm.
→ Chi phí LLM = 0, chỉ tốn wall-clock.
→ Đòn bẩy sẵn có: BabelDOC `--skip-translation` / `--only-parse-generate-pdf`; PDFTranslator patch `Gateway.call` (đúng pattern `test/test_json_translator.py:155-163`) hoặc trỏ `litellm` vào echo-proxy; PDFMathTranslate dùng translator offline.
→ **DeepL không làm được** (hộp đen) ⇒ chỉ ở chế độ (b). Nói rõ hạn chế.

**(b) Real-translation** — cùng LLM qua LiteLLM proxy, `temperature=0`, cùng cặp ngôn ngữ, cùng page range ⇒ đo cả layout lẫn chất lượng dịch đầu-cuối.

### 4.7 Thống kê

Bootstrap 1000 lần theo trang → CI 95%. Paired test (paired bootstrap hoặc Wilcoxon signed-rank) giữa PDFTranslator và **từng** baseline. Báo cáo effect size.

---

## 5. Kiến trúc harness

**Mở rộng `PDFTranslator/benchmark/`, không tạo repo mới** — bám invariant đã có ở `benchmark/translation/README.md:101`: *"sau cùng `git diff pdf2zh/translation/` phải rỗng — harness chỉ đọc & gọi."* Áp dụng rộng: **harness không được sửa `pdf2zh/`**.

```
benchmark/
  parser/          # ĐÃ CÓ — giữ nguyên
  translation/     # ĐÃ CÓ — thêm nhánh QE (CometKiwi) + adapter DoTA
    score_comet.py       # + --qe-model, chạy được cả QE lẫn reference-based
    dota_adapter.py      # ← MỚI, khuôn theo wmt24pp_adapter.py
    calibrate_qe.py      # ← MỚI, tương quan QE vs reference-based
  e2e/             # ← MỚI, toàn bộ tầng E2E
    configs/       # pin version + weight hash, seed, prompt judge, taxonomy map, LiteLLM
    datasets/
      build_doclaynet.py   # stratified sample ds4sd/DocLayNet-v1.2 → PDF + GT COCO
      build_t2_corpus.py   # arXiv API / patents / technical docs
      # T3 dùng lại parser/evaluation/download_dataset.py + run_parser/build_pdfs.py
      manifest/            # {doc_id, source_url, sha256, pages, tier, domain, license}
    runners/       # 1 adapter/hệ → out/<system>/<lang>/<doc_id>/{output.pdf, meta.json}
      pdftranslator.py   # gọi pdf2zh.e2e (KHÔNG dùng console script), warmup() 1 lần
      babeldoc.py        # 2 lượt: sạch + --debug; nhớ infix ".debug."
      pdfmathtranslate.py
      deepl_doc.py       # DeepL Document Translation API trực tiếp
      pseudo.py          # ρ-controlled echo translator cho chế độ (a)
    parse/         # detector chung lên nguồn + mọi output → layout/<system>/<doc>/<page>.json
    align/         # trích + align block nguồn↔đích, chuẩn hoá 1 schema chung
    metrics/
      eval_preserve.py   # import helper hình học từ parser/evaluation/eval_layout.py
      visual.py content.py quality.py efficiency.py
    judge/         # LLM-as-judge 2 model, anonymize + randomize
    report/        # bảng + biểu đồ + bootstrap CI (mẫu: translation/aggregate.py)
    out/           # gitignore
```

Schema chuẩn hoá chung cho box: `{page, class, bbox_norm[4], reading_order, text}`.
`meta.json` mỗi lần chạy: `{wall_seconds, parse_s, translate_s, render_s, peak_rss, tokens_in, tokens_out, usd, exit_code, error}`.

**Cô lập môi trường:** 3 hệ có yêu cầu Python/deps xung đột (PDFMathTranslate 3.11–3.12 + `pymupdf<1.25.3`; PDFTranslator 3.11 + typst; BabelDOC riêng) → mỗi runner chạy trong venv/uv env riêng, giao tiếp qua **subprocess + filesystem**, không import chéo.

---

## 6. Lộ trình P0–P6

| GĐ | Nội dung | Kết quả |
|---|---|---|
| **P0** | 4 runner + `meta.json` + smoke. Chốt LiteLLM proxy, pin version/weight hash. **Smoke DeepL chỉ trên 1 PDF gộp ~20 trang** (không 5 file lẻ — mỗi file lẻ ngốn 50k ký tự của hạn mức, xem §7.4); smoke 3 hệ OSS thì chạy thoải mái. **Kiểm 3 rủi ro:** DeepL có xử lý scanned PDF không, DoTA có arXiv ID không, batching có làm DeepL tính đúng ký tự không | Chạy được đầu-cuối, biết hệ nào crash ở đâu |
| **P1** | T1: DocLayNet 120 trang (6 PDF theo domain) + detector Docling + `eval_preserve.py` + hàng Identity/Ceiling. EN→VI và EN→ZH | **Bảng layout đầu tiên** — phần lõi luận văn |
| **P2** | Masked-SSIM, ink-profile, collision, margin, page inflation, UTB, content loss, number integrity | Vá lỗ hổng #1 |
| **P3** | CometKiwi QE + LLM judge 2 model + terminology consistency; **hiệu chuẩn QE** bằng WMT24++ `vi_VN`/`zh_CN` + DoTA | Vá lỗ hổng #1 (trục còn lại) + mục metric validation |
| **P4** | Pseudo-translator ρ ∈ {0.6, 0.8, 1.0, 1.2, 1.5} → đường cong bền vững layout | Vá lỗ hổng #3 — **đóng góp học thuật mạnh nhất** |
| **P5** | T2 (100 trang, 5–6 tài liệu ≥18 trang, mirror tỉ lệ paper) + T3 (OmniDocBench scanned 80 trang, 3 PDF) | Cross-page context + capability gap |
| **P6** | Bootstrap CI + paired test + robustness qua 3 detector + `compare_matchers` trên dữ liệu preservation + `report.md` cuối | Vá lỗ hổng #2, #4 |

---

## 7. Chi phí & compute — model định lượng

Giá đã tra ngày 2026-09-04 từ `ai.google.dev/gemini-api/docs/pricing` và DeepL. Script tính: `benchmark/e2e/configs/cost_model.py` (sẽ commit cùng harness để tái lập).

### 7.1 Giả định (khai báo rõ để kiểm chứng lại)

| Tham số | Giá trị | Căn cứ |
|---|---|---|
| Ký tự / trang | 3,000 | trang khoa học/kỹ thuật dày |
| Token text nguồn / trang | ~900 | 3,000 ký tự EN |
| Block text / trang | 20 | → số segment cho QE |
| Output VI / EN | ~2.0× token | VI giãn ~1.2× ký tự + tokenizer bất lợi |
| Output ZH / VI | 0.55× | ZH co ~0.6× ký tự |
| Ảnh 1 trang cho judge | ~1,100 token | ~1024×1400 → ~4 tile × 258 tok |
| Phase 1 PDFTranslator | 1.74 s/trang (MPS) · 0.44 (CUDA) | **đo thật** từ `test_local/make_benchmark_charts.py` |
| Hệ số lặp lại khi dev | ×2.5 | chạy lại full corpus 2–3 lần khi debug/đổi config |

Token/trang theo từng hệ (đã gồm retry): **PDFMathTranslate** ~1.45k in / 1.98k out (đơn giản nhất) · **BabelDOC** ~3.28k in / 2.24k out (glossary + cross-page context) · **PDFTranslator** ~4.02k in / 2.58k out (glossary + equation/table vision pass + retry ±15%).

### 7.2 Kết quả

| | **300 trang**<br>(T1 120 / T2 100 / T3 80) | **500 trang**<br>(T1 200 / T2 180 / T3 120) | **600 trang**<br>(T1/T2/T3 = 200) |
|---|---|---|---|
| **Token dịch** | **5.45M** (3.30 in / 2.15 out) | **9.20M** (5.56 / 3.64) | **10.04M** (6.05 / 3.99) |
| ↳ Gemini 2.5 Flash-Lite | $1.19 → **$2.98** (×2.5) | $2.01 → **$5.03** | $2.20 → **$5.50** |
| ↳ Gemini 3.1 Flash-Lite | $4.05 → **$10.13** | $6.85 → **$17.13** | $7.49 → **$18.73** |
| **Token judge** | 3,360 call · **5.04M** | 3,200 call · **4.80M** | 3,200 call · **4.80M** |
| ↳ 2 judge (2.5-Flash + 3.5-Flash) | **$13.96** | **$13.30** | **$13.30** |
| **DeepL** (đã batch) | 1.26M ký tự · **$33** | 2.10M · **$56** | 2.40M · **$65** |
| ↳ *nếu KHÔNG batch* | *12.9M ký tự ·* **$353** | *20.9M ·* **$573** | *20.9M ·* **$573** |
| **Detector inference** | 3,725 page | 6,235 | 6,600 |
| **QE segment** | 42,500 | 64,700 | 70,000 |
| **GPU-hour (CUDA)** | 0.74 h → **1.8 h** (×2.5) | 1.20 → **3.0 h** | 1.29 → **3.2 h** |
| **GPU-hour (M2 Pro/MPS)** | 3.36 h → **8.4 h** | 5.55 → **13.9 h** | 5.93 → **14.8 h** |
| ↳ *chỉ detector chính, bỏ robustness* | *CUDA 0.06 h · MPS 0.41 h* | *0.10 · 0.69* | *0.11 · 0.73* |
| **TỔNG USD — tiết kiệm** | **~$40** | **~$64** | **~$73** |
| **TỔNG USD — đầy đủ** | **~$90** | **~$143** | **~$161** |

*Tiết kiệm = Gemini 2.5 Flash-Lite ×1.5, 1 judge, DeepL ×1. Đầy đủ = Gemini 3.1 Flash-Lite ×2.5, 2 judge, DeepL ×2.*

> ✅ **Đã chọn cột 300 trang.** Con số DeepL $33 ở trên là giá Growth theo ký tự; thực tế **giảm còn $0 cho EN→VI và $26 cho EN→ZH** nhờ dùng gói Developer miễn phí — xem sổ ngân sách §7.4b. ⇒ **Tổng dự án thực tế ~$10 (tiết kiệm) → ~$50 (đầy đủ)**, GPU 1.8 h (CUDA) / 8.4 h (M2 Pro), wall-clock ~8–10 h.

Wall-clock (không phải GPU-hour — phần lớn là chờ mạng): **~3–4 h** ở 300 trang, **~5–6 h** ở 600 trang cho một lượt sạch; ×2.5 khi dev ⇒ **8–15 h**, tức 1–2 ngày chạy nền.

### 7.3 Ba kết luận quan trọng

1. **GPU không phải nút cổ chai.** 1.8–3.2 GPU-hour trên CUDA cho *toàn bộ* benchmark. Con số MPS cao (8–15 h) gần như hoàn toàn do **Surya layout trong pass robustness 3-detector** — bỏ pass đó xuống còn 0.4–0.7 h. ⇒ Chạy được trên Mac M2 Pro; chỉ cần thuê GPU vài giờ cho pass robustness + CometKiwi-XL.
2. **Token dịch rẻ đến mức không cần cân nhắc** ($3–19). **Judge đắt hơn dịch** ($13–14) vì input là ảnh trang, không phải text. Nếu cần cắt: giảm judge xuống 1 model hoặc subset 100 trang/tầng.
3. **DeepL là hạng mục chi phối, và có một cái bẫy 10× → phải thiết kế để tránh.** Xem 7.4.

### 7.4 ⚠️ Ràng buộc thiết kế bắt buộc: bẫy 50.000 ký tự của DeepL

**DeepL tính tối thiểu 50.000 ký tự cho MỖI file .pdf/.docx/.pptx/.xlsx**, kể cả file chỉ có 200 ký tự (file > 50k thì tính đúng ký tự thật; .txt/.html/.srt/.xliff không bị áp mức tối thiểu).

T1 lấy từ DocLayNet là **PDF 1 trang** (~3,000 ký tự). Gửi lẻ ⇒ bị tính 50,000 ⇒ **đội 16.7×**: $33 → **$353** ở 300 trang, $65 → **$573** ở 600 trang.

**Cách tránh — gộp trang thành PDF nhiều trang trước khi gửi:**
- **T1**: gộp theo domain → ~6 PDF/ngôn ngữ (33 trang × 3,000 ≈ 99k ký tự > 50k ⇒ tính đúng). Dùng lại `benchmark/parser/run_parser/build_pdfs.py` — nó đã đóng ảnh thành PDF 32 trang + sinh `mapping.json` để giữ ánh xạ về từng trang.
  - *Chấp nhận đánh đổi:* T1 sau khi gộp không còn mạch nội dung xuyên trang. Điều này **cố ý** — T1 là benchmark mức trang; tính năng cross-page đánh giá ở T2. Phải gộp **giống nhau cho cả 4 hệ** để công bằng.
- **T2**: chọn tài liệu **≥ 18 trang** (18 × 3,000 = 54k > 50k). Tài liệu ngắn hơn sẽ bị tính 50k ⇒ vừa tốn tiền vừa lệch.
- **T3**: đã sẵn dạng PDF nhiều trang từ `build_pdfs.py` ⇒ không vấn đề.

### 7.4b Sổ ngân sách ký tự DeepL — track EN→VI vừa khít gói Developer miễn phí

Gói **Developer = 1.000.000 ký tự tổng (một lần, miễn phí)**. Với batching ở trên:

| Bước | Cấu trúc file | Ký tự bị tính | Luỹ kế |
|---|---|---|---|
| P0 smoke | 1 PDF × 20 trang | 50k *(sàn)* | 50k |
| P1 · T1 EN→VI | 6 PDF × 20 trang (60k mỗi cái) | 360k | 410k |
| P5 · T2 EN→VI | ~6 tài liệu ≥18 trang | 300k | 710k |
| P5 · T3 EN→VI | 3 PDF × ~27 trang | 240k | **950k** |

⇒ **Toàn bộ track EN→VI 300 trang tốn ~950k / 1M ký tự ⇒ $0.** Chỉ còn ~5% dư, **không đủ cho một lần chạy lại**.

Phần cần trả tiền: **EN→ZH T1** (6 PDF × 60k = 360k ký tự) + mọi lần chạy lại ⇒ mua **Growth 1 tháng ($26, bao gồm 1M ký tự)** là đủ cho cả hai.

⇒ **Ngân sách DeepL thực tế: $0 cho EN→VI, $26 cho EN→ZH + chạy lại.** Tổng dự án hạ từ ~$40–90 xuống **~$10 (tiết kiệm) → ~$50 (đầy đủ)**.

**Kỷ luật bắt buộc:** runner DeepL phải (a) log số ký tự đã dùng sau mỗi lần gọi vào `meta.json`, (b) đọc `/v2/usage` trước mỗi batch và **dừng nếu vượt ngưỡng cấu hình**, (c) resumable tuyệt đối. Hết hạn mức giữa lúc chạy mà không có 3 thứ này là mất cả pha.

### 7.5 Rủi ro về giá & gói, cần quyết trước khi chạy

| Rủi ro | Chi tiết | Xử lý |
|---|---|---|
| **Gemini 2.5 Flash-Lite bị khai tử 16/10/2026** | Còn ~6 tuần. Sau đó rẻ nhất là 3.1 Flash-Lite ($0.25/$1.50 thay vì $0.10/$0.40) — đắt hơn ~2.5× | Lập ngân sách theo giá **3.1 Flash-Lite** (đã làm trong bảng). PDFTranslator vốn đã default `google/gemini-3.1-flash-lite` trên OpenRouter ⇒ pin luôn model này cho mọi hệ, khỏi phải chạy lại |
| **DeepL API Free/Pro không còn bán từ 7/2026** | Gói mới: **Developer** (1M ký tự *tổng*, một lần) và **Growth** ($26/tháng, ~12M ký tự/năm ≈ 1M/tháng, vượt thì $27.50/1M) | **Đã chốt: pilot bằng Developer.** Toàn bộ track EN→VI vừa khít 950k/1M ⇒ $0. Mua Growth 1 tháng ($26) khi cần EN→ZH hoặc chạy lại. Sổ chi tiết §7.4b |
| **Hết hạn mức DeepL giữa lúc chạy** | Chỉ dư ~5% sau track EN→VI | Runner DeepL bắt buộc: log ký tự vào `meta.json`, đọc `/v2/usage` trước mỗi batch, dừng khi vượt ngưỡng, resumable tuyệt đối (§7.4b) |
| **DeepL có xử lý được PDF scan (T3) không** | Chưa rõ | Kiểm thực nghiệm ở P0 trên 3 file, **không giả định**. Nếu không ⇒ T3 chỉ còn PDFTranslator, tiết kiệm thêm DeepL |
| **CometKiwi-XL cần ≥15 GB VRAM** | M2 Pro 16GB unified là biên | Fallback `wmt22-cometkiwi-da` (580M) chạy tốt trên MPS. `score_comet.py` vốn **không phụ thuộc pdf2zh** ⇒ portable sang máy GPU khi cần |
| **Runner không resumable ⇒ mất tiền khi crash** | Với ×2.5 hệ số lặp, đây là rủi ro tiền thật | Bắt buộc **skip khi output đã tồn tại** ngay từ P0 — đúng pattern `run_translate.py --resume` đã có |
| **P4 chạy lại Phase 1 5 lần cho 5 mức ρ** | Sẽ đội GPU-hour lên 5× vô ích | **Cache `phase1_parsed.json` và tái dùng cho cả 5 mức ρ** (Phase 1 độc lập với bản dịch). Với BabelDOC: tái dùng `--working-dir`. Đã tính trong bảng |

---

## 8. Thực thi trên HF Jobs (GPU thuê theo phút)

### 8.1 Vì sao HF Jobs, không phải HF Space

Space là app Gradio tương tác — không phải nơi chạy batch. **HF Jobs** (`hf jobs run`) là primitive đúng: Docker image + command + flavor hardware, **tính tiền theo phút, chỉ khi Starting/Running, không tính lúc build**.

Điểm quan trọng nhất: **`hf jobs run` nhận image từ HF Space** (`hf.co/spaces/<user>/<space>`) ⇒ **tái dùng được ngay image Space PDFTranslator đang demo**, không phải build image mới. `Dockerfile` hiện tại đã đúng chuẩn cần thiết:
- `FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04`, `TORCH_DEVICE=cuda`
- typst v0.14.2 + font Noto/CJK/Be Vietnam Pro đã bake sẵn
- `COPY . .` ⇒ code `benchmark/` sẽ tự nằm trong image
- **entrypoint đã ưu tiên `/data` cho model cache khi `/data` writable** — khớp chính xác với cơ chế mount volume của HF Jobs (xem 8.4)

`hf jobs run` **override CMD**, nên entrypoint Gradio không chạy; ta truyền thẳng command batch.

### 8.2 Hai baseline có chạy được kiểu này không? → **Có**, và rẻ hơn nhiều

| | Có dùng được GPU? | Bằng chứng | Flavor nên dùng |
|---|---|---|---|
| **BabelDOC** | ✅ tự động | `docvision/doclayout.py:52` gọi `onnxruntime.get_available_providers()` rồi append tất cả provider khả dụng ⇒ cài `onnxruntime-gpu` là tự bật `CUDAExecutionProvider` | **`cpu-upgrade`** — không cần GPU |
| **PDFMathTranslate** | ✅ tường minh | `doclayout.py:24-26` map `"cuda" → ["CUDAExecutionProvider","CPUExecutionProvider"]`, có flag `--backend cuda` và extra `[cuda] = onnxruntime-gpu` | **`cpu-upgrade`** — không cần GPU |

**Lý do cho `cpu-upgrade`:** layout model của cả hai là một DocLayout-YOLO ONNX nhỏ (~0.03–0.1 s/trang kể cả trên CPU); thời gian thực tế của chúng là **chờ mạng LLM**, không phải compute. Trả $1/h cho A10G để nó ngồi chờ HTTP là đốt tiền. GPU chỉ cần cho: **Phase 1 của PDFTranslator (Surya+Paddle)**, **detector chấm điểm**, **CometKiwi**.

Vấn đề Python xung đột ở §5 tự tan: HF Jobs **một job = một image**, nên mỗi hệ có môi trường riêng một cách tự nhiên.

### 8.3 Ma trận job + chi phí (N = 300 trang)

Giá flavor lấy từ `huggingface.co/docs/hub/jobs-pricing`. Mặc định timeout **30 phút** ⇒ **luôn phải đặt `--timeout`**.

| Job | Nội dung | Flavor | $/h | Wall-clock | Chi phí |
|---|---|---|---|---|---|
| **A** | PDFTranslator E2E: parse 300 trang (1 lần, độc lập ngôn ngữ) + translate/render 420 lượt | `t4-medium` | 0.60 | ~30–60 ph | **$0.30–0.60** |
| **B** | BabelDOC + PDFMathTranslate + DeepL runner (1,100 lượt, network-bound) | `cpu-upgrade` | 0.03 | ~1 h | **$0.03** |
| **C** | P4 ρ-sweep: 3 hệ × 5 ρ × 120 trang = 1,800 lượt **render-only** (parse đã cache) | `cpu-upgrade` | 0.03 | ~1–1.5 h | **$0.05** |
| **D** | Detector chấm điểm: 3 model × 3,725 trang | `t4-medium` | 0.60 | ~1 h | **$0.60** |
| **E** | CometKiwi-XL, 42,500 segment (cần ≥15 GB ⇒ **không dùng T4 16GB**) | `l4x1` (24 GB) | 0.80 | ~0.6 h | **$0.48** |
| **F** | LLM-as-judge (thuần API, không GPU) | `cpu-basic` *hoặc chạy ở máy* | 0.01 | ~30 ph | **$0.01** |
| | | | | **Một lượt sạch** | **≈ $1.80** |
| | | | | **×2.5 khi debug** | **≈ $4.40** |

⇒ **Ngân sách HF Jobs thực tế: $5–10** kể cả thử nghiệm thoải mái. **Không phải hạng mục cần cân nhắc.** Tổng dự án: ~$15 (tiết kiệm) → ~$60 (đầy đủ).

Nếu muốn nhanh hơn: `l4x1` ($0.80) hoặc `a10g-small` ($1.00) rút job A và D xuống ~1/2 thời gian mà tổng vẫn < $3.

### 8.4 Bốn kỷ luật vận hành bắt buộc (không có là đốt tiền hoặc mất dữ liệu)

1. **Mount volume để cache model — nếu không, MỖI job tải lại 3–5 GB Surya + Paddle và bạn bị tính tiền cho thời gian tải.**
   `Dockerfile` entrypoint đã sẵn logic: `if mkdir -p /data && [ -w /data ]` thì dùng `/data` làm `MODEL_CACHE_DIR`/`PADDLE_PDX_CACHE_HOME`/`HF_HOME`. ⇒ chỉ cần mount một **storage bucket** vào `/data:rw`. Lần đầu tải, các lần sau dùng lại.
2. **`--timeout` tường minh.** Mặc định 30 phút sẽ giết job A/C/D giữa đường. Dùng `--timeout 3h`.
3. **Secret qua `--secrets`, không bake vào image.** `LITELLM_API_KEY`, `LITELLM_BASE_URL`, `DEEPL_AUTH_KEY` — HF mã hoá server-side. (Liên quan §1.5: đang có key OpenRouter plaintext trong repo.)
4. **Runner resumable + artifact ghi ra volume rồi sync về.** Job có thể bị timeout/suspend; không resumable là chạy lại từ đầu, và với DeepL thì mất luôn hạn mức ký tự.

### 8.5 Lệnh cụ thể

```bash
# 0) Tải model một lần vào bucket cache (chạy ngắn, chỉ để warm cache)
hf jobs run --name eval-warm --flavor t4-small --timeout 30m \
  -v hf-bucket://$HF_USER/pdftranslator-eval-cache:/data:rw \
  hf.co/spaces/$HF_USER/$SPACE \
  python3 -m benchmark.e2e.runners.pdftranslator --warmup-only

# A) PDFTranslator E2E — dùng thẳng image Space đang demo
hf jobs run --name eval-pdftranslator --flavor t4-medium --timeout 3h \
  --secrets LITELLM_API_KEY=$LITELLM_API_KEY --env LITELLM_BASE_URL=$LITELLM_BASE_URL \
  -v hf-bucket://$HF_USER/pdftranslator-eval-cache:/data:rw \
  -v ./benchmark/e2e/datasets/corpus:/corpus:ro \
  -v ./benchmark/e2e/out:/out:rw \
  hf.co/spaces/$HF_USER/$SPACE \
  python3 -m benchmark.e2e.runners.pdftranslator \
    --corpus /corpus --out /out --tiers T1,T2,T3 --langs vi,zh --resume

# B) Baseline — CPU, image nhẹ dựng bằng uv script (không cần Docker build)
hf jobs uv run --name eval-babeldoc --flavor cpu-upgrade --timeout 3h \
  --secrets LITELLM_API_KEY=$LITELLM_API_KEY \
  -v ./benchmark/e2e/datasets/corpus:/corpus:ro -v ./benchmark/e2e/out:/out:rw \
  benchmark/e2e/runners/babeldoc_job.py -- --corpus /corpus --out /out --resume

# D) Detector chấm điểm
hf jobs run --name eval-detect --flavor t4-medium --timeout 3h \
  -v hf-bucket://$HF_USER/pdftranslator-eval-cache:/data:rw \
  -v ./benchmark/e2e/out:/out:rw \
  hf.co/spaces/$HF_USER/$SPACE \
  python3 -m benchmark.e2e.parse.run_detectors --out /out --detectors docling,doclayout,surya

# E) CometKiwi — cần 24 GB
hf jobs uv run --name eval-qe --flavor l4x1 --timeout 2h \
  -v ./benchmark/e2e/out:/out:rw \
  benchmark/translation/score_comet.py -- --qe-model Unbabel/wmt23-cometkiwi-da-xl --hyp /out/pairs.jsonl

# Theo dõi / dọn
hf jobs ls -a          # trạng thái
hf jobs logs <job_id>  # log
hf jobs wait <job_id>  # chờ, exit 0 nếu COMPLETED  -> dùng để chain trong run_all.sh
hf jobs cancel <job_id>
hf jobs hardware       # bảng flavor + giá, đọc từ API
```

`-v ./local:/mount` tự sync thư mục local lên bucket `jobs-artifacts` (chỉ upload file mới/đổi), và lấy kết quả về bằng `sync_bucket()` trong Python API. **Kết quả cuối đẩy lên một HF dataset repo** (`$HF_USER/pdftranslator-eval-results`) để luận văn có link tái lập vĩnh viễn.

Ghi chú: env có sẵn trong container — `JOB_ID`, `ACCELERATOR`, `CPU_CORES`, `MEMORY` ⇒ **log `ACCELERATOR` vào `meta.json`** để bảng hiệu năng nói rõ chạy trên hardware nào. Cần debug tương tác thì `hf jobs run --ssh` rồi `hf jobs ssh <job_id>`.

---

## 9. Code deliverables — file nào, chạy thế nào, lưu ở đâu

Tất cả nằm trong repo PDFTranslator dưới `benchmark/e2e/` (§5), **được commit**, kèm `docs/EVALUATION_PLAN.md` là chính plan này. Không sửa `pdf2zh/`.

### 9.1 Chuẩn bị dataset

| File | Làm gì | Lệnh |
|---|---|---|
| `datasets/build_doclaynet.py` | Lấy mẫu stratified 120 trang từ `ds4sd/DocLayNet-v1.2` (cột `pdf` binary + COCO GT), 20 trang × 6 domain, seed cố định → **gộp thành 6 PDF/domain** + `gt_coco.json` + `mapping.json` | `python -m benchmark.e2e.datasets.build_doclaynet --out datasets/corpus/T1 --per-domain 20 --seed 42` |
| `datasets/build_t2_corpus.py` | Tải arXiv (API) + patents (Google Patents) + technical docs theo `sources.yaml`; **lọc chỉ nhận tài liệu ≥18 trang** (§7.4); ghi `manifest.json` có `sha256` | `python -m benchmark.e2e.datasets.build_t2_corpus --out datasets/corpus/T2 --target-pages 100` |
| *(T3 — dùng lại code có sẵn)* | `benchmark/parser/evaluation/download_dataset.py` tải OmniDocBench → `run_parser/build_pdfs.py` đóng ảnh thành PDF | `python benchmark/parser/evaluation/download_dataset.py --out datasets/omnidoc && python benchmark/parser/run_parser/build_pdfs.py --images datasets/omnidoc/images --out datasets/corpus/T3 --per-pdf 27 --limit 80` |
| `datasets/verify_corpus.py` | Kiểm bất biến trước khi chạy: mọi PDF có text layer (T1/T2) hoặc không có (T3); **mọi PDF ≥ 50k ký tự** (bẫy DeepL); stem duy nhất không dấu chấm (bẫy surya `split(".")[0]`); `sha256` khớp manifest | `python -m benchmark.e2e.datasets.verify_corpus --corpus datasets/corpus` |

`verify_corpus.py` là **cửa chặn**: chạy nó trước mọi job, fail thì dừng. Ba bẫy nó bắt đều đã xác minh là bẫy thật, không phải giả định.

### 9.2 Runner DeepL

`deepl` SDK **đã có trong `requirements.txt` và `pyproject.toml`** của PDFTranslator. Dùng endpoint document (`/v2/document`) qua `translate_document_from_filepath`, khác hoàn toàn với `DeepLTranslator` text-mode ở Pipeline A.

`runners/deepl_doc.py` — hình dạng:

```python
import deepl, json, time, hashlib
from pathlib import Path

LANG = {"vi": "VI", "zh": "ZH"}          # DeepL thêm VI từ 06/2025

def run(pdf: Path, out_dir: Path, lang: str, client: deepl.DeepLClient,
        char_budget: int, resume: bool = True) -> dict:
    dst = out_dir / f"{pdf.stem}.{lang}.pdf"
    meta_p = out_dir / f"{pdf.stem}.{lang}.meta.json"
    if resume and dst.exists() and meta_p.exists():
        return json.loads(meta_p.read_text())          # KỶ LUẬT 1: resumable

    # KỶ LUẬT 2: chặn trước khi gọi — hết hạn mức giữa batch là mất cả pha (§7.4b)
    u = client.get_usage()
    if u.character.valid and u.character.count >= char_budget:
        raise RuntimeError(f"DeepL budget reached: {u.character.count}/{char_budget}")

    t0 = time.perf_counter()
    err = None
    try:
        client.translate_document_from_filepath(
            str(pdf), str(dst), target_lang=LANG[lang], source_lang="EN",
        )
    except deepl.DocumentTranslationException as e:
        err = f"{type(e).__name__}: {e}"                # giữ document_handle để lấy lại
    except deepl.DeepLException as e:
        err = f"{type(e).__name__}: {e}"

    after = client.get_usage()
    meta = dict(system="deepl-document", lang=lang, src=pdf.name,
                sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                wall_seconds=round(time.perf_counter()-t0, 2),
                chars_billed=after.character.count - u.character.count,   # KỶ LUẬT 3: log ký tự
                chars_total=after.character.count,
                pages_in=None, pages_out=None, error=err)
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
```

Ba điểm cần biết khi review: (1) **`chars_billed` phải log lại** để đối chiếu sổ §7.4b và phát hiện sớm nếu batching sai (thấy 50,000 cho một file 20 trang là sai); (2) DeepL trên **T3 (scan)** phải bọc try/except và ghi `error` thay vì crash — chưa xác minh nó có OCR, đây là một trong 3 rủi ro kiểm ở P0; (3) `pages_out` đo bằng PyMuPDF sau khi tải về, để tính **page inflation** (§4.1).

### 9.3 Bố cục artifact + driver một lệnh

```
benchmark/e2e/out/
  <system>/<lang>/<doc_id>/
    output.pdf              # PDF dịch (bản SẠCH — BabelDOC chạy thêm lượt --debug riêng)
    meta.json               # wall_seconds, parse_s/translate_s/render_s, tokens, usd,
                            # chars_billed (DeepL), ACCELERATOR, exit_code, error
    debug/                  # BabelDOC: layout_generator.json, typsetting.json, translate_tracking.json
  layout/<detector>/<system>/<lang>/<doc_id>/<page>.json   # box đã chuẩn hoá
  pairs.jsonl               # cặp {src, mt} đã align — input cho CometKiwi
  judge/<judge>/<...>.json  # điểm LF/TP/VA/TC + UTB
  report/
    report.md  tables/*.csv  figures/*.png   # + bootstrap CI, paired test
```

`benchmark/e2e/run_all.sh` — driver, chain job bằng `hf jobs wait`, theo khuôn `benchmark/translation/run_all.sh` đã có (env-var overridable, preflight deps + key):

```bash
TIERS=T1,T2 LANGS=vi MODEL=google/gemini-3.1-flash-lite \
DEEPL_CHAR_BUDGET=950000 FLAVOR_GPU=t4-medium \
bash benchmark/e2e/run_all.sh
```

Chạy được cả **local** (`RUNTIME=local`) và **HF Jobs** (`RUNTIME=hfjobs`) — cùng một script, chỉ khác cách phát lệnh. Local để debug trên M2 Pro, HF Jobs cho lượt chính thức.

### 9.4 Thứ tự dựng ở P0 (để có cái review sớm nhất)

1. `datasets/build_doclaynet.py` + `verify_corpus.py` → có corpus T1 và biết nó sạch.
2. `runners/deepl_doc.py` + `runners/pdftranslator.py` → 2 hệ đối đầu, chạy local trên 1 PDF gộp 20 trang.
3. `run_all.sh` với `RUNTIME=local` → xác nhận bố cục artifact.
4. Chuyển sang `RUNTIME=hfjobs`, warm cache volume, chạy job A + B.
5. Mới thêm `babeldoc.py`, `pdfmathtranslate.py`, rồi mở sang T2/T3.

---

## 10. Verification

1. **Identity test** — đưa PDF gốc làm "output" ⇒ mIoU ≈ 1.0, collision ≈ 0, masked-SSIM ≈ 1.0. Sai thì harness sai, không phải hệ thống sai.
2. **Ceiling test** — detector trên PDF gốc vs GT DocLayNet ⇒ công bố con số trần.
3. **Synthetic perturbation** — script dịch mọi box đi 5 pt / phóng 10% ⇒ metric phải biến thiên **đúng hướng và đúng độ lớn**.
4. **Matcher ablation** — chạy `compare_matchers.py` trên dữ liệu preservation ⇒ kết luận không phụ thuộc chiến lược ghép.
5. **QE calibration** — Pearson/Spearman giữa CometKiwi và COMET-DA trên WMT24++ `vi_VN`. Nếu tương quan thấp thì phải đổi metric chính, phát hiện sớm ở P3.
6. **Smoke run** — 5 PDF × 4 hệ; kiểm đủ artifact, đúng tên file: infix `.debug.` (BabelDOC), `<stem>-mono.pdf`/`<stem>-dual.pdf` (PDFMathTranslate), `translated_<uuid8>.pdf` (PDFTranslator).
7. **Đối chiếu định tính với paper** — nếu harness tái lập được xu hướng "BabelDOC > PDFMathTranslate về LF/VA" và "DeepL BIoU thấp nhưng LF cao" thì harness đáng tin.
8. **Không hồi quy** — `pytest test/ -v` + `ruff check pdf2zh test` + `black --check` vẫn xanh; **`git diff pdf2zh/` rỗng**.

---

## 11. Việc đầu tiên khi được duyệt

1. **Revoke API key OpenRouter** trong `test_local/test_pipeline.txt` và xoá khỏi file (§1.5). Chuyển hết secret sang `hf jobs --secrets`.
2. Ghi plan này thành `docs/EVALUATION_PLAN.md` — vá tham chiếu chết ở `benchmark/translation/README.md:4,98` và `__init__.py:5` (§1.5).
3. Dựng `benchmark/e2e/` theo **thứ tự §9.4**: corpus T1 + `verify_corpus.py` → runner DeepL + PDFTranslator → `run_all.sh` (`RUNTIME=local`) → chuyển `RUNTIME=hfjobs` → thêm 2 baseline → mở T2/T3.
4. **Kiểm 3 rủi ro của P0 trước khi tiêu tiền:** (a) batching có làm DeepL tính đúng ký tự thay vì sàn 50k không — xem `chars_billed` trong `meta.json`; (b) DeepL có xử lý được PDF scan (T3) không; (c) DoTA có expose arXiv ID không.
5. **Việc cần bạn làm song song:** tạo gói DeepL **Developer** (1M ký tự, miễn phí) và lấy `DEEPL_AUTH_KEY`; xác nhận HF account có credit dương để chạy Jobs.

### Không nằm trong phạm vi (nói rõ để tránh trôi scope)

Không sửa bất kỳ dòng nào trong `pdf2zh/` — harness chỉ đọc và gọi (bất biến đã có ở `benchmark/translation/README.md:101`, mở rộng ra toàn bộ). Không tối ưu PDFTranslator dựa trên kết quả benchmark trong cùng lần này: đo trước, cải thiện sau, tránh overfit vào chính bộ đo của mình.

---

## 12. Cập nhật từ P0 — những gì đo được khác với plan

Phần trên là plan đã duyệt, giữ nguyên để đối chiếu. Phần này ghi các sai lệch **đã đo**
trong lúc dựng P0, để plan không nói sai so với thực tế.

### 12.1 Sáu chỗ phải sửa so với plan

| # | Plan viết | Thực tế đo được | Hệ quả |
|---|---|---|---|
| 1 | Dataset `ds4sd/DocLayNet-v1.2` | Repo **đã bị rename**; API trả *"The dataset has been renamed."* Tên đúng: **`docling-project/DocLayNet-v1.2`** | Đã sửa trong `build_doclaynet.py` |
| 2 | *(không nêu)* | GT nằm trong không gian **COCO vuông 1025×1025**, PDF là vd 612×792 ⇒ scale **bất đẳng hướng** (x 1.675, y 1.294) | Chuẩn hoá **từng trục theo chiều của nó**. Scale đều ⇒ bbox sai hệ thống |
| 3 | "ép cùng LLM, `temperature=0` cho mọi hệ" | `pdf2zh/translation/gateway.py` **hardcode** temperature 0.7 (dòng 146) và 0.2 (dòng 246), **không có knob config** | `temperature=0` **không đặt được trong hệ**. Phải ép ở **LiteLLM proxy** — thực ra công bằng hơn vì cả 3 hệ nhận cùng override tại một điểm |
| 4 | T1 = "PDF born-digital, có text layer" | DocLayNet có trang là **ảnh scan nhúng** (0 font, 0 drawing, `get_text()` rỗng) mà vẫn có `pdf_cells` đầy. Gặp 2/150 ứng viên | `pdf_cells` **không** là bằng chứng đọc được text. Builder kiểm lại bằng `get_text()` thật rồi bù từ dự phòng (`--over`, mặc định 25%) |
| 5 | "T1: 200 trang, ~33 trang/domain" | Chốt **120 trang, 20 trang/domain** (quy mô 300 trang). Mật độ chữ DocLayNet lệch rất mạnh: patents p10=140 ký tự, p25=398; 27% trang patents và 22% trang laws dưới 500 ký tự | Thêm `--min-chars` (mặc định **500**) loại trang gần như không có gì để dịch, vẫn giữ trang nhiều hình/bảng vì đó là thứ Anchor-IoU đo. Loại 580/4999 trang |
| 6 | *(không nêu)* | **`manuals` chỉ có 7 tài liệu gốc** cho 20 trang; `government_tenders` 16. Giới hạn của dataset, không phải lựa chọn lấy mẫu | Ghi `n_source_docs` vào `mapping.json` và **phải công bố trong luận văn** |

### 12.2 Ngân sách DeepL — đo thật, đối chiếu §7.4b

| | Plan dự đoán | Đo thật |
|---|---|---|
| T1 EN→VI | 360.000 ký tự | **353.661** (lệch 2%) |
| T1 EN→VI + EN→ZH | — | **707.322** |
| Bẫy 50k nếu gửi PDF 1 trang | "đội 16.7×" | **600.000 ký tự cho 40.000 ký tự thật** (6 file × 2 ngôn ngữ) — xác nhận |

Sau khi gộp 20 trang/PDF, **4/6 file vượt sàn 50k**. Hai file còn dưới sàn:
`manuals` 38.498 và `laws_and_regulations` 41.034 ⇒ **overpay chỉ 1.2–1.3×**
(~20k ký tự lãng phí mỗi ngôn ngữ, ~2% hạn mức Developer). Chấp nhận và để lộ ra
qua cảnh báo của `verify_corpus`, thay vì phá cân bằng 20 trang/domain để chữa.

**Xác nhận phasing của §7.4b là đúng**: T1 hai ngôn ngữ đã ngốn 707k/1M, không đủ chỗ
cho T2+T3 ⇒ chạy **EN→VI toàn bộ 3 tầng trên gói Developer**, mua Growth cho lát ZH.

### 12.3 Hiệu năng đọc dataset (ảnh hưởng thời gian dựng corpus)

HfFileSystem là **bandwidth-bound**, không phải latency-bound, và **không phụ thuộc cột**:
đọc riêng 1 shard mất ~100–230s dù xin cột nào (`pdf_cells` một mình còn *nhanh hơn*
`metadata` một mình). Prune cột thêm không giúp gì. Song song 6 shard: mỗi shard chậm
lại còn ~700s nhưng tổng **~20 phút → ~12 phút (1.7×, không phải 6×)**.
⇒ `--scan-cache` mới là thứ làm việc lặp lại rẻ. Pass 2 (fetch 47 row group, 8 worker): **126s**.

### 12.4 Corpus T1 đã dựng

120 trang · 6 PDF · **1.785 box GT** · 333.193 ký tự · mọi trang có text layer (20/20 mỗi domain).

Phân bố class GT: Text 785 · List-item 400 · Section-header 166 · Page-header 105 ·
Page-footer 87 · Picture 69 · Table 60 · Footnote 36 · Title 31 · Formula 30 · Caption 16.

Nhóm **anchor** của §4.1 (Picture + Table + Formula + Page-header/footer) = **351 box** — đủ để Anchor-IoU có ý nghĩa.

### 12.5 Ba rủi ro P0 — trạng thái

| Rủi ro | Trạng thái |
|---|---|
| (a) Batching có làm DeepL tính đúng ký tự thay vì sàn 50k | **Xác nhận một phần** — forecast cho thấy 4/6 file vượt sàn. Xác nhận billing thật cần `DEEPL_AUTH_KEY` |
| (b) DeepL có xử lý được PDF scan (T3) | **Chưa kiểm** — cần T3 + key |
| (c) DoTA có expose arXiv ID | **Chưa kiểm** — `liangyupu/DoTA_dataset` tồn tại nhưng **`gated: auto`**, cần accept điều khoản trên HF. Không phải blocker: WMT24++ `vi_VN` đã chạy được và lo phần hiệu chuẩn QE chính |
