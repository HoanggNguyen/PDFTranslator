# Benchmark — Đánh giá phân tích bố cục (parser vs OmniDocBench)

Đo độ chính xác của `StageAParser` (giai đoạn phân tích cấu trúc của PDFTranslator) bằng cách đối chiếu trực tiếp với ground truth của **OmniDocBench**. Quy trình gồm **hai giai đoạn**:

1. **Sinh prediction** — chạy parser trên toàn bộ trang OmniDocBench → cần **GPU** (nên dùng Google Colab A100).
2. **Chấm điểm (eval)** — tính các độ đo (định vị, phân loại, OCR, thứ tự đọc, công thức, bảng) → chỉ cần **CPU**, chạy local. Riêng chỉ số **CDM** cho công thức cần thêm TeX Live + ImageMagick.

---

## Cấu trúc thư mục

```
benchmark/parser/
├── run_parser/          # build_pdfs.py, run_parser.py   (sinh prediction — cần GPU)
├── evaluation/          # eval_layout / eval_formula / eval_table / eval_formula_cdm
│                        # aggregate_reports.py, compare_matchers.py, download_dataset.py
│                        # requirements-eval.txt
├── data/                # OmniDocBench.json + images/ + pdfs/      (tải/tạo ở bước 1)
├── parser_results/      # batch_*.json (ParsedDocument) + mapping.json   (đầu ra parser)
└── eval_results/        # *.json report + eval_summary_*.csv            (đầu ra eval)
```

---

## 0. Chuẩn bị

```bash
git clone https://github.com/HoanggNguyen/PDFTranslator.git
cd PDFTranslator
# repo OmniDocBench chỉ cần cho CDM (đánh giá công thức):
git clone https://github.com/opendatalab/OmniDocBench.git
cd benchmark/parser
```

---

## 1. Sinh prediction bằng parser (GPU)

Giai đoạn này cần GPU lớn. Nếu GPU laptop không đủ thì dùng **GoogleColab**.

### 1a. Kiểm tra GPU và tạo môi trường ảo

```bash
nvidia-smi
```

Trên Colab, tạo môi trường ảo riêng để **tránh xung đột** với các gói cài sẵn:

```bash
pip install virtualenv
virtualenv myenv
./myenv/bin/pip install -r ../../requirements.txt
```

(Nếu chạy local có GPU đủ: dùng venv của repo và `pip install -r ../../requirements.txt`.)

### 1b. Tải dữ liệu → gộp PDF → chạy parser

```bash
# (1) tải ảnh + OmniDocBench.json về data/
./myenv/bin/python evaluation/download_dataset.py --out data

# (2) gộp ảnh thành PDF 32 trang/PDF (tạo kèm data/pdfs/mapping.json)
./myenv/bin/python run_parser/build_pdfs.py \
    --images data/images --out data/pdfs --per-pdf 32

# (3) chạy parser -> parser_results/batch_*.json + báo cáo thời gian
./myenv/bin/python run_parser/run_parser.py \
    --pdfs   data/pdfs \
    --out    parser_results \
    --timing eval_results/parser_timing.json \
    --device cuda
```

Tuỳ chọn hữu ích của `run_parser.py`:

- Batch size (điều chỉnh theo VRAM): `--layout-batch-size`, `--detection-batch-size`, `--ocr-batch-size`, `--table-batch-size`, `--page-batch-size` (mặc định hợp cho A100).
- Ngưỡng detector: `--blank-threshold`, `--text-threshold`.
- `--limit N`: chỉ chạy N PDF đầu (test nhanh); `--overwrite`: chạy lại PDF đã có JSON.

> `build_pdfs.py` ghi `mapping.json` cạnh các PDF (`data/pdfs/mapping.json`). Để các
> lệnh eval ở Giai đoạn 2 chạy nguyên trạng, sau khi chạy parser hãy copy nó vào
> `parser_results/`: `cp data/pdfs/mapping.json parser_results/`.

- Tải `parser_results/` từ Colab về máy để chấm điểm ở Giai đoạn 2.

---

## 2. Chấm điểm (eval — CPU, chạy local)

### 2a. Môi trường eval

```bash
python3 -m venv .venv
.venv/bin/pip install -r evaluation/requirements-eval.txt

sudo apt install -y texlive-latex-base texlive-latex-extra \
                    texlive-fonts-recommended imagemagick
```

Đặt cho gọn: `PY=.venv/bin/python`.

### 2b. Localization + Classification + OCR + Reading order

```bash
# fine (bung merge_list — khuyến nghị)
$PY evaluation/eval_layout.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json \
    --gt-granularity fine --out eval_results/eval_report_fine.json

# merged (box top-level như OmniDocBench)
$PY evaluation/eval_layout.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json \
    --gt-granularity merged --out eval_results/eval_report_merged.json

# fine + mask-math (thay công thức inline bằng token -> CER/WER text thuần)
$PY evaluation/eval_layout.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json \
    --gt-granularity fine --mask-math \
    --out eval_results/eval_report_fine_maskmath.json
```

### 2c. Công thức (edit distance) + Bảng (nội dung)

```bash
$PY evaluation/eval_formula.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json \
    --out eval_results/eval_report_formula.json

$PY evaluation/eval_table.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json \
    --out eval_results/eval_report_table.json
```

### 2d. Công thức CDM (chuẩn vàng — cần TeX Live + ImageMagick + pylatexenc)

```bash
$PY evaluation/eval_formula_cdm.py \
    --gt data/OmniDocBench.json --pred parser_results \
    --mapping parser_results/mapping.json \
    --omnidocbench ../../OmniDocBench \
    --out eval_results/eval_report_formula_cdm.json
# thêm --limit 200 để test nhanh
```

### 2e. Gom kết quả thành CSV

```bash
$PY evaluation/aggregate_reports.py \
    --layout  fine=eval_results/eval_report_fine.json \
    --layout  merged=eval_results/eval_report_merged.json \
    --layout  fine_maskmath=eval_results/eval_report_fine_maskmath.json \
    --formula eval_results/eval_report_formula.json \
    --report  table=eval_results/eval_report_table.json \
    --out     eval_results/eval_summary
```

### 2f. (tuỳ chọn) So sánh 3 cách matching localization

```bash
$PY evaluation/compare_matchers.py --iou 0.5 --granularity fine
```

---

## Ghi chú

- GT↔pred nối theo **tên ảnh** (`image_path`) qua `mapping.json` — khớp 1:1.
- Eval chạy **CPU** (trừ CDM cần TeX Live). Chỉ Giai đoạn 1 (parser) mới cần GPU.
- Các script trong `evaluation/` import lẫn nhau theo thư mục cạnh bên; hãy gọi bằng `python evaluation/<script>.py` (đừng đổi tên/di chuyển lẻ từng file).
