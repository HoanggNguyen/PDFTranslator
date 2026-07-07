# Benchmark parse_pdf (PDFTranslator) trên OmniDocBench — Google Colab

Đo **thời gian thật của `StageAParser.parse_pdf`** trên ảnh OmniDocBench và xuất
JSON để tự format / đo metric sau.

Cách làm: parse_pdf nhận **PDF**, nên gộp ảnh thành PDF nhiều trang (mặc định
**32 ảnh/PDF**) kèm `mapping.json` để tách kết quả về từng ảnh. Trước khi đo,
**preload cả 3 mô hình** (layout / OCR / table) + một lượt warmup để số đo không lệch.

Thư mục này (`omnidocbench/`) nằm ngay trong repo PDFTranslator:

```
PDFTranslator/
├── pdf2zh/                 # parser (StageAParser.parse_pdf) — KHÔNG chỉnh sửa
├── requirements.txt
└── omnidocbench/
    ├── download_dataset.py # tải ảnh + OmniDocBench.json từ HuggingFace
    ├── build_pdfs.py       # gộp ảnh -> PDF 32 trang + mapping.json
    ├── run_parser.py       # preload model -> đo parse_pdf từng PDF -> JSON + timing
    └── HUONG_DAN.md        # file này
```

`run_parser.py` tự dò repo root (tìm thư mục chứa `pdf2zh/` phía trên nó) nên
không cần chỉ định đường dẫn.

---

## Chạy trên Colab

Bật GPU trước: **Runtime → Change runtime type → GPU (T4)**.
Copy từng cell dưới đây vào notebook (mỗi khối = 1 cell).

### Cell 1 — Kiểm tra GPU
```python
!nvidia-smi
```

### Cell 2 — Mount Google Drive (chỉ để lưu kết quả)
Mô hình cứ tải bình thường về cache mặc định của Colab (không cache lên Drive).
Chỉ mount Drive để kết quả không mất khi hết phiên:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
RESULTS = '/content/drive/MyDrive/odb_results'
os.makedirs(RESULTS, exist_ok=True)
```

### Cell 3 — Clone repo (đúng branch)
```bash
%cd /content
!git clone -b feat/omnidocbench-benchmark https://github.com/HoanggNguyen/PDFTranslator.git
%cd /content/PDFTranslator
```

### Cell 4 — Cài thư viện
```bash
!pip install -q -r requirements.txt
```
> Nếu `paddlepaddle-gpu` báo lỗi CUDA trên Colab, xem mục **Xử lý sự cố** cuối file.
> Sau bước này Colab có thể yêu cầu **Restart runtime** — bấm restart rồi chạy
> lại **Cell 2** (mount lại Drive) trước khi tiếp tục.

### Cell 5a — Đăng nhập HuggingFace (tránh lỗi 429)
IP dùng chung của Colab hay bị HF giới hạn tốc độ (`429 Too Many Requests`).
Đăng nhập bằng token (miễn phí, quyền **Read**, tạo ở
https://huggingface.co/settings/tokens) để hết bị chặn:
```python
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxx"   # dán token của bạn
```

### Cell 5b — Tải dataset OmniDocBench
```bash
!python omnidocbench/download_dataset.py --out /content/OmniDocBench_data
```
Ra `/content/OmniDocBench_data/images/` (ảnh) và `OmniDocBench.json` (GT).
> Nếu vẫn 429: đợi vài phút rồi chạy lại — `snapshot_download` có **resume**,
> nó tải tiếp phần còn dở chứ không tải lại từ đầu.

### Cell 6 — Gộp ảnh thành PDF (32 ảnh/PDF) + mapping
```bash
!python omnidocbench/build_pdfs.py \
    --images  /content/OmniDocBench_data/images \
    --out     /content/OmniDocBench_data/pdfs \
    --per-pdf 32
```
> Test nhanh: thêm `--limit 64` (chỉ 64 ảnh → 2 PDF).

### Cell 7 — Chạy đo, lưu kết quả vào Drive
```bash
!python omnidocbench/run_parser.py \
    --pdfs   /content/OmniDocBench_data/pdfs \
    --out    /content/drive/MyDrive/odb_results/parser_json \
    --timing /content/drive/MyDrive/odb_results/parser_timing.json \
    --device cuda
```
`run_parser.py` sẽ: khởi tạo parser → **preload 3 model** → **warmup** PDF đầu
(bỏ số đo) → đo `parse_pdf` từng PDF → lưu `<pdf>.json`.

### Cell 8 — Xem nhanh kết quả thời gian
```python
import json
d = json.load(open('/content/drive/MyDrive/odb_results/parser_timing.json'))
print('load model:', d['model_load_seconds'], 's')
print('warmup    :', d['warmup_seconds'], 's')
print('tổng parse:', d['total_parse_seconds'], 's /', d['total_pages'], 'trang')
print('avg/trang :', d['avg_seconds_per_page'], 's')
print('fail      :', d['num_failed'])
```

---

## Kết quả

- **JSON mỗi PDF**: `.../odb_results/parser_json/batch_xxxxx.json`
  (`ParsedDocument`: `pages[].elements[]` với `label`, `category`, `bbox_pdf`,
  `source_text`, `cells`).
- **Thời gian**: `.../odb_results/parser_timing.json` — `model_load_seconds`,
  `warmup_seconds`, `total_parse_seconds`, `total_pages`, `avg_seconds_per_pdf`,
  `avg_seconds_per_page`, và `per_pdf[]`.
- **mapping**: `/content/OmniDocBench_data/pdfs/mapping.json` — trang thứ `k` của
  `batch_x.json` chính là ảnh `mapping.pdfs[x].images[k]`.
  > `pdfs/` nằm ở `/content` (bị xóa khi hết phiên) → nếu cần giữ `mapping.json`
  > lâu dài, copy vào Drive:
  > `!cp /content/OmniDocBench_data/pdfs/mapping.json /content/drive/MyDrive/odb_results/`.

---

## Ghi chú

- `run_parser.py` **bỏ qua** PDF đã có JSON (resume). Vì kết quả lưu trên Drive,
  chạy lại notebook sau khi Colab ngắt phiên sẽ **tiếp tục** chỗ dở. Chạy lại từ
  đầu: thêm `--overwrite`.
- Một PDF lỗi → ghi `ok:false` + `error`, chạy tiếp, không dừng.
- Đã có warmup nên PDF đầu không lệch; tắt bằng `--no-warmup`.
- `--per-pdf` đổi được (16/32/64) — chỉ đổi cách gộp, không đổi kết quả từng trang.
- Colab free hay ngắt phiên/giới hạn giờ GPU → chia nhỏ 32 ảnh/PDF + lưu Drive
  giúp resume an toàn thay vì mất sạch.

## Xử lý sự cố

- **Lỗi cài `paddlepaddle-gpu` (CUDA mismatch)**: `requirements.txt` ghim
  `paddlepaddle-gpu==3.3.1 (cu130)`. Nếu Colab dùng CUDA khác, cài bản CPU để
  vẫn chạy được (phần cell-table sẽ chậm hơn nhưng vẫn ra kết quả):
  ```bash
  !pip install -q paddlepaddle==3.3.1
  ```
- **Tải model lại mỗi phiên**: bình thường — mô hình dùng cache mặc định của
  Colab (bị xóa khi hết phiên) nên mỗi phiên mới sẽ tải lại ~3–5GB. Chấp nhận
  được cho việc chạy 1 lần; không cần cache lên Drive.
- **Muốn PDF nhẹ hơn (RAM/VRAM)**: build với `--per-pdf 16`.
