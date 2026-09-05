# Hướng dẫn chạy benchmark trên Hugging Face

Từ con số 0 đến `report.md`. Đây là tài liệu **thao tác**; lý do đằng sau từng
quyết định nằm ở [E2E_HF_PLAN.md](E2E_HF_PLAN.md), còn cách chạy ở máy thì ở
[E2E_RUNBOOK.md](E2E_RUNBOOK.md).

Mọi lệnh chạy từ **gốc repo** `PDFTranslator/`.

**Bản đồ nhanh** — bảy bước, mỗi bước có một cửa kiểm:

| | Bước | Lệnh | Xong khi |
|---|---|---|---|
| 1 | Chuẩn bị một lần | §1 | `.env` đủ, Space build xanh |
| 2 | Kiểm image | `run_hf.sh check` | 3 dòng `--version` chạy sạch |
| 3 | Nung cache | `run_hf.sh warm` | `/data` có model |
| 4 | Đẩy corpus | `run_hf.sh push-corpus` | `sync ls` thấy `corpus/` |
| 5 | Chạy 3 hệ | `run_hf.sh run` | `manifest verify` không lỗi |
| 6 | DeepL ở máy | §6 | `out/deepl-document/` đã push |
| 7 | Chấm điểm + lấy về | `run_hf.sh score` → `pull` | có `report/report.md` |

---

## 0. Trước khi bắt đầu

Cần có:

* tài khoản HF có **payment method** (Jobs tính tiền theo phút — xem
  [§11 của plan](E2E_HF_PLAN.md#11-chi-phí--thời-gian--ước-lượng-có-neo-vào-số-đo-thật):
  một lượt EN→VI ≈ **$1.55**);
* `hf` CLI. Trên máy này nó nằm trong env `thesis`, **không** có trong PATH toàn cục:

```bash
conda activate thesis
hf version && hf auth whoami
```

* corpus T1 đã dựng (`benchmark/e2e/datasets/corpus/T1/` có 6 PDF + `gt.json` + `mapping.json`).

Không cần cài BabelDOC hay PDFMathTranslate ở máy — chúng nằm trong image.

---

## 1. Chuẩn bị một lần

### 1.1 Token

Tạo token **quyền `write`** ở https://huggingface.co/settings/tokens.

Một token dùng cho ba việc, nên đừng tạo loại `read`: đẩy dataset, cho **job đẩy
ngược artifact lên**, và mở khoá model gated của CometKiwi.

```bash
hf auth login          # hoặc đặt HF_TOKEN trong .env
```

> **Model gated:** vào https://huggingface.co/Unbabel/wmt22-cometkiwi-da bấm chấp
> nhận điều khoản. Không làm bước này thì job chấm điểm chạy được ~40 phút rồi mới
> chết vì 401 ở bước cuối — mất tiền mà không có kết quả.

### 1.2 Dataset repo + bucket cache

```bash
python -m benchmark.e2e.sync init --private     # tạo $HF_EVAL_REPO
```

Bucket cache tạo ở https://huggingface.co/settings/storage (hoặc để trống — sẽ mất
tiền hơn, xem cảnh báo ở bước 3).

Hai thứ này khác nhau và đừng lẫn:

| | Dùng làm gì | Sống bao lâu |
|---|---|---|
| `HF_EVAL_REPO` (dataset) | corpus + artifact + report — **nguồn sự thật** | vĩnh viễn, có version |
| `HF_CACHE_BUCKET` (bucket) | model tải về (Surya, Paddle, RT-DETR, CometKiwi) | cache, xoá lúc nào cũng được |

### 1.3 Space benchmark

HF Docker Space **chỉ đọc file tên `Dockerfile` ở gốc repo**, nên phải đổi tên khi đẩy:

```bash
# tạo Space SDK=Docker, hardware CPU basic (Space chỉ dùng làm IMAGE, không chạy)
hf repo create $HF_BENCH_SPACE --repo-type space --space_sdk docker

git clone https://huggingface.co/spaces/$HF_BENCH_SPACE /tmp/bench-space
rsync -a --exclude .git --exclude benchmark/e2e/out \
      --exclude benchmark/e2e/datasets/corpus ./ /tmp/bench-space/
cd /tmp/bench-space && cp Dockerfile.bench Dockerfile
git add -A && git commit -m "bench image" && git push
```

Build mất **20–40 phút** (CUDA + torch + surya + paddle + docling + comet + 2 venv
baseline). Theo dõi ở tab *Logs* của Space. Chỉ khi build **xanh** mới sang bước 2.

> ⚠️ **Image là ảnh chụp của Space, không phải của máy bạn.** Sửa `benchmark/` ở máy
> mà chưa push Space thì job chạy code cũ và **không có cảnh báo nào**. Mọi lần sửa
> harness đều phải push lại Space. `_run/<run_id>.json` ghi `git_rev` để hậu kiểm.

### 1.4 `.env`

```bash
cat benchmark/e2e/.env.bench.example >> .env    # rồi điền giá trị
```

Bắt buộc: `HF_TOKEN` `HF_EVAL_REPO` `HF_BENCH_SPACE` `LITELLM_BASE_URL`
`LITELLM_API_KEY` `BENCH_MODEL`. Nên có: `HF_CACHE_BUCKET` `KEY_ALIAS_PREFIX`.
Chỉ dùng ở máy: `DEEPL_AUTH_KEY`.

### 1.5 Hai việc ở LiteLLM proxy — **không có trong repo**

1. **Ép `temperature=0` cho `$BENCH_MODEL`.** `pdf2zh/translation/gateway.py`
   hardcode 0.7 và 0.2, không có knob config, mà harness không được sửa `pdf2zh/`.
   Hai baseline gửi `temperature=0`. Không ép ở proxy thì hệ của bạn chạy ở nhiệt độ
   khác hai đối thủ — hội đồng hỏi một câu là bảng sập.
2. **Tạo 4 virtual key** khớp `KEY_ALIAS_PREFIX`: `thesis-pdftranslator`,
   `thesis-babeldoc`, `thesis-pdfmathtranslate`, `thesis-ablation`. Đây là đường
   **duy nhất** lấy được token/tiền của PDFMathTranslate — bản v1 không đếm token ở
   bất cứ đâu.

---

## 2. Kiểm image — cửa đầu tiên

```bash
bash benchmark/e2e/run_hf.sh check
```

Chạy trên `cpu-basic` nên gần như miễn phí. Phải thấy đủ:

```
--- PDFTranslator ---      <bản pymupdf>
--- BabelDOC ---           babeldoc 0.6.4
--- PDFMathTranslate ---   pdf2zh v1.9.11
--- tầng chấm điểm ---     scoring deps OK
--- cache ---              HOME=/data/home  HF_HOME=/data/huggingface
```

Đọc kỹ dòng cuối: `HOME` và `HF_HOME` **phải trỏ vào `/data`**. Nếu chúng trỏ vào
`/app/.cache` thì volume chưa mount hoặc `ENTRYPOINT` không chạy — dừng lại sửa,
đừng chạy tiếp (xem §10, đây là lỗi tốn tiền nhất).

**Đừng tiêu một token LLM nào trước khi bước này sạch.**

---

## 3. Nung cache

```bash
bash benchmark/e2e/run_hf.sh warm
```

Tải Surya + Paddle + DocLayout-YOLO của BabelDOC + Docling RT-DETR + model LID vào
`/data`. Mất 15–30 phút, **làm một lần**, và mọi job sau dùng lại.

> Không có `HF_CACHE_BUCKET` thì **mỗi job tải lại 3–5 GB trong lúc đang bị tính
> tiền** — đội khoảng 50% hoá đơn mà không đổi lấy con số nào. `run_hf.sh` in cảnh
> báo khi thiếu biến này.

**Cách xác nhận cache có tác dụng:** job đầu tiên ở bước 5 phải khởi động **nhanh
hơn ít nhất 5 phút** so với khi chưa nung. Không thấy chênh lệch = cache không dùng
được.

---

## 4. Đẩy corpus

```bash
bash benchmark/e2e/run_hf.sh push-corpus
python -m benchmark.e2e.sync ls              # kiểm
```

Phải thấy `corpus/T1` với 8 file. Corpus **bất biến từ đây**: dựng lại là đổi bài
kiểm, mọi artifact cũ hết giá trị so sánh, và `manifest verify` sẽ chặn bằng `sha256`.

---

## 5. Chạy ba hệ mã nguồn mở

```bash
bash benchmark/e2e/run_hf.sh run
```

Ba job **tuần tự**, cùng `t4-medium`. Tuần tự là cố ý: hai job cùng lúc là hai job
giành GPU và băng thông của nhau, cột giây/trang thành rác. Chậm hơn ~1 giờ, đổi lại
là số đọc được.

Mỗi job tự làm bốn việc: `sync pull` corpus → `manifest write` → chạy runner →
`sync push` artifact của riêng nó.

Theo dõi:

```bash
hf jobs ls -a                  # trạng thái mọi job
hf jobs logs <job_id>          # log
hf jobs logs <job_id> -f       # theo dõi trực tiếp
hf jobs cancel <job_id>
```

Một job hỏng thì script dừng và in 60 dòng log cuối. Sửa xong chạy lại **cùng lệnh**
— runner resume theo `(doc, lang)`, doc nào đã có `output.pdf` + `meta.json` thì bỏ qua.

Xong ba hệ, kiểm ngay:

```bash
python -m benchmark.e2e.sync pull --only out
python -m benchmark.e2e.manifest verify --out benchmark/e2e/out
```

Phải ra `OK`. Hai lỗi hay gặp và ý nghĩa:

| Báo lỗi | Nghĩa là | Xử lý |
|---|---|---|
| `model không đồng nhất giữa các hệ` | có hệ chạy khi `BENCH_MODEL` còn rỗng ⇒ rơi về `gpt-4o-mini` | xoá artifact hệ đó, chạy lại. **Đừng** dùng `--allow-drift` để lách |
| `corpus đã trôi` | corpus bị dựng lại giữa chừng | mọi artifact trước đó hết giá trị |

---

## 6. DeepL — ở máy, không lên HF

DeepL là API thuần; trả tiền GPU để ngồi chờ HTTP là vô nghĩa. Nhưng output của nó
**phải đi qua đúng bộ metric như ba hệ kia**, nên phải push lên **trước** bước 7.

```bash
# LUÔN xem tiền trước
python -m benchmark.e2e.runners.deepl_doc \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi --dry-run

# chạy thật
python -m benchmark.e2e.runners.deepl_doc \
    --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \
    --tiers T1 --langs vi --char-budget 950000

python -m benchmark.e2e.sync push --only out/deepl-document
```

Hai điều phải biết trước khi bấm:

* **T1 EN→VI tốn 353.661 ký tự** trên hạn mức 1.000.000/tháng. vi + zh = 707.322.
  **Chạy lại một lần nữa là vượt.** Nên chạy DeepL **sau cùng**, khi luồng đã trơn.
* **Document thất bại vẫn bị tính tiền** — DeepL đã nhận và đã xử lý file. Exit
  code 2 nghĩa là chạm `--char-budget`, dừng có kiểm soát chứ không phải lỗi.

---

## 7. Chấm điểm

```bash
bash benchmark/e2e/run_hf.sh score
```

Một job chạy hết chuỗi: `identity` → `render_pages` → `run_detectors` →
`eval_preserve` → `eval_visual` → `eval_text` → `extract_pairs` → `eval_qe` →
`aggregate` → `sync push`.

Job này **không phân biệt hệ nào** — nó quét mọi thư mục dưới `out/` và chấm y hệt
nhau. Đó là lý do DeepL phải được push lên trước.

Đọc log để bắt lỗi sớm:

* `render`: phải thấy `6 × (1 nguồn + 5 hệ)` tài liệu;
* `run_detectors`: nếu báo thiếu `docling_ibm_models` thì image thiếu dep — quay lại §1.3;
* `eval_preserve`: hàng `Source ceiling` phải có **mIoU ≥ 0.8**. Thấp hơn nghĩa là
  detector quá yếu và **mọi so sánh phía sau vô nghĩa** — dừng lại xử lý;
* hàng `identity` phải **trùng khít** hàng `Source ceiling`. Lệch là lỗi harness,
  không phải phát hiện khoa học.

Muốn CometKiwi bản XL (chất lượng cao hơn, cần ≥24 GB):

```bash
FLAVOR=l4x1 QE_MODEL=Unbabel/wmt23-cometkiwi-da-xl bash benchmark/e2e/run_hf.sh score
```

---

## 8. Lấy kết quả về

```bash
bash benchmark/e2e/run_hf.sh pull                 # chỉ report, vài MB
python -m benchmark.e2e.sync pull                 # đầy đủ, ~500 MB
python -m benchmark.e2e.sync pull --only out/_render   # riêng ảnh trang
```

Đọc: `benchmark/e2e/out/report/report.md` (bảng chính + CI 95% + kiểm định ghép cặp)
và `report/tables/headline.csv`.

**Cửa kiểm cuối:** xoá sạch `benchmark/e2e/out/` ở máy rồi `sync pull` — phải dựng
lại được toàn bộ bảng mà không chạy lại job nào. Không làm được nghĩa là có artifact
chỉ tồn tại ở máy và luận văn không tái lập được.

Khi nộp: `sync ls` để lấy `revision` hiện tại và trích dẫn nó, để người đọc lấy lại
đúng phiên bản artifact đã dùng.

---

## 9. Lệnh HF Jobs hay dùng

```bash
hf jobs ls -a                  # mọi job + trạng thái
hf jobs logs <job_id>          # log; thêm -f để theo dõi
hf jobs wait <job_id>          # chờ, exit 0 nếu COMPLETED — dùng để chain
hf jobs cancel <job_id>        # huỷ (đang chạy = đang tính tiền)
hf jobs hardware               # bảng flavor + giá, đọc thẳng từ API
hf jobs run --ssh ...          # rồi hf jobs ssh <job_id> để debug tương tác
```

Biến có sẵn trong container: `JOB_ID`, `ACCELERATOR`, `CPU_CORES`, `MEMORY`. Runner
ghi `ACCELERATOR` vào `meta.json` nên bảng hiệu năng luôn nói rõ chạy trên phần cứng
nào.

---

## 10. Sự cố riêng của HF

**`HOME` và `HF_HOME` trỏ vào `/app/.cache` chứ không phải `/data`** — bẫy tốn tiền
nhất. `hf jobs run` **thay `CMD`**, nên nếu logic chọn cache nằm trong `CMD` (như
`Dockerfile` của Space demo) thì nó không bao giờ chạy, và mỗi job tải lại 3–5 GB
model trong lúc đang bị tính tiền theo phút. `Dockerfile.bench` đã chuyển sang
`ENTRYPOINT` — thứ mà `hf jobs run` không thay được. Nếu vẫn thấy `/app/.cache`:
kiểm `ENTRYPOINT` còn trong Dockerfile không, và `-v hf-bucket://...:/data:rw` có
được truyền không.

**Job chết đúng phút thứ 30** — quên `--timeout`. Mặc định HF là 30 phút.
`run_hf.sh` luôn truyền `--timeout $TIMEOUT` (mặc định `3h`).

**Job chạy code cũ** — quên push Space. Đối chiếu `git_rev` trong
`out/_run/<run_id>.json` với `git rev-parse HEAD` ở máy.

**401 ở bước CometKiwi, sau khi job đã chạy 40 phút** — chưa chấp nhận điều khoản
model gated, hoặc `HF_TOKEN` không được truyền vào job. Xử lý ở §1.1.

**`sync push` trong job báo 403** — `HF_TOKEN` là loại `read`. Phải là `write`.

**`run_hf.sh run` không lấy được job id** — script bắt id bằng `hf jobs run -d ... | tail -1`.
Nếu bản CLI của bạn không có `-d` hoặc in ra định dạng khác, chạy `hf jobs run --help`
rồi hoặc bỏ `-d` và chờ trực tiếp, hoặc lấy id bằng `hf jobs ls -a | head`.

**Job "Starting" rất lâu** — image benchmark nặng (~15–25 GB). Lần kéo đầu chậm là
bình thường; các job sau trên cùng hạ tầng nhanh hơn.

---

## 11. Chạy lại từng phần

| Chạy lại cái gì | An toàn? | Vì sao |
|---|---|---|
| `score` | ✅ luôn luôn | chỉ đọc artifact, không gọi API, không tốn tiền LLM |
| xoá `out/_render`, `_layout`, `_metrics`, `report` | ✅ | sinh lại được hết |
| `run` một hệ | ✅ | resume theo `(doc, lang)`; doc đã xong bị bỏ qua |
| `run` với `--no-resume` | ⚠️ | trả tiền LLM lại từ đầu |
| DeepL | ❌ | ăn hạn mức tháng, không hoàn lại |
| dựng lại corpus | ❌ | đổi bài kiểm ⇒ mọi artifact cũ vô giá trị |

---

## 12. Checklist một trang

```
[ ] hf auth whoami chạy được, token quyền WRITE
[ ] đã accept điều khoản Unbabel/wmt22-cometkiwi-da
[ ] .env đủ: HF_TOKEN HF_EVAL_REPO HF_BENCH_SPACE HF_CACHE_BUCKET
              LITELLM_BASE_URL LITELLM_API_KEY BENCH_MODEL KEY_ALIAS_PREFIX
[ ] LiteLLM: temperature=0 cho BENCH_MODEL + 4 virtual key
[ ] Space $HF_BENCH_SPACE build XANH từ Dockerfile.bench
[ ] run_hf.sh check   -> 3 dòng --version sạch, HOME=/data/...
[ ] run_hf.sh warm    -> model nằm trên /data
[ ] run_hf.sh push-corpus -> sync ls thấy corpus/T1
[ ] run_hf.sh run     -> manifest verify = OK
[ ] DeepL ở máy: --dry-run rồi mới chạy thật, rồi sync push
[ ] run_hf.sh score   -> Source ceiling mIoU >= 0.8, identity == ceiling
[ ] run_hf.sh pull    -> report/report.md
[ ] xoá out/ rồi sync pull -> dựng lại được toàn bộ bảng
```
