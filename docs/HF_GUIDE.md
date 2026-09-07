# Chạy benchmark E2E trên Hugging Face, từ máy chưa có token

Đây là runbook cho **driver mới** `benchmark.e2e.hf_driver`. Mọi lệnh chạy từ gốc
repo. `bash benchmark/e2e/run_hf.sh ...` là wrapper tương đương. Không dùng nhánh
`RUNTIME=hfjobs` của `run_all.sh` cho quy trình này.

Có ba mức kiểm tra:

| Mức | Chạy ở đâu | Có dùng key/GPU? | Mục đích |
|---|---|---|---|
| Mock test | Máy local/CI | Không | Kiểm orchestration, artifact, resume, chống leak |
| Smoke test | HF Jobs | Có, dịch thật | 1 PDF ngắn → 3 hệ → một job scoring chung |
| Eval | HF Jobs + DeepL local | Có | Corpus đã chốt, 3 hoặc 4 hệ, lưu số liệu |

**Mock test qua không chứng minh Docker/GPU/API đã chạy được.** Phải đi qua smoke
test HF. Một PDF không đủ để kết luận hệ nào tốt hơn. Synthetic PDF chỉ dùng để
kiểm luồng; eval phải dùng corpus thật có ground truth.

## 1. Cài môi trường local và chạy mock test — chưa cần token

Dùng Python 3.10–3.12 cho môi trường điều phối; không cài vào env OCR đang dùng.
Ví dụ với Python 3.12 đã cài trên máy:

```bash
python3.12 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -r benchmark/e2e/requirements-local.txt
python -m pytest benchmark/e2e/tests benchmark/e2e/metrics/test_metrics.py -q
```

Bộ test dùng PDF thật được tạo trong thư mục tạm, thay translator/detector/QE/LID
bằng mock. Render, alignment, các metric hình học/text/visual và aggregate vẫn chạy
code thật. Test kiểm credential giả không vào log, upload từ chối file chứa key,
thiếu hệ chặn scoring, resume không gọi dịch lại, và lỗi vẫn có checkpoint.

Controller pin `huggingface_hub==1.30.0`; image model pin `0.36.0` vì
`transformers==4.56.1`/Docling yêu cầu Hub SDK `<1`. Hai môi trường tách nhau có chủ ý.

**Qua bước khi:** toàn bộ tests pass. Không cần cài Surya, Paddle, BabelDOC, COMET
hay model weights trên máy local để chạy tests.

## 2. Chuẩn bị PDF nguồn cho smoke test

Nếu đã có corpus T1, dùng corpus đó. `prepare-smoke` sẽ chọn một PDF (hoặc PDF bạn
chỉ định) và lấy tối đa 2 trang đầu; cập nhật GT/mapping/hash theo phần được giữ lại.
File corpus gốc không bị sửa.

Nếu chưa có corpus nào, tạo PDF synthetic gồm 2 trang tiếng Anh, đoạn văn, số và hình:

```bash
python -m benchmark.e2e.datasets.make_smoke \
  --out benchmark/e2e/work/demo-source
```

Không cần tải dataset để thử connectivity. Synthetic GT có thể không trùng cách
annotate của DocLayNet; không diễn giải điểm của nó như kết quả nghiên cứu.

## 3. Tạo `.env` local — lúc này mới cần điền model/endpoint

Nếu chưa có `.env`, copy template; nếu đã có thì chỉ ghép các biến còn thiếu:

```bash
cp -n benchmark/e2e/.env.bench.example .env
chmod 600 .env
```

Mở `.env` bằng editor. Chưa có token thì để các credential trống, nhưng điền:

```dotenv
LITELLM_BASE_URL=https://your-proxy.example/v1
BENCH_MODEL=exact-model-id-on-proxy
HF_EVAL_REPO=YOUR_USER/pdftranslator-eval
HF_BENCH_SPACE=YOUR_USER/pdftranslator-bench
BENCH_RUN_ID=smoke-001
```

Endpoint phải là HTTPS, không chứa username/password/query. Địa chỉ chỉ truy cập
được trong VPN/mạng công ty sẽ không tự trở nên truy cập được từ HF.

Python tự đọc `.env` ở gốc repo bằng `python-dotenv`, không thực thi file như shell.
Biến đã export ngoài terminal thắng `.env`. **HF CLI riêng (`hf ...`) không tự đọc
file này**; ở bước 5 dùng `hf auth login`. Không chạy `source .env`, `set -x`,
`printenv`, `hf env`, hoặc in `os.environ` để debug credential.

## 4. Snapshot smoke test — vẫn chưa cần HF token

Với synthetic PDF của bước 2:

```bash
python -m benchmark.e2e.hf_driver prepare-smoke \
  --run-id smoke-001 --source benchmark/e2e/work/demo-source --pages 2
```

Hoặc với corpus thật:

```bash
python -m benchmark.e2e.hf_driver prepare-smoke \
  --run-id smoke-001 --source benchmark/e2e/datasets/corpus \
  --doc-id TEN_DOC_KHONG_CO_DUOI_PDF --pages 2
```

Chọn **một** lệnh. Bỏ `--doc-id` sẽ tự chọn PDF nhỏ nhất trong T1. Chỉ có một PDF
đầu vào, không phải ba bản corpus khác nhau. Cảnh báo `DEEPL-FLOOR` được phép ở
smoke vì lượt này không gọi DeepL.

Snapshot ở:

```text
benchmark/e2e/work/smoke-001/
  corpus/
    run.json
    T1/<doc>.pdf
    T1/gt.json
    T1/mapping.json
```

`run.json` chốt model, proxy URL, systems, ngôn ngữ, flavor, QE model, đơn vị bootstrap,
hash code và hash PDF/GT/mapping. Thay những thông số này phải dùng **run ID mới**.
Không sửa JSON bằng tay. Sửa code sau prepare thì chuẩn bị run mới và build image mới.

## 5. Tạo tài khoản, token và tài nguyên HF

1. Bật billing/payment cho Jobs trong tài khoản HF; kiểm giá bằng `hf jobs hardware`.
2. Tạo token controller tại [HF tokens](https://huggingface.co/settings/tokens).
   Token cần quyền Jobs/compute, tạo/quản lý các repo benchmark và cache bucket.
   Ưu tiên fine-grained token theo quyền cần; tên quyền trong UI có thể thay đổi.
3. Đăng nhập CLI bằng prompt ẩn, không đưa token trực tiếp vào dòng lệnh:

```bash
hf auth login
hf auth whoami
hf version
```

4. Thay `YOUR_USER` trong các lệnh sau bằng username của bạn:

```bash
hf repos create YOUR_USER/pdftranslator-eval --repo-type dataset --private
hf repos create YOUR_USER/pdftranslator-bench --repo-type space --space-sdk docker --private
hf buckets create YOUR_USER/pdftranslator-cache --private
```

Bucket là tùy chọn. Không có bucket thì model tải lại giữa các job. Dataset là
nguồn lưu artifact có version; bucket chỉ là cache. Space dùng CPU basic để build/
giữ image; GPU được chọn riêng khi phát Job.

5. Vào [CometKiwi model](https://huggingface.co/Unbabel/wmt22-cometkiwi-da), chấp nhận
   điều khoản bằng tài khoản sở hữu token runtime.
6. Tạo token runtime riêng: đọc/ghi dataset eval và đọc model cần thiết, bao gồm
   gated models đã được cấp quyền. Không cấp quyền điều phối Jobs nếu không cần.
7. Điền `.env`:

```dotenv
HF_TOKEN=controller-token
HF_JOB_TOKEN=runtime-token
HF_CACHE_BUCKET=YOUR_USER/pdftranslator-cache
```

Không dùng các chuỗi placeholder như credential thật. Driver dùng `HF_TOKEN` local
để submit; chỉ ánh xạ `HF_JOB_TOKEN` thành `HF_TOKEN` trong container. Quyền mount
bucket được HF xét khi submit bằng tài khoản điều phối.

## 6. Tạo virtual keys ở proxy

Tạo ba key **khác nhau**, chỉ cho model đã chốt, có ngân sách nhỏ cho smoke:

| Alias ở proxy | Biến local |
|---|---|
| `smoke-001-pdftranslator` | `LITELLM_KEY_PDFTRANSLATOR` |
| `smoke-001-babeldoc` | `LITELLM_KEY_BABELDOC` |
| `smoke-001-pdfmathtranslate` | `LITELLM_KEY_PDFMATHTRANSLATE` |

Điền giá trị secret vào `.env`. Driver chọn một key cho một job và đặt
`LITELLM_API_KEY` trong container. Alias trong metadata không tự tạo/đổi alias ở
proxy; bạn phải cấu hình alias tương ứng trên proxy.

Ép temperature=0, tắt response cache ở proxy, giữ routing/model backend ổn định.
Harness không thể chứng minh proxy đã áp dụng các điều kiện này; lưu cấu hình
không chứa secret làm bằng chứng khi eval.

## 7. Đóng gói và build Space

```bash
python -m benchmark.e2e.hf_driver stage-space
```

Review thư mục `benchmark/e2e/space-build/`: chỉ source theo allowlist,
`Dockerfile`, README của Space và `image-source.json`. Tool quét credential trước
khi tạo staging. Không copy toàn bộ working directory bằng `rsync`.

```bash
hf upload YOUR_USER/pdftranslator-bench benchmark/e2e/space-build . --repo-type space
```

Theo dõi build ở Space hoặc:

```bash
hf spaces logs YOUR_USER/pdftranslator-bench --build
```

**Qua bước khi:** build thành công. Không nhập secrets vào Space Settings để chạy
Jobs: Space chỉ cung cấp image, còn Jobs nhận secrets khi submit. Server mặc định
chỉ phục vụ thư mục health rỗng; không serve `/app` hoặc artifact.

Image có ba interpreter: PDFTranslator/scoring, BabelDOC, PDFMathTranslate. Detector
pin `docling-ibm-models==3.9.1` và layout weights `docling-models@v2.2.0`.

Khi build lại, tạo staging mới (`--dest benchmark/e2e/space-build-v2`) và không giữ
staging cũ trong source bundle. Nên đặt staging bên ngoài repo nếu chọn tên khác:
`--dest /tmp/pdftranslator-space-v2`. Với eval cần tái lập chính xác dependency,
đặt `HF_BENCH_IMAGE` thành registry reference có digest trước `prepare-eval`;
hash code không thay thế image digest.

## 8. Push corpus → check → warm → probe

Thứ tự này quan trọng: worker đọc run contract từ dataset, nên phải push corpus
**trước** check/warm.

```bash
python -m benchmark.e2e.hf_driver push-corpus --run-id smoke-001
python -m benchmark.e2e.hf_driver check --run-id smoke-001
python -m benchmark.e2e.hf_driver warm --run-id smoke-001
python -m benchmark.e2e.hf_driver probe --run-id smoke-001
```

- `push-corpus`: tạo/kiểm dataset private, từ chối ghi đè contract khác cùng run ID.
- `check`: CPU job, import dependencies, version baseline, đọc/ghi dataset.
- `warm`: GPU job, nạp Surya/Paddle/BabelDOC, Docling, fastText và **CometKiwi ngay
  ở bước này**, để lỗi gated model xuất hiện trước khi dịch.
- `probe`: ba job tuần tự, mỗi key gửi một request dịch rất ngắn. Có phát sinh phí.
  Response không được in. Probe xác minh proxy/model truy cập được từ HF.

Nếu không có persistent cache, entrypoint nói rõ cache ephemeral; việc `/data`
tồn tại không còn bị coi là bằng chứng bucket đã được mount.

Driver in Job ID và lưu ID tại `work/<run-id>/jobs/` trước khi chờ. Muốn xem log:

```bash
hf jobs logs JOB_ID
hf jobs logs JOB_ID -f
hf jobs cancel JOB_ID
```

Ctrl-C ở terminal **không huỷ job remote**. Dùng ID đã lưu để theo dõi/huỷ; tránh
submit thêm khi job cũ vẫn chạy. Không chạy hai controller cùng run ID đồng thời.

## 9. Dịch một PDF bằng ba pipeline

```bash
python -m benchmark.e2e.hf_driver run --run-id smoke-001
python -m benchmark.e2e.hf_driver pull --run-id smoke-001
python -m benchmark.e2e.hf_driver verify --run-id smoke-001
```

Ba job chạy tuần tự cùng flavor để giữ điều kiện phần cứng nhất quán và dễ kiểm
ngân sách. HF Jobs riêng không mặc nhiên dùng chung một GPU; tuần tự không phải
bằng chứng loại bỏ mọi nhiễu timing/network.

Kết quả cần có đúng ba `output.pdf`, mỗi file thuộc một hệ. Metadata phải khớp
run signature, model và hash corpus. Mở ba PDF, kiểm có tiếng Việt, không trắng,
không mất phần lớn nội dung. Validator không tự chứng minh chất lượng dịch.

Job checkpoint sau **từng document/ngôn ngữ**, gồm trạng thái lỗi và log đã lọc.
Resume kiểm hash output/run trước khi bỏ qua. Nếu job bị kill giữa document,
checkpoint trước đó còn trên dataset; document đang chạy có thể phải trả tiền lại.

Chạy lại một hệ:

```bash
python -m benchmark.e2e.hf_driver run --run-id smoke-001 --system babeldoc
```

Sau `pull`, có thể chạy lại lệnh này để xác nhận log ghi `resume` và không có request
dịch mới ở proxy. Job/GPU vẫn có thể tốn phí khởi động dù không dịch thêm.

## 10. Scoring chung cho cả ba output

```bash
python -m benchmark.e2e.hf_driver score --run-id smoke-001
python -m benchmark.e2e.hf_driver pull --run-id smoke-001
python -m benchmark.e2e.hf_driver verify --run-id smoke-001 --require-score
```

Chuỗi chấm:

```text
kiểm đủ 3 hệ × 1 PDF × vi
  → identity
  → render nguồn/output
  → detector Docling chung
  → reading-order + chẩn đoán box
  → NT-PPR / IO-PPR / OF-harm / Page-fail (headline, không dùng detector)
  → text integrity (UTB/trang)
  → alignment
  → CometKiwi QE
  → aggregate
  → kiểm source ceiling/identity, lưu score-status và hashes
  → upload
```

Scoring không nhận LLM API key. `identity` là hàng kiểm harness, không phải pipeline
thứ tư. Đầu ra ở:

```text
benchmark/e2e/work/smoke-001/out/
  pdftranslator/vi/<doc>/output.pdf
  babeldoc/vi/<doc>/output.pdf
  pdfmathtranslate/vi/<doc>/output.pdf
  _metrics/qe/<system>.vi.json
  _run/score-status.json
  report/report.md
  report/tables/headline.csv
```

**Nghiệm thu:** `verify --require-score` qua; status completed=true; ba PDF hợp lệ;
report có ba hệ và QE, có nhãn SMOKE, không có CI/p-value suy luận từ một PDF.
Identity layout phải trùng source ceiling. Thiếu metric không được hiểu là 0;
anchor/formula không tồn tại trong PDF có thể có giá trị trống hợp lý.

Source ceiling thấp nghĩa là detector không khớp GT; xem trước khi tin vào metric
layout. Mốc 0.8 trong tài liệu thiết kế là tiêu chí nghiên cứu cần xem xét trên corpus
thật, không phải bảo đảm mọi synthetic PDF phải đạt. Số trang đầu ra thay đổi sẽ làm
một số metric layout/visual không chấm được; report phải công bố coverage đó.

## 11. Chạy eval thật khi cần

Dựng corpus thật theo `benchmark/e2e/README.md` và `datasets/build_doclaynet.py`.
Không dùng synthetic corpus; driver từ chối `prepare-eval` với dữ liệu synthetic
của công cụ trên. Chốt tier/ngôn ngữ/model/GPU và dùng run ID mới.

Ví dụ 4 hệ, T1 EN→VI:

```bash
python -m benchmark.e2e.hf_driver prepare-eval \
  --run-id eval-t1-vi-001 --source benchmark/e2e/datasets/corpus \
  --tiers T1 --langs vi \
  --systems pdftranslator,babeldoc,pdfmathtranslate,deepl-document --unit doc
python -m benchmark.e2e.hf_driver push-corpus --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver check --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver warm --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver probe --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver run --run-id eval-t1-vi-001
```

Đổi alias/budget của ba virtual key cho run mới trước `probe/run`. Chỉ muốn 3 hệ thì
bỏ `deepl-document` khỏi `--systems`, bỏ các lệnh DeepL bên dưới.

`--langs vi,zh` được hỗ trợ. `--unit doc` lấy lại mẫu theo PDF được đưa vào runner;
T1 gộp nhiều trang từ tài liệu gốc khác nhau nên cần xem mapping trước khi chọn đơn
vị suy luận. `--unit page` chỉ hợp lý nếu đã chứng minh các trang là đơn vị độc lập.
Đơn vị thống kê là một phần contract, không đổi giữa chừng.

### DeepL local và upload checkpoint

Điền `DEEPL_AUTH_KEY` trong `.env`, kiểm hạn mức/tariff thực tế của tài khoản.
Không mặc định mọi tài khoản có quota 1 triệu ký tự.

```bash
python -m benchmark.e2e.hf_driver deepl-dry-run --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver deepl --run-id eval-t1-vi-001 --char-budget 950000
```

Dry-run không gửi tài liệu; lệnh thật có phí, tự resume và push checkpoint vào
`<run-id>/out/deepl-document`. Document handle/key không được ghi vào artifact.
Budget được runner kiểm trước document; không phải hard cap giao dịch của nhà cung
cấp. Đặt limit ở tài khoản DeepL nếu cần chặn chi tiêu chắc chắn.

### Chấm và lấy kết quả

```bash
python -m benchmark.e2e.hf_driver score --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver pull --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver verify --run-id eval-t1-vi-001 --require-score
```

Mặc định yêu cầu tất cả thành công. Nếu một pipeline thất bại trên một số PDF và
bạn cần đưa failure rate vào eval, sau khi kiểm log có thể dùng:

```bash
python -m benchmark.e2e.hf_driver score --run-id eval-t1-vi-001 --allow-failures
python -m benchmark.e2e.hf_driver pull --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver verify --run-id eval-t1-vi-001 --allow-failures --require-score
```

Cờ này cho phép **failure record rõ ràng**, không cho phép thiếu system/document,
model/hash lệch hoặc output đã bị sửa. Công bố số doc chấm được, failure rate và
alignment coverage cùng với điểm trung bình. Không loại thất bại khỏi mẫu số.

QE là chỉ báo không reference; xem `metrics.eval_qe --calibrate` trước khi diễn giải
thành xếp hạng chất lượng dịch. Thời gian runner có ranh giới warmup/cache khác nhau;
report hiện gắn nhãn chẩn đoán, chưa nên dùng để tuyên bố speedup công bằng.

### Token/USD từ proxy (tuỳ chọn)

Export spend logs ở phía proxy, lọc đúng virtual key + thời gian chạy document,
bao gồm retries tính phí, loại probe. Không đưa master key hoặc raw spend log lên HF.
Chuẩn hoá CSV local thành đúng sáu cột:

```csv
run_id,system,request_id,tokens_in,tokens_out,usd
eval-t1-vi-001,pdftranslator,request-a,100,200,0.001
eval-t1-vi-001,babeldoc,request-b,150,240,0.002
eval-t1-vi-001,pdfmathtranslate,request-c,120,210,0.0015
```

Đây chỉ là ví dụ schema, không phải chi phí thực tế. Dùng một dòng mỗi request:

```bash
python -m benchmark.e2e.hf_driver import-costs --run-id eval-t1-vi-001 \
  --spend-csv /path/to/filtered-spend.csv
python -m benchmark.e2e.hf_driver score --run-id eval-t1-vi-001
python -m benchmark.e2e.hf_driver pull --run-id eval-t1-vi-001
```

Importer chặn request ID trùng, số âm/NaN, sai run/system, thiếu hệ. Chỉ upload tổng
đã chuẩn hoá; controller không tự gọi endpoint admin LiteLLM. Phí HF và DeepL lấy từ
billing tương ứng, không cộng vào USD LLM một cách ngầm định.

## 12. Tái lập và chạy lại an toàn

Dataset layout là `<run-id>/corpus` và `<run-id>/out`. Local là `work/<run-id>`.
`pull` ghim một dataset commit cho cả lần tải và ghi `download-revision.txt`.

Muốn kiểm từ thư mục mới, **đổi tên** thư mục local đang có để giữ bản sao, rồi:

```bash
python -m benchmark.e2e.hf_driver pull --run-id eval-t1-vi-001 --revision DATASET_COMMIT_SHA
python -m benchmark.e2e.hf_driver verify --run-id eval-t1-vi-001 --require-score
```

Lệnh verify yêu cầu status completed và hash report/metrics đúng snapshot. Vì sync
không xoá file remote, sau một scoring rerun lỗi có thể còn report cũ trên dataset;
**không đọc report mà bỏ qua `score-status.json`/`verify --require-score`.**

| Tình huống | Cách xử lý |
|---|---|
| Thiếu token/key | Điền `.env`; env export có thể đang đè giá trị file |
| Proxy timeout từ HF | Kiểm DNS/firewall/VPN; không dùng endpoint chỉ truy cập nội bộ |
| 401/403 ở warm | Accept gated model; kiểm scope token runtime |
| Sai hash source/image | Rebuild Space, chuẩn bị run ID mới; không sửa contract |
| Runner lỗi | Pull/check log, chạy lại `run --system ...`; checkpoint hợp lệ được bỏ qua |
| Thiếu metric/QE | Kiểm pairs, số trang/reflow và scoring logs; không điền điểm 0 |
| Scoring hết VRAM | Chuẩn bị run mới với `--flavor l4x1` hoặc chọn QE phù hợp trước prepare |
| Upload từ chối credential | Xử lý file được nêu, rotate key nếu từng upload; không bỏ scanner |
| Shell đóng giữa job | Lấy Job ID từ `work/<run-id>/jobs`, kiểm/huỷ trước khi submit lại |

## 13. Cơ chế secrets và giới hạn

```text
.env local (không commit)
  → Python controller chọn secrets theo job
  → HF Jobs secrets API (server-side encryption)
  → environment variables trong container
  → runner chỉ nhận key cần thiết
```

Không bake secrets vào image, không truyền key trong argv; BabelDOC dùng
`OPENAI_API_KEY` qua env. Baseline không thừa hưởng HF token. Worker giữ runtime
token để sync; scoring/warm không nhận LLM key. Upload dùng allowlist + staging
snapshot + scan credential; raw/config/dotfiles không được publish.

Đây là giảm nguy cơ leak, **không phải bảo đảm tuyệt đối**: code/dependency có quyền
đọc secret vẫn có thể tiết lộ nó, scanner không phát hiện mọi kiểu mã hoá/biến đổi.
Giới hạn quyền/budget, dùng private repos, chỉ cấp quyền chỉnh code/Jobs cho người
được tin cậy. Nếu đã lộ key thì revoke/rotate; xoá log hiện tại không xoá lịch sử Git.

Nguồn HF: [Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration),
[Jobs SDK](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.run_job),
[CLI](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli),
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).
