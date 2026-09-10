# Vendored: microsoft/table-transformer

`tatr_postprocess.py` là **bản sao nguyên văn** của `src/postprocess.py` từ
[microsoft/table-transformer](https://github.com/microsoft/table-transformer), giấy phép
**MIT** (xem `LICENSE.table-transformer`). Copyright (C) 2021 Microsoft Corporation.

| | |
|---|---|
| Nguồn | `https://raw.githubusercontent.com/microsoft/table-transformer/main/src/postprocess.py` |
| Commit | `9e54b815aa612bc2de46f65e3ae0cd0cba7089d8` |
| SHA-256 | `a7b3bdf0b0f9ea4b2ccd04bb0acca4bb05b1809d120edf556526a6d976d027db` |
| Ngày lấy | 2026-09-05 |
| Số dòng | 893 |

**KHÔNG sửa file này.** Giữ nguyên văn để có thể `diff` với upstream. Mọi thích ứng nằm
ở `../pubtables_gt.py`.

Lý do vendor thay vì tự viết lại: bản tự viết trước đó bỏ mất bước `align_supercells()`
(snap ô span vào chỉ số hàng/cột trước khi hấp thụ subcell), nên cho kết quả khác upstream
ở các bảng có ô span — tức 42,0% của split test.

Kiểm lại tính nguyên vẹn:

    sha256sum vendor/tatr_postprocess.py    # phải khớp bảng trên
