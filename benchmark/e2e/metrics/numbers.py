"""Number-digit recall — số ở bản gốc có sống sót qua bản dịch không.

Đây là metric rẻ nhất bắt được một lỗi LLM kinh điển: nuốt hoặc bóp méo số. Bộ
domain của DocLayNet gần như đặt hàng cho nó — 3 trong 6 domain (financial_reports,
patents, government_tenders) dày số.

**Vì sao so theo dãy chữ số, không so theo giá trị.** Tiếng Anh viết ``1,234.56``;
tiếng Việt viết ``1.234,56``. Cùng một giá trị, dấu ngược nhau hoàn toàn. Nếu chuẩn
hoá theo *giá trị* thì phải đoán dấu nào là thập phân, và ``1.234`` là ambiguous
thật: tiếng Anh là 1.234, tiếng Việt là 1234. Đoán sai ⇒ recall tụt vì **lý do quy
ước**, tức là phạt oan đúng cái hệ nào bản địa hoá số cho tử tế.

Nên chuẩn hoá bỏ hết dấu phân cách, chỉ giữ dấu âm + dãy chữ số:

    "1,234.56" -> "123456"      "1.234,56" -> "123456"      khớp nhau
    "1,5"      -> "15"          "1.5"      -> "15"           khớp nhau

Câu hỏi metric này trả lời chính xác là **"chữ số có còn nguyên không"**, chứ không
phải "giá trị có đúng không". Hạn chế phải nói rõ: ``1.5`` và ``15`` cùng chuẩn hoá
về ``"15"`` ⇒ **không phát hiện được lỗi mất dấu thập phân**. Đổi lấy việc không có
false negative do quy ước — với 4 hệ và 2 ngôn ngữ đích thì đánh đổi này đáng.
"""

from __future__ import annotations

import re
from collections import Counter

# Bắt được "1,234.56", "1.234,56", "-3.14", "42". KHÔNG bắt nghìn phân cách bằng
# khoảng trắng ("1 234") — cố bắt là dính rủi ro dán hai số rời thành một.
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*")

_SEPS = str.maketrans("", "", ".,\u00a0\u202f ")


def canon(token: str) -> str:
    """Chuẩn hoá một token số về dấu âm + dãy chữ số."""
    token = token.strip()
    neg = token.startswith("-")
    digits = token.translate(_SEPS).lstrip("+-")
    return ("-" + digits) if neg and digits else digits


def extract(text: str) -> list[str]:
    """Mọi số trong text, đã chuẩn hoá. Giữ trùng lặp — trùng lặp là thông tin:
    một trang có ``12`` ba lần mà bản dịch chỉ còn một là mất nội dung thật."""
    return [c for c in (canon(m.group()) for m in NUM_RE.finditer(text)) if c]


def recall(src_text: str, out_text: str) -> tuple[int, int]:
    """Trả (số lượng số ở nguồn, số lượng khớp được ở đích) — dạng multiset.

    Trả về **số đếm thô, không phải tỉ lệ**, để ``aggregate`` dồn theo trang / theo
    tài liệu / theo domain rồi mới chia. Trung bình của các tỉ lệ khác tỉ lệ của các
    tổng, và cái sau mới là con số muốn báo.
    """
    src, out = Counter(extract(src_text)), Counter(extract(out_text))
    found = sum(min(n, out[tok]) for tok, n in src.items())
    return sum(src.values()), found
