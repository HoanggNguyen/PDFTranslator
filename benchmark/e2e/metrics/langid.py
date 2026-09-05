"""Nhận diện ngôn ngữ cho UTB (untranslated blocks) — §4.3.

UTB là metric BabelDOC tự công bố, nên muốn đối chiếu định tính với paper thì phải
dùng cùng cách: language-ID trên text từng khối của PDF đích, khối nào vẫn là ngôn
ngữ nguồn thì tính là chưa dịch.

Hai backend, và **backend nào đang dùng luôn được ghi vào kết quả** (``lid_backend``):

* ``fasttext`` — ``lid.176.ftz`` (917 KB, không phải bản ``.bin`` 126 MB). Đây là
  backend cho lượt chính thức.
* ``heuristic`` — không cần cài gì, phân biệt EN / VI / ZH bằng dấu thanh + hư từ +
  tỉ lệ ký tự CJK. Có mặt để chạy được ngay khi chưa cài fasttext, **không** dùng cho
  số liệu đưa vào luận văn. Nó không phải LID tổng quát: chỉ đúng trên đúng 3 ngôn
  ngữ của benchmark này.

Vì sao có sàn độ dài (mặc định 30 ký tự ở ``eval_text``): khối toàn số, tên riêng,
nhãn hình ("Fig. 3", "Table 1") thì LID nào cũng đoán là tiếng Anh, và chúng chiếm
tỉ lệ đáng kể trong DocLayNet. Không có sàn thì UTB đo nhiễu chứ không đo lỗi dịch.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
CACHE_DIR = Path(os.environ.get("BENCH_CACHE_DIR",
                                Path.home() / ".cache" / "pdftranslator-bench"))

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Hư từ — chọn loại tần suất cao và KHÔNG trùng nhau giữa hai ngôn ngữ.
_VI_WORDS = frozenset("""của và các được trong cho những một là có không với để này
khi đã sẽ người như tại theo từ hoặc cũng nhưng vào ra trên dưới rằng thì mà""".split())
_EN_WORDS = frozenset("""the of and to in is that for with as are be this by on from
it an or which at was were has have will would can such between""".split())

# Chữ có dấu riêng của tiếng Việt (đã tổ hợp sẵn). Dùng phân rã Unicode nên không
# cần liệt kê đủ 134 ký tự: chỉ cần biết một chữ Latin có mang dấu phụ hay không.
_VI_BASE = frozenset("aeiouyd")


def _vi_diacritic_ratio(text: str) -> float:
    letters = diacritics = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        decomp = unicodedata.decomposition(ch)
        if decomp and decomp.split()[0].lower().lstrip("<") != "":
            base = unicodedata.normalize("NFD", ch)[0].lower()
            if base in _VI_BASE:
                diacritics += 1
        elif ch.lower() in "đăâêôơư":
            diacritics += 1
    return diacritics / letters if letters else 0.0


def _cjk_ratio(text: str) -> float:
    dense = [ch for ch in text if not ch.isspace()]
    if not dense:
        return 0.0
    cjk = sum(1 for ch in dense if "一" <= ch <= "鿿")
    return cjk / len(dense)


class LangID:
    """API tối thiểu: ``predict(text) -> (nhãn, độ tin)``. Nhãn theo mã 2 chữ."""

    def __init__(self, model_path: str | Path | None = None, allow_download: bool = True):
        self.backend = "heuristic"
        self._model = None
        try:
            import fasttext  # noqa: PLC0415 — optional dep, đừng bắt buộc
        except ImportError:
            return

        path = Path(model_path) if model_path else CACHE_DIR / "lid.176.ftz"
        if not path.exists() and allow_download:
            try:
                import urllib.request

                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[langid] tải {LID_URL} -> {path}", flush=True)
                urllib.request.urlretrieve(LID_URL, path)  # noqa: S310
            except Exception as exc:  # noqa: BLE001 — offline thì rơi về heuristic
                print(f"[langid] tải thất bại ({exc}); dùng backend heuristic", flush=True)
                return
        if not path.exists():
            return

        # fasttext in cảnh báo ra stderr khi load; nó vô hại nhưng làm bẩn log run.
        self._model = fasttext.load_model(str(path))
        self.backend = "fasttext"

    def predict(self, text: str) -> tuple[str, float]:
        text = " ".join(text.split())          # fasttext gãy nếu gặp "\n"
        if not text:
            return "un", 0.0
        if self._model is not None:
            labels, probs = self._model.predict(text, k=1)
            return labels[0].replace("__label__", ""), float(probs[0])
        return self._heuristic(text)

    @staticmethod
    def _heuristic(text: str) -> tuple[str, float]:
        if _cjk_ratio(text) >= 0.10:
            return "zh", 0.95

        words = [w.lower() for w in _WORD_RE.findall(text)]
        if not words:
            return "un", 0.0
        n = len(words)
        vi = sum(1 for w in words if w in _VI_WORDS) / n
        en = sum(1 for w in words if w in _EN_WORDS) / n
        # Dấu thanh là bằng chứng mạnh hơn hư từ: một khối tiếng Việt ngắn có thể
        # không chứa hư từ nào, nhưng gần như luôn có dấu.
        vi += 3.0 * _vi_diacritic_ratio(text)

        if vi == en == 0.0:
            return "un", 0.0
        total = vi + en
        return ("vi", vi / total) if vi > en else ("en", en / total)
