"""Trích cặp {nguồn, đích} từ PDF của MỌI hệ — đầu vào cho CometKiwi (§4.4).

Quy tắc công bằng quan trọng nhất của file này: **mọi hệ đều bị đối xử như hộp đen.**

PDFTranslator có sẵn ``phase2_translated.json`` đã align hoàn hảo; BabelDOC có
``translate_tracking.json`` nếu chạy ``--debug``; PDFMathTranslate không dump gì và
DeepL thì không mở được. Nếu dùng dump nội bộ cho hệ nào có, hệ đó được align chuẩn
miễn phí trong khi hệ khác phải gánh thêm sai số trích xuất — hai cột không còn đo
cùng một thứ. Nên ở đây tất cả đều đi qua một đường: **rút text từ ``output.pdf``.**

Ghép thế nào, và vì sao:

* **Ghép theo trang khi số trang khớp**, ghép theo cả tài liệu khi không khớp. Hệ
  reflow (DeepL) vẫn có cặp để chấm, chỉ là nhiễu hơn — và điều đó được ghi lại
  trong ``align_mode`` để đọc số cho đúng.
* **Quy hoạch động đơn điệu** (cho phép chèn/xoá), không phải ghép theo chỉ số.
  Ghép theo chỉ số thì một khối bị gộp ở đầu trang làm lệch toàn bộ phần còn lại.
* **Điểm tương đồng dựa trên CHỮ SỐ và tỉ lệ độ dài**, vì nguồn và đích khác ngôn
  ngữ nên không có gì để so về từ vựng. Số thì được giữ nguyên qua bản dịch, và độ
  dài thì tỉ lệ tương đối ổn định (EN→VI giãn ~1.15–1.3×). Đây là tín hiệu yếu
  nhưng **không thiên vị hệ nào** — đó mới là điều cần.

Đầu ra ``<out>/_pairs/<system>.<lang>.jsonl``, mỗi dòng một cặp::

    {"doc_id": ..., "page": 3, "idx": 7, "src": "...", "mt": "...",
     "align_mode": "page", "score": 0.62}

Ví dụ
-----
    python -m benchmark.e2e.align.extract_pairs \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DIGITS = re.compile(r"\d+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None)
    p.add_argument("--min-chars", type=int, default=30,
                   help="Bỏ khối ngắn hơn mức này. Nhãn hình, số trang, tiêu đề "
                        "cột — CometKiwi chấm chúng thành nhiễu thuần tuý.")
    p.add_argument("--gap-penalty", type=float, default=0.25,
                   help="Giá của một lần chèn/xoá trong quy hoạch động. Cao quá thì "
                        "ép ghép bừa; thấp quá thì bỏ cặp đúng.")
    p.add_argument("--min-score", type=float, default=0.15,
                   help="Cặp dưới ngưỡng này bị loại khỏi jsonl.")
    return p.parse_args()


def blocks(path: Path) -> list[list[dict]]:
    """Khối text theo trang, đã sắp theo thứ tự đọc (trên xuống, trái sang phải)."""
    import fitz  # PyMuPDF

    pages: list[list[dict]] = []
    with fitz.open(path) as doc:
        for page in doc:
            items = []
            for b in page.get_text("blocks"):
                if len(b) >= 7 and b[6] != 0:          # 1 = ảnh
                    continue
                text = " ".join((b[4] or "").split())
                if text:
                    items.append({"text": text, "bbox": [b[0], b[1], b[2], b[3]]})
            # Lượng tử y để hai khối cùng dòng không đảo nhau vì lệch nửa điểm.
            items.sort(key=lambda it: (round(it["bbox"][1] / 5), it["bbox"][0]))
            pages.append(items)
    return pages


def similarity(src: str, mt: str) -> float:
    """0..1. Nửa từ chữ số trùng nhau, nửa từ tỉ lệ độ dài."""
    ds, dm = set(DIGITS.findall(src)), set(DIGITS.findall(mt))
    if ds or dm:
        digit = len(ds & dm) / len(ds | dm)
    else:
        digit = 0.5                     # không có số ⇒ trung tính, không thưởng phạt
    ls, lm = len(src), len(mt)
    ratio = min(ls, lm) / max(ls, lm) if max(ls, lm) else 0.0
    return 0.5 * digit + 0.5 * ratio


def align(src: list[str], mt: list[str], gap: float) -> list[tuple[int, int, float]]:
    """Quy hoạch động đơn điệu (Needleman-Wunsch). Trả (i_src, i_mt, score)."""
    n, m = len(src), len(mt)
    if not n or not m:
        return []

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]     # 0=ghép 1=bỏ src 2=bỏ mt
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] - gap
        back[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] - gap
        back[0][j] = 2

    for i in range(1, n + 1):
        si = src[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + similarity(si, mt[j - 1])
            up = score[i - 1][j] - gap
            left = score[i][j - 1] - gap
            best = max(diag, up, left)
            score[i][j] = best
            back[i][j] = 0 if best == diag else (1 if best == up else 2)

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        move = back[i][j]
        if move == 0:
            pairs.append((i - 1, j - 1, similarity(src[i - 1], mt[j - 1])))
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def pairs_for_doc(src_pages: list[list[dict]], mt_pages: list[list[dict]],
                  args: argparse.Namespace) -> tuple[list[dict], str]:
    def keep(items):
        return [it["text"] for it in items if len(it["text"]) >= args.min_chars]

    out: list[dict] = []
    if len(src_pages) == len(mt_pages):
        mode = "page"
        for page, (sp, mp) in enumerate(zip(src_pages, mt_pages)):
            s, m = keep(sp), keep(mp)
            for k, (i, j, sc) in enumerate(align(s, m, args.gap_penalty)):
                out.append({"page": page, "idx": k, "src": s[i], "mt": m[j],
                            "score": round(sc, 4)})
    else:
        # Reflow: chỉ số trang không còn nghĩa, ghép trên toàn tài liệu.
        mode = "document"
        s = [t for p in src_pages for t in keep(p)]
        m = [t for p in mt_pages for t in keep(p)]
        for k, (i, j, sc) in enumerate(align(s, m, args.gap_penalty)):
            out.append({"page": None, "idx": k, "src": s[i], "mt": m[j],
                        "score": round(sc, 4)})
    return [p for p in out if p["score"] >= args.min_score], mode


def discover_docs(corpus: Path, tiers: list[str]) -> list[Path]:
    docs = []
    for tier in tiers:
        tier_dir = corpus / tier
        if tier_dir.is_dir():
            docs += sorted(tier_dir.glob("*.pdf"))
    return docs


def main() -> int:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    docs = discover_docs(args.corpus, tiers)
    if not docs:
        print(f"!! không thấy PDF nào dưới {args.corpus} cho tiers {tiers}")
        return 1

    systems = ([s.strip() for s in args.systems.split(",") if s.strip()]
               if args.systems else
               sorted(d.name for d in args.out.iterdir()
                      if d.is_dir() and not d.name.startswith("_")))

    dest = args.out / "_pairs"
    dest.mkdir(parents=True, exist_ok=True)

    # Text trang nguồn đọc một lần, dùng cho mọi hệ.
    src_cache = {pdf.stem: blocks(pdf) for pdf in docs}

    rows = []
    for system in systems:
        for lang in langs:
            base = args.out / system / lang
            if not base.is_dir():
                continue
            lines, modes, n_docs = [], {"page": 0, "document": 0}, 0
            for pdf in docs:
                out_pdf = base / pdf.stem / "output.pdf"
                if not out_pdf.exists():
                    continue
                try:
                    mt_pages = blocks(out_pdf)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {system}/{lang}/{pdf.stem}: {exc}", flush=True)
                    continue
                got, mode = pairs_for_doc(src_cache[pdf.stem], mt_pages, args)
                modes[mode] += 1
                n_docs += 1
                for rec in got:
                    lines.append({"system": system, "lang": lang,
                                  "doc_id": pdf.stem, "align_mode": mode, **rec})

            if not lines:
                continue
            path = dest / f"{system}.{lang}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for rec in lines:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append((system, lang, n_docs, len(lines), modes))

    if not rows:
        print(f"!! không thấy output.pdf nào dưới {args.out}/<system>/<lang>/")
        return 1

    hdr = f"{'system':22} {'lang':4} {'docs':>5} {'cặp':>7} {'ghép theo trang':>17}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for system, lang, n_docs, n_pairs, modes in rows:
        print(f"{system:22} {lang:4} {n_docs:5d} {n_pairs:7d} "
              f"{modes['page']:>8}/{n_docs}")
    print(f"\nchi tiết: {dest}/   |   'ghép theo trang' thấp = hệ đó reflow, "
          f"cặp của nó nhiễu hơn — nói rõ khi báo cáo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
