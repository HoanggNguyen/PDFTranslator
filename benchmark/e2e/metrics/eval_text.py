"""Khối metric KHÔNG cần detector — bảng so sánh 4 hệ đầu tiên.

Năm con số, tất cả chỉ cần ``meta.json`` + text rút từ PDF:

    page inflation · UTB/trang · number-digit recall · sec/page · success rate

Giá trị của việc gom đúng năm cái này vào một bước: chúng **chạy được cho cả 4 hệ,
kể cả DeepL**. Mọi metric hình học (§4.1) và visual (§4.2) đều neo vào ground-truth
theo *từng trang nguồn*, nên hệ nào reflow làm đổi số trang là metric mất định nghĩa
— không phải "điểm thấp", mà là không có phép ghép trang nào đúng. Năm metric ở đây
không neo vào trang nguồn nên không vướng chuyện đó.

**Number recall luôn tính ở mức tài liệu**, không phải mức trang. Dồn hết số của cả
doc thành một multiset rồi so. Lý do: đếm theo trang thì một con số bị đẩy từ trang 3
sang trang 4 bị tính là *mất*, trong khi nó chỉ *di chuyển* — và đó đúng là chuyện
xảy ra với hệ nào reflow. Mức tài liệu phân biệt được "mất" với "dịch chuyển", nên
con số so được cho **cả 4 hệ** mà không cần ngoại lệ nào cho DeepL.

Số đếm theo từng trang vẫn được xuất ra, nhưng **chỉ khi số trang vào bằng số trang
ra**, và chỉ để ``aggregate`` bootstrap theo trang. Nó chặt hơn (nhạy vị trí) nên
đừng đọc nó như một metric độc lập.

Đầu ra: ``<out>/_metrics/text/<system>.<lang>.json`` + ``summary.json``. Số đếm thô
được giữ nguyên (``n_src``/``n_found``), không tính sẵn tỉ lệ — trung bình của các
tỉ lệ khác tỉ lệ của các tổng, và cái sau mới là con số muốn báo.

Ví dụ
-----
    python -m benchmark.e2e.metrics.eval_text \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi --systems pdftranslator,babeldoc,pdfmathtranslate,deepl-document
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.e2e import manifest
from benchmark.e2e.metrics import numbers as N
from benchmark.e2e.metrics.langid import LangID

# Tên thư mục artifact của từng hệ = hằng SYSTEM trong runner tương ứng.
DEFAULT_SYSTEMS = ["pdftranslator", "babeldoc", "pdfmathtranslate", "deepl-document"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True,
                   help="Gốc corpus (chứa T1/ T2/ T3/) — dùng làm MẪU SỐ của "
                        "success rate: hệ crash sạch thì không để lại meta.json nào.")
    p.add_argument("--out", type=Path, required=True, help="Gốc artifact của runner.")
    p.add_argument("--tiers", default="T1")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    p.add_argument("--src-lang", default="en",
                   help="Mã ngôn ngữ nguồn; khối nào còn là ngôn ngữ này thì tính UTB.")
    p.add_argument("--min-block-chars", type=int, default=30,
                   help="Khối ngắn hơn mức này bị bỏ khỏi UTB. Khối toàn số / tên "
                        "riêng / nhãn hình thì LID nào cũng đoán là tiếng Anh.")
    p.add_argument("--lid-prob", type=float, default=0.5,
                   help="Ngưỡng độ tin để tính một khối là chưa dịch.")
    p.add_argument("--lid-model", default=None, help="Đường dẫn lid.176.ftz.")
    p.add_argument("--no-download", action="store_true",
                   help="Không tải model LID; thiếu thì rơi về backend heuristic.")
    p.add_argument("--allow-drift", action="store_true",
                   help="Chấm điểm dù cửa chặn manifest báo lỗi. CHỈ để debug: "
                        "corpus hoặc model đã trôi thì 4 cột không so được với nhau.")
    return p.parse_args()


def page_blocks(path: Path) -> list[list[str]]:
    """Text theo từng khối, theo từng trang. Chỉ khối text (block_type 0).

    ``get_text("blocks")`` gộp theo đoạn nên khối của nó là đơn vị hợp lý cho UTB —
    LID trên từng dòng lẻ thì quá ngắn để đoán đúng.
    """
    import fitz  # PyMuPDF

    out: list[list[str]] = []
    with fitz.open(path) as doc:
        for page in doc:
            blocks = []
            for b in page.get_text("blocks"):
                if len(b) >= 7 and b[6] != 0:      # 1 = ảnh
                    continue
                text = (b[4] or "").strip()
                if text:
                    blocks.append(text)
            out.append(blocks)
    return out


def load_src_text(pdf: Path, cache_dir: Path) -> list[str]:
    """Text trang nguồn, memo hoá — 4 hệ × nhiều ngôn ngữ đều đọc cùng file này."""
    cache = cache_dir / f"{pdf.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    pages = ["\n".join(b) for b in page_blocks(pdf)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def score_numbers(src_pages: list[str], out_pages: list[str]) -> dict:
    """Multiset recall của dãy chữ số. Xem docstring module về mức tài liệu."""
    n_src, n_found = N.recall("\n".join(src_pages), "\n".join(out_pages))

    # Chỉ ghép theo trang khi số trang khớp — và ngay cả lúc đó, đây là số phụ để
    # bootstrap, không phải metric để đọc: nó tính "di chuyển sang trang khác" thành
    # "mất".
    per_page = None
    if len(src_pages) == len(out_pages):
        per_page = [list(N.recall(s, o)) for s, o in zip(src_pages, out_pages)]

    return {"n_src": n_src, "n_found": n_found,
            "recall": round(n_found / n_src, 4) if n_src else None,
            "page_aligned": per_page is not None,
            "per_page": per_page}


def score_utb(out_blocks: list[list[str]], lid: LangID, src_lang: str,
              min_chars: int, min_prob: float) -> dict:
    per_page, scored, untrans = [], 0, 0
    for blocks in out_blocks:
        page_hits = page_scored = 0
        for text in blocks:
            if len(text) < min_chars:
                continue
            page_scored += 1
            label, prob = lid.predict(text)
            if label == src_lang and prob >= min_prob:
                page_hits += 1
        per_page.append(page_hits)
        scored += page_scored
        untrans += page_hits
    n_pages = len(out_blocks)
    return {"n_blocks_scored": scored, "n_untranslated": untrans,
            "utb_per_page": round(untrans / n_pages, 4) if n_pages else None,
            "per_page": per_page}


def discover(corpus: Path, tiers: list[str]) -> list[tuple[str, Path]]:
    jobs = []
    for tier in tiers:
        tier_dir = corpus / tier
        if tier_dir.is_dir():
            jobs += [(tier, pdf) for pdf in sorted(tier_dir.glob("*.pdf"))]
    return jobs


def evaluate(system: str, lang: str, jobs: list[tuple[str, Path]], out_root: Path,
             src_cache: Path, lid: LangID, args: argparse.Namespace) -> list[dict]:
    records = []
    for tier, pdf in jobs:
        dest = out_root / system / lang / pdf.stem
        meta_path, out_pdf = dest / "meta.json", dest / "output.pdf"

        rec = {"system": system, "lang": lang, "tier": tier, "doc_id": pdf.stem,
               "ok": False, "error": None, "n_pages_in": None, "n_pages_out": None,
               "page_inflation": None, "wall_seconds": None, "sec_per_page": None,
               "numbers": None, "utb": None}

        if not meta_path.exists():
            # Không có meta = runner chưa chạy hoặc chết trước khi ghi được gì. Vẫn
            # phải xuất hiện trong bảng, vì nó là MẪU SỐ của success rate.
            rec["error"] = "thiếu meta.json (chưa chạy?)"
            records.append(rec)
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("error", "n_pages_in", "n_pages_out", "page_inflation",
                    "wall_seconds"):
            rec[key] = meta.get(key)
        if rec["wall_seconds"] and rec["n_pages_in"]:
            rec["sec_per_page"] = round(rec["wall_seconds"] / rec["n_pages_in"], 2)

        if not out_pdf.exists():
            rec["error"] = rec["error"] or "thiếu output.pdf"
            records.append(rec)
            continue

        try:
            out_blocks = page_blocks(out_pdf)
            src_pages = load_src_text(pdf, src_cache)
        except Exception as exc:  # noqa: BLE001 — một doc hỏng không được giết cả lượt
            rec["error"] = f"đọc PDF: {type(exc).__name__}: {exc}"
            records.append(rec)
            continue

        rec["numbers"] = score_numbers(src_pages, ["\n".join(b) for b in out_blocks])
        rec["utb"] = score_utb(out_blocks, lid, args.src_lang,
                               args.min_block_chars, args.lid_prob)
        rec["ok"] = meta.get("error") is None
        records.append(rec)
    return records


def summarize(records: list[dict]) -> dict:
    """Dồn số đếm thô rồi mới chia. Xem docstring module."""
    n = len(records)
    ok = [r for r in records if r["ok"]]
    scored = [r for r in records if r["numbers"]]

    n_src = sum(r["numbers"]["n_src"] for r in scored)
    n_found = sum(r["numbers"]["n_found"] for r in scored)
    pages_out = sum(r["n_pages_out"] or 0 for r in scored)
    untrans = sum(r["utb"]["n_untranslated"] for r in scored)
    infl = [r["page_inflation"] for r in scored if r["page_inflation"]]
    # sec/page CHỈ lấy từ doc chạy xong. Doc crash giữa đường có wall_seconds nhỏ
    # nên nếu gộp vào, hệ nào chết sớm lại trông như hệ nhanh nhất.
    spp = [r["sec_per_page"] for r in ok if r["sec_per_page"]]

    return {
        "n_docs": n, "n_ok": len(ok), "n_docs_scored": len(scored),
        "success_rate": round(len(ok) / n, 4) if n else None,
        "page_inflation_mean": round(sum(infl) / len(infl), 4) if infl else None,
        "page_inflation_max": max(infl) if infl else None,
        "n_docs_reflowed": sum(1 for x in infl if abs(x - 1.0) > 1e-9),
        "sec_per_page_mean": round(sum(spp) / len(spp), 2) if spp else None,
        "number_recall": round(n_found / n_src, 4) if n_src else None,
        "n_numbers_src": n_src,
        "utb_per_page": round(untrans / pages_out, 4) if pages_out else None,
        "n_untranslated": untrans,
    }


def main() -> int:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    jobs = discover(args.corpus, tiers)
    if not jobs:
        print(f"!! không thấy PDF nào dưới {args.corpus} cho tiers {tiers}")
        return 1

    # Cửa chặn: 4 hệ chạy cách nhau hàng tuần nên corpus/model có thể đã trôi giữa
    # chừng mà PDF đầu ra không hề lộ ra. Chặn TRƯỚC khi chấm, không phải sau.
    print(">>> [gate] đối chiếu manifest giữa các hệ")
    errors, warns = manifest.verify(args.out, systems, langs)
    manifest.print_report(errors, warns)
    if errors and not args.allow_drift:
        print("\n!! Dừng: bảng so sánh sẽ vô nghĩa. Sửa rồi chạy lại, hoặc "
              "--allow-drift nếu chỉ muốn xem số để debug.")
        return 1

    lid = LangID(args.lid_model, allow_download=not args.no_download)
    if lid.backend != "fasttext":
        print("!! LID backend = heuristic (chưa có fasttext). Chạy được, nhưng UTB "
              "KHÔNG dùng cho số liệu luận văn — xem metrics/langid.py.", flush=True)

    metrics_dir = args.out / "_metrics" / "text"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    src_cache = args.out / "_metrics" / "_srctext"

    rows, summaries = [], {}
    for system in systems:
        for lang in langs:
            if not (args.out / system / lang).is_dir():
                continue
            records = evaluate(system, lang, jobs, args.out, src_cache, lid, args)
            summary = summarize(records)
            summary["lid_backend"] = lid.backend
            (metrics_dir / f"{system}.{lang}.json").write_text(
                json.dumps({"summary": summary, "records": records},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            summaries[f"{system}/{lang}"] = summary
            rows.append((system, lang, summary))

    if not rows:
        print(f"!! không thấy artifact nào dưới {args.out}/<system>/<lang>/. "
              f"Chạy run_all.sh trước.")
        return 1

    (args.out / "_metrics" / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    hdr = (f"{'system':22} {'lang':4} {'ok':>7} {'inflation':>10} {'reflow':>7} "
           f"{'sec/page':>9} {'UTB/trang':>10} {'num-recall':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for system, lang, s in rows:
        def f(v, spec=".3f"):
            return format(v, spec) if isinstance(v, (int, float)) else "—"
        print(f"{system:22} {lang:4} {s['n_ok']:>3}/{s['n_docs']:<3} "
              f"{f(s['page_inflation_mean']):>10} {s['n_docs_reflowed']:>7} "
              f"{f(s['sec_per_page_mean'], '.1f'):>9} "
              f"{f(s['utb_per_page']):>10} {f(s['number_recall']):>11}")
    print(f"\nlid_backend = {lid.backend}   |   chi tiết: {metrics_dir}/")
    print("Cột 'reflow' = số doc có page_inflation != 1 ⇒ doc đó BỊ LOẠI khỏi mIoU / "
          "Anchor-IoU / Masked-SSIM (áp cho mọi hệ như nhau).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
