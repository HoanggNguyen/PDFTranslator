"""Baseline PDFMathTranslate v1 (Byaidu/PDFMathTranslate) — subprocess, venv riêng.

Đây là baseline "pipeline cổ điển": DocLayout-YOLO + pdfminer, đục chữ gốc rồi vẽ
chữ dịch trở lại đúng chỗ. Nó là tiền thân của Pipeline A mà PDFTranslator kế thừa,
nên cột này trong bảng so sánh trả lời đúng câu hỏi "chuyển sang Surya + Typst được
gì" — chứ không chỉ là thêm một baseline cho đủ số.

Ba cái bẫy phải xử ở đây, cả ba đều im lặng:

* **Tên package trùng.** Package của nó cũng là ``pdf2zh``. Chạy chung interpreter
  với PDFTranslator là đo sai hệ thống mà không có lỗi nào. ``_common.child_env()``
  xoá ``PYTHONPATH`` và ``cwd`` đặt ra thư mục tạm, không phải repo root.
* **Cache dịch.** pdf2zh v1 nhớ bản dịch trong sqlite. Không truyền
  ``--ignore-cache`` thì lượt thứ hai (lang thứ hai, hay chạy lại sau timeout) ăn
  cache: latency và token tụt xuống gần 0 và cột hiệu năng thành số bịa. Runner này
  LUÔN truyền ``--ignore-cache``.
* **Config dính ở ``~/.config/PDFMathTranslate/config.json``.** ``set_envs`` ghi
  ``OPENAI_*`` vào đó rồi đọc lại ở lần sau. Chạy ở shell không có env var là nó âm
  thầm dùng base-url/model của lượt trước. Runner ép ``--config`` vào file riêng
  mỗi run nên không có state nào sống sót qua run.

``--thread`` để default 8 để khớp ``TranslatorConfig.concurrent`` của PDFTranslator:
so latency mà hai hệ thống khác mức song song thì cột sec/page ở §4.5 chỉ đang đo
chênh lệch concurrency. Chất lượng dịch không phụ thuộc tham số này.

Token không quan sát được: pdf2zh v1 không đếm token ở đâu cả. Quy về LiteLLM proxy
qua ``key_alias`` (xem ``_common.base_meta``).

Ví dụ
-----
    PDFMATHTRANSLATE_BIN=~/venvs/pdfmath/bin/pdf2zh \\
    LITELLM_KEY_ALIAS=pdfmathtranslate \\
    python -m benchmark.e2e.runners.pdfmathtranslate \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi --model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from benchmark.e2e.runners import _common as C

SYSTEM = "pdfmathtranslate"

# Mã ngôn ngữ đi vào prompt và vào download_remote_fonts(). "vi" không có trong
# noto_list của nó nên rơi về GoNotoKurrent-Regular.ttf — font này CÓ đủ dấu tiếng
# Việt, nên đây là fallback lành, không phải lỗi thiếu glyph.
LANG_OUT = {"vi": "vi", "zh": "zh"}
LANG_IN = "en"

_MODULE_EXPR = "from pdf2zh.pdf2zh import main; raise SystemExit(main())"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="Artifact root; ghi <out>/%s/<lang>/<doc_id>/." % SYSTEM)
    p.add_argument("--tiers", default="T1,T2,T3")
    p.add_argument("--langs", default="vi")
    p.add_argument("--bin", default=os.environ.get("PDFMATHTRANSLATE_BIN"),
                   help="Console script 'pdf2zh' của venv PDFMathTranslate, hoặc "
                        "python của venv đó. Default: $PDFMATHTRANSLATE_BIN, rồi PATH. "
                        "CẢNH BÁO: 'pdf2zh' trong PATH có thể là của PDFTranslator — "
                        "trỏ tường minh vào venv baseline.")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL"),
                   help="Model id; đi vào --service openai:<model>. Phải TRÙNG model "
                        "của PDFTranslator và BabelDOC.")
    p.add_argument("--base-url", default=os.environ.get("LITELLM_BASE_URL"),
                   help="Vào child qua OPENAI_BASE_URL.")
    p.add_argument("--api-key", default=None, help="Default: $LITELLM_API_KEY.")
    p.add_argument("--thread", type=int, default=8,
                   help="Số thread dịch (default 8 = concurrent của PDFTranslator).")
    p.add_argument("--timeout-s", type=int, default=3600)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def build_cmd(base: list[str], pdf: Path, raw_dir: Path, lang: str,
              args: argparse.Namespace, cfg_path: Path) -> list[str]:
    return base + [
        str(pdf),
        "--lang-in", LANG_IN,
        "--lang-out", LANG_OUT[lang],
        "--service", f"openai:{args.model}",
        "--output", str(raw_dir),
        "--thread", str(args.thread),
        "--config", str(cfg_path),
        "--ignore-cache",
    ]


def main() -> int:
    args = parse_args()

    # KHÔNG rơi về PATH: env của PDFTranslator có sẵn console script "pdf2zh" của
    # chính nó (trùng tên package), nên fallback là lặng lẽ benchmark nhầm hệ thống.
    base = C.resolve_bin(args.bin, "pdf2zh", _MODULE_EXPR, "PDFMATHTRANSLATE_BIN",
                         allow_path_fallback=False)
    if base is None:
        return 1

    if args.corpus is None or args.out is None:
        print(f"[{SYSTEM}] cần --corpus và --out")
        return 1

    api_key = args.api_key or os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        print(f"[{SYSTEM}] thiếu API key: đặt LITELLM_API_KEY hoặc --api-key")
        return 1
    if not args.model:
        print(f"[{SYSTEM}] --model là bắt buộc — để mặc định thì nó dùng "
              f"gpt-4o-mini và bảng so sánh vô nghĩa.")
        return 1

    args.corpus, args.out = C.absolutize(args.corpus, args.out)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    unknown = [x for x in langs if x not in LANG_OUT]
    if unknown:
        print(f"[{SYSTEM}] --langs lạ {unknown}; đã biết: {sorted(LANG_OUT)}")
        return 1

    jobs = C.discover(args.corpus, tiers, SYSTEM)
    if not jobs:
        print(f"[{SYSTEM}] không thấy PDF nào dưới {args.corpus} cho tiers {tiers}")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="pdfmath-bench-"))
    cfg_path = workdir / "pdf2zh_config.json"
    env = C.child_env({
        "OPENAI_BASE_URL": args.base_url,
        "OPENAI_API_KEY": api_key,
        "OPENAI_MODEL": args.model,
    })

    version_text, _ = C.probe_version(base, env, workdir)
    print(f"[{SYSTEM}] bin: {' '.join(base)}  |  version: {version_text}", flush=True)
    print(f"[{SYSTEM}] {len(jobs)} PDF x {len(langs)} lang; model={args.model} "
          f"thread={args.thread} (cache tắt)", flush=True)

    done = skipped = failed = 0
    for tier, pdf in jobs:
        for lang in langs:
            dest = args.out / SYSTEM / lang / pdf.stem
            out_pdf, meta_path = dest / "output.pdf", dest / "meta.json"
            if not args.no_resume and out_pdf.exists() and meta_path.exists():
                skipped += 1
                continue

            raw_dir = dest / "raw"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)

            meta = C.base_meta(SYSTEM, tier, lang, pdf, args.model, args.base_url)
            meta["thread"] = args.thread
            meta["ignore_cache"] = True
            meta["system_version"] = version_text

            log_path = dest / "run.log"
            cmd = build_cmd(base, pdf, raw_dir, lang, args, cfg_path)
            rc, secs, err = C.run_child(cmd, log_path, args.timeout_s, env, workdir)
            meta["wall_seconds"] = secs

            produced = C.pick_mono(raw_dir)
            if err:
                meta["error"] = err
            elif produced is None:
                meta["error"] = f"không đẻ ra PDF nào (rc={rc}); xem run.log"
            else:
                shutil.copy2(produced, out_pdf)
                meta["raw_pdf"] = produced.name
                if rc != 0:
                    meta["error"] = f"rc={rc} nhưng vẫn có PDF — xem run.log"

            C.finalize(meta, out_pdf)
            C.write_meta(meta_path, meta)

            if meta["error"] and not out_pdf.exists():
                failed += 1
                print(f"  [fail] {lang}/{pdf.name}: {meta['error'][:140]}", flush=True)
            else:
                done += 1
                flag = "!" if meta["error"] else " "
                print(f"  [ok]{flag}  {lang}/{pdf.name:30} {secs:7.1f}s  "
                      f"{meta['n_pages_in']}->{meta['n_pages_out']}p", flush=True)

    print(f"\n[{SYSTEM}] xong. {done} dịch, {skipped} bỏ qua, {failed} lỗi.", flush=True)
    print(f"  peak RSS (children): {C.peak_rss_children_mb()} MB", flush=True)
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
