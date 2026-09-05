"""Baseline BabelDOC (funstory-ai/BabelDOC) — chạy qua console script trong venv riêng.

BabelDOC là baseline quan trọng nhất: nó là hệ thống mà bảng so sánh phải thắng, và
paper của nó chính là nguồn của rubric 4 chiều dùng ở §4.4. Vì vậy cấu hình ở đây
giữ nguyên **default của BabelDOC**, không tắt bớt tính năng:

* ``--no-auto-extract-glossary`` KHÔNG được truyền. BabelDOC bật auto term extraction
  theo default; tắt đi thì rẻ hơn và giống PDFTranslator hơn, nhưng đó là dựng bù
  nhìn — reviewer sẽ bảo baseline bị làm yếu. Ai muốn đo riêng ảnh hưởng của nó thì
  dùng ``--no-glossary`` (một biến thể phụ, không phải lượt chính thức).
* ``--no-dual`` thì CÓ truyền: chỉ cần bản mono. Dual xen trang gốc nên số trang gấp
  đôi và mọi metric hình học ở §4.1-4.2 mất nghĩa. Đây là thay đổi định dạng đầu ra,
  không phải thay đổi chất lượng dịch.
* ``--no-watermark``: watermark của BabelDOC là mực đè lên trang, làm nhiễu
  masked-SSIM và ink-profile ở §4.2. Loại nó ra là điều kiện để so được về thị giác.

Token: BabelDOC tự log ``Total/Prompt/Completion tokens``, runner này bới lại từ
``run.log`` — baseline duy nhất cho token miễn phí. Tiền vẫn quy về LiteLLM proxy
qua ``key_alias`` (xem ``_common.base_meta``).

Ví dụ
-----
    BABELDOC_BIN=~/venvs/babeldoc/bin/babeldoc \\
    LITELLM_KEY_ALIAS=babeldoc \\
    python -m benchmark.e2e.runners.babeldoc \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi --model gemini-3.1-flash-lite

    python -m benchmark.e2e.runners.babeldoc --warmup-only   # chỉ tải asset onnx
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

from benchmark.e2e.runners import _common as C

SYSTEM = "babeldoc"

# Mã ngôn ngữ BabelDOC truyền vào prompt và dùng để chọn font.
LANG_OUT = {"vi": "vi", "zh": "zh"}
LANG_IN = "en"

_MODULE_EXPR = "from babeldoc.main import cli; cli()"

_TOK_RE = {
    "tokens_in": re.compile(r"Prompt tokens:\s*(\d+)"),
    "tokens_out": re.compile(r"Completion tokens:\s*(\d+)"),
    "tokens_total": re.compile(r"Total tokens:\s*(\d+)"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="Artifact root; ghi <out>/%s/<lang>/<doc_id>/." % SYSTEM)
    p.add_argument("--tiers", default="T1,T2,T3")
    p.add_argument("--langs", default="vi")
    p.add_argument("--bin", default=os.environ.get("BABELDOC_BIN"),
                   help="Console script 'babeldoc' hoặc python của venv BabelDOC. "
                        "Default: $BABELDOC_BIN, rồi PATH.")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL"),
                   help="Model id gửi qua --openai-model. Phải TRÙNG model của "
                        "PDFTranslator, nếu không bảng so sánh vô nghĩa.")
    p.add_argument("--base-url", default=os.environ.get("LITELLM_BASE_URL"),
                   help="Endpoint OpenAI-compatible (LiteLLM proxy).")
    p.add_argument("--api-key", default=None,
                   help="Default: $LITELLM_API_KEY.")
    p.add_argument("--qps", type=int, default=8,
                   help="QPS của BabelDOC. Default 8 (BabelDOC để 4) vì "
                        "--pool-max-workers của nó rơi về đúng giá trị QPS, nên đây "
                        "là núm duy nhất đặt mức song song — để 8 cho khớp "
                        "concurrent=8 của PDFTranslator và --thread 8 của "
                        "PDFMathTranslate. Không ảnh hưởng chất lượng dịch.")
    p.add_argument("--no-glossary", action="store_true",
                   help="Truyền --no-auto-extract-glossary. Biến thể phụ; lượt "
                        "chính thức KHÔNG dùng (xem docstring).")
    p.add_argument("--timeout-s", type=int, default=3600,
                   help="Trần thời gian mỗi document.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--warmup-only", action="store_true",
                   help="Chỉ chạy `babeldoc --warmup` để tải asset rồi thoát.")
    return p.parse_args()


def build_cmd(base: list[str], pdf: Path, raw_dir: Path, lang: str,
              args: argparse.Namespace, api_key: str) -> list[str]:
    cmd = base + [
        "--files", str(pdf),
        "--lang-in", LANG_IN,
        "--lang-out", LANG_OUT[lang],
        "--output", str(raw_dir),
        "--no-dual",
        "--no-watermark",
        "--openai",
        "--openai-model", args.model,
        "--openai-api-key", api_key,
    ]
    if args.base_url:
        cmd += ["--openai-base-url", args.base_url]
    if args.qps is not None:
        cmd += ["--qps", str(args.qps), "--pool-max-workers", str(args.qps)]
    if args.no_glossary:
        cmd += ["--no-auto-extract-glossary"]
    return cmd


def scrape_tokens(log_path: Path) -> dict:
    """Bới token từ log. Lấy match CUỐI: BabelDOC in tổng ở cuối, các dòng trước là
    tiến độ từng phần."""
    out: dict = {"tokens_in": None, "tokens_out": None, "tokens_total": None}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for key, rx in _TOK_RE.items():
        hits = rx.findall(text)
        if hits:
            out[key] = int(hits[-1])
    return out


def main() -> int:
    args = parse_args()

    base = C.resolve_bin(args.bin, "babeldoc", _MODULE_EXPR, "BABELDOC_BIN")
    if base is None:
        return 1

    env = C.child_env()
    workdir = Path(tempfile.mkdtemp(prefix="babeldoc-bench-"))

    # Env của PDFTranslator cài sẵn babeldoc <0.3 (nó là dependency). Baseline là
    # checkout 0.6.x. Chạy nhầm bản thì không có lỗi nào — chỉ là cột BabelDOC trong
    # bảng thuộc về phần mềm khác. Chặn ở đây, ghi phiên bản vào meta.
    version_text, version = C.probe_version(base, env, workdir)
    if version is not None and version < (0, 6, 0):
        print(f"[{SYSTEM}] !! bin này là babeldoc {version_text!r} — quá cũ.\n"
              f"   Gần như chắc chắn đang trỏ vào bản babeldoc mà PDFTranslator cài "
              f"làm dependency, không phải checkout baseline.\n"
              f"   Đặt BABELDOC_BIN=<venv baseline>/bin/babeldoc.")
        return 1
    print(f"[{SYSTEM}] bin: {' '.join(base)}  |  version: {version_text}", flush=True)

    if args.warmup_only:
        rc, secs, err = C.run_child(base + ["--warmup"],
                                    workdir / "warmup.log", args.timeout_s,
                                    env, workdir)
        print(f"[{SYSTEM}] warmup rc={rc} trong {secs}s"
              + (f" — {err}" if err else ""), flush=True)
        print(f"  log: {workdir / 'warmup.log'}", flush=True)
        return 0 if rc == 0 else 1

    if args.corpus is None or args.out is None:
        print(f"[{SYSTEM}] cần --corpus và --out (hoặc --warmup-only)")
        return 1

    api_key = args.api_key or os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        print(f"[{SYSTEM}] thiếu API key: đặt LITELLM_API_KEY hoặc --api-key")
        return 1
    if not args.model:
        print(f"[{SYSTEM}] --model là bắt buộc — phải trùng model của PDFTranslator, "
              f"để mặc định thì BabelDOC dùng gpt-4o-mini và bảng so sánh vô nghĩa.")
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

    print(f"[{SYSTEM}] {len(jobs)} PDF x {len(langs)} lang; model={args.model} "
          f"glossary={'off' if args.no_glossary else 'on (default)'}", flush=True)

    done = skipped = failed = 0
    for tier, pdf in jobs:
        for lang in langs:
            dest = args.out / SYSTEM / lang / pdf.stem
            out_pdf, meta_path = dest / "output.pdf", dest / "meta.json"
            if not args.no_resume and out_pdf.exists() and meta_path.exists():
                skipped += 1
                continue

            # Thư mục raw phải sạch trước mỗi lần chạy: pick_mono quét cả cây, sót
            # PDF của lượt trước là chọn nhầm và không hề báo lỗi.
            raw_dir = dest / "raw"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)

            meta = C.base_meta(SYSTEM, tier, lang, pdf, args.model, args.base_url)
            meta["auto_glossary"] = not args.no_glossary
            meta["qps"] = args.qps
            meta["system_version"] = version_text

            cmd = build_cmd(base, pdf, raw_dir, lang, args, api_key)
            log_path = dest / "run.log"
            rc, secs, err = C.run_child(cmd, log_path, args.timeout_s, env, workdir)
            meta["wall_seconds"] = secs
            meta.update(scrape_tokens(log_path))

            produced = C.pick_mono(raw_dir)
            if err:
                meta["error"] = err
            elif produced is None:
                meta["error"] = f"không đẻ ra PDF nào (rc={rc}); xem run.log"
            else:
                shutil.copy2(produced, out_pdf)
                meta["raw_pdf"] = produced.name
                if rc != 0:
                    # Có PDF nhưng rc khác 0: giữ lại và đánh dấu, đừng bỏ đi. Đa số
                    # là lỗi ở bước dọn dẹp sau khi đã ghi file xong.
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
                      f"{meta['n_pages_in']}->{meta['n_pages_out']}p  "
                      f"tok {meta['tokens_in']}/{meta['tokens_out']}", flush=True)

    print(f"\n[{SYSTEM}] xong. {done} dịch, {skipped} bỏ qua, {failed} lỗi.", flush=True)
    print(f"  peak RSS (children): {C.peak_rss_children_mb()} MB", flush=True)
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
