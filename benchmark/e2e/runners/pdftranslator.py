"""PDFTranslator runner — the system under test (Pipeline B: Surya -> LLM -> Typst).

Calls ``pdf2zh.e2e`` directly. It must NOT go through the ``pdf2zh`` console script:
that is the inherited legacy Pipeline A (DocLayout-YOLO + pdfminer), and for a
scanned input it writes ``<stem>_stage_a.json`` and produces no PDF at all
(``pdf2zh/high_level.py:360-374``). Benchmarking that by accident would measure the
wrong system.

Two things worth knowing before reading the numbers this produces:

* **Phase 1 output is language-independent, so it is parsed once and cached.**
  ``<out>/pdftranslator/_parse/<doc_id>/phase1_parsed.json`` is reused for every
  target language and by the rho-sweep in §4.6. Without this, a 5-point rho sweep
  would re-run Surya five times and inflate GPU cost 5x for no information.
* **Determinism cannot be set here.** ``pdf2zh/translation/gateway.py`` hardcodes
  ``temperature`` (0.7 at line 146, 0.2 at line 246) with no config knob, and the
  harness is not allowed to patch ``pdf2zh/``. Pin temperature at the LiteLLM proxy
  instead — which is strictly fairer, since BabelDOC and PDFMathTranslate then
  receive the identical override at the same single point.

Render is invoked as ``render_document`` rather than ``e2e.run_render`` because the
former returns a stats dict (``elements_fallback``, ``elements_skipped``) that is a
free, direct render-failure signal for §4.3; ``run_render`` discards it.

Example
-------
    # chạy từ repo root
    LITELLM_BASE_URL=... LITELLM_API_KEY=... \\
    python -m benchmark.e2e.runners.pdftranslator \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi,zh --provider litellm --model gemini-3.1-flash-lite

    # chỉ nạp model để làm nóng cache volume trên HF Jobs (§8.5)
    python -m benchmark.e2e.runners.pdftranslator --warmup-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

from benchmark.e2e.manifest import now_iso

SYSTEM = "pdftranslator"

# Must match pdf2zh.e2e.SUPPORTED_LANGUAGES — Phase-2 prompts interpolate the name.
LANG_NAME = {"vi": "Vietnamese", "zh": "Simplified Chinese"}
SOURCE_LANG_NAME = "English"

# e2e.font_chain appends CJK as a fallback anyway, but leading with the right family
# keeps Typst from substituting glyph-by-glyph across two fonts on every CJK line.
LANG_FONT = {"vi": "Noto Sans", "zh": "Noto Sans CJK SC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=None,
                        help="Corpus root containing tier folders T1/ T2/ T3/.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Artifact root; writes <out>/%s/<lang>/<doc_id>/." % SYSTEM)
    parser.add_argument("--tiers", default="T1,T2,T3", help="Comma-separated tiers.")
    parser.add_argument("--langs", default="vi", help="Comma-separated target langs.")
    parser.add_argument("--provider", default="litellm",
                        help="pdf2zh translation provider (default litellm proxy).")
    parser.add_argument("--model", default=None,
                        help="Model id; defaults to the provider's own default.")
    parser.add_argument("--concurrent", type=int, default=None,
                        help="Override TranslatorConfig.concurrent (default 8).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run documents that already have artifacts.")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Load the Surya/Paddle models and exit (warms a cache).")
    return parser.parse_args()


def peak_rss_mb() -> float:
    """Process peak RSS. Monotonic across the run, so read it as a run-level ceiling
    rather than a per-document figure."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def discover(corpus: Path, tiers: list[str]) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for tier in tiers:
        tier_dir = corpus / tier
        if not tier_dir.is_dir():
            print(f"[{SYSTEM}] {tier}: not built yet ({tier_dir}) — skipped", flush=True)
            continue
        for pdf in sorted(tier_dir.glob("*.pdf")):
            jobs.append((tier, pdf))
    return jobs


def parse_cached(pdf: Path, cache_dir: Path) -> tuple[dict, float, bool]:
    """Phase 1, memoised per document. Returns (parsed_dict, seconds, was_cached)."""
    from pdf2zh.e2e import run_parse

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "phase1_parsed.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), 0.0, True

    t0 = time.perf_counter()
    parsed = run_parse(str(pdf), None, cache_dir)
    return parsed, time.perf_counter() - t0, False


def translate_and_render(pdf: Path, parsed: dict, dest: Path, lang: str,
                         cfg_kwargs: dict, inst) -> dict:
    """Phase 2 + 3 for one (document, language). Records failures, never raises."""
    from pdf2zh.e2e import build_render_config
    from pdf2zh.render import render_document
    from pdf2zh.translation import TranslatorConfig, translate_document

    dest.mkdir(parents=True, exist_ok=True)
    out_pdf = dest / "output.pdf"

    meta: dict = {"translate_s": None, "render_s": None, "render_stats": None,
                  "n_req": None, "n_retry": None, "tokens_in": None,
                  "tokens_out": None, "error": None}

    mark = inst.mark()
    t0 = time.perf_counter()
    try:
        tcfg = TranslatorConfig(
            source_language=SOURCE_LANG_NAME,
            target_language=LANG_NAME[lang],
            **cfg_kwargs,
        )
        translated = translate_document(parsed, tcfg)
        meta["translate_s"] = round(time.perf_counter() - t0, 2)
        (dest / "phase2_translated.json").write_text(
            json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — a failed doc must not kill the run
        meta["translate_s"] = round(time.perf_counter() - t0, 2)
        meta["error"] = f"translate: {type(exc).__name__}: {exc}"
    finally:
        stats = inst.since(mark)
        meta.update(n_req=stats.n_req, n_retry=stats.n_retry,
                    tokens_in=stats.tok_in, tokens_out=stats.tok_out)

    if meta["error"] is None:
        t1 = time.perf_counter()
        try:
            rcfg = build_render_config(LANG_FONT.get(lang, "Noto Sans"), None)
            meta["render_stats"] = render_document(str(pdf), translated,
                                                   str(out_pdf), rcfg)
            meta["render_s"] = round(time.perf_counter() - t1, 2)
        except Exception as exc:  # noqa: BLE001
            meta["render_s"] = round(time.perf_counter() - t1, 2)
            meta["error"] = f"render: {type(exc).__name__}: {exc}"

    return meta


def page_count(path: Path) -> int:
    import fitz  # PyMuPDF

    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    args = parse_args()

    if args.warmup_only:
        from pdf2zh.e2e import warmup

        t0 = time.perf_counter()
        warmup()
        print(f"[{SYSTEM}] models loaded in {time.perf_counter() - t0:.1f}s "
              f"(peak RSS {peak_rss_mb()} MB)", flush=True)
        return 0

    if args.corpus is None or args.out is None:
        print(f"[{SYSTEM}] --corpus and --out are required (or use --warmup-only)")
        return 1

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    unknown = [lang for lang in langs if lang not in LANG_NAME]
    if unknown:
        print(f"[{SYSTEM}] unsupported --langs {unknown}; known: {sorted(LANG_NAME)}")
        return 1

    # Resolve the API key the same way pdf2zh does, so a missing key fails here
    # rather than after the first (slow, GPU-bound) parse.
    from pdf2zh.translation.config import PROVIDERS

    provider = PROVIDERS.get(args.provider)
    if provider is None:
        print(f"[{SYSTEM}] unknown --provider {args.provider!r}; "
              f"choose from {sorted(PROVIDERS)}")
        return 1
    api_key = os.environ.get(provider["env_var"], "").strip()
    if not api_key:
        print(f"[{SYSTEM}] {provider['env_var']} is not set "
              f"(provider={args.provider}).")
        return 1

    cfg_kwargs = {"provider": args.provider, "model": args.model, "api_key": api_key}
    if args.concurrent is not None:
        cfg_kwargs["concurrent"] = args.concurrent

    jobs = discover(args.corpus, tiers)
    if not jobs:
        print(f"[{SYSTEM}] no PDFs found under {args.corpus} for tiers {tiers}")
        return 1

    from benchmark.translation.instrument import Instrument
    from pdf2zh.e2e import warmup

    print(f"[{SYSTEM}] {len(jobs)} PDFs x {len(langs)} lang(s); "
          f"provider={args.provider} model={args.model or provider['model']}", flush=True)
    t_warm = time.perf_counter()
    warmup()
    print(f"[{SYSTEM}] models loaded in {time.perf_counter() - t_warm:.1f}s", flush=True)

    done = skipped = failed = 0
    with Instrument() as inst:
        for tier, pdf in jobs:
            parse_dir = args.out / SYSTEM / "_parse" / pdf.stem
            parsed = None

            for lang in langs:
                dest = args.out / SYSTEM / lang / pdf.stem
                meta_path = dest / "meta.json"
                if not args.no_resume and (dest / "output.pdf").exists() \
                        and meta_path.exists():
                    skipped += 1
                    continue

                if parsed is None:
                    try:
                        parsed, parse_s, was_cached = parse_cached(pdf, parse_dir)
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        dest.mkdir(parents=True, exist_ok=True)
                        meta_path.write_text(json.dumps({
                            "system": SYSTEM, "tier": tier, "lang": lang,
                            "ts": now_iso(),
                            "doc_id": pdf.stem, "src": pdf.name,
                            "error": f"parse: {type(exc).__name__}: {exc}",
                        }, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"  [fail] parse {pdf.name}: {exc}", flush=True)
                        break
                else:
                    parse_s, was_cached = 0.0, True

                t0 = time.perf_counter()
                meta = translate_and_render(pdf, parsed, dest, lang, cfg_kwargs, inst)
                meta.update({
                    "system": SYSTEM, "tier": tier, "lang": lang,
                    "ts": now_iso(),   # xem benchmark/e2e/manifest.py
                    "doc_id": pdf.stem, "src": pdf.name,
                    "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    "provider": args.provider,
                    "model": args.model or provider["model"],
                    "parse_s": round(parse_s, 2),
                    "parse_cached": was_cached,
                    "wall_seconds": round(parse_s + (time.perf_counter() - t0), 2),
                    "n_pages_in": page_count(pdf),
                    "n_pages_out": page_count(dest / "output.pdf"),
                    "peak_rss_mb": peak_rss_mb(),
                    "accelerator": os.environ.get("ACCELERATOR", ""),
                })
                meta["page_inflation"] = (
                    round(meta["n_pages_out"] / meta["n_pages_in"], 4)
                    if meta["n_pages_in"] else None
                )
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

                if meta["error"]:
                    failed += 1
                    print(f"  [fail] {lang}/{pdf.name}: {meta['error'][:140]}", flush=True)
                else:
                    done += 1
                    stats = meta["render_stats"] or {}
                    fallback = stats.get("elements_fallback", 0)
                    note = f"  fallback={fallback}" if fallback else ""
                    print(f"  [ok]   {lang}/{pdf.name:30} "
                          f"parse={meta['parse_s']:6.1f}s"
                          f"{'*' if was_cached else ' '} "
                          f"tr={meta['translate_s']:6.1f}s "
                          f"rd={meta['render_s']:5.1f}s  "
                          f"{meta['n_pages_in']}->{meta['n_pages_out']}p  "
                          f"tok {meta['tokens_in']}/{meta['tokens_out']}{note}",
                          flush=True)

    print(f"\n[{SYSTEM}] done. {done} translated, {skipped} skipped, {failed} failed. "
          f"(* = parse reused from cache)", flush=True)
    print(f"  peak RSS: {peak_rss_mb()} MB", flush=True)
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
