"""DeepL Document Translation runner — the commercial baseline from the BabelDOC paper.

This uses the ``/v2/document`` endpoint (upload PDF -> poll -> download PDF), which
is a different product from the text endpoint that ``pdf2zh/translator.py`` wraps in
the legacy Pipeline A. DeepL is a black box: we cannot pin its model, so it is
excluded from the pseudo-translator ablation (§4.6) and flagged in every table.

Money discipline, because DeepL is the one line item that can actually hurt (§7.4b):

* **Every PDF costs at least 50,000 characters.** DeepL bills a per-file floor for
  .pdf/.docx/.pptx/.xlsx. ``--dry-run`` prints the billed total *before* you spend
  anything, and the ledger shows the floor separately from real characters.
* **``--char-budget`` is checked against the live ``/v2/usage`` before every call**
  and the run stops rather than silently blowing through the free Developer tier.
* **Resumable.** An existing output.pdf + meta.json is skipped, so a timed-out job
  never re-pays for work already done.

Example
-------
    # xem trước tốn bao nhiêu ký tự, không gọi API
    python -m benchmark.e2e.runners.deepl_doc --corpus benchmark/e2e/datasets/corpus \\
        --out benchmark/e2e/out --tiers T1 --langs vi --dry-run

    # chạy thật (DEEPL_AUTH_KEY trong env)
    python -m benchmark.e2e.runners.deepl_doc --corpus benchmark/e2e/datasets/corpus \\
        --out benchmark/e2e/out --tiers T1,T2,T3 --langs vi --char-budget 950000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from benchmark.e2e.manifest import now_iso

SYSTEM = "deepl-document"

# DeepL target-language codes. Vietnamese was added in June 2025; validated against
# get_target_languages() on first use rather than trusted blindly.
TARGET_LANG = {"vi": "VI", "zh": "ZH"}
SOURCE_LANG = "EN"

DEEPL_CHAR_FLOOR = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Corpus root containing tier folders T1/ T2/ T3/.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Artifact root; this runner writes <out>/%s/<lang>/..." % SYSTEM)
    parser.add_argument("--tiers", default="T1,T2,T3", help="Comma-separated tiers.")
    parser.add_argument("--langs", default="vi", help="Comma-separated target langs.")
    parser.add_argument("--char-budget", type=int, default=None,
                        help="Stop before a call that would exceed this lifetime "
                             "character count (from /v2/usage). Developer tier = 1000000.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the billed-character forecast and exit. No API calls.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-translate documents that already have artifacts.")
    return parser.parse_args()


def pdf_stats(path: Path) -> tuple[int, int]:
    """(n_pages, n_chars) — n_chars is what DeepL bills on for a text-bearing PDF."""
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        texts = [page.get_text("text").strip() for page in doc]
    return len(texts), sum(len(t) for t in texts)


def discover(corpus: Path, tiers: list[str]) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for tier in tiers:
        tier_dir = corpus / tier
        if not tier_dir.is_dir():
            print(f"[deepl] {tier}: not built yet ({tier_dir}) — skipped", flush=True)
            continue
        for pdf in sorted(tier_dir.glob("*.pdf")):
            jobs.append((tier, pdf))
    return jobs


def forecast(jobs: list[tuple[str, Path]], langs: list[str]) -> int:
    """Billed characters if we ran everything now, floor included."""
    print(f"\n  {'tier':5} {'pdf':34} {'pages':>5} {'chars':>9} {'billed':>9}", flush=True)
    total = 0
    for tier, pdf in jobs:
        n_pages, n_chars = pdf_stats(pdf)
        billed = max(n_chars, DEEPL_CHAR_FLOOR)
        floor_hit = " <- floor" if billed > n_chars else ""
        print(f"  {tier:5} {pdf.name:34} {n_pages:5d} {n_chars:9,d} "
              f"{billed:9,d}{floor_hit}", flush=True)
        total += billed * len(langs)
    print(f"\n  {len(jobs)} PDFs x {len(langs)} lang(s) = "
          f"{total:,} billed characters", flush=True)
    return total


def translate_one(client, pdf: Path, dest: Path, lang: str, tier: str) -> dict:
    """One document. Never raises for a translation failure — records it instead."""
    import deepl

    dest.mkdir(parents=True, exist_ok=True)
    out_pdf = dest / "output.pdf"
    n_pages_in, n_chars_in = pdf_stats(pdf)

    before = client.get_usage()
    t0 = time.perf_counter()
    error = None
    try:
        client.translate_document_from_filepath(
            str(pdf), str(out_pdf), source_lang=SOURCE_LANG, target_lang=TARGET_LANG[lang],
        )
    except deepl.DocumentTranslationException as exc:
        # Carries document_handle — DeepL has already billed, so keep it for recovery.
        error = f"DocumentTranslationException: {exc} (handle={getattr(exc, 'document_handle', None)})"
    except deepl.DeepLException as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    after = client.get_usage()

    n_pages_out = 0
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        n_pages_out, _ = pdf_stats(out_pdf)

    return {
        "system": SYSTEM,
        "tier": tier,
        "lang": lang,
        "ts": now_iso(),          # xem benchmark/e2e/manifest.py
        "doc_id": pdf.stem,
        "src": pdf.name,
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "wall_seconds": round(wall, 2),
        "n_pages_in": n_pages_in,
        "n_pages_out": n_pages_out,
        # page_inflation feeds §4.1; DeepL reflows, so this is a real signal.
        "page_inflation": round(n_pages_out / n_pages_in, 4) if n_pages_in else None,
        "n_chars_in": n_chars_in,
        "chars_billed": after.character.count - before.character.count,
        "chars_total_after": after.character.count,
        "accelerator": os.environ.get("ACCELERATOR", ""),
        "error": error,
    }


def main() -> int:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    unknown = [lang for lang in langs if lang not in TARGET_LANG]
    if unknown:
        print(f"[deepl] unsupported --langs {unknown}; known: {sorted(TARGET_LANG)}")
        return 1

    jobs = discover(args.corpus, tiers)
    if not jobs:
        print(f"[deepl] no PDFs found under {args.corpus} for tiers {tiers}")
        return 1
    print(f"[deepl] {len(jobs)} PDFs x {len(langs)} lang(s) from {args.corpus}", flush=True)

    if args.dry_run:
        forecast(jobs, langs)
        print("\n[deepl] dry run — nothing was sent, nothing was billed.", flush=True)
        return 0

    auth_key = os.environ.get("DEEPL_AUTH_KEY", "").strip()
    if not auth_key:
        print("[deepl] DEEPL_AUTH_KEY is not set.")
        return 1

    import deepl

    client = deepl.DeepLClient(auth_key)
    supported = {lang.code.upper() for lang in client.get_target_languages()}
    missing = [f"{lang}->{TARGET_LANG[lang]}" for lang in langs
               if TARGET_LANG[lang] not in supported]
    if missing:
        print(f"[deepl] target language(s) not offered by this account: {missing}")
        return 1

    usage = client.get_usage()
    print(f"[deepl] usage before run: {usage.character.count:,}"
          + (f" / limit {usage.character.limit:,}" if usage.character.valid else "")
          + (f"  budget {args.char_budget:,}" if args.char_budget else ""), flush=True)

    done = skipped = failed = 0
    billed_total = 0
    for lang in langs:
        for tier, pdf in jobs:
            dest = args.out / SYSTEM / lang / pdf.stem
            meta_path = dest / "meta.json"
            if not args.no_resume and (dest / "output.pdf").exists() and meta_path.exists():
                skipped += 1
                continue

            if args.char_budget is not None:
                current = client.get_usage().character.count
                if current >= args.char_budget:
                    print(f"\n[deepl] STOP — budget reached: {current:,} >= "
                          f"{args.char_budget:,}. {done} done, "
                          f"{len(jobs) * len(langs) - done - skipped} remaining.", flush=True)
                    return 2

            meta = translate_one(client, pdf, dest, lang, tier)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            billed_total += meta["chars_billed"]

            if meta["error"]:
                failed += 1
                print(f"  [fail] {lang}/{pdf.name}: {meta['error'][:120]}", flush=True)
            else:
                done += 1
                # A 20-page PDF billed at exactly the floor means the merge in
                # build_doclaynet did not take — catch it on the first document.
                floor_note = "  <- BILLED AT FLOOR, check merging" \
                    if meta["chars_billed"] >= DEEPL_CHAR_FLOOR > meta["n_chars_in"] else ""
                print(f"  [ok]   {lang}/{pdf.name:34} "
                      f"{meta['n_pages_in']:3d}->{meta['n_pages_out']:3d} pages  "
                      f"{meta['wall_seconds']:6.1f}s  "
                      f"billed {meta['chars_billed']:,}{floor_note}", flush=True)

    print(f"\n[deepl] done. {done} translated, {skipped} skipped, {failed} failed.",
          flush=True)
    print(f"  characters billed this run: {billed_total:,}", flush=True)
    print(f"  usage now: {client.get_usage().character.count:,}", flush=True)
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
