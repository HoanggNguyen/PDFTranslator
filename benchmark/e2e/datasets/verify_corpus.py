"""Gate the corpus before any job runs. Fail here, not after spending money.

Each check below corresponds to a trap that was verified to be real, not
hypothetical (docs/EVALUATION_PLAN.md §7.4, §1.4):

* **DeepL's 50,000-character-per-file floor.** A PDF under the floor is billed at
  the floor. T1's source pages are single-page PDFs (~3k chars), so un-merged they
  would cost 16.7x. This check is the difference between $0 and $353.
* **A dot in the filename stem.** surya's batch loader derives a page key with
  ``os.path.basename(path).split(".")[0]``, so ``paper.v2.pdf`` becomes ``paper``
  and two files sharing a stem silently merge into one renumbered result list.
* **Text layer present (T1/T2) or absent (T3).** BabelDOC and PDFMathTranslate need
  a text layer; BabelDOC raises ``ScannedPDFError`` when >=80% of pages lack one.
  A born-digital tier that accidentally contains scans, or a scanned tier that
  accidentally contains text, invalidates the comparison rather than failing loudly.
* **sha256 matches the manifest**, so a corpus cannot drift between runs.

Exit code is 0 when only warnings fired, 1 when any error did (or any warning,
with ``--strict``).

Example
-------
    # chạy từ repo root, trước mỗi lần phát job
    python -m benchmark.e2e.datasets.verify_corpus --corpus benchmark/e2e/datasets/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEEPL_CHAR_FLOOR = 50_000

# Per-tier expectation for a text layer: True = required, False = must be absent.
TIER_WANTS_TEXT = {"T1": True, "T2": True, "T3": False}

# Fraction of pages that must satisfy the tier's text expectation.
TEXT_PAGE_THRESHOLD = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Corpus root containing tier folders T1/ T2/ T3/.")
    parser.add_argument("--tiers", default="T1,T2,T3",
                        help="Comma-separated tiers to check (default all present).")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors.")
    return parser.parse_args()


def inspect_pdf(path: Path) -> dict:
    """Open the PDF once and collect everything the checks need."""
    import fitz  # PyMuPDF

    info: dict = {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        with fitz.open(path) as doc:
            texts = [page.get_text("text").strip() for page in doc]
        info["n_pages"] = len(texts)
        info["n_chars"] = sum(len(t) for t in texts)
        info["text_pages"] = sum(1 for t in texts if t)
        info["error"] = None
    except Exception as exc:
        info.update(n_pages=0, n_chars=0, text_pages=0, error=f"{type(exc).__name__}: {exc}")
    return info


def load_manifest(tier_dir: Path) -> dict[str, str]:
    """Map pdf filename -> sha256 from mapping.json, when the builder recorded one."""
    mapping_path = tier_dir / "mapping.json"
    if not mapping_path.exists():
        return {}
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {d["pdf"]: d["sha256"] for d in mapping.get("docs", []) if d.get("sha256")}


def check_tier(tier: str, tier_dir: Path, seen_stems: dict[str, Path]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    pdfs = sorted(tier_dir.glob("*.pdf"))
    if not pdfs:
        errors.append(f"{tier}: no PDFs in {tier_dir}")
        return errors, warnings

    manifest_sha = load_manifest(tier_dir)
    wants_text = TIER_WANTS_TEXT.get(tier)

    print(f"\n[{tier}] {tier_dir}  ({len(pdfs)} PDFs)", flush=True)
    header = f"  {'pdf':36} {'pages':>5} {'chars':>9} {'text pg':>8}  checks"
    print(header, flush=True)

    for pdf in pdfs:
        info = inspect_pdf(pdf)
        flags: list[str] = []

        if info["error"]:
            errors.append(f"{tier}/{pdf.name}: cannot open — {info['error']}")
            print(f"  {pdf.name:36} {'--':>5} {'--':>9} {'--':>8}  UNREADABLE", flush=True)
            continue

        # --- stem: no dot, globally unique (surya's split(".")[0]) ---
        if "." in pdf.stem:
            errors.append(f"{tier}/{pdf.name}: stem contains '.' — surya will "
                          f"truncate the page key to {pdf.stem.split('.')[0]!r}")
            flags.append("DOT-IN-STEM")
        prior = seen_stems.get(pdf.stem)
        if prior is not None:
            errors.append(f"{tier}/{pdf.name}: stem {pdf.stem!r} already used by "
                          f"{prior} — surya would merge their results")
            flags.append("DUP-STEM")
        else:
            seen_stems[pdf.stem] = pdf

        # --- text layer matches the tier's expectation ---
        frac = info["text_pages"] / info["n_pages"] if info["n_pages"] else 0.0
        if wants_text is True and frac < TEXT_PAGE_THRESHOLD:
            errors.append(f"{tier}/{pdf.name}: only {frac:.0%} of pages have a text "
                          f"layer; {tier} must be born-digital for the baselines")
            flags.append("NO-TEXT-LAYER")
        elif wants_text is False and frac > (1.0 - TEXT_PAGE_THRESHOLD):
            errors.append(f"{tier}/{pdf.name}: {frac:.0%} of pages have a text layer; "
                          f"{tier} is the scanned tier and must not")
            flags.append("UNEXPECTED-TEXT")

        # --- DeepL per-file character floor ---
        if wants_text is True and info["n_chars"] < DEEPL_CHAR_FLOOR:
            warnings.append(f"{tier}/{pdf.name}: {info['n_chars']:,} chars < "
                            f"{DEEPL_CHAR_FLOOR:,} — DeepL bills the floor, "
                            f"{DEEPL_CHAR_FLOOR / max(info['n_chars'], 1):.1f}x overpay")
            flags.append("DEEPL-FLOOR")

        # --- sha256 vs manifest ---
        expected = manifest_sha.get(pdf.name)
        if expected and expected != info["sha256"]:
            errors.append(f"{tier}/{pdf.name}: sha256 differs from mapping.json — "
                          f"corpus drifted, rebuild it")
            flags.append("SHA-MISMATCH")

        status = " ".join(flags) if flags else "ok"
        print(f"  {pdf.name:36} {info['n_pages']:5d} {info['n_chars']:9,d} "
              f"{info['text_pages']:8d}  {status}", flush=True)

    return errors, warnings


def main() -> int:
    args = parse_args()
    if not args.corpus.is_dir():
        print(f"[verify] --corpus is not a folder: {args.corpus}")
        return 1

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    all_errors: list[str] = []
    all_warnings: list[str] = []
    seen_stems: dict[str, Path] = {}
    checked = 0

    for tier in tiers:
        tier_dir = args.corpus / tier
        if not tier_dir.is_dir():
            print(f"[verify] {tier}: not built yet ({tier_dir}) — skipped", flush=True)
            continue
        errors, warnings = check_tier(tier, tier_dir, seen_stems)
        all_errors += errors
        all_warnings += warnings
        checked += 1

    if checked == 0:
        print(f"[verify] no tier folders found under {args.corpus}")
        return 1

    print()
    for msg in all_warnings:
        print(f"[WARN ] {msg}", flush=True)
    for msg in all_errors:
        print(f"[ERROR] {msg}", flush=True)

    if all_errors:
        print(f"\n[verify] FAIL — {len(all_errors)} error(s), "
              f"{len(all_warnings)} warning(s). Do not run jobs.", flush=True)
        return 1
    if all_warnings and args.strict:
        print(f"\n[verify] FAIL (--strict) — {len(all_warnings)} warning(s).", flush=True)
        return 1
    print(f"\n[verify] OK — {checked} tier(s), {len(all_warnings)} warning(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
