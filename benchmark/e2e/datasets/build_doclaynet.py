"""Build the T1 corpus: born-digital pages from DocLayNet-v1.2 + human layout GT.

T1 is the primary layout-preservation tier. DocLayNet gives us something the
BabelDOC paper did not have: **layout ground truth drawn by humans**, so the
source-side boxes carry no parser error. See docs/EVALUATION_PLAN.md §2/§3.

Three things here are non-obvious and were verified against the live dataset:

1. **Repo name.** ``ds4sd/DocLayNet-v1.2`` is renamed; the API errors with "The
   dataset has been renamed." Use ``docling-project/DocLayNet-v1.2``.

2. **GT lives in a SQUARE 1025x1025 COCO space, the PDF page does not.** A page
   whose PDF MediaBox is 612x792 pt is rendered to 1025x1025, i.e. the x and y
   scales differ (1.675 vs 1.294). Scaling GT boxes uniformly gives wrong boxes.
   We sidestep it by normalising **each axis by its own dimension** — bbox_norm is
   then directly comparable to a detector's boxes normalised by page width/height.

3. **Pages are merged per domain into one multi-page PDF.** DeepL bills a minimum
   of 50,000 characters *per PDF file*, so sending 120 single-page PDFs would cost
   16.7x what it should (§7.4). 20 pages/PDF clears the floor. The merge is applied
   identically to every system under test, so it costs no fairness. T1 therefore has
   no cross-page coherence by design — that is what T2 is for.

Only ``metadata``, ``bboxes``, ``category_id`` and ``pdf`` are read from the
parquet shards; skipping ``image``/``segmentation``/``pdf_cells`` drops 75% of the
bytes (2422 MB -> 754 MB worst case).

Output layout::

    <out>/
      T1_financial_reports.pdf     # 20 pages
      T1_scientific_articles.pdf
      ...                          # one per domain
      gt.json                      # layout GT, bbox_norm xyxy in [0,1]
      mapping.json                 # page -> source page provenance + char counts

Both passes read shards/row groups in parallel. Expect **~12 min** for the scan
(HfFileSystem is bandwidth-bound here; see ``scan_metadata``), so always pass
``--scan-cache`` — re-sampling with a different seed or ``--min-chars`` is then
instant instead of another 12 minutes.

Example
-------
    # chạy từ repo root — 20 trang/domain x 6 domain = 120 trang
    python -m benchmark.e2e.datasets.build_doclaynet \\
        --out benchmark/e2e/datasets/corpus/T1 \\
        --scan-cache benchmark/e2e/datasets/.doclaynet_scan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

REPO_ID = "docling-project/DocLayNet-v1.2"
SPLIT = "test"
# The parquet mirror the HF datasets-server maintains, addressed as an HfFileSystem
# path with an explicit revision after '@'.
PARQUET_DIR = f"datasets/{REPO_ID}@refs/convert/parquet/default/{SPLIT}"

# Verified against the dataset card (1-based, 11 classes).
CATEGORIES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
    9: "Table", 10: "Text", 11: "Title",
}

# metadata.doc_category constants, verified against the dataset card.
# scientific_articles / manuals / patents are the three that map onto BabelDOC's
# benchmark domains; the other three widen coverage.
DOMAINS = (
    "scientific_articles",
    "manuals",
    "patents",
    "financial_reports",
    "laws_and_regulations",
    "government_tenders",
)

# DeepL's per-file minimum. A merged PDF below this is billed at the floor (§7.4).
DEEPL_CHAR_FLOOR = 50_000

# Minimum extractable characters for a page to earn a slot. Chosen from the
# measured density distribution of the test split, not guessed — see
# ``--min-chars`` and the note in ``sample()``.
MIN_CHARS_DEFAULT = 500

COLUMNS = ["bboxes", "category_id", "pdf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output folder for the merged PDFs + gt.json + mapping.json.")
    parser.add_argument("--per-domain", type=int, default=20,
                        help="Pages sampled per domain (default 20 -> 120 pages).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed (default 42). Same seed = same corpus.")
    parser.add_argument("--domains", default=",".join(DOMAINS),
                        help="Comma-separated subset of doc_category values.")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT,
                        help=f"Drop pages with fewer characters of extractable text "
                             f"(default {MIN_CHARS_DEFAULT}). DocLayNet contains many "
                             f"figure-only pages with ~100 chars, which carry no "
                             f"translation signal. Pass 0 to keep everything.")
    parser.add_argument("--over", type=int, default=None,
                        help="Extra candidates per domain to backfill pages that turn "
                             "out to be scan images (default: 25%% of --per-domain, "
                             "min 3).")
    parser.add_argument("--scan-cache", type=Path, default=None,
                        help="Cache the pass-1 scan here (~1 MB JSON). The first scan "
                             "reads pdf_cells over HTTP and takes ~15-20 min; with a "
                             "cache, re-sampling with a different seed is instant.")
    return parser.parse_args()


def shard_paths(fs) -> list[str]:
    # ls, not glob: globbing an HfFileSystem path walks the whole repo tree and
    # hangs for minutes on a dataset this size.
    paths = sorted(p for p in fs.ls(PARQUET_DIR, detail=False)
                   if p.endswith(".parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards under {PARQUET_DIR}")
    return paths


def scan_one_shard(fs, shard_idx: int, path: str) -> list[dict]:
    """Read ``metadata`` + ``pdf_cells`` for one shard and index it by row group."""
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    with fs.open(path) as handle:
        pf = pq.ParquetFile(handle)
        # Row-group boundaries: needed to map a row index to its row group.
        starts, cursor = [], 0
        for rg in range(pf.metadata.num_row_groups):
            starts.append(cursor)
            cursor += pf.metadata.row_group(rg).num_rows
        table = pf.read(columns=["metadata", "pdf_cells"])
        metas = table.column("metadata").to_pylist()
        cells = table.column("pdf_cells").to_pylist()

    out: list[dict] = []
    rg_of = 0
    for row_idx, (meta, page_cells) in enumerate(zip(metas, cells)):
        while rg_of + 1 < len(starts) and row_idx >= starts[rg_of + 1]:
            rg_of += 1
        n_chars = sum(len(cell.get("text") or "")
                      for group in (page_cells or [])
                      for cell in (group or []))
        out.append({
            "shard": shard_idx,
            "row": row_idx,
            "row_group": rg_of,
            "row_in_group": row_idx - starts[rg_of],
            "meta": meta,
            "n_chars_cells": n_chars,
        })
    print(f"  [scan] {Path(path).name}: {len(metas)} rows "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def scan_metadata(fs, paths: list[str]) -> list[dict]:
    """Read ``metadata`` + ``pdf_cells`` across all shards to drive the sampling.

    ``pdf_cells`` carries the text of every detected cell, which is the only way to
    know a page's text density *before* deciding whether to fetch its 150 KB PDF.
    That matters because DocLayNet page density is wildly skewed — a randomly drawn
    "scientific article" page is often a full-page molecule diagram with ~200
    characters, contributing almost nothing to a translation benchmark.

    Shards are read in parallel. Measured on this dataset, one shard read alone
    takes ~100-230s regardless of which columns are requested — the cost is
    HfFileSystem's many-small-range-reads overhead, not the column payload
    (``pdf_cells`` alone measured *faster* than ``metadata`` alone), so pruning
    further buys nothing. Concurrency helps but is bandwidth-bound, not
    latency-bound: with all 6 in flight each shard slows to ~700s, so the win is
    ~20 min sequential -> ~12 min parallel, not 6x. Hence ``--scan-cache``, which is
    what actually makes iteration cheap.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures = [pool.submit(scan_one_shard, fs, idx, path)
                   for idx, path in enumerate(paths)]
        per_shard = [f.result() for f in futures]   # in submission order
    return [row for shard_rows in per_shard for row in shard_rows]


def report_density(rows: list[dict], domains: list[str]) -> None:
    """Print the per-domain text-density distribution.

    Printed rather than hidden because it is the evidence behind ``--min-chars``:
    DocLayNet page density is skewed enough that the median and the 10th percentile
    of a domain can differ by an order of magnitude, and a reader of the thesis
    should be able to see why pages were excluded.
    """
    by_domain: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_domain[row["meta"]["doc_category"]].append(row["n_chars_cells"])

    print("\n  text density (chars/page, from pdf_cells)")
    print(f"  {'domain':24}{'n':>6}{'p10':>8}{'p25':>8}{'median':>8}{'p75':>8}"
          f"{'p90':>8}{'<500':>7}", flush=True)
    for domain in domains:
        values = sorted(by_domain.get(domain, []))
        if not values:
            continue
        pick = lambda frac: values[int(frac * (len(values) - 1))]  # noqa: E731
        below = 100 * sum(1 for v in values if v < MIN_CHARS_DEFAULT) / len(values)
        print(f"  {domain:24}{len(values):6d}{pick(.10):8d}{pick(.25):8d}"
              f"{pick(.50):8d}{pick(.75):8d}{pick(.90):8d}{below:6.0f}%", flush=True)
    print(flush=True)


def sample(rows: list[dict], domains: list[str], per_domain: int, seed: int,
           min_chars: int, over: int) -> list[dict]:
    """Stratified sample, preferring one page per distinct source document.

    Diversity matters more than raw count here: 20 pages from 20 different PDFs
    exercise 20 layout styles, 20 pages from one PDF exercise roughly one.

    ``min_chars`` drops pages with too little text to be informative. It is a
    deliberately *low* floor, not a bias toward prose: figure- and table-heavy
    pages are what Anchor-IoU (§4.1) is designed to measure, so we keep them —
    we only drop pages that have essentially nothing to translate.

    ``over`` extra candidates per domain are picked because ``n_chars_cells`` is
    **not** a reliable proxy for "a PDF reader can extract this text". DocLayNet
    ships some pages that are a single embedded scan image (0 fonts, 0 drawings)
    yet still carry populated ``pdf_cells``; those pass this filter but no baseline
    can translate them, which would silently break T1's born-digital premise.
    ``write_corpus`` re-checks each fetched page with a real text extraction and
    consumes the surplus to backfill.
    """
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["meta"]["doc_category"]].append(row)

    rng = random.Random(seed)
    selected: list[dict] = []
    for domain in domains:
        everything = by_domain.get(domain, [])
        if not everything:
            raise ValueError(f"doc_category {domain!r} not present in {SPLIT} split")
        pool = [r for r in everything if r["n_chars_cells"] >= min_chars]
        dropped = len(everything) - len(pool)
        if not pool:
            raise ValueError(f"{domain}: no page reaches --min-chars {min_chars} "
                             f"(max is {max(r['n_chars_cells'] for r in everything)})")
        if dropped:
            print(f"  [filter] {domain:22} {dropped}/{len(everything)} pages below "
                  f"{min_chars} chars dropped", flush=True)
        rng.shuffle(pool)

        want = per_domain + over
        picked, seen_docs, taken = [], set(), set()
        for row in pool:                                   # first pass: unique docs
            if len(picked) >= want:
                break
            name = row["meta"]["original_filename"]
            if name not in seen_docs:
                seen_docs.add(name)
                taken.add((row["shard"], row["row"]))
                picked.append(row)
        for row in pool:                                   # top up if the pool is thin
            if len(picked) >= want:
                break
            if (row["shard"], row["row"]) not in taken:
                taken.add((row["shard"], row["row"]))
                picked.append(row)

        if len(picked) < per_domain:
            print(f"  [warn] {domain}: only {len(picked)}/{per_domain} pages available",
                  flush=True)
        print(f"  [sample] {domain:22} {len(picked):3d} candidates "
              f"(want {per_domain}+{over}) from {len(seen_docs)} distinct documents",
              flush=True)
        selected.extend(picked)
    return selected


def fetch_one_group(fs, path: str, shard_idx: int, rg: int,
                    group: list[dict]) -> None:
    """Attach the payload columns to every selected row inside one row group."""
    import pyarrow.parquet as pq

    with fs.open(path) as handle:
        table = pq.ParquetFile(handle).read_row_group(rg, columns=COLUMNS)
    cols = {name: table.column(name).to_pylist() for name in COLUMNS}
    for row in group:
        i = row["row_in_group"]
        row["bboxes"] = cols["bboxes"][i]
        row["category_id"] = cols["category_id"][i]
        row["pdf"] = cols["pdf"][i]
    print(f"  [fetch] shard {shard_idx} rg {rg}: {len(group)} rows", flush=True)


def fetch_payloads(fs, paths: list[str], selected: list[dict],
                   workers: int = 8) -> None:
    """Second pass: attach ``bboxes``/``category_id``/``pdf`` to the selected rows.

    Reads whole row groups (parquet's smallest random-access unit) but only the
    columns we need, and reads them concurrently for the same reason as the scan.
    """
    from concurrent.futures import ThreadPoolExecutor

    wanted: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in selected:
        wanted[(row["shard"], row["row_group"])].append(row)

    print(f"  [fetch] {len(wanted)} row groups, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one_group, fs, paths[shard_idx], shard_idx, rg, group)
                   for (shard_idx, rg), group in sorted(wanted.items())]
        for future in futures:
            future.result()
    print(f"  [fetch] done in {time.perf_counter() - t0:.0f}s", flush=True)


def norm_elements(row: dict) -> list[dict]:
    """GT boxes -> normalised xyxy in [0,1].

    Source format is COCO ``[x, y, w, h]``, top-left origin, in the square
    1025x1025 image space. Normalising per axis absorbs the anisotropic page->image
    scale (see module docstring, point 2).
    """
    meta = row["meta"]
    cw = float(meta["coco_width"])
    ch = float(meta["coco_height"])

    elements = []
    for bbox, cat in zip(row["bboxes"], row["category_id"]):
        x, y, w, h = (float(v) for v in bbox)
        x0 = min(max(x / cw, 0.0), 1.0)
        y0 = min(max(y / ch, 0.0), 1.0)
        x1 = min(max((x + w) / cw, 0.0), 1.0)
        y1 = min(max((y + h) / ch, 0.0), 1.0)
        if x1 <= x0 or y1 <= y0:          # fully outside the page after clamping
            continue
        elements.append({
            "class": CATEGORIES.get(int(cat), f"Unknown-{cat}"),
            "category_id": int(cat),
            "bbox_norm": [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)],
        })
    return elements


def write_corpus(out: Path, selected: list[dict], domains: list[str],
                 per_domain: int) -> dict:
    """Merge each domain's pages into one PDF and emit gt.json + mapping.json.

    This is where the *authoritative* text-layer check happens: a candidate whose
    real ``get_text()`` is empty is a scan image and is dropped, backfilled from the
    surplus ``sample()`` provided. Doing it here rather than at sampling time is
    unavoidable — it needs the actual PDF bytes.
    """
    import fitz  # PyMuPDF

    out.mkdir(parents=True, exist_ok=True)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_domain[row["meta"]["doc_category"]].append(row)

    gt_docs, mapping_docs = [], []
    for domain in domains:
        rows = by_domain.get(domain, [])
        if not rows:
            continue
        doc_id = f"T1_{domain}"
        pdf_path = out / f"{doc_id}.pdf"

        merged = fitz.open()
        gt_pages, map_pages, doc_chars = [], [], 0
        no_text_dropped = 0
        try:
            for row in rows:
                if merged.page_count >= per_domain:
                    break
                with fitz.open("pdf", row["pdf"]) as src:
                    if src.page_count != 1:
                        # T1 assumes one page per record; a multi-page blob would
                        # silently desync every page index downstream.
                        print(f"  [skip] {row['meta']['original_filename']}: "
                              f"{src.page_count} pages, expected 1", flush=True)
                        continue
                    text = src[0].get_text("text")
                    if not text.strip():
                        # An embedded scan image. It passed the pdf_cells filter but
                        # no baseline can read it, so it does not belong in T1.
                        no_text_dropped += 1
                        continue
                    width, height = src[0].rect.width, src[0].rect.height
                    merged.insert_pdf(src)

                page_index = merged.page_count - 1
                n_chars = len(text.strip())
                doc_chars += n_chars
                gt_pages.append({
                    "page": page_index,
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "elements": norm_elements(row),
                })
                map_pages.append({
                    "page": page_index,
                    "original_filename": row["meta"]["original_filename"],
                    "page_no": row["meta"]["page_no"],
                    "page_hash": row["meta"]["page_hash"],
                    "doc_category": domain,
                    "n_chars": n_chars,
                })
            if merged.page_count == 0:
                print(f"  [warn] {doc_id}: no valid pages, skipped", flush=True)
                continue
            merged.save(pdf_path)
        finally:
            merged.close()

        gt_docs.append({"doc_id": doc_id, "pdf": pdf_path.name, "pages": gt_pages})
        n_source_docs = len({p["original_filename"] for p in map_pages})
        mapping_docs.append({
            "doc_id": doc_id, "pdf": pdf_path.name, "doc_category": domain,
            "n_pages": len(map_pages), "n_chars": doc_chars,
            # Disclosed because it is dataset-limited, not a sampling choice: the
            # DocLayNet test split simply contains few distinct manuals.
            "n_source_docs": n_source_docs,
            "n_dropped_no_text": no_text_dropped,
            # verify_corpus.py compares this against the file on disk so a corpus
            # cannot silently drift between runs.
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            "pages": map_pages,
        })

        n_boxes = sum(len(p["elements"]) for p in gt_pages)
        flag = "" if doc_chars >= DEEPL_CHAR_FLOOR else \
            f"  <- BILLED AT DeepL FLOOR {DEEPL_CHAR_FLOOR:,}"
        print(f"  [write] {pdf_path.name:34} {len(map_pages):3d} pages  "
              f"{n_boxes:5d} boxes  {doc_chars:7,d} chars  "
              f"{n_source_docs:2d} src docs{flag}", flush=True)
        if no_text_dropped:
            print(f"          dropped {no_text_dropped} scan-image page(s) "
                  f"(populated pdf_cells but no real text layer)", flush=True)
        if len(map_pages) < per_domain:
            print(f"          [warn] only {len(map_pages)}/{per_domain} pages — "
                  f"raise --over to backfill", flush=True)

    gt = {
        "dataset": f"{REPO_ID}/{SPLIT}",
        "bbox_format": "xyxy normalised to [0,1], per-axis (top-left origin)",
        "categories": {str(k): v for k, v in CATEGORIES.items()},
        "docs": gt_docs,
    }
    (out / "gt.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    mapping = {"tier": "T1", "dataset": f"{REPO_ID}/{SPLIT}", "docs": mapping_docs}
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return mapping


def main() -> int:
    args = parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    try:
        from huggingface_hub import HfFileSystem
    except ImportError:
        print("huggingface_hub is not installed:  pip install huggingface_hub pyarrow pymupdf")
        return 1

    fs = HfFileSystem()
    paths = shard_paths(fs)
    print(f"[build_doclaynet] {REPO_ID} split={SPLIT}  {len(paths)} parquet shards",
          flush=True)

    cache = args.scan_cache
    if cache is not None and cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[build_doclaynet] loaded {len(rows)} rows from {cache}", flush=True)
    else:
        rows = scan_metadata(fs, paths)
        print(f"[build_doclaynet] scanned {len(rows)} rows", flush=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows), encoding="utf-8")
            print(f"[build_doclaynet] scan cached to {cache}", flush=True)

    report_density(rows, domains)

    over = args.over if args.over is not None else max(3, args.per_domain // 4)
    selected = sample(rows, domains, args.per_domain, args.seed, args.min_chars, over)
    print(f"[build_doclaynet] {len(selected)} candidates for "
          f"{args.per_domain * len(domains)} slots "
          f"(seed={args.seed}, min_chars={args.min_chars}, over={over})", flush=True)

    fetch_payloads(fs, paths, selected)
    mapping = write_corpus(args.out, selected, domains, args.per_domain)

    total_pages = sum(d["n_pages"] for d in mapping["docs"])
    total_chars = sum(d["n_chars"] for d in mapping["docs"])
    print(f"\n[build_doclaynet] done. {len(mapping['docs'])} PDFs, "
          f"{total_pages} pages, {total_chars:,} chars.", flush=True)
    print(f"  corpus : {args.out}", flush=True)
    print(f"  next   : python -m benchmark.e2e.datasets.verify_corpus "
          f"--corpus {args.out.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
