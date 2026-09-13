"""Reading-order preservation for translated PDFs.

The same detector is applied to source and translated pages.  Regions are
matched one-to-one with class-group-constrained Hungarian matching, then
Kendall's tau compares their reading orders.  Documents whose page count
changes are excluded because source-page anchoring is no longer defined.

The source PDF is scored against its human annotations as a detector ceiling;
the identity system must reproduce that ceiling exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.e2e.parse.run_detectors import DOCLAYNET_GROUP, SOURCE_KEY, reading_order
from benchmark.parser.evaluation.eval_layout import iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tiers", default="T1")
    parser.add_argument("--langs", default="vi")
    parser.add_argument("--systems", default=None)
    parser.add_argument("--detector", default="docling")
    return parser.parse_args()


def hungarian(gts: list[dict], preds: list[dict]) -> list[tuple[int, int, float]]:
    """Return optimal positive-IoU matches restricted to the same class group."""
    if not gts or not preds:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"numpy and scipy are required: {exc}") from exc

    cost = np.full((len(gts), len(preds)), 2.0, dtype=float)
    for i, gt in enumerate(gts):
        for j, pred in enumerate(preds):
            if gt["group"] == pred["group"]:
                cost[i, j] = 1.0 - iou(gt["bbox_norm"], pred["bbox_norm"])

    rows, cols = linear_sum_assignment(cost)
    return [
        (int(i), int(j), 1.0 - float(cost[i, j]))
        for i, j in zip(rows, cols)
        if cost[i, j] < 1.0
    ]


def kendall_tau(a: list[int], b: list[int]) -> float | None:
    if len(a) < 2:
        return None
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            sign = (a[i] - a[j]) * (b[i] - b[j])
            concordant += sign > 0
            discordant += sign < 0
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def score_page(gt_page: dict, pred_page: dict) -> dict:
    gts, preds = gt_page["elements"], pred_page["elements"]
    matched = hungarian(gts, preds)
    tau = kendall_tau(
        [gts[i].get("reading_order", i) for i, _, _ in matched],
        [preds[j].get("reading_order", j) for _, j, _ in matched],
    )
    return {
        "page": gt_page["page"],
        "n_gt": len(gts),
        "n_pred": len(preds),
        "n_matched": len(matched),
        "tau": round(tau, 4) if tau is not None else None,
    }


def load_gt(corpus: Path, tiers: list[str]) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = {}
    for tier in tiers:
        path = corpus / tier / "gt.json"
        if not path.exists():
            print(f"  [skip] {tier}: missing gt.json", flush=True)
            continue
        for doc in json.loads(path.read_text(encoding="utf-8"))["docs"]:
            pages = []
            for page in doc["pages"]:
                elements = [
                    {
                        "class": element["class"],
                        "group": DOCLAYNET_GROUP.get(element["class"], "text"),
                        "bbox_norm": element["bbox_norm"],
                    }
                    for element in page["elements"]
                ]
                reading_order(elements)
                pages.append({"page": page["page"], "elements": elements})
            docs[doc["doc_id"]] = pages
    return docs


def load_pred(layout_dir: Path) -> dict[int, dict] | None:
    if not layout_dir.is_dir():
        return None
    pages = {}
    for path in sorted(layout_dir.glob("p*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        pages[int(record["page"])] = record
    return pages or None


def evaluate(
    system: str,
    lang: str | None,
    gt_docs: dict[str, list[dict]],
    layout_root: Path,
) -> list[dict]:
    records = []
    for doc_id, gt_pages in sorted(gt_docs.items()):
        base = (
            layout_root / system / doc_id
            if lang is None
            else layout_root / system / lang / doc_id
        )
        record = {
            "system": system,
            "lang": lang,
            "doc_id": doc_id,
            "skipped": None,
            "pages": [],
        }
        preds = load_pred(base)
        if preds is None:
            record["skipped"] = "missing detector output"
        elif len(preds) != len(gt_pages):
            record["skipped"] = (
                f"page-count mismatch: source {len(gt_pages)}, output {len(preds)}"
            )
        else:
            for gt_page in gt_pages:
                pred_page = preds.get(gt_page["page"])
                if pred_page is not None:
                    record["pages"].append(score_page(gt_page, pred_page))
        records.append(record)
    return records


def summarize(records: list[dict]) -> dict:
    scored = [record for record in records if not record["skipped"]]
    doc_taus = []
    for record in scored:
        values = [page["tau"] for page in record["pages"] if page["tau"] is not None]
        if values:
            doc_taus.append(sum(values) / len(values))
    return {
        "n_docs": len(records),
        "n_docs_scored": len(scored),
        "n_docs_with_tau": len(doc_taus),
        "n_docs_skipped": len(records) - len(scored),
        "n_pages": sum(len(record["pages"]) for record in scored),
        "reading_order_tau": (
            round(sum(doc_taus) / len(doc_taus), 4) if doc_taus else None
        ),
    }


def main() -> int:
    args = parse_args()
    tiers = [value.strip() for value in args.tiers.split(",") if value.strip()]
    langs = [value.strip() for value in args.langs.split(",") if value.strip()]
    layout_root = args.out / "_layout" / args.detector
    if not layout_root.is_dir():
        print(f"missing {layout_root}; run parse.run_detectors first")
        return 1

    gt_docs = load_gt(args.corpus, tiers)
    if not gt_docs:
        print(f"could not load ground truth from {args.corpus}")
        return 1
    systems = (
        [value.strip() for value in args.systems.split(",") if value.strip()]
        if args.systems
        else sorted(
            path.name
            for path in layout_root.iterdir()
            if path.is_dir() and path.name != SOURCE_KEY
        )
    )

    destination = args.out / "_metrics" / "layout"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []

    ceiling = evaluate(SOURCE_KEY, None, gt_docs, layout_root)
    ceiling_summary = summarize(ceiling) | {"detector": args.detector}
    (destination / f"source_ceiling.{args.detector}.json").write_text(
        json.dumps(
            {"summary": ceiling_summary, "records": ceiling},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows.append(("Source ceiling", "-", ceiling_summary))

    for system in systems:
        for lang in langs:
            if not (layout_root / system / lang).is_dir():
                continue
            records = evaluate(system, lang, gt_docs, layout_root)
            summary = summarize(records) | {"detector": args.detector}
            (destination / f"{system}.{lang}.{args.detector}.json").write_text(
                json.dumps(
                    {"summary": summary, "records": records},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rows.append((system, lang, summary))

    print(f"{'system':22} {'lang':4} {'docs':>7} {'tau':>8}")
    for system, lang, summary in rows:
        tau = summary.get("reading_order_tau")
        shown = f"{tau:.4f}" if isinstance(tau, (int, float)) else "-"
        print(
            f"{system:22} {lang:4} "
            f"{summary['n_docs_scored']:>3}/{summary['n_docs']:<3} {shown:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
