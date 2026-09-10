"""Aggregate exactly the document-level metrics reported in the paper.

Every document contributes one value to each mean, regardless of page count.
This benchmark is descriptive; it does not produce confidence intervals or
significance tests because the corpus contains only six documents.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

METRICS = (
    ("NT-PPR", "nt_ppr", "up"),
    ("IO-PPR", "io_ppr", "up"),
    ("OF-harm", "of_harm", "down"),
    ("IC-harm", "ic_harm", "down"),
    ("Order tau", "tau", "up"),
    ("UTB/page", "utb_per_page", "down"),
    ("COMETKiwi QE", "qe", "up"),
    ("Runner s/page", "sec_per_page", "down"),
)
SOURCE_CEILING = "source_ceiling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--langs", default="vi")
    parser.add_argument("--systems", default=None)
    parser.add_argument("--detector", default="docling")
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_doc_values(path: Path, keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Mean pages within each document; never pool pages across documents."""
    result = {key: {} for key in keys}
    for record in _read(path).get("records", []):
        if record.get("skipped"):
            continue
        for key in keys:
            values = [
                float(page[key])
                for page in record.get("pages", [])
                if "error" not in page and page.get(key) is not None
            ]
            if values:
                result[key][record["doc_id"]] = sum(values) / len(values)
    return result


def visual_values(path: Path) -> dict[str, dict[str, float]]:
    return _page_doc_values(path, ("nt_ppr", "io_ppr", "of_harm"))


def ink_values(path: Path) -> dict[str, dict[str, float]]:
    return _page_doc_values(path, ("ic_harm",))


def layout_values(path: Path) -> dict[str, dict[str, float]]:
    return _page_doc_values(path, ("tau",))


def text_values(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {
        "utb_per_page": {},
        "sec_per_page": {},
    }
    for record in _read(path).get("records", []):
        doc_id = record["doc_id"]
        utb = record.get("utb") or {}
        if utb.get("utb_per_page") is not None:
            result["utb_per_page"][doc_id] = float(utb["utb_per_page"])
        if record.get("ok") and record.get("sec_per_page") is not None:
            result["sec_per_page"][doc_id] = float(record["sec_per_page"])
    return result


def qe_values(path: Path) -> dict[str, dict[str, float]]:
    data = _read(path)
    return {
        "qe": {
            doc_id: float(summary["mean"])
            for doc_id, summary in data.get("by_doc", {}).items()
            if summary.get("mean") is not None
        }
    }


def _merge(target: dict[str, dict[str, float]], source: dict[str, dict[str, float]]) -> None:
    for metric, values in source.items():
        target.setdefault(metric, {}).update(values)


def load_all(
    out_root: Path,
    langs: list[str],
    detector: str,
    systems: list[str] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    layout_dir = out_root / "_metrics" / "layout"
    ceiling = layout_dir / f"{SOURCE_CEILING}.{detector}.json"
    if ceiling.exists():
        result[SOURCE_CEILING] = layout_values(ceiling)

    loaders = (
        ("layout", f"*.{{lang}}.{detector}.json", layout_values),
        ("visual", "*.{lang}.json", visual_values),
        ("ink", "*.{lang}.json", ink_values),
        ("text", "*.{lang}.json", text_values),
        ("qe", "*.{lang}.json", qe_values),
    )
    for lang in langs:
        for group, pattern, loader in loaders:
            directory = out_root / "_metrics" / group
            for path in sorted(directory.glob(pattern.format(lang=lang))):
                suffix = (
                    f".{lang}.{detector}.json"
                    if group == "layout"
                    else f".{lang}.json"
                )
                system = path.name[: -len(suffix)]
                if systems and system not in systems:
                    continue
                _merge(result.setdefault(f"{system}/{lang}", {}), loader(path))
    return result


def mean(values: dict[str, float]) -> dict[str, float | int | None]:
    sequence = list(values.values())
    return {
        "n_docs": len(sequence),
        "mean": round(sum(sequence) / len(sequence), 4) if sequence else None,
    }


def aggregate(data: dict[str, dict[str, dict[str, float]]]) -> dict:
    return {
        system: {key: mean(metrics.get(key, {})) for _, key, _ in METRICS}
        for system, metrics in data.items()
    }


def _format(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def write_report(destination: Path, table: dict, detector: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    systems = sorted(table)
    lines = [
        "# End-to-end benchmark results",
        "",
        f"Detector: `{detector}`. Values are observed document-macro means.",
        "No confidence intervals or significance tests are reported for the six-document corpus.",
        "",
        "| Metric | " + " | ".join(systems) + " |",
        "|---|" + "---:|" * len(systems),
    ]
    for label, key, direction in METRICS:
        arrow = "↑" if direction == "up" else "↓"
        cells = [_format(table[system][key]["mean"]) for system in systems]
        lines.append(f"| {label} {arrow} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Each metric is averaged within a document first and then across documents.",
        "Page-anchored metrics exclude reflowed documents; UTB/page and COMETKiwi remain defined.",
        "Runner time is diagnostic because runner boundaries and remote-service latency differ.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(destination: Path, table: dict) -> None:
    tables = destination / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "e2e_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("system", "metric", "n_documents", "document_macro_mean"))
        for system in sorted(table):
            for _, key, _ in METRICS:
                row = table[system][key]
                writer.writerow((system, key, row["n_docs"], row["mean"]))


def main() -> int:
    args = parse_args()
    langs = [value.strip() for value in args.langs.split(",") if value.strip()]
    systems = (
        [value.strip() for value in args.systems.split(",") if value.strip()]
        if args.systems
        else None
    )
    data = load_all(args.out, langs, args.detector, systems)
    if not data:
        print(f"no metrics found below {args.out / '_metrics'}")
        return 1

    table = aggregate(data)
    destination = args.out / "report"
    write_report(destination, table, args.detector)
    write_csv(destination, table)

    columns = sorted(table)
    print(f"{'metric':20}" + "".join(f"{system:>24}" for system in columns))
    for label, key, _ in METRICS:
        cells = "".join(
            _format(table[system][key]["mean"]).rjust(24) for system in columns
        )
        print(f"{label:20}{cells}")
    print(f"report: {destination / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
