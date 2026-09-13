"""Immutable corpus/run contract and artifact gates shared by local and HF workers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from benchmark.e2e.manifest import sha256

SYSTEMS = ("pdftranslator", "babeldoc", "pdfmathtranslate", "deepl-document")


def csv(value: str) -> list[str]:
    return list(dict.fromkeys(x for x in re.split(r"[\s,]+", value.strip()) if x))


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def documents(corpus: Path, tiers: list[str]) -> dict[str, Path]:
    docs = {}
    for tier in tiers:
        paths = sorted((corpus / tier).glob("*.pdf"))
        if not paths:
            raise ValueError(f"Missing PDFs in tier {tier}")
        for pdf in paths:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", pdf.stem) or pdf.stem in docs:
                raise ValueError(f"Invalid/duplicate document ID: {pdf.stem}")
            docs[pdf.stem] = pdf
    return docs


def validate_corpus(corpus: Path, tiers: list[str]) -> dict[str, str]:
    import fitz
    from benchmark.e2e.datasets.verify_corpus import check_tier

    docs = documents(corpus, tiers)
    seen = {}
    for tier in tiers:
        errors, warnings = check_tier(tier, corpus / tier, seen)
        if errors:
            raise ValueError("; ".join(errors))
        for warning in warnings:
            print(f"  warning: {warning}")
        expected = {d for d, p in docs.items() if p.parent.name == tier}
        for filename in ("gt.json", "mapping.json"):
            data = json.loads((corpus / tier / filename).read_text())
            rows = data["docs"]
            if {r["doc_id"] for r in rows} != expected or len(rows) != len(expected):
                raise ValueError(f"{tier}/{filename}: document coverage mismatch")
            for row in rows:
                pdf = docs[row["doc_id"]]
                if row["pdf"] != pdf.name:
                    raise ValueError(f"{filename}: PDF name mismatch")
                with fitz.open(pdf) as document:
                    count = len(document)
                if [p["page"] for p in row["pages"]] != list(range(count)):
                    raise ValueError(
                        f"{filename}: page coverage mismatch for {pdf.name}"
                    )
                if filename == "mapping.json" and row.get("sha256") != sha256(pdf):
                    raise ValueError(f"mapping.json: hash mismatch for {pdf.name}")
                if filename == "gt.json":
                    for page in row["pages"]:
                        if not page.get("elements"):
                            raise ValueError(f"Empty GT page in {pdf.name}")
                        for element in page["elements"]:
                            box = element["bbox_norm"]
                            if len(box) != 4 or not (
                                0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
                            ):
                                raise ValueError(f"Invalid GT box in {pdf.name}")
    files = [p for p in docs.values()]
    files += [
        corpus / tier / name for tier in tiers for name in ("gt.json", "mapping.json")
    ]
    return {str(p.relative_to(corpus)): sha256(p) for p in sorted(files)}


def load_contract(corpus: Path) -> dict:
    contract = json.loads((corpus / "run.json").read_text())
    actual = validate_corpus(corpus, contract["tiers"])
    if actual != contract["files"]:
        raise ValueError("Corpus/GT changed since preparation; use a new run ID")
    return contract


def gate(
    corpus: Path, out: Path, contract: dict, systems=None, allow_failures=False
) -> dict:
    """Require an explicit valid success or failure for every expected matrix cell."""
    import fitz

    systems = systems or contract["systems"]
    expected = documents(corpus, contract["tiers"])
    result = {"expected": 0, "success": 0, "failed": 0}
    sig = signature(contract)
    for system in systems:
        for lang in contract["langs"]:
            base = out / system / lang
            actual = {p.parent.name for p in base.glob("*/meta.json")}
            if actual != set(expected):
                raise ValueError(
                    f"{system}/{lang}: missing or unexpected document metadata"
                )
            for doc, source in expected.items():
                meta = json.loads((base / doc / "meta.json").read_text())
                result["expected"] += 1
                if any(
                    meta.get(k) != v
                    for k, v in {
                        "system": system,
                        "lang": lang,
                        "doc_id": doc,
                        "sha256": sha256(source),
                        "run_signature": sig,
                    }.items()
                ):
                    raise ValueError(
                        f"{system}/{lang}/{doc}: metadata/corpus/run drift"
                    )
                if system in SYSTEMS[:3] and meta.get("model") != contract["model"]:
                    raise ValueError(f"{system}: model drift")
                if meta.get("error"):
                    result["failed"] += 1
                    if not allow_failures:
                        raise ValueError(
                            f"{system}/{lang}/{doc}: failed; inspect sanitized logs"
                        )
                    continue
                pdf = base / doc / "output.pdf"
                if not pdf.is_file() or meta.get("output_sha256") != sha256(pdf):
                    raise ValueError(
                        f"{system}/{lang}/{doc}: missing or changed output"
                    )
                with fitz.open(pdf) as document:
                    if not len(document):
                        raise ValueError("Empty output PDF")
                result["success"] += 1
    return result


def verify_score(out: Path, contract: dict):
    record = json.loads((out / "_run/score-status.json").read_text())
    if not record.get("completed") or record.get("run_signature") != signature(
        contract
    ):
        raise ValueError(
            "Scoring incomplete or belongs to another run; do not use old reports"
        )
    if "report/report.md" not in record.get("files", {}):
        raise ValueError("Missing report in scoring snapshot")
    for name, expected in record["files"].items():
        path = out / name
        if (
            ".." in Path(name).parts
            or Path(name).is_absolute()
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError("Scoring artifact changed or missing")
