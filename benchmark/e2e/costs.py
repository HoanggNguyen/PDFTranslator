"""Import a minimal, credential-free proxy spend export; never upload raw proxy logs."""

import csv
import math
from pathlib import Path

from benchmark.e2e.security import safe_json, scan_file


def import_spend(path: Path, out: Path, contract: dict):
    scan_file(path)
    systems = [s for s in contract["systems"] if s != "deepl-document"]
    totals = {
        s: {"tokens_in": 0, "tokens_out": 0, "usd": 0.0, "requests": 0} for s in systems
    }
    seen = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {
            "run_id",
            "system",
            "request_id",
            "tokens_in",
            "tokens_out",
            "usd",
        }:
            raise ValueError("Spend CSV must have exactly the documented six columns")
        for row in reader:
            if row["run_id"] != contract["run_id"] or row["system"] not in totals:
                raise ValueError("Spend rows belong to another run/system")
            if not row["request_id"] or row["request_id"] in seen:
                raise ValueError("Missing/duplicate request ID in spend export")
            seen.add(row["request_id"])
            cost = totals[row["system"]]
            for key in ("tokens_in", "tokens_out", "usd"):
                value = float(row[key]) if key == "usd" else int(row[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid spend value")
                cost[key] += value
            cost["requests"] += 1
    if any(not v["requests"] for v in totals.values()):
        raise ValueError("Spend export must cover all OSS systems; missing is not zero")
    dest = out / "_costs"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(safe_json(totals))
    print("Imported spend totals; raw proxy logs/keys were not copied")
