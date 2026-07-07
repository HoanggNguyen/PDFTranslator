"""Download the OmniDocBench dataset (page images + GT annotations) from HuggingFace.

The dataset repo ``opendatalab/OmniDocBench`` is public, so no token is needed.
It contains:
  - ``images/``          — all benchmark page images (one image per page)
  - ``OmniDocBench.json`` — ground-truth annotations (used later for metrics)

We fetch only those two (skipping the README figures at the repo root).
``snapshot_download`` is resumable: re-running continues an interrupted download.

Run this on the LOGIN NODE (it has internet; compute nodes may not) and point
``--out`` at /media/lhbac32 so the data is staged on shared storage.

Example
-------
    python download_dataset.py --out /media/lhbac32/OmniDocBench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "opendatalab/OmniDocBench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Destination folder, e.g. /media/lhbac32/OmniDocBench",
    )
    parser.add_argument(
        "--images-only", action="store_true",
        help="Download only images/ (skip OmniDocBench.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it first:\n"
            "    pip install --user huggingface_hub\n"
            "or, with the alternative CLI:\n"
            f"    hf download {REPO_ID} --repo-type dataset "
            f"--local-dir {args.out} --include 'images/**' 'OmniDocBench.json'",
            file=sys.stderr,
        )
        return 1

    allow_patterns = ["images/**"]
    if not args.images_only:
        allow_patterns.append("OmniDocBench.json")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo={REPO_ID}  ->  {args.out}", flush=True)
    print(f"[download] patterns={allow_patterns}", flush=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.out),
        allow_patterns=allow_patterns,
        resume_download=True,
    )

    images_dir = args.out / "images"
    gt_json = args.out / "OmniDocBench.json"
    num_images = sum(1 for _ in images_dir.glob("*")) if images_dir.is_dir() else 0

    print("\n[download] done.", flush=True)
    print(f"  images dir : {images_dir}  ({num_images} files)", flush=True)
    if gt_json.exists():
        print(f"  GT json    : {gt_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
