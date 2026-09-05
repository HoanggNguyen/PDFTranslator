"""Hàng chuẩn Identity — copy PDF gốc làm "bản dịch". Không gọi API, không tốn gì.

Plan §3 bắt buộc mọi bảng phải có hàng này, và lý do không phải để so với ai: nó
**kiểm chính harness**. Vì đầu ra bằng đúng đầu vào, mọi metric phải ra giá trị lý
tưởng đã biết trước:

    page_inflation = 1.000      num-recall = 1.000      reflow = 0
    mIoU / Anchor-IoU / Masked-SSIM = đúng bằng hàng Source ceiling

Lệch một chữ số ở bất kỳ ô nào ⇒ bug trong harness (sai ánh xạ trang, sai chuẩn hoá
trục toạ độ, chọn nhầm file trong ``raw/``), không phải hệ nào dở. Đây là bài test
rẻ nhất mà bắt được nhiều lỗi nhất, nên chạy nó **trước** mỗi lượt chấm điểm.

UTB thì ngược lại: đầu ra vẫn nguyên tiếng Anh nên UTB/trang phải **cao**, và con số
đó đọc được như mật độ khối text đủ dài của corpus — một cross-check với số element
trong ``gt.json``.

Ví dụ
-----
    python -m benchmark.e2e.runners.identity \\
        --corpus benchmark/e2e/datasets/corpus --out benchmark/e2e/out \\
        --tiers T1 --langs vi
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from benchmark.e2e.runners import _common as C

SYSTEM = "identity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Artifact root; ghi <out>/%s/<lang>/<doc_id>/." % SYSTEM)
    p.add_argument("--tiers", default="T1,T2,T3")
    p.add_argument("--langs", default="vi",
                   help="Ghi một bản cho mỗi lang để hàng chuẩn xuất hiện ở mọi bảng.")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.corpus, args.out = C.absolutize(args.corpus, args.out)
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    jobs = C.discover(args.corpus, tiers, SYSTEM)
    if not jobs:
        print(f"[{SYSTEM}] không thấy PDF nào dưới {args.corpus} cho tiers {tiers}")
        return 1

    done = skipped = 0
    for tier, pdf in jobs:
        for lang in langs:
            dest = args.out / SYSTEM / lang / pdf.stem
            out_pdf, meta_path = dest / "output.pdf", dest / "meta.json"
            if not args.no_resume and out_pdf.exists() and meta_path.exists():
                skipped += 1
                continue

            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, out_pdf)

            meta = C.base_meta(SYSTEM, tier, lang, pdf, model=None, base_url=None)
            meta["wall_seconds"] = 0.0
            C.finalize(meta, out_pdf)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            done += 1
            print(f"  [ok]   {lang}/{pdf.name:30} "
                  f"{meta['n_pages_in']}->{meta['n_pages_out']}p", flush=True)

    print(f"\n[{SYSTEM}] xong. {done} copy, {skipped} bỏ qua.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
