"""Lấy mẫu PHÂN TẦNG từ split test PubTables-1M (93.834 bảng).

Vì sao không chạy full: 93.834 bảng là quá nhiều cho một số liệu chẩn đoán. Với
n=1.500, CI 95% cho một tỉ lệ chỉ khoảng ±2,5% — dư cho mọi kết luận cần rút ra.

Tầng theo hai chiều lấy trực tiếp từ GT (không phải phỏng đoán):

  * ``has_span``  — có ``table spanning cell`` hay không. Đo trên toàn split: **42,0%**
    bảng có span, nên đây là chiều quan trọng nhất.
  * ``size``      — số grid cell (row × col), chia 3 nhóm theo tercile của chính split.
    Median toàn split là 52 ô, nên nếu lấy mẫu ngẫu nhiên thuần thì bảng lớn sẽ áp đảo.

Seed cố định để lần chạy sau tái lập được. File output ghi luôn seed + phân bố tầng để
đưa vào phần phụ lục của bài.

Ví dụ
-----
    # chạy từ benchmark/parser/
    python evaluation/sample_pubtables.py \
        --ann-dir data/pubtables1m/test \
        --out     data/pubtables1m/sample_1500.json \
        --n 1500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pubtables_gt as G


def scan(ann_dir: Path) -> list[dict]:
    """Quét mọi XML, chỉ lấy metadata cần cho phân tầng (không dựng cell -> nhanh)."""
    rows = []
    files = sorted(ann_dir.glob("*.xml"))
    for i, f in enumerate(files, 1):
        if i % 10000 == 0:
            print(f"  ...{i}/{len(files)}")
        st = G.quick_stats(f)      # đếm thô, không chạy pipeline upstream
        if st is not None:
            rows.append(st)
    return rows


def stratify(rows: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    slots = sorted(r["n_slots"] for r in rows)
    q1 = slots[len(slots) // 3]
    q2 = slots[2 * len(slots) // 3]

    def size_band(v: int) -> str:
        return "small" if v <= q1 else ("medium" if v <= q2 else "large")

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        r["size_band"] = size_band(r["n_slots"])
        buckets[(r["has_span"], r["size_band"])].append(r)

    rng = random.Random(seed)
    total = len(rows)
    picked: list[dict] = []
    # Cấp phát theo tỉ lệ dân số của tầng, để mẫu giữ đúng phân bố gốc.
    for key, items in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        take = min(len(items), max(1, round(n * len(items) / total)))
        picked.extend(rng.sample(items, take))
    rng.shuffle(picked)
    picked = picked[:n]

    meta = {
        "seed": seed,
        "n_requested": n,
        "n_sampled": len(picked),
        "population": total,
        "size_tercile_cuts": {"q1_slots": q1, "q2_slots": q2},
        "population_strata": {f"{k[0]}|{k[1]}": len(v) for k, v in buckets.items()},
        "sample_strata": dict(Counter(f"{r['has_span']}|{r['size_band']}" for r in picked)),
        "sample_has_span_rate": round(
            sum(1 for r in picked if r["has_span"]) / max(len(picked), 1), 4),
        "population_has_span_rate": round(
            sum(1 for r in rows if r["has_span"]) / max(total, 1), 4),
        "sample_slots_median": statistics.median([r["n_slots"] for r in picked]) if picked else None,
        "population_slots_median": statistics.median(slots),
    }
    return picked, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[sample] quét {args.ann_dir} ...")
    rows = scan(args.ann_dir)
    print(f"[sample] {len(rows)} bảng hợp lệ")

    picked, meta = stratify(rows, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"meta": meta, "items": picked}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"\n[sample] n = {meta['n_sampled']} / {meta['population']}")
    print(f"  tercile số ô: <= {meta['size_tercile_cuts']['q1_slots']} | "
          f"<= {meta['size_tercile_cuts']['q2_slots']} | lớn hơn")
    print(f"  has_span : mẫu {meta['sample_has_span_rate']:.1%}  "
          f"vs toàn tập {meta['population_has_span_rate']:.1%}")
    print(f"  median ô : mẫu {meta['sample_slots_median']}  "
          f"vs toàn tập {meta['population_slots_median']}")
    print(f"  tầng     : {meta['sample_strata']}")
    print(f"\n[sample] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
