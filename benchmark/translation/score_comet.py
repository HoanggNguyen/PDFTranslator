"""Bước B — chấm quality (local, CPU/MPS, 0 token API).

Đọc hypotheses.jsonl, tính:
  * COMET-DA (Unbabel/wmt22-comet-da, ref-based) — metric CHÍNH
  * chrF++ (sacrebleu) — lexical baseline
  * fallback% / empty% / length-violation% — chỉ số độ tin cậy hệ thống

Aggregate theo system × (language, domain). WMT24++ là segment-level (~32 từ) nên
KHÔNG cần chunk — feed thẳng, giới hạn 512-token không cắn (xem §4).

CHẠY ĐỘC LẬP: file này KHÔNG phụ thuộc pdf2zh — chỉ cần comet + sacrebleu + torch.
Có thể copy riêng file này + hypotheses.jsonl lên GPU box (A100...) để chấm nhanh:

  # CPU (Mac):
  python -m benchmark.translation.score_comet --hyp benchmark/translation/out/_all/hypotheses.jsonl --out comet_scores.json
  # GPU (A100, nhanh gấp bội — 110k segment ~vài phút):
  pip install unbabel-comet sacrebleu
  python score_comet.py --hyp hypotheses.jsonl --out comet_scores.json --gpus 1 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def _length_violation(translation: str, source: str, tol: float) -> bool:
    """Cùng logic với pdf2zh.translation.pipeline._length_violation (inline để
    score_comet CHẠY ĐỘC LẬP trên GPU box — chỉ cần comet + sacrebleu + torch,
    không cần cài cả pdf2zh)."""
    if len(source) < 20:
        return False
    return abs(len(translation) - len(source)) / max(len(source), 1) > tol


def load_hyp(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def comet_scores(rows: list[dict], model_name: str, batch_size: int, gpus: int,
                 num_workers: int = 2) -> list[float]:
    from comet import download_model, load_from_checkpoint
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows]
    # num_workers > 0 explicitly: on CPU (gpus=0) COMET derives num_workers=2*gpus=0 but
    # still passes multiprocessing_context, which newer torch rejects.
    try:
        out = model.predict(data, batch_size=batch_size, gpus=gpus, num_workers=num_workers)
    except TypeError:  # older comet without the num_workers kwarg
        out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return list(out["scores"] if isinstance(out, dict) else out.scores)


def chrf_scores(rows: list[dict]) -> list[float]:
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)  # chrF++
    return [chrf.sentence_score(r["hypothesis"], [r["reference"]]).score for r in rows]


def _agg(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(vals), "mean": round(stats.mean(vals), 4),
            "median": round(stats.median(vals), 4)}


def summarize(rows: list[dict]) -> dict:
    """Group by system, then language(pair) and domain."""
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)

    report: dict = {}
    for system, rs in by_sys.items():
        comet = [r["_comet"] for r in rs]
        chrf = [r["_chrf"] for r in rs]
        n = len(rs)
        rel = {
            "overall": {"comet": _agg(comet), "chrf": _agg(chrf),
                        "fallback_pct": round(100 * sum(r["is_fallback"] for r in rs) / n, 2),
                        "empty_pct": round(100 * sum(r["is_empty"] for r in rs) / n, 2),
                        "len_viol_pct": round(100 * sum(
                            _length_violation(r["hypothesis"], r["source"], 0.15) for r in rs) / n, 2)},
            "by_language": {},
            "by_domain": {},
        }
        for key, field in (("by_language", "pair"), ("by_domain", "domain")):
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in rs:
                groups[r[field]].append(r)
            for g, gr in sorted(groups.items()):
                rel[key][g] = {
                    "comet": _agg([x["_comet"] for x in gr]),
                    "chrf": _agg([x["_chrf"] for x in gr]),
                    "fallback_pct": round(100 * sum(x["is_fallback"] for x in gr) / len(gr), 2),
                }
        report[system] = rel
    return report


def print_tables(report: dict) -> None:
    for system, rel in report.items():
        o = rel["overall"]
        print(f"\n=== {system} ===")
        print(f"  COMET-DA mean {o['comet']['mean']}  | chrF++ mean {o['chrf']['mean']}  "
              f"| fallback {o['fallback_pct']}%  empty {o['empty_pct']}%  len-viol {o['len_viol_pct']}%  "
              f"(n={o['comet']['n']})")
        if len(rel["by_language"]) > 1:
            print("  per-language (COMET-DA mean):")
            for lang, m in rel["by_language"].items():
                print(f"    {lang:<12} {m['comet']['mean']}   (chrF {m['chrf']['mean']}, "
                      f"fallback {m['fallback_pct']}%, n={m['comet']['n']})")
        print("  per-domain (COMET-DA mean):")
        for dom, m in rel["by_domain"].items():
            print(f"    {dom:<10} {m['comet']['mean']}   (n={m['comet']['n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", default="benchmark/translation/out/hypotheses.jsonl")
    ap.add_argument("--out", default="benchmark/translation/out/comet_scores.json")
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=0, help="0 = CPU (an toàn trên Mac); >0 nếu có CUDA")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (>0 để né bug torch mới)")
    ap.add_argument("--no-comet", action="store_true", help="Chỉ chrF (bỏ qua tải model COMET)")
    args = ap.parse_args()

    rows = load_hyp(Path(args.hyp))
    print(f"Loaded {len(rows)} segments from {args.hyp}")

    chrf = chrf_scores(rows)
    for r, c in zip(rows, chrf):
        r["_chrf"] = c
    if args.no_comet:
        for r in rows:
            r["_comet"] = float("nan")
    else:
        print(f"Scoring COMET ({args.model}, gpus={args.gpus}) — có thể mất vài phút...")
        cs = comet_scores(rows, args.model, args.batch_size, args.gpus, args.num_workers)
        for r, c in zip(rows, cs):
            r["_comet"] = c

    report = summarize(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_tables(report)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
