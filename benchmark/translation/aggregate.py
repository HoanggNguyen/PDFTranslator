"""Gộp latency + COMET -> bảng markdown cho báo cáo (§10).

  python -m benchmark.translation.aggregate --out-dir benchmark/translation/out
Đọc: latency.jsonl, comet_scores.json
Ghi:  benchmark/translation/out/report.md
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_latency(path: Path):
    """Return per-system aggregate over latency-measure records."""
    recs = defaultdict(list)
    if not path.exists():
        return {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") == "latency-measure":
            recs[r["system"]].append(r)
    agg = {}
    for sys_, rs in recs.items():
        walls = [r["wall_s"] for r in rs]
        total_w = sum(walls)
        tok_out = sum(r.get("tok_out", 0) or 0 for r in rs)
        src_words = sum(r.get("src_words", 0) or 0 for r in rs)
        # s / 1000 source words — chuẩn hoá latency theo độ dài, so được doc dài/ngắn.
        s_per_kword = [1000 * r["wall_s"] / r["src_words"]
                       for r in rs if r.get("src_words")]
        # Tỉ lệ giãn ký tự dst/src (ràng buộc ±15% là theo ký tự) — theo từng doc.
        char_ratio = [r["dst_chars"] / r["src_chars"]
                      for r in rs if r.get("src_chars") and r.get("dst_chars")]
        agg[sys_] = {
            "s_per_doc_median": round(stats.median(walls), 2),
            "s_per_doc_p25": round(_pct(walls, 25), 2),
            "s_per_doc_p95": round(_pct(walls, 95), 2),
            "src_words_per_s": round(src_words / total_w, 2) if total_w else 0,
            "s_per_kword_median": round(stats.median(s_per_kword), 2) if s_per_kword else 0,
            "char_ratio_median": round(stats.median(char_ratio), 3) if char_ratio else 0,
            "out_tok_per_s": round(tok_out / total_w, 1) if total_w else 0,
            "retries": sum(r.get("n_retry", 0) or 0 for r in rs),
            "n": len(rs),
        }
    return agg


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="benchmark/translation/out")
    args = ap.parse_args()
    d = Path(args.out_dir)

    comet = json.loads((d / "comet_scores.json").read_text()) if (d / "comet_scores.json").exists() else {}
    lat = load_latency(d / "latency.jsonl")

    md = ["# Kết quả đánh giá — bảng tổng hợp\n"]

    # (A) per-language COMET × system
    langs = sorted({lang for rel in comet.values() for lang in rel.get("by_language", {})})
    if langs:
        systems = list(comet)
        md.append("## (A) COMET-DA theo ngôn ngữ × hệ\n")
        headers = ["Target"] + systems
        rows = []
        for lang in langs:
            row = [lang]
            for s in systems:
                m = comet[s]["by_language"].get(lang, {}).get("comet", {})
                row.append(m.get("mean", "—"))
            rows.append(row)
        md.append(md_table(headers, rows) + "\n")

    # (B) system × quality × latency × cost
    md.append("## (B) So sánh hệ — quality × latency\n")
    headers = ["System", "COMET-DA", "chrF++", "s/doc (median)", "p25–p95",
               "s/1k-words", "words/s", "len ratio (dst/src)", "len-viol%",
               "out tok/s", "429 retries", "fallback%"]
    rows = []
    for s, rel in comet.items():
        o = rel["overall"]
        L = lat.get(s, {})
        rows.append([
            s, o["comet"]["mean"], o["chrf"]["mean"],
            L.get("s_per_doc_median", "—"),
            f'{L.get("s_per_doc_p25","—")}–{L.get("s_per_doc_p95","—")}',
            L.get("s_per_kword_median", "—"),
            L.get("src_words_per_s", "—"),
            L.get("char_ratio_median", "—"),
            o.get("len_viol_pct", "—"),
            L.get("out_tok_per_s", "—"), L.get("retries", "—"), o["fallback_pct"],
        ])
    md.append(md_table(headers, rows) + "\n")

    # per-domain
    for s, rel in comet.items():
        md.append(f"### per-domain — {s}\n")
        rows = [[dom, m["comet"]["mean"], m["comet"]["n"]] for dom, m in rel["by_domain"].items()]
        md.append(md_table(["domain", "COMET-DA", "n"], rows) + "\n")

    report = "\n".join(md)
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {d/'report.md'}")


if __name__ == "__main__":
    main()
