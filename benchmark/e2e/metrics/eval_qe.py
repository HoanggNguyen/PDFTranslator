"""Nhóm D — chất lượng dịch ở mức PDF, **không cần bản dịch tham chiếu**.

Không dataset PDF nào có reference translation, nên xếp hạng cross-system ở mức tài
liệu phải dùng metric **quality estimation**: CometKiwi chấm cặp (nguồn, đích) mà
không cần đáp án. Cặp lấy từ ``align/extract_pairs`` — cùng một đường trích cho cả
4 hệ, kể cả DeepL.

⭐ **Hiệu chuẩn — đây mới là lập luận làm cho phần trên hợp lệ.** Dùng một metric
không tham chiếu để xếp hạng thì câu hỏi đầu tiên sẽ là "sao biết nó đúng?".
``--calibrate`` trả lời bằng số: trên **đúng cặp ngôn ngữ và domain này**, chạy cả
CometKiwi (không tham chiếu) lẫn COMET-DA (có tham chiếu) trên WMT24++ ``vi_VN``
rồi báo cáo tương quan Pearson/Spearman ở mức segment và mức system. Có con số đó
rồi thì việc dùng QE để xếp hạng ở mức PDF là **có căn cứ**, và luận văn có một mục
"metric validation" mà paper BabelDOC không có.

Về model: ``wmt23-cometkiwi-da-xl`` (XLM-R XL, 3.5B) cần **≥15 GB VRAM ⇒ không chạy
được trên T4 16 GB một cách an toàn** — dùng ``l4x1`` (24 GB), hoặc rơi về
``wmt22-cometkiwi-da`` và **ghi rõ trong luận văn là đã dùng bản nhỏ**. Cả hai model
đều **gated trên HF**: phải chấp nhận điều khoản và có ``HF_TOKEN``, nếu không thì
``download_model`` báo 401 giữa chừng job.

Ví dụ
-----
    python -m benchmark.e2e.metrics.eval_qe --out benchmark/e2e/out --langs vi
    python -m benchmark.e2e.metrics.eval_qe --out ... \\
        --calibrate benchmark/translation/out/hypotheses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QE_MODEL = "Unbabel/wmt22-cometkiwi-da"
QE_MODEL_XL = "Unbabel/wmt23-cometkiwi-da-xl"
REF_MODEL = "Unbabel/wmt22-comet-da"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None)
    p.add_argument("--model", default=QE_MODEL,
                   help=f"Model QE (default {QE_MODEL}). Bản XL: {QE_MODEL_XL} "
                        f"— cần ≥24 GB VRAM.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpus", type=int, default=1,
                   help="0 = CPU (chậm, nhưng chạy được để debug).")
    p.add_argument("--min-score", type=float, default=0.15,
                   help="Bỏ cặp có điểm align thấp hơn mức này. Cặp ghép sai làm "
                        "CometKiwi chấm nhiễu chứ không phải chấm dịch.")
    p.add_argument("--calibrate", type=Path, default=None,
                   help="hypotheses.jsonl của benchmark/translation (có src/hyp/ref). "
                        "Chạy cả QE lẫn COMET-DA rồi báo cáo tương quan.")
    p.add_argument("--limit", type=int, default=None,
                   help="Chỉ chấm N cặp đầu — để thử luồng cho nhanh.")
    return p.parse_args()


def load_model(name: str):
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise SystemExit(
            f"!! thiếu unbabel-comet: pip install unbabel-comet  ({exc})") from exc
    print(f"  nạp {name} ...", flush=True)
    return load_from_checkpoint(download_model(name))


def predict(model, data: list[dict], batch_size: int, gpus: int) -> list[float]:
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return list(out.scores if hasattr(out, "scores") else out["scores"])


def agg(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p25": None, "median": None, "p75": None}
    s = sorted(values)
    def q(f):
        return round(s[min(len(s) - 1, int(f * len(s)))], 4)
    return {"n": len(s), "mean": round(sum(s) / len(s), 4),
            "p25": q(0.25), "median": q(0.5), "p75": q(0.75)}


def correlate(a: list[float], b: list[float]) -> dict:
    """Pearson + Spearman. Không kéo scipy vào chỉ vì hai công thức này."""
    n = len(a)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None}

    def pearson(x, y):
        mx, my = sum(x) / n, sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
        dy = sum((yi - my) ** 2 for yi in y) ** 0.5
        return num / (dx * dy) if dx and dy else None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:                       # trung bình hạng cho các giá trị bằng nhau
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    p = pearson(a, b)
    s = pearson(ranks(a), ranks(b))
    return {"n": n,
            "pearson": round(p, 4) if p is not None else None,
            "spearman": round(s, 4) if s is not None else None}


def do_calibrate(path: Path, args: argparse.Namespace) -> dict:
    """Chạy QE và COMET-DA trên cùng một tập có reference, rồi so hai bên."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("source") and r.get("hypothesis") and r.get("reference")]
    if args.limit:
        rows = rows[:args.limit]
    if len(rows) < 3:
        print(f"!! {path} chỉ có {len(rows)} dòng dùng được — không hiệu chuẩn nổi")
        return {}

    print(f">>> hiệu chuẩn trên {len(rows)} segment từ {path}")
    qe = predict(load_model(args.model),
                 [{"src": r["source"], "mt": r["hypothesis"]} for r in rows],
                 args.batch_size, args.gpus)
    ref = predict(load_model(REF_MODEL),
                  [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]}
                   for r in rows],
                  args.batch_size, args.gpus)

    seg = correlate(qe, ref)
    # Mức system: gộp theo document rồi mới tương quan. Xếp hạng hệ thống là quyết
    # định ở mức hệ thống, nên tương quan ở mức đó mới là con số đúng để trích dẫn.
    by_doc: dict[str, list[tuple[float, float]]] = {}
    for r, q, f in zip(rows, qe, ref):
        by_doc.setdefault(str(r.get("document_id", "0")), []).append((q, f))
    doc_qe = [sum(x for x, _ in v) / len(v) for v in by_doc.values()]
    doc_ref = [sum(y for _, y in v) / len(v) for v in by_doc.values()]

    result = {"source": str(path), "qe_model": args.model, "ref_model": REF_MODEL,
              "segment_level": seg, "document_level": correlate(doc_qe, doc_ref),
              "qe": agg(qe), "ref": agg(ref)}
    print(f"  segment : pearson {seg['pearson']}  spearman {seg['spearman']}  (n={seg['n']})")
    d = result["document_level"]
    print(f"  document: pearson {d['pearson']}  spearman {d['spearman']}  (n={d['n']})")
    return result


def main() -> int:
    args = parse_args()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    dest = args.out / "_metrics" / "qe"
    dest.mkdir(parents=True, exist_ok=True)

    if args.calibrate is not None:
        result = do_calibrate(args.calibrate, args)
        if result:
            (dest / "calibration.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nghi: {dest / 'calibration.json'}")
        return 0 if result else 1

    pairs_dir = args.out / "_pairs"
    if not pairs_dir.is_dir():
        print(f"!! chưa có {pairs_dir} — chạy align.extract_pairs trước")
        return 1

    files = sorted(pairs_dir.glob("*.jsonl"))
    if args.systems:
        want = {s.strip() for s in args.systems.split(",") if s.strip()}
        files = [f for f in files if f.name.split(".")[0] in want]
    files = [f for f in files if f.stem.split(".")[-1] in langs]
    if not files:
        print(f"!! không thấy file cặp nào dưới {pairs_dir} cho langs {langs}")
        return 1

    model = load_model(args.model)
    rows = []
    for path in files:
        system, lang = path.stem.rsplit(".", 1)
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs = [r for r in recs if r.get("score", 1.0) >= args.min_score]
        if args.limit:
            recs = recs[:args.limit]
        if not recs:
            print(f"  [bỏ qua] {path.name}: không còn cặp nào sau lọc")
            continue

        print(f">>> {system}/{lang}: {len(recs)} cặp", flush=True)
        scores = predict(model, [{"src": r["src"], "mt": r["mt"]} for r in recs],
                         args.batch_size, args.gpus)

        by_doc: dict[str, list[float]] = {}
        for r, s in zip(recs, scores):
            by_doc.setdefault(r["doc_id"], []).append(s)

        summary = {
            "system": system, "lang": lang, "model": args.model,
            "n_pairs": len(recs),
            "align_modes": {m: sum(1 for r in recs if r.get("align_mode") == m)
                            for m in ("page", "document")},
            "segment": agg(scores),
            "by_doc": {d: agg(v) for d, v in sorted(by_doc.items())},
            # Trung bình của trung bình theo doc: mỗi tài liệu một phiếu, tài liệu
            # dài không nuốt mất tài liệu ngắn.
            "doc_macro_mean": round(
                sum(sum(v) / len(v) for v in by_doc.values()) / len(by_doc), 4),
        }
        (dest / f"{system}.{lang}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(summary)

    hdr = (f"{'system':22} {'lang':4} {'cặp':>7} {'QE mean':>9} {'QE median':>10} "
           f"{'doc-macro':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in rows:
        print(f"{s['system']:22} {s['lang']:4} {s['n_pairs']:7d} "
              f"{s['segment']['mean']:>9} {s['segment']['median']:>10} "
              f"{s['doc_macro_mean']:>10}")
    print(f"\nmodel = {args.model}   |   chi tiết: {dest}/")
    print("Chỉ trích dẫn xếp hạng QE kèm theo kết quả --calibrate; không có nó thì "
          "chưa chứng minh được QE tương quan với chất lượng thật trên cặp ngôn ngữ này.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
