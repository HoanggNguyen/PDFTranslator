"""Gộp mọi nhóm metric -> bảng + khoảng tin cậy + kiểm định ghép cặp -> report.md.

Ba việc mà một bảng trung bình trần không làm được:

1. **Khoảng tin cậy 95% bằng bootstrap.** "PDFTranslator 0.62 vs BabelDOC 0.60" nói
   được gì? Không gì cả, cho đến khi biết độ rộng của khoảng. Bootstrap lại theo
   đơn vị lấy mẫu rồi báo cáo CI.

2. **Kiểm định GHÉP CẶP, không phải so hai khoảng tin cậy.** Hai hệ chấm trên **cùng
   những trang đó**, nên phương sai giữa các trang (trang này khó hơn trang kia) bị
   triệt tiêu khi lấy hiệu. So hai CI rời nhau là vứt mất lợi thế đó và kết luận
   yếu đi rất nhiều. Ở đây bootstrap trên **hiệu từng trang**.

3. **``--unit doc`` cho tương lai.** Với T1 hiện tại, mỗi trang đến từ một tài liệu
   gốc **khác nhau** (xem ``mapping.json``: 20 trang / 20 doc), nên trang là đơn vị
   độc lập và ``--unit page`` là đúng. Nếu sau này corpus đổi sang gộp trang liền
   mạch của cùng một tài liệu thì các trang trong một tài liệu **tương quan với
   nhau** (chung template, chung font, chung độ rộng cột) — lúc đó bootstrap theo
   trang cho CI hẹp một cách giả tạo và **phải** đổi sang ``--unit doc``.

Đọc bảng: mọi con số layout/visual đọc **tương đối so với hàng ``Source ceiling``**.
Hàng ``identity`` phải trùng khít ceiling; lệch là lỗi của harness chứ không phải
phát hiện khoa học.

Ví dụ
-----
    python -m benchmark.e2e.metrics.aggregate --out benchmark/e2e/out --langs vi
    python -m benchmark.e2e.metrics.aggregate --out ... --baseline pdftranslator
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Metric đưa vào bootstrap: (nhãn, nhóm, khoá, chiều tốt hơn)
HEADLINE = [
    ("mIoU",            "layout", "iou",        "up"),
    ("Anchor-IoU",      "layout", "anchor_iou", "up"),
    ("Text containment", "layout", "contain",   "up"),
    ("Collisions/trang", "layout", "collisions", "down"),
    ("Margin violation", "layout", "margin",    "down"),
    ("Reading-order τ", "layout", "tau",        "up"),
    ("Masked-SSIM",     "visual", "ssim_masked", "up"),
    ("Ink distance",    "visual", "ink",        "down"),
    ("UTB/trang",       "text",   "utb",        "down"),
    ("Number recall",   "text",   "numbers",    "up"),
]

CEILING = "source_ceiling"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None)
    p.add_argument("--detector", default="docling")
    p.add_argument("--baseline", default="pdftranslator",
                   help="Hệ được đem so với mọi hệ còn lại trong kiểm định ghép cặp.")
    p.add_argument("--unit", choices=("page", "doc"), default="page",
                   help="Đơn vị lấy lại mẫu. Xem docstring: 'doc' là bắt buộc khi "
                        "các trang trong một tài liệu tương quan với nhau.")
    p.add_argument("--iters", type=int, default=1000, help="Số lần bootstrap.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Rút chuỗi giá trị theo từng trang, khoá (doc_id, page) để ghép cặp chính xác   #
# --------------------------------------------------------------------------- #
def series_layout(path: Path) -> dict[str, dict[tuple, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[tuple, float]] = {k: {} for k in
                                          ("iou", "anchor_iou", "contain",
                                           "collisions", "margin", "tau")}
    for rec in data["records"]:
        if rec.get("skipped"):
            continue
        for i, p in enumerate(rec["pages"]):
            key = (rec["doc_id"], p.get("page", i))
            if p["n_matched"]:
                out["iou"][key] = p["sum_iou"] / p["n_matched"]
            if p["n_matched_anchor"]:
                out["anchor_iou"][key] = p["sum_iou_anchor"] / p["n_matched_anchor"]
            if p["n_contain"]:
                out["contain"][key] = p["sum_contain"] / p["n_contain"]
            out["collisions"][key] = float(p["n_collisions"])
            if p["n_pred"]:
                out["margin"][key] = p["n_margin_out"] / p["n_pred"]
            if p["tau"] is not None:
                out["tau"][key] = p["tau"]
    return out


def series_visual(path: Path) -> dict[str, dict[tuple, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[tuple, float]] = {"ssim_masked": {}, "ink": {}}
    for rec in data["records"]:
        if rec.get("skipped"):
            continue
        for p in rec["pages"]:
            if "error" in p:
                continue
            key = (rec["doc_id"], p["page"])
            if p.get("ssim_masked") is not None:
                out["ssim_masked"][key] = p["ssim_masked"]
            if p.get("ink_mean") is not None:
                out["ink"][key] = p["ink_mean"]
    return out


def series_text(path: Path) -> dict[str, dict[tuple, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[tuple, float]] = {"utb": {}, "numbers": {}}
    for rec in data["records"]:
        utb = rec.get("utb") or {}
        for page, n in enumerate(utb.get("per_page") or []):
            out["utb"][(rec["doc_id"], page)] = float(n)
        nums = rec.get("numbers") or {}
        # per_page chỉ có khi số trang vào = số trang ra; hệ reflow không có cột này.
        for page, pair in enumerate(nums.get("per_page") or []):
            n_src, n_found = pair
            if n_src:
                out["numbers"][(rec["doc_id"], page)] = n_found / n_src
    return out


def load_all(out_root: Path, langs: list[str], detector: str,
             systems: list[str] | None) -> dict[str, dict[str, dict[tuple, float]]]:
    """(system/lang) -> khoá metric -> {(doc, page): giá trị}."""
    result: dict[str, dict[str, dict[tuple, float]]] = {}

    layout_dir = out_root / "_metrics" / "layout"
    ceiling_path = layout_dir / f"{CEILING}.{detector}.json"
    if ceiling_path.exists():
        result[CEILING] = series_layout(ceiling_path)

    for lang in langs:
        for path in sorted(layout_dir.glob(f"*.{lang}.{detector}.json")):
            system = path.name[:-len(f".{lang}.{detector}.json")]
            if systems and system not in systems:
                continue
            result.setdefault(f"{system}/{lang}", {}).update(series_layout(path))
        for path in sorted((out_root / "_metrics" / "visual").glob(f"*.{lang}.json")):
            system = path.name[:-len(f".{lang}.json")]
            if systems and system not in systems:
                continue
            result.setdefault(f"{system}/{lang}", {}).update(series_visual(path))
        for path in sorted((out_root / "_metrics" / "text").glob(f"*.{lang}.json")):
            system = path.name[:-len(f".{lang}.json")]
            if systems and system not in systems:
                continue
            result.setdefault(f"{system}/{lang}", {}).update(series_text(path))
    return result


# --------------------------------------------------------------------------- #
# Bootstrap                                                                     #
# --------------------------------------------------------------------------- #
def group_keys(keys: list[tuple], unit: str) -> list[list[tuple]]:
    """Đơn vị lấy lại mẫu: từng trang, hoặc cả tài liệu (clustered)."""
    if unit == "page":
        return [[k] for k in keys]
    by_doc: dict[str, list[tuple]] = {}
    for k in keys:
        by_doc.setdefault(k[0], []).append(k)
    return [v for _, v in sorted(by_doc.items())]


def ci(values: dict[tuple, float], unit: str, iters: int, rng: random.Random) -> dict:
    keys = sorted(values)
    if not keys:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    clusters = group_keys(keys, unit)
    point = sum(values[k] for k in keys) / len(keys)

    means = []
    for _ in range(iters):
        picked = [rng.choice(clusters) for _ in range(len(clusters))]
        flat = [values[k] for c in picked for k in c]
        if flat:
            means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int(0.025 * len(means))] if means else None
    hi = means[min(len(means) - 1, int(0.975 * len(means)))] if means else None
    return {"n": len(keys), "mean": round(point, 4),
            "lo": round(lo, 4) if lo is not None else None,
            "hi": round(hi, 4) if hi is not None else None}


def paired(a: dict[tuple, float], b: dict[tuple, float], unit: str, iters: int,
           rng: random.Random) -> dict:
    """Bootstrap trên HIỆU từng trang. Trả CI của hiệu + p hai phía."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return {"n": len(keys), "diff": None, "lo": None, "hi": None, "p": None}
    diffs = {k: a[k] - b[k] for k in keys}
    point = sum(diffs.values()) / len(keys)

    clusters = group_keys(keys, unit)
    samples = []
    for _ in range(iters):
        picked = [rng.choice(clusters) for _ in range(len(clusters))]
        flat = [diffs[k] for c in picked for k in c]
        if flat:
            samples.append(sum(flat) / len(flat))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    n_le = sum(1 for s in samples if s <= 0)
    n_ge = sum(1 for s in samples if s >= 0)
    p = 2 * min(n_le, n_ge) / len(samples)
    return {"n": len(keys), "diff": round(point, 4), "lo": round(lo, 4),
            "hi": round(hi, 4), "p": round(min(1.0, p), 4)}


# --------------------------------------------------------------------------- #
# Báo cáo                                                                       #
# --------------------------------------------------------------------------- #
def fmt(v, spec=".3f"):
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def write_report(dest: Path, table: dict, tests: dict, args: argparse.Namespace,
                 summaries: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Kết quả benchmark E2E",
        "",
        f"detector `{args.detector}` · bootstrap {args.iters} lần theo "
        f"`{args.unit}` · seed {args.seed} · hệ đối chứng `{args.baseline}`",
        "",
        "Mọi con số layout/visual đọc **tương đối so với hàng `source_ceiling`** — "
        "đó là trần thật của detector, không phải 1.0. Hàng `identity` phải trùng "
        "khít ceiling; lệch là lỗi harness.",
        "",
        "## Bảng chính (trung bình, CI 95%)",
        "",
    ]
    systems = sorted(table)
    head = "| metric | " + " | ".join(systems) + " |"
    lines += [head, "|" + "---|" * (len(systems) + 1)]
    for label, _, key, direction in HEADLINE:
        cells = []
        for s in systems:
            c = table[s].get(key)
            if not c or c["mean"] is None:
                cells.append("—")
            else:
                cells.append(f"{fmt(c['mean'])} [{fmt(c['lo'])}, {fmt(c['hi'])}]")
        arrow = "↑" if direction == "up" else "↓"
        lines.append(f"| {label} {arrow} | " + " | ".join(cells) + " |")

    lines += ["", f"## Kiểm định ghép cặp — `{args.baseline}` so với từng baseline", "",
              "Bootstrap trên **hiệu từng trang** (cùng trang, cùng bài kiểm), nên "
              "phương sai giữa các trang bị triệt tiêu. `p` là hai phía.", ""]
    for other, rows in sorted(tests.items()):
        lines += [f"### vs `{other}`", "",
                  "| metric | hiệu | CI 95% | p | n |", "|---|---|---|---|---|"]
        for label, _, key, _direction in HEADLINE:
            r = rows.get(key)
            if not r or r["diff"] is None:
                continue
            star = " **" if r["p"] is not None and r["p"] < 0.05 else " "
            lines.append(f"| {label} | {fmt(r['diff'], '+.4f')}{star.strip()} | "
                         f"[{fmt(r['lo'], '+.4f')}, {fmt(r['hi'], '+.4f')}] | "
                         f"{fmt(r['p'], '.4f')} | {r['n']} |")
        lines.append("")

    lines += ["## Tóm tắt mức tài liệu", "",
              "| hệ | doc chấm được | doc bị loại (reflow) | success rate | sec/trang |",
              "|---|---|---|---|---|"]
    for key in sorted(summaries):
        s = summaries[key]
        lines.append(f"| {key} | {s.get('n_docs_scored', '—')} | "
                     f"{s.get('n_docs_skipped', '—')} | "
                     f"{fmt(s.get('success_rate'))} | "
                     f"{fmt(s.get('sec_per_page_mean'), '.1f')} |")

    lines += ["", "---", "",
              "Doc bị loại khỏi metric layout/visual vì số trang đầu ra khác số "
              "trang nguồn — không có phép ghép trang nào đúng. Áp cho mọi hệ như "
              "nhau; với hệ reflow thì con số đó **chính là một phát hiện**.", ""]
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(dest: Path, table: dict) -> None:
    (dest / "tables").mkdir(parents=True, exist_ok=True)
    rows = ["system,metric,n,mean,ci_lo,ci_hi"]
    for system in sorted(table):
        for label, _, key, _d in HEADLINE:
            c = table[system].get(key)
            if not c:
                continue
            rows.append(f"{system},{key},{c['n']},{c['mean']},{c['lo']},{c['hi']}")
    (dest / "tables" / "headline.csv").write_text("\n".join(rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    systems = ([s.strip() for s in args.systems.split(",") if s.strip()]
               if args.systems else None)

    data = load_all(args.out, langs, args.detector, systems)
    if not data:
        print(f"!! chưa có metric nào dưới {args.out}/_metrics/ — chạy "
              f"eval_preserve / eval_visual / eval_text trước")
        return 1

    rng = random.Random(args.seed)
    table = {name: {key: ci(series.get(key, {}), args.unit, args.iters, rng)
                    for _l, _g, key, _d in HEADLINE}
             for name, series in data.items()}

    # Kiểm định ghép cặp: hệ đối chứng so với mọi hệ khác cùng ngôn ngữ.
    tests: dict[str, dict] = {}
    for lang in langs:
        base_key = f"{args.baseline}/{lang}"
        if base_key not in data:
            continue
        for other in sorted(data):
            if other in (base_key, CEILING) or not other.endswith(f"/{lang}"):
                continue
            tests[other] = {
                key: paired(data[base_key].get(key, {}), data[other].get(key, {}),
                            args.unit, args.iters, rng)
                for _l, _g, key, _d in HEADLINE}

    # Số mức tài liệu lấy thẳng từ summary của từng nhóm.
    summaries: dict[str, dict] = {}
    for group in ("text", "layout"):
        for path in (args.out / "_metrics" / group).glob("*.json"):
            try:
                s = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
            except Exception:  # noqa: BLE001
                continue
            name = path.stem.replace(f".{args.detector}", "")
            summaries.setdefault(name, {}).update(
                {k: v for k, v in s.items()
                 if k in ("n_docs_scored", "n_docs_skipped", "success_rate",
                          "sec_per_page_mean")})

    dest = args.out / "report"
    write_report(dest, table, tests, args, summaries)
    write_csv(dest, table)

    hdr = f"{'metric':20}" + "".join(f"{s:>26}" for s in sorted(table))
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, _g, key, _d in HEADLINE:
        cells = ""
        for s in sorted(table):
            c = table[s].get(key)
            cells += (f"{fmt(c['mean'])} [{fmt(c['lo'])},{fmt(c['hi'])}]".rjust(26)
                      if c and c["mean"] is not None else "—".rjust(26))
        print(f"{label:20}{cells}")
    print(f"\nbáo cáo: {dest / 'report.md'}   |   bảng: {dest / 'tables'}/")
    if len(data) < 3:
        print("!! mới có ít hơn 3 hàng — bảng chưa dùng được, chạy đủ 4 hệ đã.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
