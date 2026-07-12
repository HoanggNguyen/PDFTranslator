"""So sánh 3 cách MATCHING cho localization (class-agnostic) trên cùng dữ liệu:

  (0) HIỆN TẠI  — containment + union, hai chiều tách rời (recall-side / precision-side).
  (A) COCO 1-1  — ghép 1-1 tham lam theo IoU giảm dần, mỗi GT/pred dùng 1 lần (chuẩn
                  detection; không có confidence nên sort theo IoU).
  (B) COMPONENT — đồ thị chồng lấn (cạnh khi IoU>=.5 hoặc containment>=.5 hai chiều) →
                  thành phần liên thông; cụm "khớp" nếu IoU(union_GT, union_pred)>=t →
                  mọi GT/pred trong cụm tính TP (một phép ghép nhất quán, xử lý N-M).

Báo P/R/F1@IoU cho tổng thể + vài lát cắt. Chỉ để đối chiếu, KHÔNG thay eval_layout.

    .venv/bin/python compare_matchers.py [--iou 0.5] [--granularity fine]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_layout as E

# benchmark/parser/ — chứa data/ (GT) và parser_results/ (batch_*.json + mapping.json)
BASE = Path(__file__).resolve().parents[1]


# ---- (A) COCO 1-1 greedy ----
def coco_tp(gts, preds, t):
    pairs = []
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            v = E.iou(g["box"], p["box"])
            if v >= t:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    ug, up, tp = set(), set(), 0
    for v, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); tp += 1
    return tp, tp   # tp_gt == tp_pred (ghép 1-1)


# ---- (B) connected components ----
def comp_tp(gts, preds, t):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(len(gts)):
        find(("g", i))
    for j in range(len(preds)):
        find(("p", j))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            gb, pb = g["box"], p["box"]
            if (E.iou(gb, pb) >= 0.5 or E.contain_ratio(gb, pb) >= 0.5
                    or E.contain_ratio(pb, gb) >= 0.5):
                union(("g", i), ("p", j))
    comps = defaultdict(lambda: {"g": [], "p": []})
    for i, g in enumerate(gts):
        comps[find(("g", i))]["g"].append(g["box"])
    for j, p in enumerate(preds):
        comps[find(("p", j))]["p"].append(p["box"])
    tp_g = tp_p = 0
    for c in comps.values():
        if not c["g"] or not c["p"]:
            continue
        if E.iou(E.union_box(c["g"]), E.union_box(c["p"])) >= t:
            tp_g += len(c["g"]); tp_p += len(c["p"])
    return tp_g, tp_p


# ---- (0) current two-sided ----
def cur_tp(gts, preds, t):
    rec = E.match_side(gts, preds, 0.5)
    prc = E.match_side(preds, gts, 0.5)
    tp_g = sum(1 for r in rec if r["union_iou"] >= t)
    tp_p = sum(1 for r in prc if r["union_iou"] >= t)
    return tp_g, tp_p


def prf(tp_g, tp_p, n_gt, n_pred):
    r = tp_g / n_gt if n_gt else 0.0
    p = tp_p / n_pred if n_pred else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--granularity", default="fine")
    ap.add_argument("--gt", type=Path, default=BASE / "data" / "OmniDocBench.json")
    ap.add_argument("--mapping", type=Path,
                    default=BASE / "parser_results" / "mapping.json")
    ap.add_argument("--pred", type=Path, default=BASE / "parser_results",
                    help="Thư mục chứa batch_*.json của parser.")
    args = ap.parse_args()
    t = args.iou

    gt = E.load_gt(args.gt, args.granularity)
    mp = json.load(open(args.mapping))
    idx = E.build_pred_index(args.pred, mp)

    METHODS = {"current": cur_tp, "coco_1to1": coco_tp, "component": comp_tp}
    # acc[slice][method] = [tp_g, tp_p, n_gt, n_pred]
    acc = defaultdict(lambda: {m: [0, 0, 0, 0] for m in METHODS})

    for name, pp in idx.items():
        g = gt.get(name)
        if g is None:
            continue
        gts = E.prep_gt(g, True)
        preds = E.prep_pred(pp)
        keys = ["all"]
        lang = g["attr"].get("language"); lay = g["attr"].get("layout")
        if lang: keys.append(f"language={lang}")
        if lay: keys.append(f"layout={lay}")
        per = {}
        for m, fn in METHODS.items():
            per[m] = fn(gts, preds, t)
        for k in keys:
            for m in METHODS:
                a = acc[k][m]
                a[0] += per[m][0]; a[1] += per[m][1]
                a[2] += len(gts); a[3] += len(preds)

    def show(k):
        print(f"\n### {k}")
        print(f"  {'method':10s}  P@{t}   R@{t}   F1@{t}")
        for m in METHODS:
            tp_g, tp_p, n_gt, n_pred = acc[k][m]
            p, r, f = prf(tp_g, tp_p, n_gt, n_pred)
            print(f"  {m:10s}  {p:.3f}  {r:.3f}  {f:.3f}")

    print(f"So sánh matching @IoU={t}, granularity={args.granularity}, pred={args.pred}")
    show("all")
    for k in ["layout=single_column", "layout=double_column", "layout=three_column",
              "language=english", "language=simplified_chinese"]:
        if k in acc:
            show(k)


if __name__ == "__main__":
    main()
