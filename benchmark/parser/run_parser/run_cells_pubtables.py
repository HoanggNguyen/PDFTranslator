"""Chạy CHỈ cell detector trên crop bảng PubTables-1M.

Vì sao không dùng ``parse_pdf``: ảnh của subset Structure **đã là crop bảng**, nên layout
detection / text detection / recognition đều không cần. Gọi trực tiếp
``PaddleCellTableModule`` cô lập đúng phần cần đo. Hệ quả phải nói rõ trong bài: đây là đo
**module cell detection**, KHÔNG phải đo pipeline (bước phát hiện bảng không tham gia).

Hai chi tiết quyết định tính đúng của số đo:

1. **Crop về GT ``table`` bbox trước khi feed.** Ảnh gốc có viền trắng rất đều — đo trên
   300 ảnh: left ~36,8px, top ~36,9px, right ~38,8px, bottom ~38,9px. Nếu feed cả viền,
   detector thấy lề trắng mà pipeline thật không có (pipeline crop sát bbox đã detect).
   Box trả về được cộng offset để quay lại toạ độ ảnh gốc, cho khớp với GT.
2. **Giữ ``score`` của từng box.** ``PaddleCellTableModule.postprocess()`` bỏ score, mà
   không có score thì không tính được AP. Nên ở đây ta tự bóc ``coordinate`` + ``score``
   từ raw output, rồi áp prune sau (tuỳ cờ).

``--no-prune`` bỏ ``_prune_nested_cell_boxes`` (containment 0,8). Prune là một phần của hệ
thống và ảnh hưởng trực tiếp precision, nên nên chạy cả hai chế độ và báo cả hai.

Ví dụ
-----
    # chạy từ benchmark/parser/
    python run_parser/run_cells_pubtables.py \
        --sample data/pubtables1m/sample_1500.json \
        --images data/pubtables1m/images \
        --ann    data/pubtables1m/test \
        --out    pubtables_results/cells_prune.json

    python run_parser/run_cells_pubtables.py ... --no-prune \
        --out pubtables_results/cells_noprune.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import pubtables_gt as G  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter_area(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x1 - x0) * (y1 - y0) if x1 > x0 and y1 > y0 else 0.0


def prune_boxes(boxes: list[list[float]], mode: str, thr: float = 0.8) -> list[list[float]]:
    """Các biến thể của ``_prune_nested_cell_boxes``.

    Hàm gốc trong ``pdf2zh/parser/ai_models/table.py`` có HAI lượt:

      * lượt 1 — bỏ box mà >= thr DIỆN TÍCH CỦA CHÍNH NÓ nằm trong một box đã giữ
        (duyệt theo diện tích tăng dần, nên box nhỏ được giữ trước). Đây là lọc trùng lặp.
      * lượt 2 — bỏ box nào **CHỨA** một box nhỏ hơn với >= thr diện tích box nhỏ.
        Đây là lượt xoá mất ô gộp: một spanning cell đúng thì đương nhiên chứa các
        grid cell bên trong, nên bị loại, và xung đột được giải theo hướng GIỮ Ô NHỎ.

    Các mode:
      ``system``           cả hai lượt — hành vi hiện tại của hệ thống
      ``nocontainer``      chỉ lượt 1 — giữ box chứa, vẫn lọc trùng lặp
      ``prefer-container`` lượt 1, rồi bỏ box BỊ CHỨA thay vì box chứa — ưu tiên ô gộp
      ``none``             không lọc gì
    """
    if mode == "none" or len(boxes) < 2:
        return list(boxes)

    # --- lượt 1: lọc trùng lặp (giữ nguyên logic gốc) ---
    kept: list[list[float]] = []
    for box in sorted(boxes, key=_area):
        a = max(1.0, _area(box))
        if not any(_inter_area(box, k) / a >= thr for k in kept):
            kept.append(box)

    if mode == "nocontainer":
        return kept

    if mode == "system":
        out = []
        for box in kept:
            ba = max(1.0, _area(box))
            contains = any(
                _area(o) < ba and _inter_area(box, o) / max(1.0, _area(o)) >= thr
                for o in kept if o is not box)
            if not contains:
                out.append(box)
        return out

    if mode == "prefer-container":
        drop = set()
        for i, small in enumerate(kept):
            sa = max(1.0, _area(small))
            for j, big in enumerate(kept):
                if i == j or _area(big) <= _area(small):
                    continue
                if _inter_area(big, small) / sa >= thr:
                    drop.add(i)
                    break
        return [b for i, b in enumerate(kept) if i not in drop]

    raise ValueError(f"prune mode không hợp lệ: {mode}")


def _extract(raw_results: list, prune_mode: str) -> list[list[dict]]:
    """Raw Paddle output -> [[{box, score}, ...], ...]; giữ score để tính AP."""
    out = []
    for res in raw_results:
        if res is None:
            out.append([])
            continue
        items = []
        for cell in res.get("boxes", []):
            coords = cell.get("coordinate")
            if not coords:
                continue
            items.append({
                "box": [float(v) for v in coords],
                "score": float(cell.get("score", 1.0)),
            })
        if prune_mode != "none" and items:
            kept = prune_boxes([it["box"] for it in items], prune_mode)
            keys = {tuple(round(v, 4) for v in b) for b in kept}
            items = [it for it in items
                     if tuple(round(v, 4) for v in it["box"]) in keys]
        out.append(items)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True, help="output của sample_pubtables.py")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--prune-mode",
                    choices=("system", "nocontainer", "prefer-container", "none"),
                    default="system",
                    help="system = hành vi hiện tại; xem docstring prune_boxes()")
    ap.add_argument("--no-prune", action="store_true",
                    help="alias của --prune-mode none")
    ap.add_argument("--no-crop", action="store_true",
                    help="feed cả viền trắng thay vì crop về GT table bbox (biến thể robustness)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from pdf2zh.parser.ai_models.table import PaddleCellTableModule

    prune_mode = "none" if args.no_prune else args.prune_mode
    items = json.loads(args.sample.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    print(f"[cells] {len(items)} bảng | batch={args.batch_size} thr={args.threshold} "
          f"prune_mode={prune_mode} crop_to_table={not args.no_crop}")

    model = PaddleCellTableModule()
    results: dict[str, dict] = {}
    skipped: Counter = Counter()
    t0 = time.time()

    for start in range(0, len(items), args.batch_size):
        chunk = items[start: start + args.batch_size]
        crops, metas = [], []
        for it in chunk:
            img_p = args.images / f"{it['name']}.jpg"
            if not img_p.exists():
                img_p = args.images / f"{it['name']}.png"
            if not img_p.exists():
                skipped["no_image"] += 1
                continue
            # KHÔNG bọc try/except rộng ở đây: một lỗi API im lặng từng làm cả run
            # trả về 0 bảng mà vẫn báo "thành công". Lỗi đọc XML phải nổ ra.
            tb = G.table_bbox(args.ann / it["xml"])
            if tb is None:
                skipped["no_table_object"] += 1
                continue
            img = Image.open(img_p).convert("RGB")
            if args.no_crop:
                crop, off = img, (0.0, 0.0)
            else:
                x0 = max(0, int(tb[0])); y0 = max(0, int(tb[1]))
                x1 = min(img.width, int(round(tb[2]))); y1 = min(img.height, int(round(tb[3])))
                if x1 <= x0 or y1 <= y0:
                    skipped["degenerate_crop"] += 1
                    continue
                crop, off = img.crop((x0, y0, x1, y1)), (float(x0), float(y0))
            crops.append(crop)
            metas.append((it, off))

        if not crops:
            continue

        prepared = model.prepare(crops)
        if model.model is None:
            model.load_model()
        raw = model.predict(prepared, batch_size=args.batch_size, threshold=args.threshold)
        per_img = _extract(raw, prune_mode)

        for (it, off), cells in zip(metas, per_img):
            ox, oy = off
            results[it["name"]] = {
                "xml": it["xml"],
                "offset": [ox, oy],
                # cộng offset -> quay về toạ độ ảnh gốc, khớp không gian của GT
                "cells": [{"box": [c["box"][0] + ox, c["box"][1] + oy,
                                   c["box"][2] + ox, c["box"][3] + oy],
                           "score": c["score"]} for c in cells],
            }

        done = start + len(chunk)
        if done % (args.batch_size * 20) == 0 or done >= len(items):
            el = time.time() - t0
            print(f"  {done}/{len(items)}  {el:.0f}s  ({done/max(el,1e-9):.1f} bảng/s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "threshold": args.threshold, "prune_mode": prune_mode,
            "crop_to_table": not args.no_crop, "batch_size": args.batch_size,
            "model": "RT-DETR-L_wireless_table_cell_det",
        },
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    n_cells = sum(len(v["cells"]) for v in results.values())
    print(f"\n[cells] {len(results)}/{len(items)} bảng, {n_cells} cell "
          f"({n_cells/max(len(results),1):.1f}/bảng) trong {time.time()-t0:.0f}s")
    if skipped:
        print(f"[cells] BỎ QUA: {dict(skipped)}")
    if not results:
        raise SystemExit("[cells] LỖI: không xử lý được bảng nào — dừng thay vì ghi file rỗng.")
    print(f"[cells] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
