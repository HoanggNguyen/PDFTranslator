"""Chạy detector chấm điểm lên ảnh trang, ra box đã chuẩn hoá.

Detector chính là **Docling layout (RT-DETR)** vì hai lý do độc lập (§3):

* **Không hệ nào dưới bài kiểm dùng nó.** BabelDOC và PDFMathTranslate dùng chung
  DocLayout-YOLO; PDFTranslator dùng Surya. Chấm bằng một trong hai là thiên vị.
* **Nó train trên DocLayNet *train*, corpus lấy từ *test*** ⇒ cùng phân bố nhãn với
  GT người vẽ nên trần đo rất cao, mà không rò rỉ dữ liệu. Trần cao thì mọi sụt
  giảm quy được cho translator chứ không cho detector — đó chính là điều cần.

``--detectors surya`` có mặt để làm kiểm định robustness ở P6: **thứ hạng không đổi
qua nhiều detector ⇒ kết luận vững; đổi ⇒ phải nói ra.** Đây đúng chỗ paper BabelDOC
im lặng. Đừng dùng surya làm detector chính — nó là bộ não của PDFTranslator.

Đầu ra ``_layout/<detector>/<system>/<lang>/<doc_id>/p<NNN>.json``:

    {"page": 3, "width": 1275, "height": 1650,
     "elements": [{"class": "Text", "group": "text",
                   "bbox_norm": [x0,y0,x1,y1], "score": 0.97, "reading_order": 0}]}

``bbox_norm`` xyxy trong [0,1], gốc toạ độ **góc trên trái**, chuẩn hoá **từng trục
theo chiều của nó** — cùng quy ước với ``gt.json`` (xem bẫy #4 ở README), nên box
của detector và box GT so trực tiếp được, không cần biết DPI hay khổ giấy.

Ví dụ
-----
    python -m benchmark.e2e.parse.run_detectors --out benchmark/e2e/out
    python -m benchmark.e2e.parse.run_detectors --out ... --detectors docling,surya
    python -m benchmark.e2e.parse.run_detectors --warmup-only     # chỉ tải model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_KEY = "_source"

# Bộ nhãn rút gọn của §4.1. Mọi detector và cả GT đều quy về đây trước khi ghép,
# để "Picture" của detector này và "Figure" của detector kia không thành hai lớp.
GROUPS = ("text", "title", "list", "table", "figure", "formula", "caption",
          "furniture")

# Nhóm ANCHOR: những thứ PHẢI đứng yên tuyệt đối khi dịch. Anchor-IoU chỉ tính trên
# đây nên nó là tín hiệu sạch nhất — không nhiễu bởi text nở ra hay co lại.
ANCHOR_GROUPS = ("figure", "table", "formula", "furniture")

# DocLayNet 11 lớp — dùng chung cho GT người vẽ và cho Docling RT-DETR (nó train
# đúng trên taxonomy này).
DOCLAYNET_GROUP = {
    "Caption": "caption",
    "Footnote": "text",
    "Formula": "formula",
    "List-item": "list",
    "Page-footer": "furniture",
    "Page-header": "furniture",
    "Picture": "figure",
    "Section-header": "title",
    "Table": "table",
    "Text": "text",
    "Title": "title",
}

# Surya 0.17 (surya/layout/label.py). Nhãn pin theo mã nguồn, KHÔNG theo README —
# README của surya liệt kê thiếu và lệch tên.
SURYA_GROUP = {
    "Caption": "caption", "Footnote": "text", "Formula": "formula",
    "List-item": "list", "Page-footer": "furniture", "Page-header": "furniture",
    "Picture": "figure", "Figure": "figure", "Section-header": "title",
    "Table": "table", "Text": "text", "Title": "title",
    "Text-inline-math": "text", "Code": "text", "Form": "text",
    "Table-of-contents": "list", "Handwriting": "text",
    "PageHeader": "furniture", "PageFooter": "furniture",
    "SectionHeader": "title", "ListItem": "list", "TextInlineMath": "text",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None,
                   help="Gốc artifact; đọc <out>/_render/, ghi <out>/_layout/.")
    p.add_argument("--detectors", default="docling",
                   help="Cách nhau bởi dấu phẩy. 'docling' chấm chính, 'surya' để "
                        "kiểm robustness (§P6) — đừng đổi vai.")
    p.add_argument("--langs", default="vi")
    p.add_argument("--systems", default=None,
                   help="Mặc định: mọi hệ có ảnh trong _render/.")
    p.add_argument("--batch", type=int, default=4, help="Số trang mỗi lượt suy luận.")
    p.add_argument("--min-score", type=float, default=0.30,
                   help="Bỏ box dưới ngưỡng tin cậy. Đặt thấp có chủ ý: box thừa bị "
                        "phạt qua precision, còn box thiếu thì không cứu được.")
    p.add_argument("--force", action="store_true", help="Chạy lại cả trang đã có.")
    p.add_argument("--warmup-only", action="store_true",
                   help="Chỉ nạp model rồi thoát — dùng để nung cache /data.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Adapter detector. Mỗi cái nhận list ảnh PIL, trả list[list[dict]] cùng thứ tự. #
# --------------------------------------------------------------------------- #
class DoclingDetector:
    name = "docling"

    def __init__(self) -> None:
        # Cố ý dùng lớp THẤP thay vì DocumentConverter: ta chỉ cần box, không cần
        # nó convert cả tài liệu (OCR, table structure, ghép markdown) — vừa chậm
        # gấp bội vừa thêm biến không kiểm soát được vào phép đo.
        try:
            from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
        except ImportError as exc:
            raise SystemExit(
                "!! thiếu docling-ibm-models:  pip install docling docling-ibm-models\n"
                f"   ({exc})") from exc
        self._predict = self._build(LayoutPredictor)

    @staticmethod
    def _build(LayoutPredictor):
        """API của LayoutPredictor đổi giữa các bản; thử các chữ ký đã biết.

        Không đoán mò: mỗi nhánh là một chữ ký có thật ở một bản đã phát hành. Hỏng
        cả ba thì báo rõ chứ không im lặng trả về rỗng — rỗng sẽ được đọc thành
        "hệ này mất sạch layout", tức là một kết luận sai nghiêm trọng.
        """
        from huggingface_hub import snapshot_download

        repo = "ds4sd/docling-models"
        try:
            predictor = LayoutPredictor(snapshot_download(repo_id=repo))
        except TypeError:
            try:
                predictor = LayoutPredictor(artifact_path=snapshot_download(repo_id=repo))
            except TypeError:
                predictor = LayoutPredictor()
        return predictor.predict

    def __call__(self, images: list) -> list[list[dict]]:
        out = []
        for image in images:
            boxes = []
            for r in self._predict(image):
                label = str(r.get("label", r.get("class", ""))).strip()
                boxes.append({
                    "class": label,
                    "group": DOCLAYNET_GROUP.get(label, "text"),
                    "xyxy": [float(r["l"]), float(r["t"]),
                             float(r["r"]), float(r["b"])],
                    "score": float(r.get("confidence", r.get("score", 1.0))),
                })
            out.append(boxes)
        return out


class SuryaDetector:
    name = "surya"

    def __init__(self) -> None:
        try:
            from surya.layout import LayoutPredictor
        except ImportError as exc:
            raise SystemExit(f"!! thiếu surya-ocr: {exc}") from exc
        self._predictor = LayoutPredictor()

    def __call__(self, images: list) -> list[list[dict]]:
        results = self._predictor(images)
        out = []
        for res in results:
            boxes = []
            for b in res.bboxes:
                label = str(getattr(b, "label", "")).strip()
                x0, y0, x1, y1 = (float(v) for v in b.bbox)
                boxes.append({
                    "class": label,
                    "group": SURYA_GROUP.get(label, "text"),
                    "xyxy": [x0, y0, x1, y1],
                    "score": float(getattr(b, "confidence", 1.0) or 1.0),
                })
            out.append(boxes)
        return out


DETECTORS = {"docling": DoclingDetector, "surya": SuryaDetector}


def reading_order(elements: list[dict]) -> None:
    """Gán ``reading_order`` tại chỗ: trên xuống dưới, trái sang phải.

    Cố tình đơn giản và **tất định**. Kendall's tau ở §4.1 so thứ tự nguồn với thứ
    tự đích, nên điều quan trọng là cùng một hàm áp cho cả hai phía — không phải là
    hàm này đoán đúng thứ tự đọc thật của một trang hai cột. Lượng tử y theo 1%
    chiều cao trang để hai box cùng dòng không bị đảo vì lệch nửa pixel.
    """
    order = sorted(range(len(elements)),
                   key=lambda i: (round(elements[i]["bbox_norm"][1] * 100),
                                  elements[i]["bbox_norm"][0]))
    for rank, idx in enumerate(order):
        elements[idx]["reading_order"] = rank


def page_records(boxes: list[dict], width: int, height: int, page: int,
                 min_score: float) -> dict:
    elements = []
    for b in boxes:
        if b["score"] < min_score:
            continue
        x0, y0, x1, y1 = b["xyxy"]
        # Chuẩn hoá TỪNG TRỤC theo chiều của nó — cùng quy ước với gt.json.
        bbox = [min(max(x0 / width, 0.0), 1.0), min(max(y0 / height, 0.0), 1.0),
                min(max(x1 / width, 0.0), 1.0), min(max(y1 / height, 0.0), 1.0)]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        elements.append({"class": b["class"], "group": b["group"],
                         "bbox_norm": [round(v, 6) for v in bbox],
                         "score": round(b["score"], 4)})
    reading_order(elements)
    return {"page": page, "width": width, "height": height, "elements": elements}


def detect_doc(detector, img_dir: Path, dest: Path, batch: int, min_score: float,
               force: bool) -> tuple[int, int]:
    """Trả (số trang chạy mới, số trang dùng lại)."""
    from PIL import Image

    pages = sorted(img_dir.glob("p*.png"))
    if not pages:
        return 0, 0
    dest.mkdir(parents=True, exist_ok=True)

    todo = [p for p in pages if force or not (dest / f"{p.stem}.json").exists()]
    cached = len(pages) - len(todo)

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        images = [Image.open(p).convert("RGB") for p in chunk]
        try:
            results = detector(images)
        finally:
            for im in images:
                im.close()
        for path, image_boxes in zip(chunk, results):
            with Image.open(path) as im:
                w, h = im.size
            rec = page_records(image_boxes, w, h, int(path.stem[1:]), min_score)
            (dest / f"{path.stem}.json").write_text(
                json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return len(todo), cached


def main() -> int:
    args = parse_args()
    names = [d.strip() for d in args.detectors.split(",") if d.strip()]
    unknown = [d for d in names if d not in DETECTORS]
    if unknown:
        print(f"!! detector lạ {unknown}; đã biết: {sorted(DETECTORS)}")
        return 1

    if args.warmup_only:
        for name in names:
            print(f">>> nạp {name}...", flush=True)
            DETECTORS[name]()
            print(f"    {name} sẵn sàng", flush=True)
        return 0

    if args.out is None:
        print("!! cần --out (hoặc --warmup-only)")
        return 1

    render_root = args.out / "_render"
    if not render_root.is_dir():
        print(f"!! chưa có {render_root} — chạy parse.render_pages trước")
        return 1

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    systems = ([s.strip() for s in args.systems.split(",") if s.strip()]
               if args.systems else
               sorted(d.name for d in render_root.iterdir()
                      if d.is_dir() and d.name != SOURCE_KEY))

    for name in names:
        print(f"\n>>> detector: {name}", flush=True)
        detector = DETECTORS[name]()
        layout_root = args.out / "_layout" / name
        total_new = total_cached = 0

        # Trang nguồn TRƯỚC: nó là hàng `Source ceiling`, và mọi metric preservation
        # đều đọc tương đối so với hàng đó.
        for img_dir in sorted((render_root / SOURCE_KEY).glob("*")):
            if not img_dir.is_dir():
                continue
            new, cached = detect_doc(detector, img_dir,
                                     layout_root / SOURCE_KEY / img_dir.name,
                                     args.batch, args.min_score, args.force)
            total_new += new
            total_cached += cached
            print(f"  {SOURCE_KEY}/{img_dir.name:34} {new:3d} mới, {cached:3d} cache",
                  flush=True)

        for system in systems:
            for lang in langs:
                base = render_root / system / lang
                if not base.is_dir():
                    continue
                for img_dir in sorted(base.glob("*")):
                    if not img_dir.is_dir():
                        continue
                    new, cached = detect_doc(
                        detector, img_dir,
                        layout_root / system / lang / img_dir.name,
                        args.batch, args.min_score, args.force)
                    total_new += new
                    total_cached += cached
                    print(f"  {system}/{lang}/{img_dir.name:28} "
                          f"{new:3d} mới, {cached:3d} cache", flush=True)

        print(f"  [{name}] {total_new} trang chạy mới, {total_cached} dùng lại "
              f"-> {layout_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
