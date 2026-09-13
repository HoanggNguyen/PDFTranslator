"""Test cho khối metric: eval_text + eval_preserve (reading-order).

Chỉ test hàm thuần — phần đọc PDF cần PyMuPDF nên để cho smoke test của driver.
Chạy: ``python -m pytest benchmark/e2e/metrics/test_metrics.py -q``
"""

from benchmark.e2e.metrics import eval_text as E
from benchmark.e2e.metrics.eval_preserve import kendall_tau, hungarian, score_page
from benchmark.e2e.parse.run_detectors import reading_order
from benchmark.e2e.metrics.langid import LangID


# ── eval_text ──────────────────────────────────────────────────────────────── #

class TestLangID:
    lid = LangID(allow_download=False)

    def test_heuristic_phan_biet_duoc_ba_ngon_ngu(self):
        assert self.lid.predict("the of and to in is that for with as are")[0] == "en"
        assert self.lid.predict("của và các được trong cho những một là")[0] == "vi"
        assert self.lid.predict("本公司今年的收入增长而成本得到控制")[0] == "zh"

    def test_khoi_khong_ket_luan_duoc_thi_tra_un(self):
        """'un' quan trọng: nó KHÔNG bị tính là chưa dịch."""
        assert self.lid.predict("Fig. 3")[0] == "un"
        assert self.lid.predict("")[0] == "un"


class TestScoreUTB:
    lid = LangID(allow_download=False)

    def test_san_do_dai_loai_khoi_nhieu(self):
        blocks = [["Fig. 3", "the of and to in is that for with as are be this"]]
        r = E.score_utb(blocks, self.lid, "en", min_chars=30, min_prob=0.5)
        assert r["n_blocks_scored"] == 1        # "Fig. 3" bị loại
        assert r["n_untranslated"] == 1

    def test_khoi_da_dich_khong_bi_tinh(self):
        blocks = [["của và các được trong cho những một là có không với để này"]]
        r = E.score_utb(blocks, self.lid, "en", min_chars=30, min_prob=0.5)
        assert r["n_untranslated"] == 0
        assert r["utb_per_page"] == 0.0


class TestSummarize:
    def _rec(self, pages, untrans, ok=True, infl=1.0):
        return {"ok": ok, "n_pages_out": pages, "page_inflation": infl,
                "sec_per_page": 10.0,
                "utb": {"n_untranslated": untrans}}

    def test_dem_doc_reflow(self):
        s = E.summarize([self._rec(1, 0, infl=1.0),
                         self._rec(1, 0, infl=1.15)])
        assert s["n_docs_reflowed"] == 1

    def test_sec_per_page_bo_qua_doc_chet(self):
        """Doc crash sớm có wall nhỏ; gộp vào là hệ chết sớm trông như hệ nhanh."""
        chet = self._rec(1, 0, ok=False)
        chet["sec_per_page"] = 1.0
        s = E.summarize([self._rec(1, 0), chet])
        assert s["sec_per_page_mean"] == 10.0


# ── eval_preserve (reading-order) ──────────────────────────────────────────── #

class TestKendallTau:
    def test_hoan_hao_dong_thu_tu(self):
        assert kendall_tau([0, 1, 2], [0, 1, 2]) == 1.0

    def test_nguoc_hoan_toan(self):
        assert kendall_tau([0, 1, 2], [2, 1, 0]) == -1.0

    def test_khong_phan_quyet(self):
        """Trùng rank ở cả hai phía ⇒ concordant + discordant = 0 ⇒ None."""
        assert kendall_tau([1, 1], [2, 2]) is None

    def test_mot_phan_tu(self):
        """3 cặp: (0,1)=+1, (0,2)=+1, (1,2)=−1 → tau = (2−1)/3."""
        assert abs(kendall_tau([0, 1, 2], [0, 2, 1]) - 1 / 3) < 1e-9

    def test_rieng_le(self):
        assert kendall_tau([0], [0]) is None

    def test_rong(self):
        assert kendall_tau([], []) is None


class TestReadingOrder:
    def test_tren_xuong_duoi(self):
        elems = [
            {"bbox_norm": [0.1, 0.5, 0.9, 0.6]},
            {"bbox_norm": [0.1, 0.1, 0.9, 0.2]},
            {"bbox_norm": [0.1, 0.3, 0.9, 0.4]},
        ]
        reading_order(elems)
        orders = [e["reading_order"] for e in elems]
        assert orders == [2, 0, 1]

    def test_cung_dong_trai_phai(self):
        elems = [
            {"bbox_norm": [0.5, 0.1, 0.9, 0.2]},
            {"bbox_norm": [0.1, 0.1, 0.4, 0.2]},
        ]
        reading_order(elems)
        orders = [e["reading_order"] for e in elems]
        assert orders == [1, 0]

    def test_1_phan_tram_cao_do(self):
        """Hai box lệch 0.5% chiều cao → phải kể là khác dòng."""
        elems = [
            {"bbox_norm": [0.1, 0.100, 0.5, 0.200]},
            {"bbox_norm": [0.5, 0.106, 0.9, 0.206]},   # 0.6% lệch
        ]
        reading_order(elems)
        orders = [e["reading_order"] for e in elems]
        assert orders == [0, 1]

    def test_rong(self):
        elems = []
        reading_order(elems)
        assert elems == []


class TestHungarian:
    def test_ghep_cung_nhom(self):
        gts  = [{"group": "text", "bbox_norm": [0.0, 0.0, 0.5, 0.5]},
                {"group": "text", "bbox_norm": [0.5, 0.5, 1.0, 1.0]}]
        preds = [{"group": "text", "bbox_norm": [0.0, 0.0, 0.5, 0.5]},
                 {"group": "text", "bbox_norm": [0.5, 0.5, 1.0, 1.0]}]
        matched = hungarian(gts, preds)
        assert len(matched) == 2
        ious = [iou for _, _, iou in matched]
        assert all(i > 0.99 for i in ious)

    def test_khong_ghep_khac_nhom(self):
        gts  = [{"group": "text", "bbox_norm": [0.0, 0.0, 0.5, 0.5]}]
        preds = [{"group": "table", "bbox_norm": [0.0, 0.0, 0.5, 0.5]}]
        matched = hungarian(gts, preds)
        assert len(matched) == 0

    def test_rong(self):
        assert hungarian([], [{"group": "text", "bbox_norm": [0, 0, 1, 1]}]) == []
        assert hungarian([{"group": "text", "bbox_norm": [0, 0, 1, 1]}], []) == []


class TestScorePage:
    def test_identity_tau_1(self):
        page = {"page": 1, "elements": [
            {"group": "text", "bbox_norm": [0.1, 0.1, 0.5, 0.5], "reading_order": 0},
            {"group": "text", "bbox_norm": [0.1, 0.6, 0.5, 0.9], "reading_order": 1},
        ]}
        r = score_page(page, page)
        assert r["tau"] == 1.0
        assert r["n_matched"] == 2
