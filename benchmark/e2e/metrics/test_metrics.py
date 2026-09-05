"""Test cho khối metric không cần detector.

Chỉ test hàm thuần — phần đọc PDF cần PyMuPDF nên để cho smoke test của driver.
Chạy: ``python -m pytest benchmark/e2e/metrics/test_metrics.py -q``
"""

from benchmark.e2e.metrics import eval_text as E
from benchmark.e2e.metrics import numbers as N
from benchmark.e2e.metrics.langid import LangID


class TestNumbers:
    def test_quy_uoc_en_va_vi_cho_cung_ket_qua(self):
        """Lý do metric so theo dãy chữ số chứ không theo giá trị."""
        assert N.canon("1,234.56") == N.canon("1.234,56") == "123456"
        assert N.canon("1,5") == N.canon("1.5") == "15"

    def test_giu_dau_am(self):
        assert N.canon("-3.14") == "-314"
        assert N.canon("+42") == "42"

    def test_khong_dan_hai_so_cach_nhau_boi_khoang_trang(self):
        assert N.extract("1 234") == ["1", "234"]

    def test_trung_lap_la_thong_tin(self):
        """Nguồn có '12' ba lần mà đích còn một là mất nội dung thật."""
        assert N.recall("12 12 12", "12") == (3, 1)

    def test_so_moc_them_o_dich_khong_lam_tang_recall(self):
        assert N.recall("12", "12 12 12") == (1, 1)

    def test_nguon_khong_co_so(self):
        assert N.recall("khong co so nao", "cung vay") == (0, 0)


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


class TestScoreNumbers:
    def test_so_trang_khop_thi_xuat_them_so_theo_trang(self):
        r = E.score_numbers(["1 2", "3"], ["1 2", "3"])
        assert r["page_aligned"] is True
        assert r["per_page"] == [[2, 2], [1, 1]]
        assert r["recall"] == 1.0

    def test_so_trang_lech_thi_khong_co_so_theo_trang(self):
        r = E.score_numbers(["1 2", "3"], ["1 2 3"])
        assert r["page_aligned"] is False
        assert r["per_page"] is None
        assert (r["n_src"], r["n_found"]) == (3, 3)

    def test_so_di_chuyen_sang_trang_khac_khong_bi_tinh_la_mat(self):
        """Lý do headline number là mức tài liệu: reflow đẩy số sang trang sau."""
        r = E.score_numbers(["1 2 3", "4"], ["1 2", "3 4"])
        assert r["recall"] == 1.0                    # mức doc: không mất gì
        assert r["per_page"] == [[3, 2], [1, 1]]     # mức trang: tưởng mất 1


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
    def _rec(self, n_src, n_found, pages, untrans, ok=True, infl=1.0):
        return {"ok": ok, "n_pages_out": pages, "page_inflation": infl,
                "sec_per_page": 10.0,
                "numbers": {"n_src": n_src, "n_found": n_found},
                "utb": {"n_untranslated": untrans}}

    def test_dung_ti_le_cua_tong_khong_phai_trung_binh_cac_ti_le(self):
        """Doc 1: 1/100. Doc 2: 1/1. Trung bình các tỉ lệ = 0.505 (sai),
        tỉ lệ của các tổng = 2/101 = 0.0198 (đúng)."""
        s = E.summarize([self._rec(100, 1, 1, 0), self._rec(1, 1, 1, 0)])
        assert s["number_recall"] == 0.0198

    def test_dem_doc_reflow(self):
        s = E.summarize([self._rec(1, 1, 1, 0, infl=1.0),
                         self._rec(1, 1, 1, 0, infl=1.15)])
        assert s["n_docs_reflowed"] == 1

    def test_sec_per_page_bo_qua_doc_chet(self):
        """Doc crash sớm có wall nhỏ; gộp vào là hệ chết sớm trông như hệ nhanh."""
        chet = self._rec(1, 1, 1, 0, ok=False)
        chet["sec_per_page"] = 1.0
        s = E.summarize([self._rec(1, 1, 1, 0), chet])
        assert s["sec_per_page_mean"] == 10.0

    def test_success_rate_tinh_ca_doc_chet(self):
        dead = {"ok": False, "n_pages_out": None, "page_inflation": None,
                "sec_per_page": None, "numbers": None, "utb": None}
        s = E.summarize([self._rec(1, 1, 1, 0), dead])
        assert s["success_rate"] == 0.5
        assert s["n_docs"] == 2
