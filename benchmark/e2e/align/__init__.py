"""Trích và ghép cặp câu nguồn↔đích, dùng CHUNG cho cả 4 hệ.

Điểm mấu chốt về tính công bằng: PDFTranslator có sẵn ``phase2_translated.json``
đã align hoàn hảo, BabelDOC có ``translate_tracking.json`` nếu chạy ``--debug``,
còn PDFMathTranslate không dump gì và DeepL là hộp đen. Dùng dump nội bộ cho hệ
nào có, và trích từ PDF cho hệ nào không, là **so hai thứ khác nhau** — hệ có dump
được align chuẩn miễn phí trong khi hệ kia gánh thêm sai số trích xuất.

Nên ở đây mọi hệ đều bị đối xử như hộp đen: trích text từ ``output.pdf``, ghép với
trang nguồn theo bbox + thứ tự đọc. Dump nội bộ chỉ dùng để **đối chiếu chéo** chất
lượng của chính module này, không dùng để chấm.
"""
