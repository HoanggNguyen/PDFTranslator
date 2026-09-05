"""Tầng tri giác dùng chung: biến PDF của MỌI hệ thành cùng một dạng box.

Hai bước, cố ý tách rời:

* ``render_pages`` — PDF -> PNG 150 DPI. Cùng DPI, cùng khổ cho nguồn lẫn mọi
  output. Ảnh này dùng ở HAI chỗ: đầu vào của detector, và Masked-SSIM (§4.2).
  Render một lần, dùng hai lần.
* ``run_detectors`` — PNG -> box đã chuẩn hoá ``{page, class, bbox_norm, reading_order}``.

Vì sao phải là detector THỨ BA, không phải detector của hệ nào: BabelDOC và
PDFMathTranslate cùng dùng DocLayout-YOLO, PDFTranslator dùng Surya. Chấm bằng
Surya là thiên vị PDFTranslator, chấm bằng DocLayout-YOLO là thiên vị hai hệ kia.
Docling (RT-DETR) **không hệ nào dưới bài kiểm dùng**, và nó train trên DocLayNet
*train* trong khi corpus lấy từ *test* ⇒ trần đo cao mà không rò rỉ.
"""
