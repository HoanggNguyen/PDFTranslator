"""Metric cho benchmark E2E. Thiết kế: docs/EVALUATION_PLAN.md §4.

Chia theo thứ hạng phụ thuộc, không theo nhóm A–E của plan:

* ``eval_text``  — không cần detector, không cần GT, không cần render ảnh. Chạy được
  ngay khi runner xong, và chạy cho **cả 4 hệ** kể cả DeepL: page inflation, UTB,
  number-digit recall, sec/page, success rate.
* ``eval_preserve`` (chưa dựng) — cần detector chung ⇒ mIoU, Anchor-IoU.
* ``eval_visual``  (chưa dựng) — cần detector + ảnh render ⇒ Masked-SSIM.
* ``eval_qe``      (chưa dựng) — cần cặp segment từ ``eval_preserve`` ⇒ CometKiwi.
"""
