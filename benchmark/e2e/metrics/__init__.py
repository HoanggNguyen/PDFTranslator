"""Metric cho benchmark E2E. Thiết kế: docs/EVALUATION_PLAN.md §4.

Chia theo thứ hạng phụ thuộc, không theo nhóm A–E của plan:

* ``eval_text``  — page inflation, UTB/page và runner seconds/page.
* ``eval_preserve`` — detector chung ⇒ reading-order tau.
* ``eval_visual``  — GT + ảnh/PDF ⇒ NT-PPR, IO-PPR và OF-harm.
* ``eval_ink``     — GT + ảnh ⇒ IC-harm.
* ``eval_qe``      — cặp segment đã align ⇒ CometKiwi QE.
* ``aggregate``    — trung bình document-macro đúng như bảng trong paper.
"""
