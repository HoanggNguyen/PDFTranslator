"""One adapter per system under test. Each writes the same artifact shape::

    out/<system>/<lang>/<doc_id>/output.pdf
    out/<system>/<lang>/<doc_id>/meta.json

so the metric layer never needs to know which system produced a PDF. Runners are
independently invocable (one process per system) because the systems have
conflicting Python/dependency requirements — see docs/EVALUATION_PLAN.md §5.

Every runner must be resumable: an existing output.pdf + meta.json pair is skipped.
Jobs get timed out and suspended, and for DeepL a re-run costs real quota.
"""
