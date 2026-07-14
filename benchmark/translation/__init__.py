"""Evaluation harness for the Phase-2 translation core.

Everything here lives OUTSIDE pdf2zh/translation and only *calls into* the public
entrypoint (translate_document) + observes it (timers, read-only httpx hooks). See
docs/EVALUATION_PLAN.md. The translation core is never modified.
"""
