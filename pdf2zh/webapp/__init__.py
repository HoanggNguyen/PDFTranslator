"""Gradio web app for end-to-end PDF translation.

Submodules: ``config`` and ``runner`` are framework-agnostic (importable without
Gradio); ``ui`` holds the Gradio layout. Kept import-light on purpose so the
agnostic modules can be unit-tested without pulling in Gradio.
"""
