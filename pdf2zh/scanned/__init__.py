"""Scanned PDF translation pipeline - Stage A.

This package provides parsing and analysis for scanned (image-based) PDFs
using Surya for layout detection and OCR.

Main exports:
- PDFTypeDetector: Detect if PDF is scanned, digital, or mixed
- StageAParser and phase result objects: Main parser for Stage A processing
"""
from pdf2zh.scanned.detector import PDFTypeDetector

__all__ = [
    # Detector
    "PDFTypeDetector",

    # Parser
    "StageAParser"
]
