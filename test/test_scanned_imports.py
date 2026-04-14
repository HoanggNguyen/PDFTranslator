"""
Smoke test for scanned OCR module imports.
Ensures core modules can be imported without errors.
"""

import unittest


class TestScannedImports(unittest.TestCase):
    def test_scanned_module_imports(self):
        """Verify scanned module and core components can be imported."""
        try:
            from pdf2zh import scanned  # noqa: F401
        except ImportError as e:
            self.fail(f"Failed to import pdf2zh.scanned: {e}")

    def test_stage_a_parser_imports(self):
        """Verify StageAParser can be imported."""
        try:
            from pdf2zh.scanned.parser import StageAParser  # noqa: F401
        except ImportError as e:
            self.fail(f"Failed to import StageAParser: {e}")

    def test_ocr_utils_imports(self):
        """Verify OCR utilities can be imported."""
        try:
            from pdf2zh.scanned.utils.ocr_text import collect_ocr_text  # noqa: F401
        except ImportError as e:
            self.fail(f"Failed to import collect_ocr_text: {e}")


if __name__ == "__main__":
    unittest.main()
