"""Backend selection tests that do not require PyTorch or model weights."""

import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

from indextts.utils.runtime import rocm_inference_options


class RuntimeOptionsTests(unittest.TestCase):
    def options(self, hip, kernel=None, deepspeed=False, accel=False):
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip=hip))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            return rocm_inference_options(kernel, deepspeed, accel)

    def test_nvidia_options_are_preserved(self):
        self.assertEqual(self.options(None, None, True, True), (None, True, True))
        self.assertEqual(self.options(None, True), (True, False, False))

    def test_rocm_disables_automatic_custom_kernel(self):
        self.assertEqual(self.options("6.4"), (False, False, False))

    def test_explicit_custom_kernel_reports_fallback(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.options("6.4", True), (False, False, False))
        self.assertIn("BigVGAN", str(caught[0].message))

    def test_rocm_rejects_unvalidated_acceleration(self):
        for deepspeed, accel in ((True, False), (False, True), (True, True)):
            with self.subTest(deepspeed=deepspeed, accel=accel):
                with self.assertRaisesRegex(ValueError, "ROCm"):
                    self.options("6.4", deepspeed=deepspeed, accel=accel)


if __name__ == "__main__":
    unittest.main()
