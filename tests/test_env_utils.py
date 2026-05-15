from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.env_utils import load_env_file


class EnvUtilsTests(unittest.TestCase):
    def test_load_env_file_sets_missing_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\nOTHER_VALUE=abc\n", encoding="utf-8")
            previous_key = os.environ.pop("OPENAI_API_KEY", None)
            previous_other = os.environ.pop("OTHER_VALUE", None)
            try:
                loaded = load_env_file(env_path)
                self.assertEqual(loaded["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["OTHER_VALUE"], "abc")
            finally:
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("OTHER_VALUE", None)
                if previous_key is not None:
                    os.environ["OPENAI_API_KEY"] = previous_key
                if previous_other is not None:
                    os.environ["OTHER_VALUE"] = previous_other

    def test_load_env_file_handles_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.local"
            env_path.write_text("\ufeffGEMINI_API_KEY=test-gemini-key\n", encoding="utf-8")
            previous = os.environ.pop("GEMINI_API_KEY", None)
            try:
                loaded = load_env_file(env_path)
                self.assertEqual(loaded["GEMINI_API_KEY"], "test-gemini-key")
                self.assertEqual(os.environ["GEMINI_API_KEY"], "test-gemini-key")
            finally:
                os.environ.pop("GEMINI_API_KEY", None)
                if previous is not None:
                    os.environ["GEMINI_API_KEY"] = previous


if __name__ == "__main__":
    unittest.main()
