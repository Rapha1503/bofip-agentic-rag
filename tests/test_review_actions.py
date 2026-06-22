from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_review_actions import extract_actions


class ReviewActionsTests(unittest.TestCase):
    def test_extracts_recommended_fixes(self):
        review = """
## Verdict
Partial success.

## Recommended next fixes
- [high][retrieval] Preserve source carry-over across relaunches.
- [medium][eval] Add RFPI and ENR validation cases.

## Minimal validation set
- TVA B2B territoriality.

END_OF_RESPONSE
"""
        actions = extract_actions(review)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].severity, "high")
        self.assertEqual(actions[0].area, "retrieval")
        self.assertIn("Preserve source", actions[0].recommendation)
        self.assertEqual(actions[1].severity, "medium")
        self.assertEqual(actions[1].area, "eval")

    def test_ignores_review_without_end_marker(self):
        with self.assertRaises(ValueError):
            extract_actions("## Recommended next fixes\n- [high][rag] Fix")

    def test_cli_writes_default_outputs_next_to_review(self):
        review = """## Recommended next fixes
- [high][retrieval] Preserve source carry-over across relaunches.

END_OF_RESPONSE
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            review_path = Path(tmpdir) / "review.md"
            review_path.write_text(review, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "extract_review_actions.py"), str(review_path)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_json = review_path.with_name("review_actions.json")
            output_md = review_path.with_name("review_actions.md")
            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["severity"], "high")
            self.assertIn("[high][retrieval] Preserve source", output_md.read_text(encoding="utf-8"))

    def test_cli_writes_custom_outputs(self):
        review = """## Recommended next fixes
- [medium][eval] Add RFPI validation cases.

END_OF_RESPONSE
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            review_path = tmp_path / "review.md"
            json_path = tmp_path / "custom" / "actions.json"
            md_path = tmp_path / "custom" / "actions.md"
            review_path.write_text(review, encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "extract_review_actions.py"),
                    str(review_path),
                    "--output-json",
                    str(json_path),
                    "--output-md",
                    str(md_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["area"], "eval")


if __name__ == "__main__":
    unittest.main()
