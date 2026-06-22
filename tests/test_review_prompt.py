import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_review_prompt import build_review_context, build_review_prompt, select_evidence_cards


class ReviewPromptTests(unittest.TestCase):
    def test_select_evidence_cards_prioritizes_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = root / "evidence_cards"
            cards.mkdir()
            (cards / "Q1.md").write_text("# Q1\nStatus: supported", encoding="utf-8")
            (cards / "Q2.md").write_text("# Q2\nStatus: partial", encoding="utf-8")

            selected = select_evidence_cards(root, max_cards=1, preferred_ids=["Q2"])

            self.assertEqual(selected[0].name, "Q2.md")

    def test_prompt_requires_sections_and_review_instructions(self):
        prompt = build_review_prompt(["# Card"], run_id="run1")

        for required in [
            "Verdict",
            "Remaining blockers",
            "Recommended next fixes",
            "Minimal validation set",
            "Overfit and leakage risks",
            "END_OF_RESPONSE",
        ]:
            self.assertIn(required, prompt)

        self.assertIn("separate retrieval failures from generation failures", prompt)
        self.assertIn("flag overfit", prompt)
        self.assertIn("avoid fiscal hardcoding", prompt)
        self.assertIn("different BOFiP families", prompt)
        self.assertIn("mark uncertain claims clearly", prompt)

    def test_context_mentions_no_runtime_gold_leakage(self):
        context = build_review_context(run_id="run1", summary={"total_queries": 3})

        self.assertIn("Do not assume gold labels were shown to the runtime", context)

    def test_main_writes_prompt_from_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cards = run_dir / "evidence_cards"
            cards.mkdir()
            (cards / "Q1.md").write_text("# Q1\nStatus: supported", encoding="utf-8")
            (cards / "Q2.md").write_text("# Q2\nStatus: partial", encoding="utf-8")
            (run_dir / "summary.json").write_text(json.dumps({"run_id": "run1", "total_queries": 2}), encoding="utf-8")

            output_path = run_dir / "review_prompt.md"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_review_prompt.py"),
                    str(run_dir),
                    "--preferred-id",
                    "Q2",
                    "--max-cards",
                    "1",
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            prompt = output_path.read_text(encoding="utf-8")
            self.assertIn("# Q2", prompt)
            self.assertNotIn("# Q1", prompt)
            self.assertIn("Run id: run1", prompt)


if __name__ == "__main__":
    unittest.main()
