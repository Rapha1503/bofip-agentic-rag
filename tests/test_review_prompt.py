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
        context = build_review_context(
            run_id="run1",
            summary={"total_queries": 3},
            config={"provider": "deepseek", "model": "deepseek-chat", "max_iterations": 2},
        )

        self.assertIn("BOFiP Agentic RAG", context)
        self.assertIn("Run id: run1", context)
        self.assertIn("Total queries: 3", context)
        self.assertIn("provider: deepseek", context)
        self.assertIn("model: deepseek-chat", context)
        self.assertIn("max_iterations: 2", context)
        self.assertIn("ChatGPT reviewer-only", context)
        self.assertIn("Do not assume gold labels were shown to the runtime", context)
        self.assertIn("distinguish retrieval failures from generation failures", context)
        self.assertIn("avoid fiscal hardcoding", context)
        self.assertIn("mark uncertainty", context)

    def test_main_writes_context_and_prompts_from_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cards = run_dir / "evidence_cards"
            cards.mkdir()
            (cards / "Q1.md").write_text("# Q1\nStatus: supported", encoding="utf-8")
            (cards / "Q2.md").write_text("# Q2\nStatus: partial", encoding="utf-8")
            (run_dir / "summary.json").write_text(json.dumps({"run_id": "run1", "total_queries": 2}), encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps({"provider": "deepseek", "model": "deepseek-chat"}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_review_prompt.py"),
                    "--run-dir",
                    str(run_dir),
                    "--preferred-id",
                    "Q2",
                    "--max-cards",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            review_dir = run_dir / "chatgpt-review"
            context = (review_dir / "context.md").read_text(encoding="utf-8")
            prompts = (review_dir / "prompts.md").read_text(encoding="utf-8")
            self.assertIn("Run id: run1", context)
            self.assertIn("provider: deepseek", context)
            self.assertIn("# Q2", prompts)
            self.assertNotIn("# Q1", prompts)
            self.assertIn("END_OF_RESPONSE", prompts)

    def test_main_accepts_summary_json_with_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cards = run_dir / "evidence_cards"
            cards.mkdir()
            (cards / "Q1.md").write_text("# Q1\nStatus: supported", encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps({"run_id": "run-bom", "total_queries": 1}),
                encoding="utf-8-sig",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_review_prompt.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            review_dir = run_dir / "chatgpt-review"
            self.assertTrue((review_dir / "context.md").exists())
            self.assertTrue((review_dir / "prompts.md").exists())
            context = (review_dir / "context.md").read_text(encoding="utf-8")
            self.assertIn("Run id: run-bom", context)

    def test_main_rejects_secret_like_evidence_without_writing_packet_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cards = run_dir / "evidence_cards"
            cards.mkdir()
            (cards / "Q1.md").write_text("# Q1\nStatus: partial\nToken: sk-1234567890abcdef", encoding="utf-8")
            (run_dir / "summary.json").write_text(json.dumps({"run_id": "run1", "total_queries": 1}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_review_prompt.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            review_dir = run_dir / "chatgpt-review"
            self.assertFalse((review_dir / "context.md").exists())
            self.assertFalse((review_dir / "prompts.md").exists())


if __name__ == "__main__":
    unittest.main()
