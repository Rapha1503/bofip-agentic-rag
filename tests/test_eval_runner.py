import json
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_runner import (
    build_run_id,
    compute_basic_summary,
    load_question_bank,
    source_from_agent_chunk,
)


class EvalRunnerTests(unittest.TestCase):
    def test_load_question_bank_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "Q1", "question": "A", "theme": "TVA"}),
                        json.dumps({"id": "Q2", "question": "B", "theme": "ENR"}),
                    ]
                ),
                encoding="utf-8",
            )
            questions = load_question_bank(path, limit=1)
            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0].id, "Q1")

    def test_summary_counts_statuses_and_coverage(self):
        summary = compute_basic_summary(
            [
                {"answer_status": "supported", "coverage": 1.0, "total_s": 10.0},
                {"answer_status": "partial", "coverage": 0.5, "total_s": 20.0},
            ]
        )
        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["supported"], 1)
        self.assertEqual(summary["partial"], 1)
        self.assertEqual(summary["avg_coverage"], 0.75)
        self.assertEqual(summary["latency_s"]["p50"], 20.0)

    def test_source_from_agent_chunk_handles_missing_fields(self):
        source = source_from_agent_chunk(
            {
                "chunk_id": "c1",
                "boi_reference": "BOI-TVA",
                "title": "Title",
                "score": 3,
                "text": "Long text",
            }
        )
        self.assertEqual(source.id, "c1")
        self.assertEqual(source.boi_reference, "BOI-TVA")
        self.assertEqual(source.snippet, "Long text")

    def test_build_run_id_is_filesystem_safe(self):
        run_id = build_run_id("smoke test")
        self.assertNotIn(" ", run_id)
        self.assertIn("smoke-test", run_id)


if __name__ == "__main__":
    unittest.main()
