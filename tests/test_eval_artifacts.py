from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_artifacts import (
    assert_no_secrets,
    compute_summary,
    write_evidence_card,
    write_json,
    write_public_csv,
    write_summary_markdown,
)
from bofip_agentic.eval_schema import AgenticScores, EvalSource, PerQueryResult


class EvalArtifactsTests(unittest.TestCase):
    def test_write_json_uses_utf8_and_rejects_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            write_json(path, {"text": "Réponse sourcée"})
            self.assertIn("Réponse", path.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                write_json(path, {"key": "sk-" + "1234567890abcdef"})

    def test_evidence_card_contains_agentic_metrics_and_sources(self):
        result = PerQueryResult(
            id="Q001",
            question="Question fiscale",
            theme="TVA",
            answer_status="supported",
            auto_verdict="candidate_pass",
            coverage=1.0,
            iterations=2,
            total_s=12.5,
            conclusion="Conclusion.",
            scores=AgenticScores(required_doc_recall=1.0, trace_score=1.0, has_plan=True, has_retrieval=True),
            sources=[EvalSource(chunk_id="c1", boi_reference="BOI-TVA-BASE-10", title="TVA", snippet="Doctrine")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_evidence_card(Path(tmp) / "Q001.md", result)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Agentic Coverage", text)
        self.assertIn("BOI-TVA-BASE-10", text)

    def test_summary_and_public_csv_are_github_friendly(self):
        result = PerQueryResult(
            id="Q001",
            question="Question fiscale",
            theme="TVA",
            answer_status="supported",
            auto_verdict="candidate_pass",
            coverage=1.0,
            total_s=3,
            scores=AgenticScores(required_doc_recall=1.0, trace_score=1.0),
        )
        summary = compute_summary([result])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary_markdown(root / "summary.md", summary, title="Eval")
            write_public_csv(root / "per_query_public.csv", [result])
            self.assertIn("Gold metadata", (root / "summary.md").read_text(encoding="utf-8"))
            csv_text = (root / "per_query_public.csv").read_text(encoding="utf-8")
            self.assertIn("required_doc_recall", csv_text)
            self.assertIn("effective_verdict", csv_text)

    def test_assert_no_secrets_rejects_key_like_text(self):
        with self.assertRaises(ValueError):
            assert_no_secrets("authorization: Bearer sk-" + "1234567890abcdef")


if __name__ == "__main__":
    unittest.main()
