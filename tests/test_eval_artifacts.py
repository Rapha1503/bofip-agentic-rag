import json
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_artifacts import (
    assert_no_secrets,
    write_evidence_card,
    write_json,
    write_jsonl,
    write_summary_markdown,
)
from bofip_agentic.eval_schema import EvalSource, PerQueryResult


class EvalArtifactsTests(unittest.TestCase):
    def test_write_json_and_jsonl_use_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "summary.json", {"text": "reponse sourcee"})
            write_jsonl(root / "rows.jsonl", [{"id": "A"}, {"id": "B"}])
            self.assertIn("reponse", (root / "summary.json").read_text(encoding="utf-8"))
            rows = (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0])["id"], "A")

    def test_evidence_card_contains_answer_sources_and_axes(self):
        result = PerQueryResult(
            id="Q1",
            question="Question fiscale",
            theme="RFPI",
            difficulty="medium",
            question_type="nuanced",
            answer_status="partial",
            coverage=0.67,
            iterations=2,
            total_s=20.0,
            conclusion="Conclusion",
            justification_bullets=["Bullet"],
            axes_requis=["Micro-foncier"],
            axes_couverts=["Location nue"],
            axes_manquants=["Charges"],
            sources=[
                EvalSource(
                    id="s1",
                    boi_reference="BOI-RFPI-BASE-20-70",
                    title="Charges",
                    section="Copro",
                    score=3.0,
                    snippet="Provisions",
                )
            ],
            retrieved_docs=["BOI-RFPI-BASE-20-70"],
            trace=[{"label": "Review", "fields": [{"label": "Axes", "value": "Charges"}]}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Q1.md"
            write_evidence_card(path, result)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Question fiscale", text)
            self.assertIn("BOI-RFPI-BASE-20-70", text)
            self.assertIn("Axes manquants", text)

    def test_summary_markdown_lists_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            write_summary_markdown(
                path,
                {"total_queries": 2, "supported": 1, "partial": 1, "avg_coverage": 0.75},
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("Total queries", text)
            self.assertIn("75%", text)

    def test_secret_scan_rejects_keys(self):
        with self.assertRaises(ValueError):
            assert_no_secrets("DEEPSEEK_API_KEY=sk-1234567890abcdef")
