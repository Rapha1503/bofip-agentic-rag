import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_eval_report import summarize_eval_report


class EvalReportSummaryTests(unittest.TestCase):
    def _write_run(self, run_dir: Path) -> None:
        summary = {
            "generated_at": "2026-06-22T10:00:00+00:00",
            "config": {
                "run_id": "eval-public",
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "authorization": "Bearer sk-1234567890abcdef",
            },
            "summary": {
                "total_queries": 2,
                "supported": 1,
                "partial": 1,
                "insufficient_evidence": 0,
                "errors": 0,
                "avg_coverage": 0.75,
                "latency_s": {"avg": 12.3, "p50": 10.0, "p95": 14.6},
            },
            "per_query_path": str(run_dir / "per_query.jsonl"),
        }
        rows = [
            {
                "id": "Q1",
                "question": "Question publique",
                "theme": "RFPI",
                "difficulty": "medium",
                "question_type": "nuanced",
                "answer_status": "supported",
                "coverage": 1.0,
                "iterations": 2,
                "total_s": 10.0,
                "conclusion": "Conclusion sans secret",
                "justification_bullets": ["Bullet public"],
                "axes_requis": ["Axe requis"],
                "axes_couverts": ["Axe couvert"],
                "axes_manquants": [],
                "sources": [
                    {
                        "id": "s1",
                        "boi_reference": "BOI-RFPI-BASE-20-70",
                        "title": "Titre source",
                        "section": "Section",
                        "score": 4.2,
                        "snippet": "RAW SOURCE SNIPPET " * 40,
                    }
                ],
                "retrieved_docs": ["BOI-RFPI-BASE-20-70"],
                "trace": [{"label": "raw prompt", "content": "Authorization: Bearer sk-1234567890abcdef"}],
            },
            {
                "id": "Q2",
                "question": "Question partielle",
                "theme": "BIC",
                "difficulty": "hard",
                "question_type": "edge",
                "answer_status": "partial",
                "coverage": 0.5,
                "iterations": 1,
                "total_s": 14.6,
                "conclusion": "OPENAI_API_KEY=sk-abcdef1234567890",
                "justification_bullets": [],
                "axes_requis": ["Axe requis"],
                "axes_couverts": [],
                "axes_manquants": ["Axe manquant"],
                "sources": [],
                "retrieved_docs": [],
                "trace": [{"label": "tool", "content": "hf_abcdefghijklmnopqrstuvwxyz"}],
            },
        ]
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "per_query.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    def test_summarize_eval_report_writes_sanitized_public_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            output_dir = root / "public"
            run_dir.mkdir()
            self._write_run(run_dir)

            summarize_eval_report(run_dir, output_dir)

            expected = {
                "summary.json",
                "summary.md",
                "per_query_public.csv",
                "failure_review.md",
            }
            self.assertEqual(expected, {path.name for path in output_dir.iterdir() if path.is_file()})
            public_text = "\n".join(path.read_text(encoding="utf-8") for path in output_dir.iterdir())
            self.assertNotIn("RAW SOURCE SNIPPET", public_text)
            self.assertNotIn("trace", public_text.lower())
            self.assertNotIn("sk-1234567890abcdef", public_text)
            self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz", public_text)
            self.assertNotIn("OPENAI_API_KEY", public_text)
            self.assertNotIn("Authorization", public_text)
            self.assertIn("[REDACTED_SECRET]", public_text)

            public_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(public_summary["summary"]["total_queries"], 2)
            self.assertNotIn("authorization", json.dumps(public_summary).lower())

            with (output_dir / "per_query_public.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(["Q1", "Q2"], [row["id"] for row in rows])
            self.assertEqual("BOI-RFPI-BASE-20-70", rows[0]["retrieved_docs"])

            failure_review = (output_dir / "failure_review.md").read_text(encoding="utf-8")
            self.assertIn("Q2", failure_review)
            self.assertIn("partial", failure_review)

    def test_cli_help_documents_run_and_output_dirs(self):
        script = PROJECT_ROOT / "scripts" / "summarize_eval_report.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
