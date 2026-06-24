from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_runner import (
    auto_verdict,
    build_run_id,
    load_question_bank,
    per_query_from_agent_result,
    run_eval,
)
from bofip_agentic.eval_schema import EvalQuestion, PerQueryResult


class FakeAgent:
    def __init__(self):
        self.seen_questions: list[str] = []

    def run(self, question: str) -> dict:
        self.seen_questions.append(question)
        return {
            "answer_status": "supported",
            "coverage": 1.0,
            "iterations": 1,
            "total_s": 0.1,
            "conclusion": "Conclusion sourcée.",
            "justification_bullets": ["Point cité."],
            "axes_requis": ["Axe"],
            "axes_couverts": ["Axe"],
            "axes_manquants": [],
            "sources": [
                {
                    "chunk_id": "c1",
                    "boi_reference": "BOI-TVA-BASE-10-20240101",
                    "title": "TVA",
                    "section_path": ["Base"],
                    "score": 4.2,
                    "text": "Doctrine BOFiP.",
                    "retrieval_stage": "final",
                }
            ],
            "trace": [{"stage": "plan_and_route", "routes": [{"facet": "TVA"}], "source_review": {"covered_axes": ["Axe"]}}],
            "step_timings": [
                {"label": "Plan fiscal produit"},
                {"label": "Recherche par axe"},
                {"label": "Critique des sources"},
                {"label": "Question posée au modèle de réponse"},
            ],
        }


class EvalRunnerTests(unittest.TestCase):
    def test_load_question_bank_uses_user_question_schema_and_sample(self):
        rows = [
            {"id": f"Q{i:03d}", "domain": "TVA", "user_question": f"Question {i}", "must_include_sources": ["BOI-TVA-BASE-10"]}
            for i in range(1, 8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            questions = load_question_bank(path, sample=3, seed=7)
        self.assertEqual(len(questions), 3)
        self.assertNotEqual([q.id for q in questions], ["Q001", "Q002", "Q003"])
        self.assertEqual(questions[0].theme, "TVA")

    def test_load_question_bank_supports_chatgpt_web_v2_json_schema(self):
        payload = {
            "name": "bofip_agentic_rag_50_human_questions_v2",
            "cases": [
                {
                    "id": "CASE-001",
                    "theme": "TVA",
                    "difficulty": "intermediate",
                    "question_type": "option_regime",
                    "runtime_question": "Question runtime uniquement.",
                    "gold_eval": {
                        "expected_status": "supported_with_limits",
                        "expected_answer_points": ["NE PAS ENVOYER AU RUNTIME"],
                        "expected_calculation": "Calcul attendu.",
                        "expected_bofip_refs": {
                            "must_include": ["BOI-TVA-CHAMP-30-10-50"],
                            "should_include": ["BOI-TVA-CHAMP-50-10"],
                            "avoid_as_primary": ["BOI-CF"],
                        },
                        "failure_signals": ["Signal externe."],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            questions = load_question_bank(path)

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].id, "CASE-001")
        self.assertEqual(questions[0].question, "Question runtime uniquement.")
        self.assertEqual(questions[0].required_docs, ["BOI-TVA-CHAMP-30-10-50"])
        self.assertEqual(questions[0].optional_docs, ["BOI-TVA-CHAMP-50-10"])
        self.assertEqual(questions[0].expected_answer_core, ["NE PAS ENVOYER AU RUNTIME"])

    def test_load_question_bank_supports_runtime_question_jsonl_schema(self):
        row = {"id": "CASE-001", "runtime_question": "Question runtime uniquement."}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            questions = load_question_bank(path)

        self.assertEqual(questions[0].id, "CASE-001")
        self.assertEqual(questions[0].question, "Question runtime uniquement.")

    def test_per_query_result_scores_trace_and_doc_recall(self):
        question = EvalQuestion(id="Q001", question="Question", required_docs=["BOI-TVA-BASE-10"])
        result = per_query_from_agent_result(question, FakeAgent().run("Question"))
        self.assertEqual(result.scores.required_doc_recall, 1.0)
        self.assertGreaterEqual(result.scores.trace_score, 0.75)
        self.assertEqual(auto_verdict(result), "candidate_pass")

    def test_run_eval_sends_only_question_to_agent_and_writes_artifacts(self):
        row = {
            "id": "Q001",
            "domain": "TVA",
            "user_question": "Question runtime uniquement.",
            "expected_answer_core": ["NE PAS ENVOYER AU RUNTIME"],
            "failure_signals": ["NE PAS ENVOYER NON PLUS"],
            "must_include_sources": ["BOI-TVA-BASE-10"],
        }
        fake_agent = FakeAgent()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "bank.jsonl"
            bank.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            result = run_eval(
                question_bank=bank,
                output_dir=root / "run",
                provider="codex",
                sample=0,
                limit=1,
                runtime_factory=lambda **_: object(),
                agent_factory=lambda **_: fake_agent,
            )

            self.assertEqual(fake_agent.seen_questions, ["Question runtime uniquement."])
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "per_query" / "Q001.json").exists())
            self.assertTrue((run_dir / "per_query_public.csv").exists())
            self.assertNotIn("NE PAS ENVOYER", "\n".join(fake_agent.seen_questions))

    def test_run_eval_resolves_provider_api_key_from_environment_without_artifact_leak(self):
        row = {
            "id": "Q001",
            "domain": "TVA",
            "user_question": "Question runtime uniquement.",
            "must_include_sources": ["BOI-TVA-BASE-10"],
        }
        captured: dict[str, str] = {}

        def factory(**kwargs):
            captured["api_key"] = kwargs.get("api_key", "")
            return FakeAgent()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "bank.jsonl"
            bank.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            previous = os.environ.get("DEEPSEEK_API_KEY")
            os.environ["DEEPSEEK_API_KEY"] = "sk-" + "1234567890abcdef"
            try:
                result = run_eval(
                    question_bank=bank,
                    output_dir=root / "run",
                    provider="deepseek",
                    limit=1,
                    runtime_factory=lambda **_: object(),
                    agent_factory=factory,
                )
            finally:
                if previous is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = previous

            self.assertEqual(captured["api_key"], "sk-" + "1234567890abcdef")
            artifact_text = (Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("1234567890abcdef", artifact_text)

    def test_build_run_id_is_filesystem_safe(self):
        self.assertRegex(build_run_id("Eval DeepSeek / hybrid"), r"^\d{8}_\d{6}_eval-deepseek-hybrid$")


if __name__ == "__main__":
    unittest.main()
