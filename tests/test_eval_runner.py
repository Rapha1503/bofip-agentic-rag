import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bofip_agentic.eval_runner import (
    _CodexCliClient,
    build_run_id,
    compute_basic_summary,
    load_question_bank,
    run_eval,
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

    def test_run_eval_uses_safe_artifact_names_for_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "bank.jsonl"
            bank.write_text(
                json.dumps({"id": "../escape", "question": "Question?", "theme": "TVA"}) + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"

            self._run_mocked_eval(bank=bank, run_dir=run_dir, root=root)

            self.assertFalse((run_dir / "escape.json").exists())
            self.assertFalse((run_dir / "escape.md").exists())
            trace_files = list((run_dir / "traces").glob("*.json"))
            evidence_files = list((run_dir / "evidence_cards").glob("*.md"))
            self.assertEqual(len(trace_files), 1)
            self.assertEqual(len(evidence_files), 1)
            self.assertEqual(trace_files[0].resolve().parent, (run_dir / "traces").resolve())
            self.assertEqual(evidence_files[0].resolve().parent, (run_dir / "evidence_cards").resolve())

    def test_run_eval_resume_filters_rows_to_active_question_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "bank.jsonl"
            bank.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "Q1", "question": "Active?", "theme": "TVA"}),
                        json.dumps({"id": "Q2", "question": "Limited out?", "theme": "TVA"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "per_query.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "Q1", "answer_status": "supported", "coverage": 1, "total_s": 1}),
                        json.dumps({"id": "STALE", "answer_status": "supported", "coverage": 1, "total_s": 1}),
                        json.dumps({"id": "Q2", "answer_status": "supported", "coverage": 1, "total_s": 1}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runtime_builder = Mock()
            fake_rag_runtime = types.ModuleType("bofip_agentic.rag_runtime")
            fake_rag_runtime.RagRuntime = SimpleNamespace(from_local_corpus=runtime_builder)
            with patch.dict(sys.modules, {"bofip_agentic.rag_runtime": fake_rag_runtime}):
                run_eval(
                    question_bank=bank,
                    run_dir=run_dir,
                    run_id="resume",
                    limit=1,
                    resume=True,
                    project_root=root,
                )

            runtime_builder.assert_not_called()
            rows = [
                json.loads(line)
                for line in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["id"] for row in rows], ["Q1"])
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["total_queries"], 1)

    def test_run_eval_writes_artifacts_and_passes_only_question_to_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "bank.jsonl"
            bank.write_text(
                json.dumps(
                    {
                        "id": "Q1",
                        "question": "Quelle TVA appliquer?",
                        "theme": "TVA",
                        "required_docs": ["BOI-SECRET-GOLD"],
                        "must_include": ["gold metadata"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"

            agent = self._run_mocked_eval(bank=bank, run_dir=run_dir, root=root)

            agent.run.assert_called_once_with("Quelle TVA appliquer?")
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "per_query.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "traces" / "Q1.json").exists())
            self.assertTrue((run_dir / "evidence_cards" / "Q1.md").exists())

    def test_codex_cli_error_includes_output_context_without_prompt(self):
        prompt_secret = "prompt must stay private"

        def fake_run(*args, **kwargs):
            self.assertIn(prompt_secret, kwargs["input"])
            return SimpleNamespace(
                returncode=17,
                stdout=f"stdout detail {prompt_secret}",
                stderr=f"stderr detail {prompt_secret}",
            )

        client = _CodexCliClient()
        with patch("subprocess.run", side_effect=fake_run):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat.completions.create(messages=[{"role": "user", "content": prompt_secret}])

        message = str(ctx.exception)
        self.assertIn("exit 17", message)
        self.assertIn("stdout detail", message)
        self.assertIn("stderr detail", message)
        self.assertNotIn(prompt_secret, message)

    def _run_mocked_eval(self, *, bank: Path, run_dir: Path, root: Path):
        agent = SimpleNamespace()
        agent.run = Mock(
            return_value={
                "answer_status": "supported",
                "coverage": 1.0,
                "iterations": 1,
                "total_s": 0.2,
                "conclusion": "Conclusion",
                "justification_bullets": ["Justification"],
                "axes_requis": ["Axe"],
                "axes_couverts": ["Axe"],
                "axes_manquants": [],
                "sources": [
                    {
                        "chunk_id": "chunk-1",
                        "boi_reference": "BOI-TVA",
                        "title": "Title",
                        "score": 1,
                        "text": "Snippet",
                    }
                ],
                "trace": [{"label": "done"}],
            }
        )
        fake_agent_rag = types.ModuleType("bofip_agentic.agent_rag")
        fake_agent_rag.AgenticRAG = Mock(return_value=agent)
        fake_rag_runtime = types.ModuleType("bofip_agentic.rag_runtime")
        fake_rag_runtime.RagRuntime = SimpleNamespace(from_local_corpus=Mock(return_value=object()))
        with patch.dict(
            sys.modules,
            {
                "bofip_agentic.agent_rag": fake_agent_rag,
                "bofip_agentic.rag_runtime": fake_rag_runtime,
            },
        ):
            run_eval(
                question_bank=bank,
                run_dir=run_dir,
                run_id="test-run",
                project_root=root,
                provider="codex",
            )
        return agent


if __name__ == "__main__":
    unittest.main()
