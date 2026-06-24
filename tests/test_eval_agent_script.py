from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_eval_agent_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_agent.py"
    spec = importlib.util.spec_from_file_location("eval_agent_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class EvalAgentScriptTests(unittest.TestCase):
    def test_load_queries_normalizes_chatgpt_eval_schema(self):
        module = _load_eval_agent_module()
        row = {
            "id": "Q001",
            "domain": "RFPI",
            "type": "calculation",
            "user_question": "Question utilisateur.",
            "must_include_sources": ["BOI-RFPI-DECLA-10"],
            "should_include_sources": ["BOI-RFPI-CHAMP-10-10"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            queries = module.load_queries(path)

        self.assertEqual(queries[0]["id"], "Q001")
        self.assertEqual(queries[0]["question"], "Question utilisateur.")
        self.assertEqual(queries[0]["theme"], "RFPI")
        self.assertEqual(queries[0]["question_type"], "calculation")
        self.assertEqual(queries[0]["required_docs"], ["BOI-RFPI-DECLA-10"])
        self.assertEqual(queries[0]["optional_docs"], ["BOI-RFPI-CHAMP-10-10"])


if __name__ == "__main__":
    unittest.main()
