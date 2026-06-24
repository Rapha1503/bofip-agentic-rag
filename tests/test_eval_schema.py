from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_schema import (
    AgenticScores,
    EvalRunConfig,
    PerQueryResult,
    as_jsonable,
    boi_matches,
    contains_secret,
    hash_file,
    normalize_boi_family,
    repair_mojibake,
    redact_secrets,
)


class EvalSchemaTests(unittest.TestCase):
    def test_dataclasses_serialize_to_plain_json(self):
        config = EvalRunConfig(
            run_id="20260624_eval",
            provider="codex",
            model="gpt-5.5",
            judge_provider="none",
            judge_model="",
            corpus="commentary",
            question_bank="data/eval/chatgpt_50_cases_v1.jsonl",
            limit=3,
            sample=3,
            seed=42,
            retrieval_mode="lexical",
            reranker=False,
            device="cpu",
            max_iterations=2,
        )
        payload = as_jsonable(config)
        self.assertEqual(payload["provider"], "codex")
        json.dumps(payload)

    def test_reference_family_strips_date_suffix(self):
        self.assertEqual(normalize_boi_family("BOI-TVA-CHAMP-20-50-20-20230621"), "BOI-TVA-CHAMP-20-50-20")
        self.assertTrue(boi_matches("BOI-TVA-CHAMP-20-50-20-20230621", "BOI-TVA-CHAMP-20-50-20"))
        self.assertTrue(boi_matches("BOI-ENR-DMTG-10-40-10-50", "BOI-ENR-DMTG-10-40-10"))
        self.assertTrue(boi_matches("BOI-ENR-DMTG-10-40-10", "BOI-ENR-DMTG-10-40-10-50"))
        self.assertFalse(boi_matches("BOI-TVA-CHAMP-20-50-10", "BOI-TVA-CHAMP-20-50-20"))

    def test_secret_redaction_detects_api_like_values(self):
        text = "DEEPSEEK_API_KEY=" + "sk-" + "1234567890abcdef and hf_" + "abcdefghijklmnopqrstuvwxyz"
        redacted = redact_secrets(text)
        self.assertNotIn("sk-" + "1234567890abcdef", redacted)
        self.assertIn("[REDACTED_SECRET]", redacted)
        self.assertTrue(contains_secret(text))

    def test_repair_mojibake_fixes_legacy_french_labels(self):
        self.assertEqual(repair_mojibake("Question posÃ©e au modÃ¨le"), "Question posée au modèle")

    def test_per_query_result_serializes_nested_scores(self):
        result = PerQueryResult(
            id="Q001",
            question="Question",
            answer_status="supported",
            scores=AgenticScores(required_doc_recall=1.0, has_plan=True),
        )
        payload = as_jsonable(result)
        self.assertEqual(payload["scores"]["required_doc_recall"], 1.0)
        json.dumps(payload)

    def test_hash_file_is_stable_sha256_prefixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.txt"
            path.write_text("abc", encoding="utf-8")
            self.assertTrue(hash_file(path).startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
