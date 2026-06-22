import json
import unittest

from bofip_agentic.eval_schema import (
    EvalQuestion,
    EvalRunConfig,
    EvalSource,
    PerQueryResult,
    ReviewAction,
    as_jsonable,
    hash_file,
    normalize_boi_family,
    redact_secrets,
)


class EvalSchemaTests(unittest.TestCase):
    def test_dataclasses_serialize_to_plain_json(self):
        config = EvalRunConfig(
            run_id="20260622-120000-smoke",
            provider="codex",
            model="gpt-5.5",
            corpus="commentary",
            question_bank="data/eval/tax_eval_50.jsonl",
            limit=3,
            lexical_only=True,
            git_commit="abc123",
            corpus_manifest_hash="sha256:manifest",
            eval_set_hash="sha256:evaluation",
        )
        payload = as_jsonable(config)
        self.assertEqual(payload["provider"], "codex")
        self.assertEqual(payload["limit"], 3)
        json.dumps(payload)

    def test_reference_family_strips_date_suffix(self):
        self.assertEqual(
            normalize_boi_family("BOI-TVA-CHAMP-20-50-20-20230621"),
            "BOI-TVA-CHAMP-20-50-20",
        )
        self.assertEqual(normalize_boi_family("BOI-RFPI-BASE-20-70"), "BOI-RFPI-BASE-20-70")

    def test_redact_secrets_removes_api_like_values(self):
        text = "key sk-1234567890abcdef and hf_abcdefghijklmnopqrstuvwxyz"
        redacted = redact_secrets(text)
        self.assertNotIn("sk-1234567890abcdef", redacted)
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertIn("[REDACTED_SECRET]", redacted)

    def test_per_query_result_carries_sources_and_trace(self):
        result = PerQueryResult(
            id="Q1",
            question="Question fiscale",
            theme="TVA",
            difficulty="medium",
            question_type="direct",
            answer_status="partial",
            coverage=0.5,
            iterations=2,
            total_s=12.3,
            conclusion="Conclusion",
            justification_bullets=["Point"],
            axes_requis=["Axe"],
            axes_couverts=[],
            axes_manquants=["Axe"],
            sources=[
                EvalSource(
                    id="src1",
                    boi_reference="BOI-TVA-CHAMP-20-50-20",
                    title="TVA",
                    section="Territorialite",
                    score=4.2,
                    snippet="Doctrine",
                )
            ],
            retrieved_docs=["BOI-TVA-CHAMP-20-50-20"],
            trace=[{"label": "Plan", "fields": []}],
        )
        payload = as_jsonable(result)
        self.assertEqual(payload["sources"][0]["boi_reference"], "BOI-TVA-CHAMP-20-50-20")
        self.assertEqual(payload["trace"][0]["label"], "Plan")

    def test_review_action_schema(self):
        action = ReviewAction(
            severity="high",
            area="retrieval",
            title="Carry useful sources",
            recommendation="Preserve useful chunks across relaunches.",
        )
        self.assertEqual(as_jsonable(action)["area"], "retrieval")
