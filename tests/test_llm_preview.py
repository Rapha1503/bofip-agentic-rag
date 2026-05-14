from __future__ import annotations

import unittest
from unittest import mock

from bofip_cleanroom.llm_preview import (
    _extract_retry_delay_seconds,
    PREVIEW_ANSWER_CONTRACT_VERSION,
    PreviewAnswer,
    build_citation_prompt,
    generate_preview_answer_with_retry,
    has_api_key,
    normalize_preview_answer,
    parse_structured_preview_answer,
    preview_row_is_valid,
    review_batch_preview_payload,
    validate_structured_preview_answer,
)
from bofip_cleanroom.preview_runtime import PreviewChunk, PreviewRetrievalResult, PreviewStage1Hit


class LlmPreviewTests(unittest.TestCase):
    def _retrieval_result(self) -> PreviewRetrievalResult:
        return PreviewRetrievalResult(
            query="Qui doit reverser la TVA ?",
            lexical_query="Qui doit reverser la TVA ?",
            acronym_expansions=[],
            source_confidences={"base": 0.4},
            stage1_hits=[
                PreviewStage1Hit(
                    rank=1,
                    score=0.9,
                    boi_reference="BOI-TVA-DECLA-10-10-20-20251022",
                    title="TVA - Redevables",
                    sources=["base"],
                    ranks={"base": 1},
                )
            ],
            family_selection={"anchor_references": [], "prefixes": [], "members": []},
            stage2_chunks=[
                PreviewChunk(
                    citation_id=1,
                    boi_reference="BOI-TVA-DECLA-10-10-20-20251022",
                    title="TVA - Redevables",
                    section_path="I. Redevable",
                    chunk_id="chunk-1",
                    chunk_kind="paragraph_window",
                    text="La TVA est due par la personne qui realise l'operation.",
                    publication_date="2025-10-22",
                ),
                PreviewChunk(
                    citation_id=2,
                    boi_reference="BOI-TVA-DECLA-10-10-20-20251022",
                    title="TVA - Redevables",
                    section_path="II. Exceptions",
                    chunk_id="chunk-2",
                    chunk_kind="paragraph_window",
                    text="Des cas particuliers existent pour certaines operations.",
                    publication_date="2025-10-22",
                ),
            ],
        )

    def test_build_citation_prompt_describes_json_contract(self):
        prompt = build_citation_prompt(self._retrieval_result())

        self.assertIn("Qui doit reverser la TVA ?", prompt)
        self.assertIn("[1] BOI: BOI-TVA-DECLA-10-10-20-20251022", prompt)
        self.assertIn('"answer_status": "supported" ou "insufficient_evidence"', prompt)
        self.assertIn('"justification_bullets"', prompt)
        self.assertIn("Tu dois renvoyer un objet JSON valide et rien d'autre.", prompt)

    def test_build_citation_prompt_compact_retry_clips_evidence_and_requests_single_line_json(self):
        retrieval = PreviewRetrievalResult(
            query="Question longue",
            lexical_query="Question longue",
            acronym_expansions=[],
            source_confidences={"base": 0.4},
            stage1_hits=self._retrieval_result().stage1_hits,
            family_selection={"anchor_references": [], "prefixes": [], "members": []},
            stage2_chunks=[
                PreviewChunk(
                    citation_id=1,
                    boi_reference="BOI-TVA-DECLA-10-10-20-20251022",
                    title="TVA - Redevables",
                    section_path="I. Redevable",
                    chunk_id="chunk-1",
                    chunk_kind="paragraph_window",
                    text="mot " * 200,
                    publication_date="2025-10-22",
                )
            ],
        )

        prompt = build_citation_prompt(
            retrieval,
            validation_errors=["json output appears truncated before the closing brace"],
        )

        self.assertIn("Mode compact obligatoire", prompt)
        self.assertIn("Renvoie le JSON sur une seule ligne.", prompt)
        self.assertIn("…", prompt)
        self.assertNotIn(("mot " * 120).strip(), prompt)

    def test_parse_structured_preview_answer_accepts_json(self):
        parsed = parse_structured_preview_answer(
            """
            {
              "answer_status": "supported",
              "conclusion": "Oui.",
              "justification_bullets": [
                "La TVA est due par l'assujetti [1].",
                "Des exceptions existent [2]."
              ],
              "limits": "aucune limite majeure dans les extraits fournis."
            }
            """
        )

        self.assertEqual(parsed["parsed_from"], "json")
        self.assertEqual(parsed["structured_answer"]["answer_status"], "supported")
        self.assertEqual(len(parsed["structured_answer"]["justification_bullets"]), 2)

    def test_parse_structured_preview_answer_reports_truncated_json_explicitly(self):
        parsed = parse_structured_preview_answer(
            """
            {
              "answer_status": "insufficient_evidence",
              "conclusion": "Les extraits sont insuffisants.",
              "justification_bullets": [
                "Aucun extrait ne traite directement le sujet."
              ],
            """
        )

        self.assertIsNone(parsed["structured_answer"])
        self.assertTrue(any("truncated" in error for error in parsed["errors"]))

    def test_extract_retry_delay_seconds_prefers_provider_hint(self):
        error = RuntimeError(
            "RateLimitError: retry in 12.5s. details: {'retryDelay': '53s'}"
        )

        self.assertEqual(_extract_retry_delay_seconds(error), 53.0)

    def test_normalize_preview_answer_falls_back_to_legacy_markdown(self):
        retrieval_payload = {
            "stage2_chunks": [
                {"citation_id": 1},
                {"citation_id": 2},
            ]
        }

        answer_text, structured_answer, validation = normalize_preview_answer(
            (
                "Conclusion: Oui.\n\n"
                "Justification:\n"
                "- La TVA est due par l'assujetti [1].\n"
                "- Des exceptions existent [2].\n\n"
                "Limites: aucune limite majeure dans les extraits fournis."
            ),
            retrieval_payload=retrieval_payload,
        )

        self.assertEqual(structured_answer["answer_status"], "supported")
        self.assertEqual(validation["parsed_from"], "legacy_markdown")
        self.assertTrue(validation["valid"])
        self.assertIn("Conclusion: Oui.", answer_text)
        self.assertIn("Limites: aucune limite majeure", answer_text)

    def test_validate_structured_preview_answer_rejects_invalid_citations(self):
        validation = validate_structured_preview_answer(
            {
                "answer_status": "supported",
                "conclusion": "Oui.",
                "justification_bullets": [
                    "Premier point [1].",
                    "Deuxieme point [3].",
                ],
                "limits": "aucune limite majeure dans les extraits fournis.",
            },
            retrieval_payload={"stage2_chunks": [{"citation_id": 1}, {"citation_id": 2}]},
        )

        self.assertFalse(validation["valid"])
        self.assertTrue(any("unknown retrieval excerpts" in error for error in validation["errors"]))

    def test_review_batch_preview_payload_uses_local_validator(self):
        review = review_batch_preview_payload(
            {
                "source_report": "phase9_batch_preview_eval_gemini_v1.json",
                "rows": [
                    {
                        "case_id": "sh001",
                        "category": "answerable",
                        "answer_text": (
                            "Conclusion: Oui.\n\n"
                            "Justification:\n"
                            "- Premier point [1].\n"
                            "- Deuxieme point [2].\n\n"
                            "Limites: aucune limite majeure dans les extraits fournis."
                        ),
                        "retrieval": {"stage2_chunks": [{"citation_id": 1}, {"citation_id": 2}]},
                    },
                    {
                        "case_id": "sh002",
                        "category": "unsupported",
                        "answer_text": "Conclusion: Je ne sais pas.",
                        "retrieval": {"stage2_chunks": [{"citation_id": 1}]},
                    },
                ],
            }
        )

        self.assertEqual(review["case_count"], 2)
        self.assertEqual(review["format_valid_count"], 1)
        self.assertEqual(review["has_conclusion_count"], 2)
        self.assertEqual(review["with_any_citation_count"], 1)
        self.assertEqual(review["format_invalid_count"], 1)
        self.assertEqual(review["provider_rate_limit_count"], 0)
        self.assertEqual(review["rows"][0]["parsed_from"], "legacy_markdown")
        self.assertEqual(review["rows"][0]["failure_kind"], "valid")
        self.assertEqual(review["rows"][1]["failure_kind"], "format_invalid")
        self.assertFalse(review["rows"][1]["format_valid"])

    def test_preview_row_is_valid_requires_matching_provider_model_and_valid_flag(self):
        row = {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "answer_validation": {"valid": True},
        }

        self.assertTrue(preview_row_is_valid(row))
        self.assertTrue(preview_row_is_valid(row, provider="gemini", model="gemini-2.5-flash"))
        self.assertFalse(preview_row_is_valid(row, provider="openai"))
        self.assertFalse(preview_row_is_valid(row, model="gpt-5.4-mini"))
        self.assertFalse(preview_row_is_valid({"provider": "gemini", "model": "gemini-2.5-flash"}))
        self.assertFalse(preview_row_is_valid({"answer_validation": {"valid": False}}))

    def test_generate_preview_answer_with_retry_tracks_attempt_count_for_truncated_retry(self):
        invalid_preview = PreviewAnswer(
            provider="gemini",
            model="gemini-2.5-flash",
            answer_text='{"answer_status":"supported"',
            raw_answer_text='{"answer_status":"supported"',
            prompt_text="prompt-1",
            retrieval_payload={"stage2_chunks": [{"citation_id": 1}, {"citation_id": 2}]},
            api_called=True,
            contract_version=PREVIEW_ANSWER_CONTRACT_VERSION,
            structured_answer=None,
            answer_validation={
                "valid": False,
                "answer_status": None,
                "has_conclusion": False,
                "has_justification": False,
                "has_limits": False,
                "bullet_count": 0,
                "citation_ids": [],
                "citation_count": 0,
                "with_any_citation": False,
                "errors": ["json output appears truncated before the closing brace; return a shorter complete JSON object"],
                "warnings": [],
                "parsed_from": None,
            },
            response_metadata={"compact_prompt": False},
        )
        valid_preview = PreviewAnswer(
            provider="gemini",
            model="gemini-2.5-flash",
            answer_text=(
                "Conclusion: Oui.\n\n"
                "Justification:\n"
                "- Premier point [1].\n"
                "- Deuxieme point [2].\n\n"
                "Limites: aucune limite majeure dans les extraits fournis."
            ),
            raw_answer_text=(
                '{"answer_status":"supported","conclusion":"Oui.",'
                '"justification_bullets":["Premier point [1].","Deuxieme point [2]."],'
                '"limits":"aucune limite majeure dans les extraits fournis."}'
            ),
            prompt_text="prompt-2",
            retrieval_payload={"stage2_chunks": [{"citation_id": 1}, {"citation_id": 2}]},
            api_called=True,
            contract_version=PREVIEW_ANSWER_CONTRACT_VERSION,
            structured_answer={
                "answer_status": "supported",
                "conclusion": "Oui.",
                "justification_bullets": ["Premier point [1].", "Deuxieme point [2]."],
                "limits": "aucune limite majeure dans les extraits fournis.",
            },
            answer_validation={
                "valid": True,
                "answer_status": "supported",
                "has_conclusion": True,
                "has_justification": True,
                "has_limits": True,
                "bullet_count": 2,
                "citation_ids": [1, 2],
                "citation_count": 2,
                "with_any_citation": True,
                "errors": [],
                "warnings": [],
                "parsed_from": "json",
            },
            response_metadata={"compact_prompt": True},
        )

        call_errors: list[list[str] | None] = []

        def fake_generate(*args, **kwargs):
            validation_errors = kwargs.get("validation_errors")
            call_errors.append(None if validation_errors is None else list(validation_errors))
            return invalid_preview if len(call_errors) == 1 else valid_preview

        with mock.patch("bofip_cleanroom.llm_preview.generate_preview_answer", side_effect=fake_generate):
            with mock.patch("bofip_cleanroom.llm_preview.time.sleep", return_value=None):
                preview = generate_preview_answer_with_retry(
                    self._retrieval_result(),
                    max_attempts=3,
                    base_delay_seconds=0.0,
                )

        self.assertEqual(preview.attempt_count, 2)
        self.assertEqual(call_errors[0], None)
        self.assertIsNotNone(call_errors[1])
        self.assertTrue(any("truncated" in error for error in call_errors[1]))

    def test_has_api_key_supports_gemini_provider(self):
        with mock.patch("bofip_cleanroom.llm_preview.load_default_env_files", return_value={"GEMINI_API_KEY": "x"}):
            with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "x"}, clear=False):
                self.assertTrue(has_api_key("gemini"))
                self.assertFalse(has_api_key("openai"))


if __name__ == "__main__":
    unittest.main()
