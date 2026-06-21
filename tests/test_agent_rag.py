from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from bofip_agentic.agent_rag import AgenticRAG, _classify_domain


class _FakeChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeCompletions:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return SimpleNamespace(choices=[_FakeChoice(self.responses.pop(0))])


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def retrieve(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        hit = SimpleNamespace(rank=1, score=1.0, boi_reference="BOI-TVA-BASE", title="TVA")
        chunk = SimpleNamespace(
            boi_reference="BOI-TVA-BASE",
            title="TVA",
            publication_date="2026-01-01",
            section_path="I. Champ",
            text="Extrait BOFiP.",
            chunk_id=f"chunk-{len(self.calls)}",
        )
        return SimpleNamespace(stage1_hits=[hit], stage2_chunks=[chunk])


class AgenticRAGTests(unittest.TestCase):
    def test_domain_classifier_extracts_boi_prefix_from_sentence(self):
        def call_llm(prompt, system, json_mode=True):
            return {"_raw": "Le pr?fixe le plus pr?cis est BOI-TVA-LIQ-30-20-20."}

        self.assertEqual(_classify_domain("question", call_llm), "TVA-LIQ-30-20-20")

    def test_domain_classifier_falls_back_to_family_keywords(self):
        def call_llm(prompt, system, json_mode=True):
            return {"_raw": ""}

        self.assertEqual(_classify_domain("Quel taux de TVA appliquer ?", call_llm), "TVA")

    def test_missing_axes_trigger_reformulation_and_second_retrieval(self):
        responses = [
            "TVA",
            json.dumps(
                {
                    "answer_status": "partial",
                    "conclusion": "Réponse partielle.",
                    "axes_requis": ["taux applicable"],
                    "axes_couverts": [],
                    "axes_manquants": ["taux applicable"],
                    "justification_bullets": [],
                    "limits": "",
                }
            ),
            json.dumps({"bofip_family": "TVA", "search_query": "taxe valeur ajoutee taux applicable"}),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Réponse supportée.",
                    "axes_requis": ["taux applicable"],
                    "axes_couverts": ["taux applicable"],
                    "axes_manquants": [],
                    "justification_bullets": ["Axe couvert."],
                    "limits": "",
                }
            ),
        ]
        runtime = _FakeRuntime()
        agent = AgenticRAG(runtime, client=_FakeClient(responses), max_iterations=2, use_reranker=False)

        result = agent.run("Quel taux de TVA appliquer ?")

        self.assertEqual(result["answer_status"], "supported")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(len(runtime.calls), 2)
        self.assertTrue(runtime.calls[0]["query"].startswith("TVA "))
        self.assertEqual(runtime.calls[0]["boost_prefix"], "TVA")
        self.assertFalse(runtime.calls[0]["use_reranker"])
        self.assertIn("taxe valeur ajoutee taux applicable", runtime.calls[1]["query"])
        self.assertEqual(result["trace"][0]["domain_prefix"], "TVA")
        self.assertIn("reformulated_query", result["trace"][0])
        self.assertEqual(result["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
