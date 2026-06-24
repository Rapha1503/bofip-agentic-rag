from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from bofip_agentic.agent_rag import (
    AgenticRAG,
    SearchFacet,
    SearchPlan,
    _build_facet_chunk_query,
    _build_expected_evidence_chunk_query,
    _build_evidence_matrix,
    _build_answer_question,
    _build_facet_retrieval_query,
    _build_retrieval_header,
    _build_retrieval_query,
    _candidate_refs_for_missing_axis,
    _chunks_for_source_review,
    _clean_answer_status,
    _complete_source_review_with_plan_gaps,
    _compute_coverage,
    _fallback_domain_from_question,
    _fallback_plan,
    _normalize_prefix,
    _normalize_plan,
    _normalize_source_review,
    _select_reviewed_chunks,
)
from bofip_agentic.prompt_utils import build_system_prompt


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
        self.intra_calls = []
        self.intra_results = []

    def retrieve(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        call_index = len(self.calls)
        prefix = (kwargs.get("boost_prefix") or "GEN").strip() or "GEN"
        boi_reference = f"BOI-{prefix}-DOC"
        hit = SimpleNamespace(rank=1, score=1.0, boi_reference=boi_reference, title=f"Source {prefix}")
        chunk = SimpleNamespace(
            boi_reference=boi_reference,
            title=f"Source {prefix}",
            publication_date="2026-01-01",
            section_path="I. Champ",
            text="Extrait BOFiP.",
            chunk_id=f"chunk-{call_index}",
        )
        return SimpleNamespace(query=query, stage1_hits=[hit], stage2_chunks=[chunk])

    def retrieve_within_documents(self, query: str, boi_references: list[str], **kwargs):
        self.intra_calls.append({"query": query, "boi_references": boi_references, **kwargs})
        if self.intra_results:
            return self.intra_results.pop(0)
        return SimpleNamespace(query=query, stage1_hits=[], stage2_chunks=[])


class AgenticRAGTests(unittest.TestCase):
    def test_fallback_taxonomy_is_generic_not_question_specific(self):
        self.assertEqual(_fallback_domain_from_question("Honoraires en micro-BNC"), "BNC")
        self.assertEqual(_fallback_domain_from_question("Vente de titres avec moins-value sur CTO"), "RPPM")
        self.assertEqual(
            _fallback_domain_from_question("Location nue avec loyers, interets d'emprunt et micro-foncier"),
            "RFPI",
        )

    def test_fallback_taxonomy_keeps_employment_income_at_family_level(self):
        question = "Dans quelles conditions une indemnite de rupture conventionnelle est exoneree d'impot ?"

        self.assertEqual(_normalize_prefix("BOI-RSA-CHAMP-20-40-10"), "RSA-CHAMP-20-40-10")
        self.assertEqual(_fallback_domain_from_question(question), "RSA")

        header = _build_retrieval_header(question, "")
        self.assertIn("RSA", header)
        self.assertNotIn("RSA-CHAMP-20-40-10-30", header)
        self.assertNotIn("rupture conventionnelle", header.lower())
        self.assertIn("rupture conventionnelle", _build_retrieval_query(question, "").lower())

    def test_fallback_taxonomy_routes_common_questions_to_families_only(self):
        cases = [
            (
                "Quelle est la regle de territorialite TVA pour une prestation B2B intracommunautaire ?",
                "TVA",
            ),
            (
                "Comment sont imposes les revenus d'une location meublee non professionnelle ?",
                "BIC",
            ),
            (
                "Quelles conditions permettent l'exoneration de la plus-value sur residence principale ?",
                "RFPI",
            ),
            (
                "Dans quelles conditions l'indemnite versee a un dirigeant mandataire social lors de la cessation de ses fonctions est exoneree ?",
                "RSA",
            ),
            (
                "Quel est le regime d'exoneration d'une indemnite de licenciement versee hors plan de sauvegarde de l'emploi ?",
                "RSA",
            ),
        ]

        for question, expected_prefix in cases:
            with self.subTest(question=question):
                self.assertEqual(_fallback_domain_from_question(question), expected_prefix)
                self.assertIn(expected_prefix, _build_retrieval_header(question, ""))

    def test_fallback_plan_stays_taxonomy_level_for_specific_tax_question(self):
        question = (
            "Je loue un appartement vide, j'ai des loyers, des travaux de peinture "
            "et des charges de copropriete. Est-ce que je reste au micro-foncier "
            "ou est-ce que le regime reel est preferable ?"
        )

        plan = _fallback_plan(question)

        facet_names = [facet.name for facet in plan.facets]
        self.assertIn("revenus fonciers et plus-values immobilières", facet_names)
        evidence_text = " ".join(" ".join(facet.expected_evidence) for facet in plan.facets)
        self.assertIn("règle applicable", evidence_text)
        self.assertNotIn("abattement representatif", evidence_text)
        self.assertNotIn("charges deductibles", evidence_text)
        self.assertNotIn("30", evidence_text)

    def test_facet_chunk_query_excludes_user_amount_noise(self):
        question = (
            "Micro-entrepreneur cree en 2024, chiffre affaires 3200 euros, "
            "avis CFE 2025, exonere ou cotisation minimum ?"
        )
        facet = SearchFacet(
            name="CFE faible chiffre d'affaires",
            goal="Verifier cotisation minimum et faible chiffre d'affaires",
            query="cotisation minimum faible chiffre affaires",
            prefix="IF",
            expected_evidence=["exoneration faible chiffre d'affaires"],
        )

        document_query = _build_facet_retrieval_query(question, facet)
        chunk_query = _build_facet_chunk_query(facet, question)

        self.assertIn("3200", document_query)
        self.assertNotIn("3200", chunk_query)
        self.assertIn("chiffre", chunk_query)
        self.assertIn("cotisation minimum", chunk_query)
        self.assertIn("faible chiffre", chunk_query)

    def test_expected_evidence_query_expands_abattement_forfaitaire_without_rate_hardcoding(self):
        facet = SearchFacet(
            name="Calcul comparatif micro-foncier",
            goal="Comparer les regimes",
            query="micro-foncier regime reel",
            prefix="RFPI",
            expected_evidence=["abattement forfaitaire micro-foncier"],
        )

        query = _build_expected_evidence_chunk_query(facet, "abattement forfaitaire micro-foncier")

        self.assertIn("application abattement", query)
        self.assertIn("conditions abattement", query)
        self.assertNotIn("30", query)

    def test_single_expected_evidence_rescue_runs_for_forfait_calculation(self):
        runtime = _FakeRuntime()
        runtime.intra_results = [
            SimpleNamespace(
                query="forfait frais",
                stage1_hits=[],
                stage2_chunks=[
                    SimpleNamespace(
                        boi_reference="BOI-RFPI-PVI",
                        title="Frais",
                        publication_date="2026-01-01",
                        section_path="Frais d'acquisition",
                        text="Preuve du forfait applicable.",
                        chunk_id="forfait-proof",
                    )
                ],
            )
        ]
        agent = AgenticRAG(runtime, client=_FakeClient([]), max_iterations=1, use_reranker=False)
        facet = SearchFacet(
            name="Frais d'acquisition",
            goal="Trouver le forfait applicable",
            query="frais acquisition forfait 7.5% plus-value immobiliere",
            prefix="RFPI-PVI",
            expected_evidence=["Option forfaitaire ou reelle applicable"],
            role="calculation",
        )

        _result, chunks = agent._retrieve_for_facet(facet, "Question fiscale.", set())

        self.assertEqual(len(runtime.intra_calls), 1)
        self.assertIn("frais acquisition forfait", runtime.intra_calls[0]["query"])
        self.assertIn("forfait-proof", [chunk["chunk_id"] for chunk in chunks])

    def test_facet_retrieval_query_filters_taxonomy_hints_to_requested_family(self):
        question = (
            "Micro-entrepreneur francais en franchise en base, prestation de conseil "
            "pour une societe allemande avec numero TVA intracommunautaire valide."
        )
        facet = SearchFacet(
            name="Territorialite TVA services B2B",
            goal="Verifier le lieu des prestations entre assujettis",
            query="territorialite TVA prestation services B2B preneur assujetti regle generale",
            prefix="TVA",
            expected_evidence=["lieu du preneur assujetti"],
        )

        query = _build_facet_retrieval_query(question, facet)

        self.assertIn("TVA", query)
        self.assertNotIn("BIC", query)
        self.assertNotIn("bénéfices industriels", query)

    def test_tva_taxonomy_header_stays_neutral_for_facturation_facets(self):
        facet = SearchFacet(
            name="Autoliquidation facture intracommunautaire",
            goal="Verifier la mention autoliquidation et le redevable par le preneur",
            query="facture prestation services intracommunautaire autoliquidation preneur redevable mention facture",
            prefix="TVA",
            expected_evidence=["mention autoliquidation", "preneur redevable"],
        )

        query = _build_facet_chunk_query(facet).lower()

        self.assertNotIn("taux", query)
        self.assertNotIn("livraison", query)
        self.assertNotIn("exonération", query)

    def test_retrieval_query_can_combine_market_and_pea_facets_without_answer_hardcoding(self):
        question = "CTO plus value 19000 moins value 5000 et gain PEA apres quatre ans"
        header = _build_retrieval_header(question, "RPPM-PVBMI")
        query = _build_retrieval_query(question, "RPPM-PVBMI")

        self.assertIn("RPPM-PVBMI", header)
        self.assertNotIn("RPPM-RCM-40-50", header)
        self.assertNotIn(question, header)
        self.assertIn("RPPM-PVBMI", query)
        self.assertIn(question, query)
        self.assertNotIn("14000", query)
        self.assertNotIn("749", query)

    def test_public_progress_events_are_decision_oriented(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Quel taux de TVA appliquer ?",
                    "facts": ["question sur un taux de TVA"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "TVA applicable",
                            "goal": "Identifier le taux applicable",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA taux applicable operation",
                            "priority": 1,
                            "expected_evidence": ["taux applicable"],
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": ["TVA applicable"],
                    "missing_axes": [],
                }
            ),
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
        events = []
        runtime = _FakeRuntime()
        agent = AgenticRAG(
            runtime,
            client=_FakeClient(responses),
            max_iterations=2,
            use_reranker=False,
            progress_callback=lambda label, payload: events.append((label, payload)),
        )

        agent.run("Quel taux de TVA appliquer ?")

        labels = [label for label, _payload in events]
        self.assertIn("Question posée au planneur fiscal", labels)
        self.assertIn("Plan fiscal produit", labels)
        self.assertIn("Recherche par axe", labels)
        self.assertIn("Critique des sources", labels)
        self.assertIn("Question posée au modèle de réponse", labels)
        self.assertIn("Auto-évaluation de couverture", labels)
        self.assertNotIn("Recherche documentaire", labels)
        self.assertFalse(any("BM25" in label for label in labels))
        self.assertTrue(any(payload.get("fields") for _label, payload in events))
        self.assertTrue(all("step_s" in payload for _label, payload in events))
        self.assertTrue(all("elapsed_s" in payload for _label, payload in events))

    def test_agent_result_exposes_step_timings_without_ui_callback(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Quel taux de TVA appliquer ?",
                    "facts": ["question sur un taux de TVA"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "TVA applicable",
                            "goal": "Identifier le taux applicable",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA taux applicable operation",
                            "priority": 1,
                            "expected_evidence": ["taux applicable"],
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": ["TVA applicable"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Reponse supportee.",
                    "axes_requis": ["taux applicable"],
                    "axes_couverts": ["taux applicable"],
                    "axes_manquants": [],
                    "justification_bullets": ["Axe couvert."],
                    "limits": "",
                }
            ),
        ]
        agent = AgenticRAG(_FakeRuntime(), client=_FakeClient(responses), max_iterations=1, use_reranker=False)

        result = agent.run("Quel taux de TVA appliquer ?")

        timings = result.get("step_timings", [])
        self.assertGreaterEqual(len(timings), 6)
        self.assertEqual(timings[0]["label"], "Question posée au planneur fiscal")
        self.assertEqual(timings[-1]["label"], "Auto-évaluation de couverture")
        self.assertTrue(all(isinstance(item["step_s"], float) for item in timings))
        self.assertTrue(all(isinstance(item["elapsed_s"], float) for item in timings))
        self.assertEqual(result["trace"][0]["step_timings"], timings)

    def test_facet_retrieval_rescues_each_expected_evidence_inside_candidate_docs(self):
        runtime = _FakeRuntime()
        runtime.intra_results = [
            SimpleNamespace(
                query="seuil applicable",
                stage1_hits=[],
                stage2_chunks=[
                    SimpleNamespace(
                        boi_reference="BOI-TVA-DOC",
                        title="Source TVA",
                        publication_date="2026-01-01",
                        section_path="I. Seuil",
                        text="Preuve du seuil applicable.",
                        chunk_id="evidence-seuil",
                    )
                ],
            ),
            SimpleNamespace(
                query="taux applicable",
                stage1_hits=[],
                stage2_chunks=[
                    SimpleNamespace(
                        boi_reference="BOI-TVA-DOC",
                        title="Source TVA",
                        publication_date="2026-01-01",
                        section_path="II. Taux",
                        text="Preuve du taux applicable.",
                        chunk_id="evidence-taux",
                    )
                ],
            ),
        ]
        agent = AgenticRAG(runtime, client=_FakeClient([]), max_iterations=1, use_reranker=False)
        facet = SearchFacet(
            name="Axe calcul",
            goal="Comparer deux preuves",
            query="regime et calcul",
            prefix="TVA",
            expected_evidence=["seuil applicable", "taux applicable"],
        )

        _result, chunks = agent._retrieve_for_facet(facet, "Question fiscale.", set())

        self.assertEqual(len(runtime.intra_calls), 2)
        self.assertIn("seuil applicable", runtime.intra_calls[0]["query"])
        self.assertIn("taux applicable", runtime.intra_calls[1]["query"])
        self.assertEqual(runtime.intra_calls[0]["boi_references"], ["BOI-TVA-DOC"])
        evidence_chunks = [chunk for chunk in chunks if chunk.get("retrieval_stage") == "evidence_rescue"]
        self.assertEqual([chunk["chunk_id"] for chunk in evidence_chunks], ["evidence-seuil", "evidence-taux"])

    def test_prompts_keep_agentic_pragmatism_for_non_blocking_limits(self):
        system_prompt = build_system_prompt()

        self.assertIn("question principale", system_prompt)
        self.assertIn("reserves, hypotheses, exceptions non declenchees, options ou precisions", system_prompt)
        self.assertIn("partial est deconseille", system_prompt)
        self.assertIn("axes_manquants n'est pas un verdict de verite", system_prompt)

        responses = [
            json.dumps(
                {
                    "reformulated_question": "Question fiscale.",
                    "facts": ["fait pertinent"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "Axe principal",
                            "goal": "Prouver la reponse principale",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA axe principal",
                            "priority": 1,
                            "expected_evidence": ["preuve principale"],
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": ["Axe principal"],
                    "missing_axes": [
                        {
                            "axis": "Precision non bloquante",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA precision annexe",
                            "why_needed": "utile en limite mais ne change pas la conclusion principale",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Reponse principale sourcee.",
                    "axes_requis": ["Axe principal"],
                    "axes_couverts": ["Axe principal"],
                    "axes_manquants": [],
                    "justification_bullets": ["Preuve principale [1]."],
                    "limits": "Precision annexe a verifier.",
                }
            ),
        ]
        runtime = _FakeRuntime()
        client = _FakeClient(responses)
        agent = AgenticRAG(runtime, client=client, max_iterations=1, use_reranker=False)

        result = agent.run("Question fiscale.")

        source_review_prompt = client.chat.completions.calls[1]["messages"][1]["content"]
        answer_prompt = client.chat.completions.calls[2]["messages"][1]["content"]
        self.assertIn("coverage_status='ready' si les passages permettent de repondre a la question principale", source_review_prompt)
        self.assertIn("Le reviewer documentaire a seulement servi a trouver plus de sources", answer_prompt)
        self.assertEqual(result["answer_status"], "supported")
        self.assertEqual(result["axes_manquants"], [])

    def test_missing_axes_trigger_reformulation_and_second_retrieval(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Quel taux de TVA appliquer ?",
                    "facts": ["question sur un taux de TVA"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "TVA applicable",
                            "goal": "Identifier le taux applicable",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA taux applicable operation",
                            "priority": 1,
                            "expected_evidence": ["taux applicable"],
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "needs_more_sources",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": [],
                    "missing_axes": [
                        {
                            "axis": "taux applicable",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA taux applicable",
                            "why_needed": "trouver le taux",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1", "chunk-2"],
                    "rejected_chunks": [],
                    "covered_axes": ["taux applicable"],
                    "missing_axes": [],
                }
            ),
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
        self.assertEqual(len(runtime.intra_calls), 1)
        self.assertTrue(runtime.calls[0]["query"].startswith("TVA "))
        self.assertEqual(runtime.calls[0]["boost_prefix"], "TVA")
        self.assertIn("chunk_query", runtime.calls[0])
        self.assertFalse(runtime.calls[0]["use_reranker"])
        self.assertIn("TVA taux applicable", runtime.calls[1]["query"])
        self.assertEqual(runtime.calls[1]["boost_prefix"], "TVA")
        self.assertEqual(result["trace"][0]["stage"], "plan_and_route")
        self.assertEqual(len(result["trace"][0]["relaunches"]), 1)
        self.assertEqual(result["coverage"], 1.0)

    def test_silent_blocking_plan_axis_triggers_targeted_relaunch(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Question fiscale multi-axes.",
                    "facts": ["fait utilisateur"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "Axe couvert",
                            "goal": "Prouver le premier axe",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA premier axe",
                            "priority": 1,
                            "role": "core",
                            "blocking": True,
                        },
                        {
                            "name": "Axe silencieux",
                            "goal": "Prouver le second axe",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA second axe",
                            "priority": 2,
                            "role": "core",
                            "blocking": True,
                        },
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": ["Axe couvert"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1", "chunk-3"],
                    "rejected_chunks": [],
                    "covered_axes": ["Axe couvert", "Axe silencieux"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Reponse complete.",
                    "axes_requis": ["Axe couvert", "Axe silencieux"],
                    "axes_couverts": ["Axe couvert", "Axe silencieux"],
                    "axes_manquants": [],
                    "justification_bullets": ["Deux axes couverts."],
                    "limits": "",
                }
            ),
        ]
        runtime = _FakeRuntime()
        agent = AgenticRAG(runtime, client=_FakeClient(responses), max_iterations=2, use_reranker=False)

        result = agent.run("Question multi-axes.")

        self.assertEqual(result["answer_status"], "supported")
        self.assertEqual(len(runtime.calls), 3)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["trace"][0]["relaunches"][0]["facet"]["name"], "Axe silencieux")

    def test_broad_covered_axis_does_not_mask_blocking_plan_facet_without_matching_useful_source(self):
        plan = SearchPlan(
            reformulated_question="Question multi-axes.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(name="Regime principal", goal="Prouver le regime", query="regime principal", prefix="TVA"),
                SearchFacet(name="Regime seuil", goal="Prouver le seuil", query="regime seuil", prefix="TVA"),
            ],
        )
        chunks = [
            {
                "rank": 1,
                "chunk_id": "principal-proof",
                "boi_reference": "BOI-TVA-A",
                "title": "Regime principal",
                "facet": "Regime principal",
                "section_path": "Regime principal",
                "text": "Preuve du regime principal.",
            }
        ]
        review = {
            "coverage_status": "ready",
            "useful_chunk_ids": ["principal-proof"],
            "covered_axes": ["Regime"],
            "missing_axes": [],
            "rejected_chunks": [],
        }

        completed = _complete_source_review_with_plan_gaps(review, plan, chunks)

        self.assertEqual(completed["coverage_status"], "needs_more_sources")
        self.assertIn("Regime seuil", [item["axis"] for item in completed["missing_axes"]])

    def test_missing_axis_candidate_survives_when_useful_ids_are_empty(self):
        chunks = [
            {
                "chunk_id": f"noise-{idx}",
                "rank": idx,
                "boi_reference": "BOI-NOISE",
                "title": "Bruit",
                "facet": "Bruit",
                "section_path": "Bruit",
                "text": "Generalites sans rapport.",
            }
            for idx in range(1, 14)
        ]
        chunks.append(
            {
                "chunk_id": "missing-proof",
                "rank": 50,
                "boi_reference": "BOI-TVA-SPEC",
                "title": "Axe manquant",
                "facet": "Axe manquant",
                "section_path": "Axe manquant",
                "text": "Preuve specifique du point manquant.",
            }
        )
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": [],
            "covered_axes": [],
            "missing_axes": [
                {
                    "axis": "Axe manquant",
                    "bofip_prefix": "TVA",
                    "search_query": "preuve specifique point manquant",
                    "why_needed": "necessaire pour conclure",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertIn("missing-proof", [chunk["chunk_id"] for chunk in selected])

    def test_missing_axis_searches_within_existing_candidate_before_global_relaunch(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "La cotisation minimum CFE est-elle due ?",
                    "facts": ["micro-entrepreneur avec faible chiffre d'affaires"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "Cotisation minimum CFE",
                            "goal": "Verifier la cotisation minimum et les exonerations",
                            "bofip_prefix": "IF",
                            "search_query": "CFE cotisation minimum faible chiffre affaires",
                            "priority": 1,
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "needs_more_sources",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": ["cotisation minimum"],
                    "missing_axes": [
                        {
                            "axis": "exoneration faible chiffre d'affaires",
                            "bofip_prefix": "IF",
                            "search_query": "exoneration cotisation minimum chiffre affaires recettes 5000",
                            "why_needed": "verifier si un seuil de chiffre d'affaires supprime la cotisation minimum",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1", "rescue-chunk"],
                    "rejected_chunks": [],
                    "covered_axes": ["cotisation minimum", "exoneration faible chiffre d'affaires"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Reponse sourcee.",
                    "axes_requis": ["cotisation minimum", "exoneration faible chiffre d'affaires"],
                    "axes_couverts": ["cotisation minimum", "exoneration faible chiffre d'affaires"],
                    "axes_manquants": [],
                    "justification_bullets": ["Le passage local couvre l'axe manquant."],
                    "limits": "",
                }
            ),
        ]
        runtime = _FakeRuntime()
        runtime.intra_results.append(
            SimpleNamespace(
                query="IF exoneration cotisation minimum chiffre affaires recettes 5000",
                stage1_hits=[
                    SimpleNamespace(rank=1, score=1.0, boi_reference="BOI-IF-DOC", title="Source IF")
                ],
                stage2_chunks=[
                    SimpleNamespace(
                        boi_reference="BOI-IF-DOC",
                        title="Source IF",
                        publication_date="2026-01-01",
                        section_path="Cotisation minimum > Exoneration faible chiffre d'affaires",
                        text="Passage local sur le seuil de chiffre d'affaires.",
                        chunk_id="rescue-chunk",
                    )
                ],
            )
        )
        client = _FakeClient(responses)
        agent = AgenticRAG(runtime, client=client, max_iterations=2, use_reranker=False)

        result = agent.run("Micro-entrepreneur faible chiffre d'affaires et avis CFE.")

        self.assertEqual(result["answer_status"], "supported")
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(len(runtime.intra_calls), 1)
        self.assertEqual(runtime.intra_calls[0]["boi_references"], ["BOI-IF-DOC"])
        self.assertEqual(result["trace"][0]["relaunches"][0]["stage"], "intra_document")
        second_review_prompt = client.chat.completions.calls[2]["messages"][1]["content"]
        self.assertIn("rescue-chunk", second_review_prompt)

    def test_relaunch_chunks_are_prioritized_in_source_review_budget(self):
        chunks = [
            {"chunk_id": f"initial-{idx}", "rank": idx, "boi_reference": "BOI-A", "title": "Initial"}
            for idx in range(20)
        ]
        rescued = {
            "chunk_id": "rescued",
            "rank": 1,
            "boi_reference": "BOI-A",
            "title": "Rescued",
            "retrieval_stage": "intra_document",
        }
        chunks.append(rescued)

        selected = _chunks_for_source_review(chunks, limit=16)

        self.assertEqual(selected[0]["chunk_id"], "rescued")
        self.assertEqual(len(selected), 16)

    def test_evidence_rescue_chunks_are_prioritized_in_source_review_budget(self):
        chunks = [
            {"chunk_id": f"initial-{idx}", "rank": idx, "boi_reference": "BOI-A", "title": "Initial"}
            for idx in range(20)
        ]
        rescued = {
            "chunk_id": "evidence-rescued",
            "rank": 1,
            "boi_reference": "BOI-A",
            "title": "Evidence",
            "retrieval_stage": "evidence_rescue",
        }
        chunks.append(rescued)

        selected = _chunks_for_source_review(chunks, limit=16)

        self.assertEqual(selected[0]["chunk_id"], "evidence-rescued")
        self.assertEqual(len(selected), 16)

    def test_unrejected_relaunch_chunks_remain_available_for_final_answer(self):
        chunks = [
            {"chunk_id": "useful-old", "rank": 1, "boi_reference": "BOI-A", "title": "Old"},
            {
                "chunk_id": "rescued",
                "rank": 2,
                "boi_reference": "BOI-A",
                "title": "Rescued",
                "retrieval_stage": "intra_document",
            },
            {"chunk_id": "noise", "rank": 3, "boi_reference": "BOI-B", "title": "Noise"},
        ]
        review = {
            "useful_chunk_ids": ["useful-old"],
            "rejected_chunks": [{"chunk_id": "noise", "reason": "hors sujet"}],
            "covered_axes": ["axe deja couvert"],
            "missing_axes": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertEqual([chunk["chunk_id"] for chunk in selected], ["useful-old", "rescued"])

    def test_unrejected_evidence_rescue_chunks_remain_available_for_final_answer(self):
        chunks = [
            {
                "chunk_id": "useful-old",
                "rank": 1,
                "boi_reference": "BOI-A",
                "title": "Old",
                "facet": "Axe calcul",
                "section_path": "Regle generale",
                "text": "Preuve generale.",
            },
            {
                "chunk_id": "evidence-rescued",
                "rank": 20,
                "boi_reference": "BOI-A",
                "title": "Evidence",
                "facet": "Axe calcul",
                "section_path": "Taux applicable",
                "text": "Preuve specifique du taux applicable.",
                "retrieval_stage": "evidence_rescue",
            },
            {"chunk_id": "noise", "rank": 3, "boi_reference": "BOI-B", "title": "Noise"},
        ]
        review = {
            "useful_chunk_ids": ["useful-old"],
            "rejected_chunks": [{"chunk_id": "noise", "reason": "hors sujet"}],
            "covered_axes": ["Axe calcul"],
            "missing_axes": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertEqual([chunk["chunk_id"] for chunk in selected], ["useful-old", "evidence-rescued"])

    def test_relaunch_chunks_survive_even_when_many_useful_chunks_exist(self):
        chunks = [
            {
                "chunk_id": f"useful-{idx}",
                "rank": idx,
                "boi_reference": "BOI-A",
                "title": "Initial",
                "facet": "Axe initial",
                "section_path": "Axe initial",
                "text": "Preuve initiale.",
            }
            for idx in range(1, 16)
        ]
        chunks.append(
            {
                "chunk_id": "rescued-axis",
                "rank": 99,
                "boi_reference": "BOI-B",
                "title": "Rescue",
                "facet": "Axe relance",
                "section_path": "Axe relance",
                "text": "Preuve relancee.",
                "retrieval_stage": "intra_document",
            }
        )
        review = {
            "coverage_status": "ready",
            "useful_chunk_ids": [f"useful-{idx}" for idx in range(1, 16)],
            "covered_axes": ["Axe initial"],
            "missing_axes": [],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertIn("rescued-axis", [chunk["chunk_id"] for chunk in selected])
        self.assertLessEqual(len(selected), 12)

    def test_covered_axes_promote_matching_evidence_before_truncation(self):
        chunks = [
            {
                "chunk_id": f"same-axis-{idx}",
                "rank": idx,
                "boi_reference": "BOI-A",
                "title": "A",
                "facet": "Axe A",
                "section_path": "A",
                "text": "Preuve axe A.",
            }
            for idx in range(1, 14)
        ]
        chunks.append(
            {
                "chunk_id": "axis-b-proof",
                "rank": 50,
                "boi_reference": "BOI-B",
                "title": "B",
                "facet": "Axe B",
                "section_path": "B",
                "text": "Preuve axe B.",
            }
        )
        review = {
            "coverage_status": "ready",
            "useful_chunk_ids": [f"same-axis-{idx}" for idx in range(1, 14)],
            "covered_axes": ["Axe A", "Axe B"],
            "missing_axes": [],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertIn("axis-b-proof", [chunk["chunk_id"] for chunk in selected])
        self.assertLess(selected.index(next(chunk for chunk in selected if chunk["chunk_id"] == "axis-b-proof")), 12)

    def test_blocking_missing_axes_promote_candidate_evidence(self):
        chunks = [
            {
                "chunk_id": f"generic-{idx}",
                "rank": idx,
                "boi_reference": "BOI-GEN",
                "title": "General",
                "facet": "Axe general",
                "section_path": "General",
                "text": "Generalites.",
            }
            for idx in range(1, 13)
        ]
        chunks.append(
            {
                "chunk_id": "missing-axis-candidate",
                "rank": 50,
                "boi_reference": "BOI-SPEC",
                "title": "Specific",
                "facet": "Axe manquant",
                "section_path": "Axe manquant",
                "text": "Preuve specifique du taux et du regime.",
            }
        )
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": [f"generic-{idx}" for idx in range(1, 13)],
            "covered_axes": ["Axe general"],
            "missing_axes": [
                {
                    "axis": "Axe manquant",
                    "search_query": "preuve specifique taux regime",
                    "why_needed": "peut changer le calcul",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertIn("missing-axis-candidate", [chunk["chunk_id"] for chunk in selected])

    def test_blocking_missing_axis_promotes_specific_candidate_despite_generic_selected_match(self):
        chunks = [
            {
                "chunk_id": "generic-selected",
                "rank": 1,
                "boi_reference": "BOI-GEN",
                "title": "General",
                "facet": "Axe general",
                "section_path": "General",
                "text": "Taux et regime mentionnes de facon generale.",
            },
            {
                "chunk_id": "specific-missing-proof",
                "rank": 20,
                "boi_reference": "BOI-SPEC",
                "title": "Specific",
                "facet": "Axe manquant",
                "section_path": "Axe manquant",
                "text": "Preuve specifique du taux et du regime.",
            },
        ]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": ["generic-selected"],
            "covered_axes": [],
            "missing_axes": [
                {
                    "axis": "Axe manquant",
                    "search_query": "preuve specifique taux regime",
                    "why_needed": "le point peut changer la conclusion",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        selected_ids = [chunk["chunk_id"] for chunk in selected]
        self.assertIn("specific-missing-proof", selected_ids)
        self.assertLess(selected_ids.index("specific-missing-proof"), selected_ids.index("generic-selected"))

    def test_axis_promotion_prefers_facet_alignment_over_lexical_noise(self):
        chunks = [
            {
                "chunk_id": "initial",
                "rank": 1,
                "boi_reference": "BOI-INIT",
                "title": "Initial",
                "facet": "Axe initial",
                "section_path": "Initial",
                "text": "Preuve initiale.",
            },
            {
                "chunk_id": "lexical-noise",
                "rank": 2,
                "boi_reference": "BOI-NOISE",
                "title": "Bruit lexical",
                "facet": "Autre axe",
                "section_path": "Autre axe",
                "text": "Preuve specifique taux regime calcul conclusion.",
            },
            {
                "chunk_id": "facet-aligned",
                "rank": 30,
                "boi_reference": "BOI-SPEC",
                "title": "Axe fiscal specifique",
                "facet": "Axe fiscal specifique",
                "section_path": "Axe fiscal specifique",
                "text": "Preuve concise.",
            },
        ]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": ["initial"],
            "covered_axes": [],
            "missing_axes": [
                {
                    "axis": "Axe fiscal specifique",
                    "search_query": "preuve specifique taux regime calcul",
                    "why_needed": "necessaire pour conclure",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertEqual(selected[0]["chunk_id"], "facet-aligned")

    def test_missing_axis_promotion_prefers_matching_bofip_prefix(self):
        chunks = [
            {
                "chunk_id": "initial",
                "rank": 1,
                "boi_reference": "BOI-BNC-DECLA-20-10",
                "title": "Initial",
                "facet": "Axe initial",
                "section_path": "Initial",
                "text": "Preuve initiale.",
            },
            {
                "chunk_id": "off-family-noise",
                "rank": 2,
                "boi_reference": "BOI-IF-TFNB-50-10-20",
                "title": "Bruit hors famille",
                "facet": "Axe fiscal specifique",
                "section_path": "Axe fiscal specifique",
                "text": "Preuve specifique taux regime calcul conclusion recettes.",
            },
            {
                "chunk_id": "same-family-proof",
                "rank": 30,
                "boi_reference": "BOI-BNC-DECLA-20-20",
                "title": "Preuve BNC",
                "facet": "Axe fiscal specifique",
                "section_path": "Axe fiscal specifique",
                "text": "Preuve specifique BNC.",
            },
        ]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": ["initial"],
            "covered_axes": [],
            "missing_axes": [
                {
                    "axis": "Axe fiscal specifique",
                    "bofip_prefix": "BNC",
                    "search_query": "preuve specifique taux regime calcul",
                    "why_needed": "necessaire pour conclure",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertEqual(selected[0]["chunk_id"], "same-family-proof")

    def test_relaunch_append_prefers_prefix_without_hiding_off_prefix_evidence(self):
        chunks = [
            {
                "chunk_id": "useful-initial",
                "rank": 1,
                "boi_reference": "BOI-BNC-DECLA-20-10",
                "title": "Initial",
                "facet": "Axe initial",
                "section_path": "Initial",
                "text": "Preuve initiale.",
            },
            {
                "chunk_id": "off-prefix-relaunch",
                "rank": 2,
                "boi_reference": "BOI-IF-TFNB-50-10-20",
                "title": "Bruit hors famille",
                "facet": "Axe fiscal specifique",
                "section_path": "Axe fiscal specifique",
                "text": "Preuve specifique taux regime calcul.",
                "retrieval_stage": "global_relaunch",
            },
            {
                "chunk_id": "same-prefix-relaunch",
                "rank": 30,
                "boi_reference": "BOI-BNC-DECLA-20-20",
                "title": "Preuve BNC",
                "facet": "Axe fiscal specifique",
                "section_path": "Axe fiscal specifique",
                "text": "Preuve specifique BNC.",
                "retrieval_stage": "global_relaunch",
            },
        ]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": ["useful-initial"],
            "covered_axes": [],
            "missing_axes": [
                {
                    "axis": "Axe fiscal specifique",
                    "bofip_prefix": "BNC",
                    "search_query": "preuve specifique taux regime calcul",
                    "why_needed": "necessaire pour conclure",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)
        selected_ids = [chunk["chunk_id"] for chunk in selected]

        self.assertIn("same-prefix-relaunch", selected_ids)
        self.assertIn("off-prefix-relaunch", selected_ids)

    def test_relaunch_append_keeps_off_prefix_chunk_when_it_matches_missing_axis(self):
        chunks = [
            {
                "chunk_id": "useful-initial",
                "rank": 1,
                "boi_reference": "BOI-BNC-DECLA-20-10",
                "title": "Initial",
                "facet": "Axe initial",
                "section_path": "Initial",
                "text": "Preuve initiale.",
            },
            {
                "chunk_id": "off-prefix-relaunch",
                "rank": 2,
                "boi_reference": "BOI-IF-TFNB-50-10-20",
                "title": "Preuve textuelle hors prefixe",
                "facet": "Axe fiscal specifique",
                "section_path": "Axe fiscal specifique",
                "text": "Preuve specifique taux regime calcul.",
                "retrieval_stage": "global_relaunch",
            },
        ]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": ["useful-initial"],
            "covered_axes": [],
            "missing_axes": [
                {
                    "axis": "Axe fiscal specifique",
                    "bofip_prefix": "BNC",
                    "search_query": "preuve specifique taux regime calcul",
                    "why_needed": "necessaire pour conclure",
                    "blocking": True,
                }
            ],
            "rejected_chunks": [],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertIn("off-prefix-relaunch", [chunk["chunk_id"] for chunk in selected])

    def test_evidence_matrix_marks_candidate_only_axis(self):
        plan = SearchPlan(
            reformulated_question="Question test",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(name="Axe A", goal="Prouver A", query="axe a", prefix="TVA"),
                SearchFacet(name="Axe B", goal="Prouver B", query="axe b", prefix="TVA"),
            ],
        )
        chunks = [
            {
                "chunk_id": "a-final",
                "rank": 1,
                "boi_reference": "BOI-A",
                "title": "A",
                "facet": "Axe A",
                "section_path": "A",
                "text": "Preuve A",
            },
            {
                "chunk_id": "b-candidate",
                "rank": 2,
                "boi_reference": "BOI-B",
                "title": "B",
                "facet": "Axe B",
                "section_path": "B",
                "text": "Preuve B",
            },
        ]
        review = {
            "coverage_status": "ready",
            "covered_axes": ["Axe A", "Axe B"],
            "missing_axes": [],
            "useful_chunk_ids": ["a-final"],
            "rejected_chunks": [],
        }

        matrix = _build_evidence_matrix(plan, chunks, review, [chunks[0]])

        by_axis = {row["axis"]: row for row in matrix}
        self.assertEqual(by_axis["Axe A"]["status"], "covered_final")
        self.assertEqual(by_axis["Axe B"]["status"], "candidate_only")
        self.assertEqual(by_axis["Axe B"]["candidate_refs"], ["BOI-B"])

    def test_plan_routes_multi_axis_queries_without_one_global_prefix(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Question fiscale mixte.",
                    "facts": ["fait A"],
                    "ambiguities": ["hypothèse à confirmer"],
                    "facets": [
                        {
                            "name": "Axe IS",
                            "goal": "Trouver la base et le taux IS",
                            "bofip_prefix": "IS",
                            "search_query": "IS taux resultat fiscal",
                            "priority": 1,
                            "expected_evidence": ["taux", "base"],
                        },
                        {
                            "name": "Axe TVA",
                            "goal": "Trouver le régime TVA",
                            "bofip_prefix": "TVA",
                            "search_query": "TVA prestation service regime",
                            "priority": 2,
                            "expected_evidence": ["champ", "taux"],
                        },
                    ],
                    "excluded_axes": [{"axis": "Axe hors sujet", "reason": "mot ambigu seulement"}],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1", "chunk-2"],
                    "rejected_chunks": [],
                    "covered_axes": ["Axe IS", "Axe TVA"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Réponse supportée.",
                    "axes_requis": ["Axe IS", "Axe TVA"],
                    "axes_couverts": ["Axe IS", "Axe TVA"],
                    "axes_manquants": [],
                    "justification_bullets": ["Deux axes couverts."],
                    "limits": "",
                }
            ),
        ]
        runtime = _FakeRuntime()
        agent = AgenticRAG(runtime, client=_FakeClient(responses), max_iterations=2, use_reranker=False)

        result = agent.run("Question fiscale mixte")

        self.assertEqual([call["boost_prefix"] for call in runtime.calls], ["IS", "TVA"])
        self.assertIn("IS taux resultat fiscal", runtime.calls[0]["query"])
        self.assertIn("TVA prestation service regime", runtime.calls[1]["query"])
        self.assertEqual(len(result["trace"][0]["routes"]), 2)
        self.assertEqual(result["source_review"]["covered_axes"], ["Axe IS", "Axe TVA"])

    def test_planner_output_sanitizes_invalid_prefix_without_dropping_axis(self):
        plan = _normalize_plan(
            "Question hors taxonomie locale",
            {
                "reformulated_question": "Question hors taxonomie locale",
                "facts": ["fait neutre"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Axe inconnu",
                        "goal": "Chercher sans préfixe fiable",
                        "bofip_prefix": "XYZ-FAKE",
                        "search_query": "termes techniques fournis par le planneur",
                        "priority": 1,
                        "expected_evidence": ["preuve"],
                    }
                ],
                "excluded_axes": [],
            },
        )

        self.assertEqual(len(plan.facets), 1)
        self.assertEqual(plan.facets[0].prefix, "")
        self.assertEqual(plan.facets[0].query, "termes techniques fournis par le planneur")

    def test_planner_keeps_rsa_bofip_prefix(self):
        plan = _normalize_plan(
            "Rupture conventionnelle",
            {
                "reformulated_question": "Regime fiscal d'une rupture conventionnelle.",
                "facts": ["rupture conventionnelle"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Indemnite de rupture",
                        "goal": "Trouver les conditions d'exoneration",
                        "bofip_prefix": "BOI-RSA-CHAMP-20-40-10",
                        "search_query": "BOI-RSA-CHAMP-20-40-10 indemnite rupture conventionnelle exoneration",
                        "priority": 1,
                        "expected_evidence": ["conditions", "limites"],
                    }
                ],
                "excluded_axes": [],
            },
        )

        self.assertEqual(plan.facets[0].prefix, "RSA-CHAMP-20-40-10")

    def test_planner_preserves_axis_role_and_blocking_flag(self):
        plan = _normalize_plan(
            "Don manuel a un enfant majeur",
            {
                "reformulated_question": "Don manuel parent enfant.",
                "facts": ["don a un enfant majeur"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Abattement parent-enfant",
                        "goal": "Verifier l'abattement de droit commun",
                        "bofip_prefix": "ENR",
                        "search_query": "donation ligne directe abattement enfant",
                        "role": "core",
                        "blocking": True,
                    },
                    {
                        "name": "Present d'usage",
                        "goal": "A signaler seulement si les faits le suggerent",
                        "bofip_prefix": "ENR",
                        "search_query": "present usage donation",
                        "role": "alternative",
                        "blocking": False,
                    },
                ],
            },
        )

        self.assertEqual(plan.facets[0].role, "core")
        self.assertTrue(plan.facets[0].blocking)
        self.assertEqual(plan.facets[1].role, "alternative")
        self.assertFalse(plan.facets[1].blocking)

    def test_planner_keeps_model_prefix_as_soft_signal_without_rewriting_cfe_to_if(self):
        plan = _normalize_plan(
            "Avis CFE 2025 pour micro-entrepreneur sans local",
            {
                "reformulated_question": "CFE micro-entrepreneur",
                "facts": ["avis CFE", "micro-entrepreneur", "sans local"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Cotisation minimum CFE",
                        "goal": "Verifier la cotisation minimum et l'exoneration faible chiffre d'affaires",
                        "bofip_prefix": "CF",
                        "search_query": "CFE cotisation minimum faible chiffre affaires",
                        "role": "core",
                        "blocking": True,
                    }
                ],
            },
        )

        self.assertEqual(plan.facets[0].prefix, "CF")

    def test_planner_keeps_model_prefix_as_soft_signal_without_rewriting_location_nue(self):
        plan = _normalize_plan(
            "Je loue nu un appartement avec loyers et interets d'emprunt.",
            {
                "reformulated_question": "Location nue revenus fonciers.",
                "facts": ["location nue", "loyers", "interets d'emprunt"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Regime reel revenus fonciers",
                        "goal": "Verifier les charges deductibles en location nue",
                        "bofip_prefix": "RPPM",
                        "search_query": "regime reel location nue charges deductibles interets emprunt",
                        "role": "core",
                        "blocking": True,
                    }
                ],
            },
        )

        self.assertEqual(plan.facets[0].prefix, "RPPM")

    def test_planner_keeps_model_prefix_as_soft_signal_without_rewriting_real_estate_gain(self):
        plan = _normalize_plan(
            "J'ai vendu ma residence principale apres un demenagement recent.",
            {
                "reformulated_question": "Plus-value immobiliere residence principale.",
                "facts": ["vente residence principale"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Exoneration residence principale",
                        "goal": "Verifier l'exoneration de plus-value immobiliere",
                        "bofip_prefix": "IR",
                        "search_query": "plus-value immobiliere residence principale exoneration",
                        "role": "core",
                        "blocking": True,
                    }
                ],
            },
        )

        self.assertEqual(plan.facets[0].prefix, "IR")

    def test_planner_does_not_demote_contingent_axis_without_model_role_signal(self):
        question = "J'ai vendu ma residence principale apres un demenagement recent. Est-ce imposable ?"
        plan = _normalize_plan(
            question,
            {
                "reformulated_question": question,
                "facts": ["cession residence principale"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Calcul de la plus-value imposable",
                        "goal": "Si l'exoneration n'est pas applicable, calculer la plus-value imposable",
                        "bofip_prefix": "RFPI",
                        "search_query": "calcul plus-value immobiliere abattement duree detention",
                        "role": "calculation",
                        "blocking": True,
                    }
                ],
            },
        )

        self.assertEqual(plan.facets[0].role, "calculation")
        self.assertTrue(plan.facets[0].blocking)

    def test_planner_promotes_threshold_axis_when_question_asks_liability(self):
        plan = _normalize_plan(
            "Suis-je exonere ou soumis a la cotisation avec 3 200 euros de chiffre d'affaires ?",
            {
                "reformulated_question": "Question sur seuil de cotisation.",
                "facts": ["chiffre d'affaires 3 200 euros"],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Cotisation minimum",
                        "goal": "Verifier seuil de chiffre d'affaires et cotisation minimum",
                        "bofip_prefix": "IF",
                        "search_query": "cotisation minimum seuil chiffre affaires",
                        "role": "alternative",
                        "blocking": False,
                    }
                ],
            },
        )

        self.assertEqual(plan.facets[0].role, "calculation")
        self.assertTrue(plan.facets[0].blocking)

    def test_planner_demotes_regime_change_axis_when_question_assumes_regime_for_calculation(self):
        plan = _normalize_plan(
            "Quel benefice imposable dois-je retenir si je releve du regime applicable aux petites exploitations agricoles ?",
            {
                "reformulated_question": "Calculer le benefice imposable sous un regime fiscal suppose applicable.",
                "facts": ["Le contribuable indique relever du regime applicable aux petites exploitations."],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Determination du benefice imposable",
                        "goal": "Identifier la methode de calcul du benefice imposable.",
                        "bofip_prefix": "BA",
                        "search_query": "micro BA benefice imposable abattement recettes",
                        "priority": 1,
                        "role": "calculation",
                        "blocking": True,
                    },
                    {
                        "name": "Changement ou sortie de regime",
                        "goal": "Rechercher les regles applicables si les conditions du regime ne sont pas confirmees ou cessent de s'appliquer.",
                        "bofip_prefix": "BA",
                        "search_query": "changement regime micro BA sortie regime reel recettes",
                        "priority": 2,
                        "role": "core",
                        "blocking": True,
                    },
                ],
            },
        )

        self.assertEqual(plan.facets[0].role, "calculation")
        self.assertTrue(plan.facets[0].blocking)
        self.assertEqual(plan.facets[1].role, "reserve")
        self.assertFalse(plan.facets[1].blocking)

    def test_planner_demotes_untriggered_optional_exception_axis(self):
        question = (
            "J'ai achete des actions en 2016 pour 20 000 euros. "
            "Je les ai revendues en 2024 pour 50 000 euros. "
            "Si j'opte pour l'imposition au bareme, quel montant de plus-value "
            "serait retenu apres l'abattement pour duree de detention ?"
        )
        plan = _normalize_plan(
            question,
            {
                "reformulated_question": question,
                "facts": ["Actions acquises en 2016 et revendues en 2024."],
                "ambiguities": ["La nature exacte des titres n'est pas precisee."],
                "facets": [
                    {
                        "name": "Abattement duree de detention",
                        "goal": "Trouver le taux et la methode de l'abattement de droit commun.",
                        "bofip_prefix": "RPPM",
                        "search_query": "abattement duree detention titres acquis avant 2018 bareme",
                        "priority": 1,
                        "role": "calculation",
                        "blocking": True,
                    },
                    {
                        "name": "Abattement renforce eventuel",
                        "goal": "Verifier seulement si des conditions particulieres non precisees pourraient modifier l'abattement.",
                        "bofip_prefix": "RPPM",
                        "search_query": "abattement renforce duree detention titres PME conditions cession actions",
                        "priority": 2,
                        "role": "calculation",
                        "blocking": True,
                    },
                ],
            },
        )

        optional_facet = next(facet for facet in plan.facets if facet.name == "Abattement renforce eventuel")
        self.assertEqual(optional_facet.role, "reserve")
        self.assertFalse(optional_facet.blocking)

    def test_planner_demotes_scope_clarification_axis_not_requested_by_question(self):
        plan = _normalize_plan(
            "Quel montant de dividendes sera retenu apres l'abattement ? Ne calcule pas l'impot final.",
            {
                "reformulated_question": "Determiner le montant retenu apres abattement.",
                "facts": ["La question demande une base apres abattement, pas une autre imposition."],
                "ambiguities": [],
                "facets": [
                    {
                        "name": "Abattement sur dividendes",
                        "goal": "Trouver le taux d'abattement et calculer la base retenue.",
                        "bofip_prefix": "RPPM",
                        "search_query": "dividendes abattement bareme base retenue",
                        "priority": 1,
                        "role": "calculation",
                        "blocking": True,
                    },
                    {
                        "name": "Prelevements sociaux",
                        "goal": "Verifier uniquement si le BOFiP distingue cette assiette afin d'eviter une confusion.",
                        "bofip_prefix": "RPPM",
                        "search_query": "distinction assiette impot revenu prelevements sociaux utile seulement pour cadrer la reponse",
                        "priority": 2,
                        "role": "calculation",
                        "blocking": True,
                    },
                ],
            },
        )

        self.assertEqual(plan.facets[0].role, "calculation")
        self.assertTrue(plan.facets[0].blocking)
        self.assertEqual(plan.facets[1].role, "reserve")
        self.assertFalse(plan.facets[1].blocking)

    def test_candidate_refs_for_missing_axis_keeps_exact_boi_outside_prefix(self):
        missing = {
            "axis": "Seuil franchise TVA services",
            "search_query": "seuil franchise TVA prestations services BOI-BAREME-000036",
            "bofip_prefix": "TVA",
            "why_needed": "retrouver le bareme exact mentionne par la doctrine",
        }
        route_log = [
            {
                "facet": {"prefix": "TVA"},
                "stage1_refs": ["BOI-TVA-DECLA-40-10-20-20230118"],
            }
        ]
        chunks = [
            {
                "rank": 1,
                "chunk_id": "franchise",
                "boi_reference": "BOI-TVA-DECLA-40-10-20-20230118",
                "title": "Franchise en base",
                "section_path": "Seuils",
                "text": "Renvoie au bareme applicable.",
            }
        ]

        refs = _candidate_refs_for_missing_axis(missing, chunks, route_log)

        self.assertEqual(refs[0], "BOI-BAREME-000036")

    def test_source_review_rejections_are_not_reintroduced_when_useful_ids_are_empty(self):
        chunks = [
            {"rank": 1, "chunk_id": "bad", "boi_reference": "BOI-CF-1", "title": "Hors sujet"},
            {"rank": 2, "chunk_id": "good", "boi_reference": "BOI-TVA-1", "title": "Utile"},
        ]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": [],
            "covered_axes": [],
            "missing_axes": [{"axis": "TVA", "search_query": "TVA franchise", "bofip_prefix": "TVA"}],
            "rejected_chunks": [{"chunk_id": "bad", "reason": "mot ambigu hors sujet"}],
        }

        selected = _select_reviewed_chunks(chunks, review)

        self.assertEqual([chunk["chunk_id"] for chunk in selected], ["good"])

    def test_source_review_without_useful_ids_triggers_more_sources(self):
        chunks = [{"rank": 1, "chunk_id": "candidate", "boi_reference": "BOI-A-1", "title": "Candidate"}]
        review = {
            "coverage_status": "ready",
            "useful_chunk_ids": [],
            "covered_axes": ["Seuil applicable"],
            "missing_axes": [],
            "rejected_chunks": [],
        }

        normalized = _normalize_source_review(review, chunks)

        self.assertEqual(normalized["coverage_status"], "needs_more_sources")
        self.assertEqual(normalized["covered_axes"], [])
        self.assertEqual(normalized["missing_axes"][0]["axis"], "Seuil applicable")
        self.assertEqual(normalized["missing_axes"][0]["search_query"], "Seuil applicable")

    def test_source_review_can_infer_useful_ids_from_matching_facet(self):
        chunks = [
            {
                "rank": 1,
                "chunk_id": "micro",
                "boi_reference": "BOI-RFPI-DECLA-10",
                "title": "Micro-foncier",
                "facet": "Eligibilite micro-foncier et option reel",
                "section_path": "Micro-foncier",
                "text": "Regime micro-foncier et option pour le reel.",
            }
        ]
        review = {
            "coverage_status": "ready",
            "useful_chunk_ids": [],
            "covered_axes": ["Eligibilite micro-foncier et option reel"],
            "missing_axes": [],
            "rejected_chunks": [],
        }

        normalized = _normalize_source_review(review, chunks)

        self.assertEqual(normalized["coverage_status"], "ready")
        self.assertEqual(normalized["useful_chunk_ids"], ["micro"])
        self.assertEqual(normalized["missing_axes"], [])

    def test_source_review_covered_axes_need_useful_chunk_ids_even_when_not_ready(self):
        chunks = [{"rank": 1, "chunk_id": "candidate", "boi_reference": "BOI-A-1", "title": "Candidate"}]
        review = {
            "coverage_status": "needs_more_sources",
            "useful_chunk_ids": [],
            "covered_axes": ["Territorialite B2B"],
            "missing_axes": [{"axis": "Franchise", "search_query": "franchise TVA", "bofip_prefix": "TVA"}],
            "rejected_chunks": [],
        }

        normalized = _normalize_source_review(review, chunks)

        self.assertEqual(normalized["covered_axes"], [])
        self.assertIn("Territorialite B2B", [item["axis"] for item in normalized["missing_axes"]])

    def test_empty_source_review_relaunches_blocking_plan_axes(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Question CFE.",
                    "facts": ["avis CFE"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "Cotisation minimum CFE",
                            "goal": "Verifier cotisation minimum et seuil chiffre affaires",
                            "bofip_prefix": "CF",
                            "search_query": "CFE cotisation minimum seuil chiffre affaires",
                            "role": "calculation",
                            "blocking": True,
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": [],
                    "rejected_chunks": [],
                    "covered_axes": [],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["chunk-1", "chunk-2"],
                    "rejected_chunks": [],
                    "covered_axes": ["Cotisation minimum CFE"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Reponse sourcee.",
                    "axes_requis": ["Cotisation minimum CFE"],
                    "axes_couverts": ["Cotisation minimum CFE"],
                    "axes_manquants": [],
                    "justification_bullets": ["Axe couvert."],
                    "limits": "",
                }
            ),
        ]
        runtime = _FakeRuntime()
        agent = AgenticRAG(runtime, client=_FakeClient(responses), max_iterations=2, use_reranker=False)

        result = agent.run("Avis CFE et faible chiffre d'affaires.")

        self.assertEqual(result["answer_status"], "supported")
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual([call["boost_prefix"] for call in runtime.calls], ["CF", "CF"])
        self.assertIn("CFE", runtime.calls[0]["query"])
        self.assertIn("CFE", runtime.calls[1]["query"])
        self.assertEqual(result["trace"][0]["relaunches"][0]["stage"], "global_relaunch")

    def test_source_review_keeps_non_blocking_axes_out_of_relaunch_queue(self):
        chunks = [{"rank": 1, "chunk_id": "candidate", "boi_reference": "BOI-ENR-1", "title": "Don manuel"}]
        review = {
            "coverage_status": "ready",
            "useful_chunk_ids": ["candidate"],
            "covered_axes": ["Declaration don manuel", "Abattement parent-enfant"],
            "missing_axes": [
                {
                    "axis": "Present d'usage",
                    "search_query": "present usage donation",
                    "bofip_prefix": "ENR",
                    "why_needed": "alternative a mentionner seulement si les faits le suggerent",
                    "role": "alternative",
                    "blocking": False,
                }
            ],
            "rejected_chunks": [],
        }

        normalized = _normalize_source_review(review, chunks)

        self.assertEqual(normalized["coverage_status"], "ready")
        self.assertEqual(normalized["missing_axes"], [])
        self.assertEqual(normalized["non_blocking_axes"][0]["axis"], "Present d'usage")

    def test_candidate_refs_for_missing_axis_prioritizes_axis_overlap_over_route_order(self):
        missing = {
            "axis": "Charges de copropriete deductibles",
            "search_query": "provisions charges copropriete syndic regularisation locataire",
            "bofip_prefix": "RFPI",
            "why_needed": "qualifier les charges deductibles au regime reel",
        }
        route_log = [
            {
                "facet": {"prefix": "RFPI"},
                "stage1_refs": ["BOI-RFPI-CHAMP-10", "BOI-RFPI-BASE-20-70"],
            }
        ]
        chunks = [
            {
                "rank": 1,
                "chunk_id": "generic",
                "boi_reference": "BOI-RFPI-CHAMP-10",
                "title": "Revenus fonciers - Champ d'application",
                "section_path": "Champ",
                "text": "Regles generales des revenus fonciers.",
            },
            {
                "rank": 2,
                "chunk_id": "copro",
                "boi_reference": "BOI-RFPI-BASE-20-70",
                "title": "Charges de copropriete",
                "section_path": "Provisions pour charges de copropriete",
                "text": "Provisions versees au syndic puis regularisation des charges recuperables.",
            },
        ]

        refs = _candidate_refs_for_missing_axis(missing, chunks, route_log)

        self.assertEqual(refs[0], "BOI-RFPI-BASE-20-70")

    def test_candidate_refs_for_missing_axis_keeps_off_prefix_textual_match(self):
        missing = {
            "axis": "Axe documentaire precis",
            "search_query": "preuve documentaire precise",
            "bofip_prefix": "TVA",
            "why_needed": "retrouver la source utile meme si le prefixe LLM est faux",
        }
        route_log = [{"facet": {"prefix": "TVA"}, "stage1_refs": ["BOI-TVA-GENERIC"]}]
        chunks = [
            {
                "rank": 1,
                "chunk_id": "off-prefix",
                "boi_reference": "BOI-BIC-SPEC",
                "title": "Preuve documentaire precise",
                "section_path": "Axe documentaire precis",
                "text": "Preuve documentaire precise utile.",
            },
        ]

        refs = _candidate_refs_for_missing_axis(missing, chunks, route_log)

        self.assertIn("BOI-BIC-SPEC", refs)

    def test_candidate_refs_for_missing_axis_prioritizes_stage2_candidate_docs(self):
        missing = {
            "axis": "Axe documentaire precis",
            "search_query": "preuve documentaire precise",
            "bofip_prefix": "TVA",
            "why_needed": "retrouver le document qui a deja produit des passages candidats",
        }
        route_log = [
            {
                "facet": {"prefix": "TVA"},
                "stage1_refs": ["BOI-TVA-GENERIC", "BOI-TVA-SPEC"],
                "stage2_refs": ["BOI-TVA-SPEC"],
                "selected_refs": [],
            }
        ]

        refs = _candidate_refs_for_missing_axis(missing, [], route_log)

        self.assertEqual(refs[0], "BOI-TVA-SPEC")

    def test_candidate_refs_for_missing_axis_prefers_axis_matching_route_refs(self):
        missing = {
            "axis": "Cloture du plan en cas de retrait PEA avant cinq ans",
            "search_query": "PEA retrait partiel avant cinq ans cloture du plan gain net consequences fiscales",
            "bofip_prefix": "RPPM",
            "why_needed": "verifier les consequences propres au retrait PEA",
        }
        route_log = [
            {
                "facet": {
                    "name": "Imposition des revenus de capitaux mobiliers",
                    "goal": "Verifier le regime general d'imposition au bareme",
                    "query": "revenus distribues option bareme abattement",
                    "prefix": "RPPM",
                },
                "selected_refs": [
                    "BOI-RPPM-RCM-20-10-20-50",
                    "BOI-RPPM-PVBMI-30-20",
                ],
            },
            {
                "facet": {
                    "name": "Retrait partiel PEA avant cinq ans",
                    "goal": "Verifier si le retrait entraine la cloture du plan et le traitement du gain net",
                    "query": "PEA retrait partiel avant cinq ans cloture plan gain net",
                    "prefix": "RPPM",
                },
                "stage2_refs": [
                    "BOI-RPPM-RCM-40-50-40",
                    "BOI-ANNX-000072",
                ],
            },
        ]

        refs = _candidate_refs_for_missing_axis(missing, [], route_log)

        self.assertLess(refs.index("BOI-RPPM-RCM-40-50-40"), refs.index("BOI-RPPM-RCM-20-10-20-50"))
        self.assertIn("BOI-ANNX-000072", refs[:3])

    def test_candidate_refs_include_sibling_document_under_same_parent_prefix(self):
        missing = {
            "axis": "Cotisation minimum et seuil de chiffre d'affaires",
            "search_query": "cotisation minimum CFE chiffre affaires recettes faible seuil",
            "bofip_prefix": "IF-CFE-20-20-40-20",
            "why_needed": "retrouver le passage de fond dans un document voisin du meme parent BOFiP",
        }
        route_log = [
            {
                "facet": {"prefix": "IF"},
                "stage1_refs": [
                    "BOI-IF-CFE-20-20-40-20-20180905",
                    "BOI-IF-CFE-20-20-40-10-20230628",
                ],
            }
        ]
        chunks = [
            {
                "rank": 1,
                "chunk_id": "generic-minimum",
                "boi_reference": "BOI-IF-CFE-20-20-40-20-20180905",
                "title": "Dispositif de convergence des bases minimum",
                "section_path": "Cotisation minimum",
                "text": "Regles transitoires.",
            },
            {
                "rank": 2,
                "chunk_id": "sibling-threshold",
                "boi_reference": "BOI-IF-CFE-20-20-40-10-20230628",
                "title": "Cotisation minimum - Regles generales",
                "section_path": "Exoneration de cotisation minimum des contribuables a faible chiffre d'affaires",
                "text": "Seuil de chiffre d'affaires ou recettes.",
            },
        ]

        refs = _candidate_refs_for_missing_axis(missing, chunks, route_log)

        self.assertIn("BOI-IF-CFE-20-20-40-10-20230628", refs[:3])

    def test_answer_prompt_forbids_treating_missing_axes_as_proven(self):
        plan = SearchPlan(
            reformulated_question="Question CFE.",
            facts=["domicile sans local"],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Base locative CFE",
                    goal="Verifier l'effet de l'absence de local dedie",
                    query="CFE domicile sans local valeur locative",
                    prefix="IF",
                )
            ],
        )
        review = {
            "covered_axes": ["Cotisation minimum faible chiffre d'affaires"],
            "missing_axes": [
                {
                    "axis": "Base locative CFE",
                    "why_needed": "l'absence de local peut changer le raisonnement",
                }
            ],
        }

        prompt = _build_answer_question("Question originale.", plan, review)

        self.assertNotIn("AXES FRAGILES A NE PAS AFFIRMER COMME PROUVES", prompt)
        self.assertNotIn("l'absence de local peut changer le raisonnement", prompt)
        self.assertIn("Le reviewer documentaire a seulement servi a trouver plus de sources", prompt)

    def test_validated_initial_sources_survive_after_relaunch_review(self):
        responses = [
            json.dumps(
                {
                    "reformulated_question": "Revenus fonciers avec travaux et charges.",
                    "facts": ["location nue", "travaux", "charges de copropriete"],
                    "ambiguities": [],
                    "facets": [
                        {
                            "name": "Travaux deductibles",
                            "goal": "Verifier les travaux d'entretien et reparation",
                            "bofip_prefix": "RFPI",
                            "search_query": "revenus fonciers travaux entretien reparation deductibles",
                            "priority": 1,
                        }
                    ],
                    "excluded_axes": [],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "needs_more_sources",
                    "useful_chunk_ids": ["chunk-1"],
                    "rejected_chunks": [],
                    "covered_axes": ["Travaux deductibles"],
                    "missing_axes": [
                        {
                            "axis": "Charges copropriete",
                            "bofip_prefix": "RFPI",
                            "search_query": "provisions charges copropriete syndic regularisation",
                            "why_needed": "verifier les charges au reel",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "coverage_status": "ready",
                    "useful_chunk_ids": ["rescue-copro"],
                    "rejected_chunks": [],
                    "covered_axes": ["Charges copropriete"],
                    "missing_axes": [],
                }
            ),
            json.dumps(
                {
                    "answer_status": "supported",
                    "conclusion": "Reponse sourcee.",
                    "axes_requis": ["Travaux deductibles", "Charges copropriete"],
                    "axes_couverts": ["Travaux deductibles", "Charges copropriete"],
                    "axes_manquants": [],
                    "justification_bullets": ["Les deux sources sont transmises."],
                    "limits": "",
                }
            ),
        ]
        runtime = _FakeRuntime()
        runtime.intra_results.append(
            SimpleNamespace(
                query="RFPI provisions charges copropriete syndic regularisation",
                stage1_hits=[
                    SimpleNamespace(rank=1, score=1.0, boi_reference="BOI-RFPI-DOC", title="Charges copropriete")
                ],
                stage2_chunks=[
                    SimpleNamespace(
                        boi_reference="BOI-RFPI-DOC",
                        title="Charges copropriete",
                        publication_date="2026-01-01",
                        section_path="Charges > Copropriete",
                        text="Passage sur les provisions de copropriete.",
                        chunk_id="rescue-copro",
                    )
                ],
            )
        )
        client = _FakeClient(responses)
        agent = AgenticRAG(runtime, client=client, max_iterations=2, use_reranker=False)

        result = agent.run("Location nue avec travaux et charges de copropriete.")

        self.assertEqual(result["answer_status"], "supported")
        answer_prompt = client.chat.completions.calls[-1]["messages"][1]["content"]
        self.assertIn("Source RFPI", answer_prompt)
        self.assertIn("Extrait BOFiP.", answer_prompt)
        self.assertIn("Charges copropriete", answer_prompt)
        self.assertIn("Passage sur les provisions de copropriete.", answer_prompt)

    def test_partial_answer_with_all_required_axes_covered_is_repaired_to_supported(self):
        answer = {
            "answer_status": "partial",
            "axes_requis": ["IS", "TVA"],
            "axes_couverts": ["IS", "TVA"],
            "axes_manquants": [],
        }

        cleaned = _clean_answer_status(answer)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])
        self.assertEqual(_compute_coverage(cleaned), 1.0)

    def test_supported_answer_with_declared_missing_axis_is_not_downgraded(self):
        answer = {
            "answer_status": "supported",
            "axes_requis": [],
            "axes_couverts": ["Regle principale"],
            "axes_manquants": ["Axe substantif non prouve"],
        }

        cleaned = _clean_answer_status(answer)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_empty_supported_answer_is_rejected_without_principal_signal(self):
        answer = {
            "answer_status": "supported",
            "axes_requis": ["Regle principale"],
            "axes_couverts": [],
            "axes_manquants": [],
            "justification_bullets": [],
            "limits": "",
        }

        cleaned = _clean_answer_status(answer)

        self.assertEqual(cleaned["answer_status"], "insufficient_evidence")
        self.assertTrue(cleaned["axes_manquants"])

    def test_supported_answer_is_not_downgraded_from_limit_text_without_missing_axis(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "Le calcul applique un abattement, mais le taux n'est pas explicitement cite dans les extraits.",
            "axes_requis": ["Calcul de la base"],
            "axes_couverts": ["Calcul de la base"],
            "axes_manquants": [],
            "justification_bullets": ["Calcul effectue avec un taux non prouve par les extraits."],
            "limits": "",
        }

        cleaned = _clean_answer_status(answer)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_partial_answer_without_structured_missing_axis_is_promoted(self):
        answer = {
            "answer_status": "partial",
            "conclusion": "Le taux d'abattement forfaitaire n'est pas fourni dans les extraits.",
            "axes_requis": ["Calcul de la base"],
            "axes_couverts": ["Calcul de la base"],
            "axes_manquants": [],
            "justification_bullets": [],
            "limits": "",
        }

        cleaned = _clean_answer_status(answer)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_unrequested_limit_confession_does_not_force_partial(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "Les loyers sont sous le seuil du micro-foncier.",
            "axes_requis": ["Micro-foncier eligibilite"],
            "axes_couverts": ["Micro-foncier eligibilite"],
            "axes_manquants": [],
            "justification_bullets": ["Le seuil de recettes est documente."],
            "limits": "Le calcul du revenu imposable, abattement non precise dans les extraits, n'est pas aborde.",
        }
        plan = SearchPlan(
            reformulated_question="Eligibilite micro-foncier pour 12 000 euros de loyers nus.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Micro-foncier eligibilite",
                    goal="Verifier si les loyers peuvent etre declares au regime micro-foncier.",
                    query="micro-foncier seuil recettes location nue",
                    prefix="RFPI",
                    expected_evidence=["Seuil de recettes annuelles", "Location nue"],
                )
            ],
        )

        cleaned = _clean_answer_status(answer, plan=plan)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_requested_evidence_confession_without_structured_missing_axis_is_not_forced_partial(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "Le calcul applique un abattement, mais le taux n'est pas explicitement cite dans les extraits.",
            "axes_requis": ["Calcul de l'abattement"],
            "axes_couverts": ["Calcul de l'abattement"],
            "axes_manquants": [],
            "justification_bullets": [],
            "limits": "",
        }
        plan = SearchPlan(
            reformulated_question="Calculer une base apres abattement.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Calcul de l'abattement",
                    goal="Calculer le montant apres application de l'abattement.",
                    query="taux abattement calcul base",
                    prefix="RPPM",
                    expected_evidence=["Taux de l'abattement applicable"],
                )
            ],
        )

        cleaned = _clean_answer_status(answer, plan=plan)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_supported_answer_is_not_downgraded_when_source_review_has_blocking_missing_axis(self):
        answer = {
            "answer_status": "supported",
            "axes_requis": ["Territorialite B2B", "Franchise en base"],
            "axes_couverts": ["Territorialite B2B"],
            "axes_manquants": [],
        }
        review = {
            "missing_axes": [
                {"axis": "Franchise en base", "blocking": True, "why_needed": "source de fond manquante"}
            ]
        }

        cleaned = _clean_answer_status(answer, source_review=review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_supported_answer_is_not_downgraded_by_axis_label_mismatch_alone(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "La reponse principale est calculee et sourcee.",
            "axes_requis": ["Base apres abattement"],
            "axes_couverts": ["Calcul de la base taxable"],
            "axes_manquants": [],
            "justification_bullets": ["Le calcul demande est cite."],
            "limits": "Reserve d'eligibilite non bloquante.",
        }
        source_review = {
            "coverage_status": "ready",
            "missing_axes": [],
            "non_blocking_axes": [],
        }

        cleaned = _clean_answer_status(answer, source_review=source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_partial_status_is_not_recreated_from_plan_facets_without_blocking_review(self):
        plan = SearchPlan(
            reformulated_question="Calculer une base fiscale principale.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Qualification fiscale du revenu",
                    goal="Qualifier le regime.",
                    query="qualification fiscale revenu",
                    prefix="RPPM",
                    role="core",
                    blocking=True,
                ),
                SearchFacet(
                    name="Calcul de l'assiette",
                    goal="Calculer l'assiette demandee.",
                    query="calcul assiette abattement",
                    prefix="RPPM",
                    role="calculation",
                    blocking=True,
                ),
            ],
        )
        answer = {
            "answer_status": "partial",
            "conclusion": "La base demandee est calculee.",
            "axes_requis": ["Base demandee"],
            "axes_couverts": ["Base demandee"],
            "axes_manquants": [],
            "justification_bullets": ["Sources citees."],
            "limits": "Reserve non bloquante.",
        }
        source_review = {
            "coverage_status": "ready",
            "missing_axes": [],
            "non_blocking_axes": [],
        }

        cleaned = _clean_answer_status(answer, plan, source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_final_supported_answer_drops_stale_missing_axes_already_covered(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "Reponse sourcee.",
            "axes_requis": [
                "Assujettissement a la CFE du micro-entrepreneur",
                "Exoneration de CFE pour faible chiffre d'affaires (cotisation minimum)",
            ],
            "axes_couverts": [
                "Assujettissement a la CFE du micro-entrepreneur",
                "Exoneration de CFE pour faible chiffre d'affaires (cotisation minimum)",
            ],
            "axes_manquants": [
                "Assujettissement a la CFE du micro-entrepreneur",
                "Cotisation minimum de CFE",
                "Exoneration temporaire creation",
                "Absence de local professionnel",
            ],
            "justification_bullets": ["Sources citees."],
            "limits": "Reserves non bloquantes.",
        }
        source_review = {
            "missing_axes": [
                {"axis": "Assujettissement a la CFE du micro-entrepreneur", "blocking": True},
                {"axis": "Cotisation minimum de CFE", "blocking": True},
                {"axis": "Exoneration temporaire creation", "blocking": True},
                {"axis": "Absence de local professionnel", "blocking": True},
            ]
        }

        cleaned = _clean_answer_status(answer, source_review=source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_complete_calculation_with_qualification_reserve_is_supported(self):
        answer = {
            "answer_status": "partial",
            "conclusion": "La plus-value brute avant abattements est de 55 000 euros.",
            "axes_requis": [
                "Prix de cession",
                "Prix d'acquisition",
                "Frais d'acquisition non justifies",
                "Travaux sans factures",
            ],
            "axes_couverts": [
                "Prix de cession",
                "Prix d'acquisition",
                "Frais d'acquisition non justifies",
                "Travaux sans factures",
            ],
            "axes_manquants": ["Champ plus-value immobiliere particuliers"],
            "justification_bullets": [
                "Prix d'acquisition majore: 200000 + 15000 + 30000.",
                "Plus-value brute: 300000 - 245000 = 55000.",
            ],
            "limits": "Sous reserve que le vendeur releve du regime des plus-values immobilieres des particuliers.",
        }
        source_review = {
            "missing_axes": [
                {
                    "axis": "Champ plus-value immobiliere particuliers",
                    "why_needed": "Qualification du regime applicable a signaler en reserve.",
                    "blocking": True,
                }
            ]
        }

        cleaned = _clean_answer_status(answer, source_review=source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_complete_pvbmi_calculation_with_untriggered_enhanced_abatement_is_supported(self):
        plan = SearchPlan(
            reformulated_question="Calculer une plus-value mobiliere apres abattement de droit commun.",
            facts=["Actions acquises en 2016 et revendues en 2024."],
            ambiguities=["La nature exacte des titres n'est pas precisee."],
            facets=[
                SearchFacet(
                    name="Plus-values mobilieres des particuliers",
                    goal="Identifier le regime applicable.",
                    query="plus-values cession actions personne physique",
                    prefix="RPPM",
                    role="core",
                    blocking=True,
                ),
                SearchFacet(
                    name="Option pour le bareme",
                    goal="Verifier l'effet de l'option pour le bareme.",
                    query="option bareme plus-values mobilieres abattement",
                    prefix="RPPM",
                    role="calculation",
                    blocking=True,
                ),
                SearchFacet(
                    name="Abattement duree de detention",
                    goal="Calculer le taux d'abattement applicable.",
                    query="abattement duree detention 65 titres acquis avant 2018",
                    prefix="RPPM",
                    role="calculation",
                    blocking=True,
                ),
                SearchFacet(
                    name="Abattement renforce eventuel",
                    goal="Reserve sur des conditions particulieres non precisees par la question.",
                    query="abattement renforce conditions particulieres",
                    prefix="RPPM",
                    role="reserve",
                    blocking=False,
                ),
            ],
        )
        answer = {
            "answer_status": "partial",
            "conclusion": "La plus-value retenue apres abattement serait de 10 500 euros.",
            "axes_requis": [
                "Plus-values mobilieres des particuliers",
                "Option pour le bareme",
                "Abattement duree de detention",
                "Determination plus-value brute",
                "Abattement renforce eventuel",
            ],
            "axes_couverts": [
                "Plus-values mobilieres des particuliers",
                "Option pour le bareme",
                "Abattement duree de detention",
                "Determination plus-value brute",
            ],
            "axes_manquants": ["Abattement renforce eventuel"],
            "justification_bullets": ["30 000 euros x 65 % = 19 500 euros."],
            "limits": "Le calcul suppose que les titres sont eligibles et ne traite pas l'abattement renforce.",
        }
        source_review = {
            "missing_axes": [
                {
                    "axis": "Abattement renforce eventuel",
                    "search_query": "abattement renforce duree detention titres PME conditions",
                    "why_needed": "conditions particulieres non precisees par les faits utilisateur",
                    "role": "calculation",
                    "blocking": True,
                }
            ]
        }

        cleaned = _clean_answer_status(answer, plan, source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])
        self.assertNotIn("Abattement renforce eventuel", cleaned["axes_requis"])

    def test_supported_answer_is_not_downgraded_for_reserve_only_in_limits(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "La plus-value brute avant abattements est de 55 000 euros.",
            "axes_requis": ["Calcul de la plus-value brute"],
            "axes_couverts": ["Calcul de la plus-value brute"],
            "axes_manquants": [],
            "justification_bullets": ["Les sources prouvent le calcul demande."],
            "limits": "Sous reserve que le vendeur releve du regime des plus-values immobilieres des particuliers.",
        }
        source_review = {
            "missing_axes": [
                {
                    "axis": "Champ plus-value immobiliere particuliers",
                    "search_query": "champ application plus-values immobilieres particuliers",
                    "why_needed": "Qualification du regime applicable.",
                    "blocking": True,
                }
            ]
        }

        cleaned = _clean_answer_status(answer, source_review=source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_nonblocking_plan_reserve_axis_does_not_force_partial_status(self):
        plan = SearchPlan(
            reformulated_question="Calculer un benefice imposable sous un regime suppose applicable.",
            facts=["Le regime est pose comme hypothese par la question."],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Determination du benefice imposable",
                    goal="Identifier la methode de calcul.",
                    query="benefice imposable abattement recettes",
                    prefix="BA",
                    role="calculation",
                    blocking=True,
                ),
                SearchFacet(
                    name="Changement ou sortie de regime",
                    goal="Reserve sur les exclusions, options ou sorties du regime.",
                    query="changement sortie regime option exclusion",
                    prefix="BA",
                    role="reserve",
                    blocking=False,
                ),
            ],
        )
        answer = {
            "answer_status": "partial",
            "conclusion": "Le benefice imposable est de 6 500 euros.",
            "axes_requis": ["Determination du benefice imposable", "Changement ou sortie de regime"],
            "axes_couverts": ["Determination du benefice imposable"],
            "axes_manquants": ["Changement ou sortie de regime"],
            "justification_bullets": ["Moyenne triennale puis abattement prouves."],
            "limits": "Calcul sous reserve d'absence d'exclusion, option, creation, cessation ou changement de regime.",
        }
        source_review = {
            "missing_axes": [
                {
                    "axis": "Changement ou sortie de regime",
                    "search_query": "changement sortie regime option exclusion",
                    "why_needed": "reserve sur les exclusions ou options non declenchees par les faits",
                    "blocking": True,
                }
            ]
        }

        cleaned = _clean_answer_status(answer, plan, source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])
        self.assertEqual(cleaned["axes_requis"], ["Determination du benefice imposable"])
        self.assertEqual(_compute_coverage(cleaned), 1.0)

    def test_nonblocking_source_reserve_axis_does_not_force_partial_status(self):
        answer = {
            "answer_status": "partial",
            "conclusion": "La reponse principale est prouvee.",
            "axes_requis": ["Regle principale", "Exception locale non demandee"],
            "axes_couverts": ["Regle principale"],
            "axes_manquants": ["Exception locale non demandee"],
            "justification_bullets": ["La regle principale est citee."],
            "limits": "L'exception locale n'est pas traitee car aucun fait ne la declenche.",
        }
        source_review = {
            "missing_axes": [],
            "non_blocking_axes": [
                {
                    "axis": "Exception locale non demandee",
                    "why_needed": "precision utile mais non bloquante",
                    "role": "reserve",
                    "blocking": False,
                }
            ],
        }

        cleaned = _clean_answer_status(answer, source_review=source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_requis"], ["Regle principale"])
        self.assertEqual(cleaned["axes_manquants"], [])
        self.assertEqual(_compute_coverage(cleaned), 1.0)

    def test_scope_clarification_missing_axis_does_not_force_partial_status(self):
        plan = SearchPlan(
            reformulated_question="Calculer le montant retenu apres abattement.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Abattement sur dividendes",
                    goal="Calculer la base retenue apres abattement.",
                    query="dividendes abattement base retenue",
                    prefix="RPPM",
                    role="calculation",
                    blocking=True,
                ),
                SearchFacet(
                    name="Prelevements sociaux",
                    goal="Verifier uniquement la distinction pour eviter une confusion hors perimetre.",
                    query="prelevements sociaux distinction assiette",
                    prefix="RPPM",
                    role="reserve",
                    blocking=False,
                ),
            ],
        )
        answer = {
            "answer_status": "partial",
            "conclusion": "Le montant retenu apres abattement est de 3 600 euros.",
            "axes_requis": ["Abattement sur dividendes", "Prelevements sociaux"],
            "axes_couverts": ["Abattement sur dividendes"],
            "axes_manquants": ["Prelevements sociaux"],
            "justification_bullets": ["Le calcul demande est source."],
            "limits": "Cette reponse ne traite pas les prelevements sociaux.",
        }
        source_review = {
            "missing_axes": [
                {
                    "axis": "Prelevements sociaux",
                    "search_query": "prelevements sociaux assiette dividendes",
                    "why_needed": "utile seulement pour eviter une confusion hors perimetre",
                    "role": "core",
                    "blocking": True,
                }
            ]
        }

        cleaned = _clean_answer_status(answer, plan, source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_requis"], ["Abattement sur dividendes"])
        self.assertEqual(cleaned["axes_manquants"], [])
        self.assertEqual(_compute_coverage(cleaned), 1.0)

    def test_source_review_demotes_missing_axis_that_matches_nonblocking_plan_facet(self):
        plan = SearchPlan(
            reformulated_question="Calculer une base principale.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Regle principale",
                    goal="Prouver le calcul demande.",
                    query="regle principale",
                    prefix="RPPM",
                    role="calculation",
                    blocking=True,
                ),
                SearchFacet(
                    name="Precision hors perimetre",
                    goal="Clarification utile mais non bloquante.",
                    query="precision hors perimetre",
                    prefix="RPPM",
                    role="reserve",
                    blocking=False,
                ),
            ],
        )
        review = {
            "coverage_status": "needs_more_sources",
            "covered_axes": ["Regle principale"],
            "useful_chunk_ids": ["chunk-1"],
            "missing_axes": [
                {
                    "axis": "Precision hors perimetre",
                    "bofip_prefix": "RPPM",
                    "search_query": "precision hors perimetre",
                    "why_needed": "utile seulement pour cadrer la reponse",
                    "role": "core",
                    "blocking": True,
                }
            ],
            "non_blocking_axes": [],
        }
        chunks = [
            {
                "chunk_id": "chunk-1",
                "facet": "Regle principale",
                "text": "Source utile pour la regle principale.",
            }
        ]

        completed = _complete_source_review_with_plan_gaps(review, plan, chunks)

        self.assertEqual(completed["missing_axes"], [])
        self.assertEqual(completed["coverage_status"], "ready")
        self.assertEqual(completed["non_blocking_axes"][0]["axis"], "Precision hors perimetre")
        self.assertFalse(completed["non_blocking_axes"][0]["blocking"])

    def test_nonblocking_review_axis_does_not_satisfy_blocking_plan_facet(self):
        plan = SearchPlan(
            reformulated_question="Comparer deux regimes.",
            facts=[],
            ambiguities=[],
            facets=[
                SearchFacet(
                    name="Micro-foncier : conditions et abattement",
                    goal="Prouver le seuil et l'abattement.",
                    query="micro-foncier seuil abattement forfaitaire",
                    prefix="RFPI",
                    role="core",
                    blocking=True,
                )
            ],
        )
        review = {
            "coverage_status": "ready",
            "covered_axes": [],
            "useful_chunk_ids": [],
            "missing_axes": [],
            "non_blocking_axes": [
                {
                    "axis": "Micro-foncier : conditions et abattement",
                    "role": "reserve",
                    "blocking": False,
                    "why_needed": "utile en limite",
                }
            ],
        }

        completed = _complete_source_review_with_plan_gaps(review, plan, [])

        self.assertEqual(completed["coverage_status"], "needs_more_sources")
        self.assertEqual(completed["missing_axes"][0]["axis"], "Micro-foncier : conditions et abattement")

    def test_supported_answer_is_not_vetoed_by_source_review_missing_axis(self):
        answer = {
            "answer_status": "supported",
            "conclusion": "La base locative est nulle, donc aucune cotisation n'est due.",
            "axes_requis": ["Cotisation minimum faible chiffre d'affaires"],
            "axes_couverts": ["Cotisation minimum faible chiffre d'affaires"],
            "axes_manquants": [],
            "justification_bullets": ["La base locative CFE serait nulle."],
            "limits": "",
        }
        source_review = {
            "missing_axes": [
                {
                    "axis": "Valeur locative activite domicile",
                    "why_needed": "confirmer la base nette en l'absence de local",
                    "blocking": True,
                }
            ]
        }

        cleaned = _clean_answer_status(answer, source_review=source_review)

        self.assertEqual(cleaned["answer_status"], "supported")
        self.assertEqual(cleaned["axes_manquants"], [])

    def test_numeric_chunk_query_adds_generic_threshold_terms_without_user_amount(self):
        facet = SearchFacet(
            name="Cotisation minimum CFE",
            goal="Verifier cotisation minimum selon chiffre d'affaires",
            query="CFE cotisation minimum chiffre affaires",
            prefix="IF",
            expected_evidence=["seuil applicable"],
        )

        query = _build_facet_chunk_query(
            facet,
            "Micro-entrepreneur avec chiffre d'affaires de 3 200 euros.",
        )

        self.assertIn("faible chiffre affaires", query)
        self.assertIn("seuil", query)
        self.assertNotIn("3200", query)

    def test_facet_queries_do_not_include_boi_like_expected_evidence(self):
        facet = SearchFacet(
            name="Axe fiscal",
            goal="Identifier la regle de fond applicable",
            query="regle applicable",
            prefix="IF",
            expected_evidence=[
                "BOI-CFE-EXO-20-10 sur une reference plausible mais non valide",
                "preuve textuelle du champ d'application",
            ],
        )

        retrieval_query = _build_facet_retrieval_query("Question utilisateur", facet)
        chunk_query = _build_facet_chunk_query(facet, "Question utilisateur")

        self.assertNotIn("BOI-CFE-EXO-20-10", retrieval_query)
        self.assertNotIn("BOI-CFE-EXO-20-10", chunk_query)
        self.assertIn("preuve textuelle", retrieval_query)
        self.assertIn("preuve textuelle", chunk_query)


if __name__ == "__main__":
    unittest.main()
