from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from bofip_agentic.eval_harness import QueryGold
from scripts import eval_retrieval_layers as eval_script


class EvalRetrievalLayersScriptTests(unittest.TestCase):
    def test_normalize_doc_ref_removes_live_bofip_date_suffix(self) -> None:
        self.assertEqual(
            "BOI-TVA-DECLA-10-10-20",
            eval_script.normalize_doc_ref("BOI-TVA-DECLA-10-10-20-20251022"),
        )
        self.assertEqual("BOI-IF-CFE-20-20-40-10", eval_script.normalize_doc_ref("boi-if-cfe-20-20-40-10"))

    def test_normalize_doc_refs_in_queries_leaves_chunk_ids_exact(self) -> None:
        queries = [
            QueryGold(
                query_id="q1",
                query="Question",
                category="direct",
                gold_doc_refs=["BOI-TVA-DECLA-10-10-20-20251022"],
                gold_chunk_ids=["3218-PGP__section_window__legacy"],
            )
        ]

        normalized = eval_script.normalize_doc_refs_in_queries(queries)

        self.assertEqual(["BOI-TVA-DECLA-10-10-20"], normalized[0].gold_doc_refs)
        self.assertEqual(["3218-PGP__section_window__legacy"], normalized[0].gold_chunk_ids)

    def test_build_gold_compatibility_reports_exact_and_normalized_doc_matches(self) -> None:
        queries = [
            QueryGold(
                query_id="q1",
                query="Question",
                category="direct",
                gold_doc_refs=["BOI-TVA-DECLA-10-10-20-20251022"],
                gold_chunk_ids=["legacy-chunk"],
            )
        ]

        report = eval_script.build_gold_compatibility_report(
            queries,
            active_doc_refs={"BOI-TVA-DECLA-10-10-20"},
            active_chunk_ids={"new-chunk"},
        )

        self.assertEqual(1, report["gold_doc_refs"])
        self.assertEqual(0, report["gold_doc_refs_exact_matches"])
        self.assertEqual(1, report["gold_doc_refs_normalized_matches"])
        self.assertEqual(1, report["gold_chunk_ids"])
        self.assertEqual(0, report["gold_chunk_ids_exact_matches"])

    def test_load_eval_queries_supports_v2_expected_fields(self) -> None:
        row = {
            "query_id": "tva_001",
            "question": "Quel taux de TVA pour un livre numérique ?",
            "theme": "TVA",
            "expected_doc_refs": ["BOI-TVA-LIQ-30-10-40"],
            "expected_section_terms": ["livres numériques", "taux réduit"],
            "expected_text_terms": ["support physique", "téléchargement"],
            "must_not_match_refs": ["BOI-RSA-BASE-10-10"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            queries = eval_script.load_eval_queries(path)

        self.assertEqual(1, len(queries))
        self.assertEqual("tva_001", queries[0].query_id)
        self.assertEqual(["BOI-TVA-LIQ-30-10-40"], queries[0].gold_doc_refs)
        self.assertEqual(["livres numériques", "taux réduit"], queries[0].expected_section_terms)
        self.assertEqual(["support physique", "téléchargement"], queries[0].expected_text_terms)
        self.assertEqual(["BOI-RSA-BASE-10-10"], queries[0].must_not_match_refs)

    def test_term_coverage_rank_accumulates_terms_across_top_k(self) -> None:
        rank = eval_script.term_coverage_rank(
            [
                "Règles générales de liquidation.",
                "Le taux réduit vise les livres numériques fournis par téléchargement.",
            ],
            ["livres numériques", "téléchargement"],
        )

        self.assertEqual(2, rank)

    def test_runtime_layers_exposes_candidate_text_and_disable_anchor_filter(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="chunk-1",
            section_path=["TVA", "Taux réduit"],
            text="Les livres numériques peuvent relever du taux réduit.",
        )

        class FakeRuntime:
            chunks = [chunk]

            def __init__(self) -> None:
                self.kwargs = {}

            def retrieve(self, query: str, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    stage1_hits=[SimpleNamespace(boi_reference="BOI-TVA-LIQ-30-10-40")],
                    stage2_chunks=[
                        SimpleNamespace(
                            boi_reference="BOI-TVA-LIQ-30-10-40",
                            chunk_id="chunk-1",
                            section_path="TVA > Taux réduit",
                            text="Les livres numériques peuvent relever du taux réduit.",
                        )
                    ],
                    pipeline_log={
                        "stage2_candidate_doc_refs": ["BOI-TVA-LIQ-30-10-40"],
                        "stage2_candidate_chunk_ids": ["chunk-1"],
                    },
                )

        runtime = FakeRuntime()

        layers = eval_script._runtime_layers(
            runtime,
            "question",
            mode="hybrid",
            use_reranker=False,
            normalize_doc_refs=True,
            top_docs=12,
            chunks_per_doc=8,
            max_chunks=12,
            use_anchor_filter=False,
        )

        self.assertFalse(runtime.kwargs["use_anchor_filter"])
        self.assertEqual(["TVA > Taux réduit"], layers["stage2_candidate_sections"])
        self.assertEqual(["Les livres numériques peuvent relever du taux réduit."], layers["final_texts"])

    def test_build_corpus_sanity_report_counts_missing_chunks_and_families(self) -> None:
        docs = [
            SimpleNamespace(
                document_id="doc-tva",
                boi_reference="BOI-TVA-LIQ-30-10-40",
                document_type="BOI",
            ),
            SimpleNamespace(
                document_id="doc-rsa",
                boi_reference="BOI-RSA-BASE-30-10-20",
                document_type="BOI",
            ),
        ]
        chunks = [
            SimpleNamespace(
                chunk_id="chunk-tva",
                document_id="doc-tva",
                boi_reference="BOI-TVA-LIQ-30-10-40",
                section_path=["TVA"],
                paragraph_range=["1", "2"],
                text="Taux réduit.",
            )
        ]

        report = eval_script.build_corpus_sanity_report(docs, chunks)

        self.assertEqual(2, report["documents_count"])
        self.assertEqual(1, report["chunks_count"])
        self.assertEqual(1, report["documents_without_chunks_count"])
        self.assertEqual({"BOI": 1}, report["documents_without_chunks_by_type"])
        self.assertEqual(1, report["family_distribution"]["TVA"])
        self.assertEqual(1, report["family_distribution"]["RSA"])
        self.assertEqual(0, report["chunk_required_field_missing_count"])

    def test_build_regression_report_lists_worsened_queries(self) -> None:
        baseline = {
            "metrics": {
                "per_query": [
                    {
                        "query_id": "q1",
                        "layers": {"final_docs": {"rank": 1}},
                    }
                ]
            }
        }
        current = {
            "metrics": {
                "per_query": [
                    {
                        "query_id": "q1",
                        "layers": {"final_docs": {"rank": None}},
                    }
                ]
            }
        }

        comparison = eval_script.build_regression_report(baseline, current, layer_name="final_docs")

        self.assertEqual(["q1"], [item["query_id"] for item in comparison["worsened"]])
        self.assertEqual(0, len(comparison["improved"]))


if __name__ == "__main__":
    unittest.main()
