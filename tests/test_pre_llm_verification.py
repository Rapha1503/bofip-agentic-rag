from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from bofip_cleanroom.pre_llm_verification import (
    build_stage1_replay_command,
    compare_numeric_metrics,
    infer_doc_dense_cache_prefix,
    summarize_chunk_document_coverage,
    summarize_order_match,
    validate_retrieval_payload,
)


class PreLlmVerificationTests(unittest.TestCase):
    def test_infer_doc_dense_cache_prefix_matches_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            interim_dir = Path(tmp_dir)
            cache_prefix = interim_dir / "doc_dense_cache_5666_sections_firstpara_e5"
            cache_prefix.with_suffix(".npy").write_bytes(b"placeholder")

            inferred = infer_doc_dense_cache_prefix(
                interim_dir=interim_dir,
                document_count=5666,
                dense_mode="sections_firstpara",
                model_name="intfloat/multilingual-e5-base",
            )

            self.assertEqual(inferred, cache_prefix)

    def test_build_stage1_replay_command_uses_reference_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            interim_dir = root / "data" / "interim"
            interim_dir.mkdir(parents=True)
            (interim_dir / "doc_dense_cache_5666_sections_firstpara_e5.npy").write_bytes(b"placeholder")

            payload = {
                "raw_docs_path": str(root / "raw_docs.jsonl"),
                "queries_path": str(root / "queries.jsonl"),
                "model_name": "intfloat/multilingual-e5-base",
                "lexical_modes": ["base", "sections_leads", "sections_leads_stem"],
                "dense_mode": "sections_firstpara",
                "weights": {
                    "base": 1.0,
                    "sections_leads": 2.0,
                    "sections_leads_stem": 1.0,
                    "dense": 1.0,
                    "chunk_dense": 2.0,
                },
                "candidate_k": 20,
                "rank_constant": 60,
                "fusion_mode": "confidence",
                "confidence_top_n": 5,
                "confidence_alpha": 1.0,
                "score_alpha": 0.5,
                "document_count": 5666,
                "query_acronym_expansion": True,
                "query_acronym_max_expansions": 3,
                "chunk_dense_enabled": True,
                "chunk_dense_cache": str(root / "chunk_dense.npy"),
                "chunks_path": str(root / "chunks.jsonl"),
                "stem_lexical": False,
                "local_title_rerank_top_n": 0,
                "specificity_rerank_top_n": 0,
            }
            output_path = root / "replay.json"

            command = build_stage1_replay_command(
                python_executable="python",
                project_root=root,
                interim_dir=interim_dir,
                reference_payload=payload,
                output_path=output_path,
            )

            self.assertIn("--device", command)
            self.assertIn("cpu", command)
            self.assertIn("--query-acronym-expansion", command)
            self.assertIn("--cache-prefix", command)
            self.assertIn(str((interim_dir / "doc_dense_cache_5666_sections_firstpara_e5").resolve()), command)
            self.assertIn("--chunk-dense-cache", command)
            self.assertIn(str(output_path.resolve()), command)

    def test_compare_numeric_metrics_detects_drift(self):
        result = compare_numeric_metrics(
            expected={"hit@1": 0.8, "hit@5": 1.0},
            observed={"hit@1": 0.80005, "hit@5": 0.98},
            metric_keys=["hit@1", "hit@5"],
            tolerance=1e-4,
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["metrics"]["hit@1"]["within_tolerance"])
        self.assertFalse(result["metrics"]["hit@5"]["within_tolerance"])

    def test_summarize_chunk_document_coverage_uses_document_id_not_boi_reference(self):
        raw_docs_rows = [
            {"document_id": "doc-1", "boi_reference": "BOI-A"},
            {"document_id": "doc-2", "boi_reference": "BOI-A"},
            {"document_id": "doc-3", "boi_reference": "BOI-B"},
        ]
        chunk_rows = [
            {"document_id": "doc-1", "boi_reference": "BOI-A"},
            {"document_id": "doc-2", "boi_reference": "BOI-A"},
            {"document_id": "doc-3", "boi_reference": "BOI-B"},
            {"document_id": "doc-3", "boi_reference": "BOI-B"},
        ]

        result = summarize_chunk_document_coverage(raw_docs_rows, chunk_rows)

        self.assertEqual(result["raw_document_count"], 3)
        self.assertEqual(result["raw_document_id_count"], 3)
        self.assertEqual(result["chunk_document_id_count"], 3)
        self.assertEqual(result["raw_unique_boi_reference_count"], 2)
        self.assertEqual(result["chunk_unique_boi_reference_count"], 2)
        self.assertEqual(result["raw_duplicate_boi_reference_count"], 1)
        self.assertEqual(result["missing_document_id_count"], 0)
        self.assertEqual(result["extra_document_id_count"], 0)

    def test_validate_retrieval_payload_accepts_valid_payload(self):
        payload = {
            "query": "question",
            "lexical_query": "question acronym expanded",
            "acronym_expansions": [{"acronym": "CIR", "phrase": "credit impot recherche"}],
            "source_confidences": {"base": 0.1},
            "stage1_hits": [
                {
                    "rank": 1,
                    "score": 0.8,
                    "boi_reference": "BOI-TEST-1",
                    "title": "Titre",
                    "sources": ["base"],
                    "ranks": {"base": 1},
                }
            ],
            "family_selection": {"anchor_references": [], "prefixes": [], "members": []},
            "stage2_chunks": [
                {
                    "citation_id": 1,
                    "boi_reference": "BOI-TEST-1",
                    "title": "Titre",
                    "publication_date": "2025-01-01",
                    "section_path": "I. Section",
                    "chunk_id": "chunk-1",
                    "chunk_kind": "paragraph_window",
                    "text": "Texte non vide",
                }
            ],
        }

        self.assertEqual(validate_retrieval_payload(payload), [])

    def test_validate_retrieval_payload_rejects_broken_citations_and_fields(self):
        payload = {
            "query": "question",
            "lexical_query": "question",
            "acronym_expansions": [],
            "source_confidences": {"base": 0.1},
            "stage1_hits": [
                {
                    "rank": 1,
                    "score": 0.5,
                    "boi_reference": "BOI-TEST-1",
                    "title": "Titre",
                    "sources": ["base"],
                }
            ],
            "family_selection": {},
            "stage2_chunks": [
                {
                    "citation_id": 2,
                    "boi_reference": "BOI-TEST-1",
                    "title": "Titre",
                    "publication_date": "2025-01-01",
                    "section_path": "I. Section",
                    "chunk_id": "chunk-1",
                    "chunk_kind": "paragraph_window",
                    "text": "",
                },
                {
                    "citation_id": 2,
                    "boi_reference": "BOI-TEST-2",
                    "title": "Titre 2",
                    "publication_date": "2025-01-02",
                    "section_path": "II. Section",
                    "chunk_id": "chunk-1",
                    "chunk_kind": "paragraph_window",
                    "text": "Texte",
                },
            ],
        }

        errors = validate_retrieval_payload(payload)

        self.assertTrue(any("stage1_hits[1] missing field: ranks" in error for error in errors))
        self.assertTrue(any("duplicate chunk_id" in error for error in errors))
        self.assertTrue(any("citation_id sequence" in error for error in errors))
        self.assertTrue(any("text must be non-empty" in error for error in errors))

    def test_summarize_order_match_reports_exact_sequence(self):
        result = summarize_order_match(["a", "b"], ["a", "c"])

        self.assertFalse(result["matches"])
        self.assertEqual(result["expected"], ["a", "b"])
        self.assertEqual(result["observed"], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
