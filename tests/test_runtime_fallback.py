from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bofip_agentic.models import ChunkNode, RawDocument
from bofip_agentic.rag_runtime import (
    CORPUS_PATHS,
    DEFAULT_RERANKER_CANDIDATE_LIMIT,
    DEFAULT_RERANKER_TEXT_LIMIT,
    RagRuntime,
    _prefix_overlap_rankings,
    _reference_matches_prefix,
)
from bofip_agentic.reranker import RankedItem


def _doc(ref: str, title: str, subject: str) -> RawDocument:
    return RawDocument(
        document_id=ref,
        boi_reference=ref,
        title=title,
        document_type="BOI",
        content_type="commentary",
        publication_date="2026-01-01",
        source_url=None,
        language="fr",
        subjects=[subject],
        category_path=[subject],
    )


def _chunk(ref: str, text: str) -> ChunkNode:
    return ChunkNode(
        chunk_id=f"{ref}-chunk",
        source_type="commentary",
        document_id=ref,
        boi_reference=ref,
        doc_version=None,
        strategy="test",
        section_id=None,
        parent_chunk_id=None,
        section_path=["I. Régime"],
        paragraph_range=[],
        text=text,
        token_count=10,
        chunk_kind="text",
        legal_refs=[],
    )


def _chunk_with_section(ref: str, chunk_id: str, section_path: list[str], text: str) -> ChunkNode:
    return ChunkNode(
        chunk_id=chunk_id,
        source_type="commentary",
        document_id=ref,
        boi_reference=ref,
        doc_version=None,
        strategy="test",
        section_id=None,
        parent_chunk_id=None,
        section_path=section_path,
        paragraph_range=[],
        text=text,
        token_count=10,
        chunk_kind="text",
        legal_refs=[],
    )


class _RecordingReranker:
    def __init__(self):
        self.calls: list[dict] = []

    def rerank(self, query, items, *, get_text, top_k):
        texts = [get_text(item) for item in items]
        self.calls.append(
            {
                "query": query,
                "count": len(items),
                "top_k": top_k,
                "max_text_len": max((len(text) for text in texts), default=0),
            }
        )
        return [RankedItem(item=item, score=float(len(items) - idx)) for idx, item in enumerate(items)]


class RuntimeFallbackTests(unittest.TestCase):
    def test_commentary_corpus_uses_full_corpus_artifact_names(self):
        paths = CORPUS_PATHS["commentary"]

        self.assertEqual("data/interim/raw_docs.jsonl", paths["raw_docs"])
        self.assertEqual("data/interim/chunks.jsonl", paths["chunks"])
        self.assertEqual("data/interim/doc_dense_cache.npy", paths["doc_dense_cache"])
        self.assertEqual("data/interim/chunk_dense_cache.npy", paths["chunk_dense_cache"])
        self.assertFalse(any("5666" in value or "sample" in value for value in paths.values()))

    def test_from_local_corpus_lexical_only_allows_mismatched_chunk_embedding_cache(self):
        document = _doc("BOI-TEST-10", "Document test", "TEST")
        chunks = [
            _chunk_with_section("BOI-TEST-10", "chunk-1", ["Section"], "seuil montant"),
            _chunk_with_section("BOI-TEST-10", "chunk-2", ["Section"], "autre texte"),
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp_path = Path(tmp)
            raw_docs_path = tmp_path / "raw_docs.jsonl"
            chunks_path = tmp_path / "chunks.jsonl"
            doc_dense_path = tmp_path / "doc_dense.npy"
            chunk_dense_path = tmp_path / "chunk_dense.npy"
            raw_docs_path.write_text(json.dumps(document.to_dict()) + "\n", encoding="utf-8")
            chunks_path.write_text(
                "".join(json.dumps(chunk.to_dict()) + "\n" for chunk in chunks),
                encoding="utf-8",
            )
            np.save(doc_dense_path, np.zeros((1, 2), dtype=np.float32))
            np.save(chunk_dense_path, np.zeros((1, 2), dtype=np.float32))

            runtime = RagRuntime.from_local_corpus(
                raw_docs_path=raw_docs_path,
                chunks_path=chunks_path,
                doc_dense_path=doc_dense_path,
                chunk_dense_path=chunk_dense_path,
                load_dense=False,
                load_reranker=False,
                allow_lexical_fallback=True,
                device="cpu",
            )

            result = runtime.retrieve("seuil montant", use_dense=False, use_chunk_dense=False, use_reranker=False)
            del runtime

        self.assertEqual(["chunk-1"], [chunk.chunk_id for chunk in result.stage2_chunks[:1]])
        self.assertEqual(result.pipeline_log["dense_status"], "unavailable")

    def test_retrieve_works_when_dense_encoder_is_unavailable(self):
        documents = [
            _doc("BOI-TVA-TEST", "TVA taux réduit", "TVA"),
            _doc("BOI-IR-TEST", "Impôt sur le revenu", "IR"),
        ]
        chunks = [
            _chunk("BOI-TVA-TEST", "La pose d'une pompe à chaleur peut relever de la TVA."),
            _chunk("BOI-IR-TEST", "Le foyer fiscal relève de l'impôt sur le revenu."),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((2, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve("TVA pompe à chaleur", use_reranker=False)

        self.assertTrue(result.stage1_hits)
        self.assertTrue(result.stage2_chunks)
        self.assertEqual(result.pipeline_log["dense_status"], "unavailable")

    def test_boost_prefix_is_soft_and_does_not_hide_stronger_nonmatching_document(self):
        documents = [
            _doc("BOI-CVAE-TEST", "CVAE seuil chiffre affaires alpha unique", "CVAE"),
            _doc("BOI-TVA-TEST", "TVA franchise", "TVA"),
            _doc("BOI-IR-TEST", "Impot sur le revenu", "IR"),
        ]
        chunks = [
            _chunk("BOI-CVAE-TEST", "Seuil alpha unique de chiffre affaires pour la CVAE."),
            _chunk("BOI-TVA-TEST", "Franchise en base de TVA."),
            _chunk("BOI-IR-TEST", "Foyer fiscal."),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((3, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((3, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "seuil alpha unique chiffre affaires",
            boost_prefix="TVA",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=3,
        )

        self.assertTrue(result.stage1_hits)
        refs = [hit.boi_reference for hit in result.stage1_hits]
        self.assertIn("BOI-CVAE-TEST", refs)
        self.assertFalse(all(ref.startswith("BOI-TVA") for ref in refs))

    def test_chunk_query_does_not_drive_document_prefix_ranking(self):
        documents = [
            _doc("BOI-TVA-TARGET", "TVA alpha target", "TVA"),
            _doc("BOI-TVA-DISTRACTOR", "TVA omega omega omega forfaitaire", "TVA"),
        ]
        chunks = [
            _chunk("BOI-TVA-TARGET", "alpha target preuve utile"),
            _chunk("BOI-TVA-DISTRACTOR", "omega omega omega forfaitaire preuve non pertinente"),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((2, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "alpha target",
            chunk_query="omega omega omega forfaitaire",
            boost_prefix="TVA",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=2,
            max_chunks=2,
        )

        self.assertEqual(result.stage1_hits[0].boi_reference, "BOI-TVA-TARGET")

    def test_document_candidates_are_stable_across_chunk_query_variants(self):
        documents = [
            _doc("BOI-BIC-CHG-10-10-10", "BIC charges frais de repas exploitant individuel", "BIC"),
            _doc("BOI-BIC-MICRO-10", "BIC micro entreprise chiffre affaires regime", "BIC"),
            _doc("BOI-BIC-FORFAIT-10", "BIC montant forfaitaire deduction generale", "BIC"),
        ]
        chunks = [
            _chunk("BOI-BIC-CHG-10-10-10", "frais supplementaires de repas eloignement domicile"),
            _chunk("BOI-BIC-MICRO-10", "micro entreprise chiffre affaires seuils"),
            _chunk("BOI-BIC-FORFAIT-10", "montant forfaitaire deduction forfaitaire"),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((3, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((3, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )
        doc_query = "BIC frais de repas exploitant individuel eloignement domicile"
        chunk_query_variants = [
            "frais supplementaires de repas conditions deduction",
            "montant forfaitaire part personnelle non deductible",
            "micro entreprise chiffre affaires seuil regime simplifie",
        ]

        stage1_orders = []
        for chunk_query in chunk_query_variants:
            result = runtime.retrieve(
                doc_query,
                chunk_query=chunk_query,
                boost_prefix="BIC",
                use_dense=False,
                use_chunk_dense=False,
                use_reranker=False,
                top_docs=3,
                max_chunks=3,
            )
            stage1_orders.append([hit.boi_reference for hit in result.stage1_hits])

        self.assertTrue(stage1_orders)
        self.assertTrue(all(order == stage1_orders[0] for order in stage1_orders))
        self.assertEqual(stage1_orders[0][0], "BOI-BIC-CHG-10-10-10")

    def test_reranker_scores_bounded_candidate_pool(self):
        documents = [
            _doc("BOI-TVA-A", "TVA taxe prestations", "TVA"),
            _doc("BOI-TVA-B", "TVA taxe livraisons", "TVA"),
        ]
        chunks = [
            _chunk_with_section(
                ref,
                f"{ref}-chunk-{idx}",
                ["TVA", "Taxe"],
                f"taxe valeur ajoutee prestation livraison operation imposable numero {idx}",
            )
            for ref in ("BOI-TVA-A", "BOI-TVA-B")
            for idx in range(10)
        ]
        reranker = _RecordingReranker()
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((20, 4), dtype=np.float32),
            reranker=reranker,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "TVA taxe prestation livraison",
            boost_prefix="TVA",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=True,
            top_docs=2,
            chunks_per_doc=8,
            max_chunks=4,
        )

        self.assertGreaterEqual(reranker.calls[0]["count"], max(12, result.pipeline_log["final_chunks"] * 3))
        self.assertLessEqual(reranker.calls[0]["count"], DEFAULT_RERANKER_CANDIDATE_LIMIT)
        self.assertLessEqual(reranker.calls[0]["max_text_len"], DEFAULT_RERANKER_TEXT_LIMIT)
        self.assertEqual(result.pipeline_log["reranker_candidates_scored"], reranker.calls[0]["count"])
        self.assertEqual(result.pipeline_log["reranker_text_limit"], DEFAULT_RERANKER_TEXT_LIMIT)

    def test_stage2_chunk_title_uses_chunk_document_when_boi_reference_is_duplicated(self):
        documents = [
            RawDocument(
                document_id="doc-new",
                boi_reference="BOI-DUP-1",
                title="Current duplicate title",
                document_type="BOI",
                content_type="commentary",
                publication_date="2026-01-01",
                source_url=None,
                language="fr",
                subjects=["DUP"],
                category_path=["DUP"],
            ),
            RawDocument(
                document_id="doc-old",
                boi_reference="BOI-DUP-1",
                title="Stale duplicate title",
                document_type="BOI",
                content_type="commentary",
                publication_date="2025-01-01",
                source_url=None,
                language="fr",
                subjects=["DUP"],
                category_path=["DUP"],
            ),
        ]
        chunks = [
            ChunkNode(
                "current-chunk",
                "BOFIP",
                "doc-new",
                "BOI-DUP-1",
                None,
                "section_window",
                None,
                None,
                ["Regle courante"],
                ["p1"],
                "regle courante applicable",
                10,
                "paragraph_window",
            )
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((1, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve("regle courante", use_dense=False, use_chunk_dense=False, use_reranker=False)

        self.assertEqual(result.stage2_chunks[0].title, "Current duplicate title")
        self.assertEqual(result.stage2_chunks[0].publication_date, "2026-01-01")

    def test_pipeline_log_exposes_layer_candidate_ids_for_eval(self):
        documents = [
            _doc("BOI-A-10", "Document A", "A"),
            _doc("BOI-B-10", "Document B", "B"),
        ]
        chunks = [
            _chunk_with_section("BOI-A-10", "a-1", ["A"], "alpha seuil montant"),
            _chunk_with_section("BOI-A-10", "a-2", ["A"], "alpha autre"),
            _chunk_with_section("BOI-B-10", "b-1", ["B"], "beta seuil montant"),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((3, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "seuil montant",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=2,
            chunks_per_doc=2,
            max_chunks=2,
        )

        self.assertEqual(
            result.pipeline_log["stage1_doc_refs"],
            [hit.boi_reference for hit in result.stage1_hits],
        )
        self.assertEqual(
            result.pipeline_log["final_chunk_ids"],
            [hit.chunk_id for hit in result.stage2_chunks],
        )
        self.assertGreaterEqual(len(result.pipeline_log["stage2_candidate_chunk_ids"]), 2)
        self.assertIn("stage2_candidate_doc_refs", result.pipeline_log)

    def test_prefix_matching_keeps_rescripts_for_matching_family(self):
        self.assertTrue(_reference_matches_prefix("BOI-RES-TVA-TEST", "TVA"))
        self.assertFalse(_reference_matches_prefix("BOI-RES-IR-TEST", "TVA"))

    def test_query_with_parent_boi_reference_promotes_child_documents(self):
        documents = [
            _doc(
                "BOI-IR-PAS-50-10-20-10",
                "IR PAS rupture conventionnelle fraction imposable",
                "IR",
            ),
            _doc(
                "BOI-RSA-CHAMP-20-40-10-30",
                "RSA sommes percues en cas de rupture du contrat de travail",
                "RSA",
            ),
            _doc("BOI-IR-RICI-TEST", "Reduction impot hors sujet", "IR"),
        ]
        chunks = [
            _chunk(
                "BOI-IR-PAS-50-10-20-10",
                "Ce passage mentionne rupture conventionnelle mais renvoie au regime fiscal de fond.",
            ),
            _chunk(
                "BOI-RSA-CHAMP-20-40-10-30",
                "Indemnites versees a l'occasion de la rupture conventionnelle.",
            ),
            _chunk("BOI-IR-RICI-TEST", "Autre avantage fiscal."),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((3, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((3, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "BOI-RSA-CHAMP-20-40-10 rupture conventionnelle exoneration limites",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=1,
        )

        self.assertEqual(result.stage1_hits[0].boi_reference, "BOI-RSA-CHAMP-20-40-10-30")
        self.assertEqual(result.stage2_chunks[0].boi_reference, "BOI-RSA-CHAMP-20-40-10-30")

    def test_detailed_boost_prefix_uses_query_overlap_to_order_child_documents(self):
        documents = [
            _doc(
                "BOI-RSA-CHAMP-20-40-10-20",
                "RSA modalites particulieres representants",
                "RSA",
            ),
            _doc(
                "BOI-RSA-CHAMP-20-40-10-30",
                "RSA rupture du contrat de travail",
                "RSA",
            ),
        ]
        chunks = [
            _chunk("BOI-RSA-CHAMP-20-40-10-20", "Indemnite de clientele des representants."),
            _chunk(
                "BOI-RSA-CHAMP-20-40-10-30",
                "Indemnites de rupture conventionnelle et limites d'exoneration.",
            ),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((2, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "rupture conventionnelle exoneration limites",
            boost_prefix="RSA-CHAMP-20-40-10",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=1,
        )

        self.assertEqual(result.stage1_hits[0].boi_reference, "BOI-RSA-CHAMP-20-40-10-30")

    def test_retrieve_within_documents_searches_section_inside_known_boi(self):
        documents = [
            _doc("BOI-IF-CFE-20-20-40-10", "Cotisation minimum CFE", "IF"),
            _doc("BOI-IF-CFE-20-50-10", "Creation et extension", "IF"),
        ]
        chunks = [
            _chunk_with_section(
                "BOI-IF-CFE-20-20-40-10",
                "cfe-general",
                ["Cotisation minimum", "Regles generales"],
                "La cotisation minimum est due au lieu du principal etablissement.",
            ),
            _chunk_with_section(
                "BOI-IF-CFE-20-20-40-10",
                "cfe-threshold",
                ["Cotisation minimum", "Exoneration des contribuables a faible chiffre d'affaires"],
                "Les redevables qui realisent un chiffre d'affaires ou des recettes "
                "inferieurs ou egaux a 5 000 euros sont exoneres de cotisation minimum.",
            ),
            _chunk_with_section(
                "BOI-IF-CFE-20-50-10",
                "cfe-creation",
                ["Creation d'etablissement"],
                "La premiere annee avec chiffre d'affaires constitue l'annee de creation.",
            ),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((2, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((3, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve_within_documents(
            "exoneration cotisation minimum chiffre affaires recettes inferieur egal 5000",
            ["BOI-IF-CFE-20-20-40-10"],
            chunks_per_doc=1,
            max_chunks=1,
        )

        self.assertEqual(result.stage2_chunks[0].chunk_id, "cfe-threshold")
        self.assertEqual(result.pipeline_log["retrieval_scope"], "intra_document")

    def test_retrieve_within_documents_ranks_children_of_broad_parent_before_cutoff(self):
        documents = [_doc("BOI-ENR-DMTG", "Mutations a titre gratuit", "ENR")]
        chunks = [_chunk("BOI-ENR-DMTG", "Regles generales des mutations a titre gratuit.")]
        for idx in range(1, 10):
            ref = f"BOI-ENR-DMTG-10-{idx:02d}"
            documents.append(_doc(ref, f"Document general {idx}", "ENR"))
            chunks.append(_chunk(ref, "Disposition generale de succession."))

        documents.append(_doc("BOI-ENR-DMTG-10-50-20", "Abattements en ligne directe", "ENR"))
        chunks.append(
            _chunk_with_section(
                "BOI-ENR-DMTG-10-50-20",
                "line-direct-child-abatement",
                ["I. Abattement applicable en ligne directe", "A. Quotite"],
                "Article 779 du CGI: abattement en ligne directe de 100 000 euros "
                "sur la part de chacun des enfants.",
            )
        )

        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((len(documents), 4), dtype=np.float32),
            chunk_embeddings=np.zeros((len(chunks), 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve_within_documents(
            "BOI-ENR-DMTG abattement donation parent enfant article 779 ligne directe 100000",
            ["BOI-ENR-DMTG"],
            chunks_per_doc=1,
            max_chunks=8,
        )

        self.assertIn(
            "line-direct-child-abatement",
            [chunk.chunk_id for chunk in result.stage2_chunks],
        )
        self.assertIn("BOI-ENR-DMTG-10-50-20", result.pipeline_log["searched_documents"])

    def test_family_prefix_uses_document_and_section_overlap_without_exact_chapter_hint(self):
        documents = [
            _doc("BOI-IF-CFE-20-20-40-10", "Cotisation minimum CFE", "IF"),
            _doc("BOI-IF-TH-10", "Taxe habitation", "IF"),
            _doc("BOI-TVA-TEST", "TVA franchise", "TVA"),
        ]
        chunks = [
            _chunk_with_section(
                "BOI-IF-CFE-20-20-40-10",
                "cfe-threshold",
                ["Cotisation minimum", "Exoneration des contribuables a faible chiffre d'affaires"],
                "Les redevables qui realisent un chiffre d'affaires ou des recettes "
                "inferieurs ou egaux a 5 000 euros sont exoneres de cotisation minimum.",
            ),
            _chunk_with_section(
                "BOI-IF-TH-10",
                "th-generic",
                ["Taxe habitation"],
                "Regles generales de taxe d'habitation.",
            ),
            _chunk_with_section(
                "BOI-TVA-TEST",
                "tva-generic",
                ["Franchise TVA"],
                "Regles de franchise en base de TVA.",
            ),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((3, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((3, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "micro entrepreneur avis CFE",
            boost_prefix="IF",
            chunk_query="exoneration cotisation minimum chiffre affaires recettes inferieur egal",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=2,
            chunks_per_doc=1,
            max_chunks=1,
        )

        self.assertEqual(result.stage1_hits[0].boi_reference, "BOI-IF-CFE-20-20-40-10")
        self.assertEqual(result.stage2_chunks[0].chunk_id, "cfe-threshold")

    def test_prefix_overlap_prefers_general_rule_over_derogation_when_query_asks_general_rule(self):
        documents = [
            _doc(
                "BOI-TVA-CHAMP-20-50-40-10",
                "TVA - Lieu des prestations de services - Derogations a la regle generale",
                "TVA",
            ),
            _doc(
                "BOI-TVA-CHAMP-20-50-20",
                "TVA - Lieu des prestations de services - Regles generales",
                "TVA",
            ),
        ]
        chunks = [
            _chunk_with_section(
                "BOI-TVA-CHAMP-20-50-40-10",
                "derogation",
                ["Derogations a la regle generale", "Prestations utilisees en France"],
                "Prestations de services, territorialite, preneur assujetti, lieu d'etablissement.",
            ),
            _chunk_with_section(
                "BOI-TVA-CHAMP-20-50-20",
                "general",
                ["Regles generales", "Lieu du preneur assujetti"],
                "Prestations de services entre assujettis et lieu d'etablissement du preneur.",
            ),
        ]

        ranked = _prefix_overlap_rankings(
            "territorialite prestation services B2B preneur assujetti lieu etablissement regle generale",
            "TVA",
            documents,
            {"BOI-TVA-CHAMP-20-50-40-10": [chunks[0]], "BOI-TVA-CHAMP-20-50-20": [chunks[1]]},
        )

        self.assertEqual(ranked[0].boi_reference, "BOI-TVA-CHAMP-20-50-20")

    def test_prefix_overlap_demotes_rescripts_for_generic_facturation_query(self):
        documents = [
            _doc(
                "BOI-RES-TVA-000128",
                "RES - Taxe sur la valeur ajoutee - Taux applicable aux prestations du secteur des services",
                "TVA",
            ),
            _doc(
                "BOI-TVA-DECLA-10-10-10",
                "TVA - Redevable de la taxe - Livraisons de biens et prestations",
                "TVA",
            ),
            _doc(
                "BOI-TVA-DECLA-30-20-20-10",
                "TVA - Regles relatives aux factures - Mentions obligatoires generales",
                "TVA",
            ),
            _doc(
                "BOI-TVA-DECLA-30-20-20-30",
                "TVA - Regles relatives aux factures - Mentions specifiques a certaines operations",
                "TVA",
            ),
        ]
        chunks = [
            _chunk_with_section(
                "BOI-RES-TVA-000128",
                "res-sector",
                ["Question sectorielle", "Taux applicable"],
                "Facture prestation services taxe valeur ajoutee taux applicable entreprise secteur services.",
            ),
            _chunk_with_section(
                "BOI-TVA-DECLA-10-10-10",
                "redevable-general",
                ["Redevable de la taxe"],
                "Livraisons de biens prestations services redevable taxe preneur facture.",
            ),
            _chunk_with_section(
                "BOI-TVA-DECLA-30-20-20-10",
                "invoice-general",
                ["Mentions obligatoires generales"],
                "Facture mention obligatoire numero date taxe valeur ajoutee.",
            ),
            _chunk_with_section(
                "BOI-TVA-DECLA-30-20-20-30",
                "invoice-mention",
                ["Mentions specifiques", "Prestations de services pour lesquelles la taxe est due par le preneur"],
                "Prestations de services intracommunautaires soumises a autoliquidation et mention sur facture.",
            ),
        ]

        ranked = _prefix_overlap_rankings(
            "facture prestation services intracommunautaire autoliquidation preneur redevable mention facture",
            "TVA",
            documents,
            {"BOI-RES-TVA-000128": [chunks[0]], "BOI-TVA-DECLA-30-20-20-30": [chunks[1]]},
        )

        self.assertEqual(ranked[0].boi_reference, "BOI-TVA-DECLA-30-20-20-30")

    def test_retrieve_keeps_structural_prefix_candidate_ahead_of_rescript_noise(self):
        documents = [
            _doc(
                "BOI-RES-TVA-000128",
                "RES - Taxe sur la valeur ajoutee - Taux applicable aux prestations du secteur des services",
                "TVA",
            ),
            _doc(
                "BOI-TVA-DECLA-10-10-10",
                "TVA - Redevable de la taxe - Livraisons de biens et prestations",
                "TVA",
            ),
            _doc(
                "BOI-TVA-DECLA-30-20-20-10",
                "TVA - Regles relatives aux factures - Mentions obligatoires generales",
                "TVA",
            ),
            _doc(
                "BOI-TVA-DECLA-30-20-20-30",
                "TVA - Regles relatives aux factures - Mentions specifiques a certaines operations",
                "TVA",
            ),
        ]
        chunks = [
            _chunk_with_section(
                "BOI-RES-TVA-000128",
                "res-sector",
                ["Question sectorielle", "Taux applicable"],
                "Facture prestation services taxe valeur ajoutee taux applicable entreprise secteur services.",
            ),
            _chunk_with_section(
                "BOI-TVA-DECLA-10-10-10",
                "redevable-general",
                ["Redevable de la taxe"],
                "Livraisons de biens prestations services redevable taxe preneur facture.",
            ),
            _chunk_with_section(
                "BOI-TVA-DECLA-30-20-20-10",
                "invoice-general",
                ["Mentions obligatoires generales"],
                "Facture mention obligatoire numero date taxe valeur ajoutee.",
            ),
            _chunk_with_section(
                "BOI-TVA-DECLA-30-20-20-30",
                "invoice-mention",
                ["Mentions specifiques", "Prestations de services pour lesquelles la taxe est due par le preneur"],
                "Prestations de services intracommunautaires soumises a autoliquidation et mention sur facture.",
            ),
        ]
        runtime = RagRuntime(
            documents=documents,
            chunks=chunks,
            doc_encoder=None,
            chunk_encoder=None,
            document_embeddings=np.zeros((4, 4), dtype=np.float32),
            chunk_embeddings=np.zeros((4, 4), dtype=np.float32),
            reranker=None,
            dense_error="dense model unavailable in test",
        )

        result = runtime.retrieve(
            "facture prestation services intracommunautaire autoliquidation preneur redevable mention facture",
            boost_prefix="TVA",
            use_dense=False,
            use_chunk_dense=False,
            use_reranker=False,
            top_docs=4,
            max_chunks=1,
        )

        self.assertEqual(result.stage1_hits[0].boi_reference, "BOI-TVA-DECLA-30-20-20-30")
        self.assertEqual(result.stage2_chunks[0].chunk_id, "invoice-mention")


if __name__ == "__main__":
    unittest.main()
