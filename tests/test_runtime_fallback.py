from __future__ import annotations

import unittest

import numpy as np

from bofip_agentic.models import ChunkNode, RawDocument
from bofip_agentic.rag_runtime import RagRuntime


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


class RuntimeFallbackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
