from __future__ import annotations

import unittest

import numpy as np

from bofip_agentic.dense_retrieval import (
    DenseDocumentIndex,
    DenseIndex,
    _dense_prompt_style,
    _passage_text,
    _query_text,
    build_dense_chunk_text,
    build_dense_document_text,
)
from bofip_agentic.models import ChunkNode, RawDocument, RawSectionNode


class DenseRetrievalTests(unittest.TestCase):
    def test_build_dense_chunk_text_includes_metadata(self) -> None:
        chunk = ChunkNode(
            chunk_id="c1",
            source_type="BOFIP",
            document_id="doc1",
            boi_reference="BOI-TEST-0001",
            doc_version="2024-01-01",
            strategy="section_window",
            section_id="sec-1",
            parent_chunk_id=None,
            section_path=["Titre", "Section A"],
            paragraph_range=["p1", "p2"],
            text="Le texte métier principal.",
            token_count=10,
            chunk_kind="paragraph_window",
            legal_refs=["article 44 sexies-0 A du CGI"],
        )

        text = build_dense_chunk_text(chunk)

        self.assertIn("BOI-TEST-0001", text)
        self.assertIn("Titre Section A", text)
        self.assertIn("article 44 sexies-0 A du CGI", text)
        self.assertIn("texte métier principal", text)

    def test_build_dense_chunk_text_supports_body_and_leaf_modes(self) -> None:
        chunk = ChunkNode(
            chunk_id="c1",
            source_type="BOFIP",
            document_id="doc1",
            boi_reference="BOI-TEST-0001",
            doc_version="2024-01-01",
            strategy="section_window",
            section_id="sec-1",
            parent_chunk_id=None,
            section_path=["Titre", "Section A"],
            paragraph_range=["p1", "p2"],
            text="Le texte métier principal.",
            token_count=10,
            chunk_kind="paragraph_window",
            legal_refs=["article 44 sexies-0 A du CGI"],
        )

        body = build_dense_chunk_text(chunk, mode="body")
        leaf = build_dense_chunk_text(chunk, mode="leaf")

        self.assertEqual(body, "Le texte métier principal.")
        self.assertNotIn("BOI-TEST-0001", leaf)
        self.assertIn("Section A", leaf)
        self.assertIn("article 44 sexies-0 A du CGI", leaf)

    def test_build_dense_chunk_text_rejects_unknown_mode(self) -> None:
        chunk = ChunkNode("c1", "BOFIP", "doc1", "BOI-A", None, "section_window", None, None, ["A"], ["p1"], "alpha", 1, "paragraph")

        with self.assertRaises(ValueError):
            build_dense_chunk_text(chunk, mode="unknown")

    def test_dense_index_returns_highest_dot_product_first(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A", None, "section_window", None, None, ["A"], ["p1"], "alpha", 1, "paragraph"),
            ChunkNode("c2", "BOFIP", "doc2", "BOI-B", None, "section_window", None, None, ["B"], ["p2"], "beta", 1, "paragraph"),
        ]
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = DenseIndex(chunks, embeddings)

        hits = index.search_from_vector(np.asarray([0.8, 0.2], dtype=np.float32), top_k=2)

        self.assertEqual([hit.chunk.boi_reference for hit in hits], ["BOI-A", "BOI-B"])

    def test_search_documents_dedupes_same_boi_reference(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A", None, "section_window", None, None, ["A"], ["p1"], "alpha", 1, "paragraph"),
            ChunkNode("c2", "BOFIP", "doc1", "BOI-A", None, "section_window", None, None, ["A"], ["p2"], "alpha secondaire", 1, "paragraph"),
            ChunkNode("c3", "BOFIP", "doc2", "BOI-B", None, "section_window", None, None, ["B"], ["p3"], "beta", 1, "paragraph"),
        ]
        embeddings = np.asarray([[0.95, 0.05], [0.90, 0.10], [0.0, 1.0]], dtype=np.float32)
        index = DenseIndex(chunks, embeddings)

        hits = index.search_documents_from_vector(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)

        self.assertEqual([hit.boi_reference for hit in hits], ["BOI-A", "BOI-B"])

    def test_build_dense_document_text_supports_structure_modes(self) -> None:
        document = RawDocument(
            document_id="doc1",
            boi_reference="BOI-TEST-0001",
            title="TVA - Redevable",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date="2024-01-01",
            source_url=None,
            language=None,
            category_path=["Commentaire", "TVA"],
            subjects=["TVA"],
            html_title="Titre HTML",
            sections=[
                RawSectionNode("s1", None, 1, 1, "I. Principes", None, ["I. Principes"]),
            ],
            paragraphs=[],
        )
        text = build_dense_document_text(document, mode="sections")
        self.assertIn("BOI-TEST-0001", text)
        self.assertIn("TVA - Redevable", text)
        self.assertIn("I. Principes", text)

    def test_dense_document_index_returns_top_document(self) -> None:
        documents = [
            RawDocument("doc1", "BOI-A", "TVA redevable", "Contenu", "Commentaire", None, None, None),
            RawDocument("doc2", "BOI-B", "TVA exoneree", "Contenu", "Commentaire", None, None, None),
        ]
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = DenseDocumentIndex(documents, embeddings)

        hits = index.search_from_vector(np.asarray([0.9, 0.1], dtype=np.float32), top_k=2)

        self.assertEqual([hit.boi_reference for hit in hits], ["BOI-A", "BOI-B"])

    def test_e5_prompt_style_uses_query_and_passage_prefixes(self) -> None:
        self.assertEqual(_dense_prompt_style("intfloat/multilingual-e5-base"), "e5")
        self.assertEqual(_query_text("TVA redevable", prompt_style="e5"), "query: TVA redevable")
        self.assertEqual(_passage_text("Texte doc", prompt_style="e5"), "passage: Texte doc")

    def test_non_e5_prompt_style_keeps_plain_text(self) -> None:
        self.assertEqual(_dense_prompt_style("BAAI/bge-m3"), "plain")
        self.assertEqual(_query_text("TVA redevable", prompt_style="plain"), "TVA redevable")
        self.assertEqual(_passage_text("Texte doc", prompt_style="plain"), "Texte doc")


if __name__ == "__main__":
    unittest.main()
