from __future__ import annotations

import unittest

from bofip_cleanroom.models import ChunkNode, RawDocument, RawSectionNode
from bofip_cleanroom.lexical_retrieval import chunk_search_text_body
from bofip_cleanroom.two_stage_retrieval import TwoStageLexicalRetriever


class TwoStageRetrievalTests(unittest.TestCase):
    def test_search_keeps_best_chunk_of_top_document_first(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-A-10",
                title="TVA Redevable de la taxe Détermination du redevable",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
            RawDocument(
                document_id="doc2",
                boi_reference="BOI-B-10",
                title="TVA Exonérations bateaux aéronefs",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
        ]
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p1"], "détermination du redevable de la taxe sur la valeur ajoutée", 10, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p2"], "le preneur des services peut être redevable", 10, "paragraph_window"),
            ChunkNode("c3", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p3"], "prestations portant sur les aéronefs", 10, "paragraph_window"),
        ]

        retriever = TwoStageLexicalRetriever(documents, chunks)
        result = retriever.search("détermination du redevable de la TVA", top_docs=2, chunks_per_doc=2, max_chunks=3)

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-A-10")
        self.assertEqual(result.chunk_hits[0].boi_reference, "BOI-A-10")

    def test_search_round_robins_across_top_documents(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-A-10",
                title="TVA Redevable determination du redevable",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
            RawDocument(
                document_id="doc2",
                boi_reference="BOI-B-10",
                title="TVA Redevable preneur des services",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
            RawDocument(
                document_id="doc3",
                boi_reference="BOI-C-10",
                title="TVA Redevable cas particuliers",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
        ]
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p1"], "determination du redevable de la taxe sur la valeur ajoutee", 10, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p2"], "le preneur des services peut etre redevable", 10, "paragraph_window"),
            ChunkNode("c3", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p3"], "redevable du preneur pour prestations de services", 10, "paragraph_window"),
            ChunkNode("c4", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p4"], "prestations de services et territorialite", 10, "paragraph_window"),
            ChunkNode("c5", "BOFIP", "doc3", "BOI-C-10", None, "section_window", None, None, ["Titre C"], ["p5"], "cas particuliers du redevable en TVA", 10, "paragraph_window"),
        ]

        retriever = TwoStageLexicalRetriever(documents, chunks)
        result = retriever.search("redevable TVA", top_docs=3, chunks_per_doc=2, max_chunks=4)

        self.assertEqual(result.chunk_hits[0].boi_reference, result.document_hits[0].boi_reference)
        self.assertEqual([hit.local_rank for hit in result.chunk_hits[:3]], [1, 1, 1])
        self.assertEqual(
            {hit.boi_reference for hit in result.chunk_hits[:3]},
            {"BOI-A-10", "BOI-B-10", "BOI-C-10"},
        )

    def test_default_stage_two_uses_body_focused_local_index(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-A-10",
                title="Crédit d'impôt recherche - Dépenses de fonctionnement",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            )
        ]
        chunks = [
            ChunkNode(
                "c1",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Crédit d'impôt recherche - Dépenses de fonctionnement"],
                ["p1"],
                "Actualité liée : publication d'une mise à jour doctrinale",
                12,
                "paragraph",
            ),
            ChunkNode(
                "c2",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["I. Principes applicables"],
                ["p2"],
                "Les dépenses de fonctionnement éligibles correspondent notamment à une quote-part des amortissements et dépenses de personnel.",
                24,
                "paragraph_window",
            ),
        ]

        retriever = TwoStageLexicalRetriever(documents, chunks)
        self.assertEqual(retriever.local_chunk_mode, "body")
        self.assertIs(retriever.local_chunk_search_text_fn, chunk_search_text_body)

    def test_document_mode_sections_uses_section_titles_for_stage_one(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-A-10",
                title="TVA",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
                sections=[
                    RawSectionNode("s1", None, 1, 1, "Determination du redevable", None, ["Determination du redevable"]),
                ],
            ),
            RawDocument(
                document_id="doc2",
                boi_reference="BOI-B-10",
                title="TVA",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
                sections=[
                    RawSectionNode("s2", None, 1, 1, "Exonerations", None, ["Exonerations"]),
                ],
            ),
        ]
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", "s1", None, ["Determination du redevable"], ["p1"], "Le preneur peut etre redevable.", 8, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "doc2", "BOI-B-10", None, "section_window", "s2", None, ["Exonerations"], ["p1"], "Les aeronefs beneficient d'une exoneration.", 8, "paragraph_window"),
        ]

        retriever = TwoStageLexicalRetriever(documents, chunks, document_mode="sections")
        result = retriever.search("determination du redevable", top_docs=2, max_chunks=2)

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-A-10")

    def test_section_then_chunk_prioritizes_matching_section_before_chunk(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-A-10",
                title="Credit d'impot recherche",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            )
        ]
        chunks = [
            ChunkNode(
                "c1",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                "intro",
                None,
                ["Actualite liee"],
                ["p1"],
                "Actualite liee et publication d'une mise a jour.",
                9,
                "paragraph",
            ),
            ChunkNode(
                "c2",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                "rules",
                None,
                ["II. Depenses eligibles"],
                ["p2"],
                "Les depenses de normalisation sont retenues pour l'assiette du CIR.",
                12,
                "paragraph_window",
            ),
            ChunkNode(
                "c3",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                "rules",
                None,
                ["II. Depenses eligibles"],
                ["p3"],
                "Les depenses de personnel et d'amortissement sont aussi prises en compte.",
                13,
                "paragraph_window",
            ),
            ChunkNode(
                "c4",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                "other",
                None,
                ["III. Depenses exclues"],
                ["p4"],
                "Les amendes et penalites ne sont pas prises en compte.",
                10,
                "paragraph_window",
            ),
        ]

        retriever = TwoStageLexicalRetriever(documents, chunks, local_strategy="section_then_chunk")
        result = retriever.search(
            "depenses de normalisation",
            top_docs=1,
            sections_per_doc=1,
            chunks_per_section=2,
            max_chunks=2,
        )

        self.assertEqual(result.section_hits[0].section_path, ["II. Depenses eligibles"])
        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "c2")
        self.assertEqual(result.chunk_hits[0].section_rank, 1)


if __name__ == "__main__":
    unittest.main()
