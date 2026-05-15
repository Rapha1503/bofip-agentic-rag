from __future__ import annotations

import unittest

from bofip_agentic.lexical_retrieval import (
    DocumentLexicalIndex,
    LexicalBM25Index,
    get_document_search_text_fn,
    tokenize,
)
from bofip_agentic.models import ChunkNode, RawDocument, RawParagraph, RawSectionNode


class LexicalRetrievalTests(unittest.TestCase):
    def test_tokenize_is_accent_insensitive(self) -> None:
        self.assertEqual(tokenize("deplacements routiers"), ["deplacements", "routiers"])
        self.assertEqual(tokenize("Ariege"), ["ariege"])

    def test_tokenize_can_apply_french_stemming(self) -> None:
        self.assertEqual(tokenize("application applique", stem=True), ["appliqu", "appliqu"])

    def test_search_documents_dedupes_same_boi_reference(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A", None, "section_window", "s1", None, ["Titre A"], ["p1"], "credit impot recherche remboursement", 10, "paragraph"),
            ChunkNode("c2", "BOFIP", "doc1", "BOI-A", None, "section_window", "s2", None, ["Titre A"], ["p2"], "remboursement immediat du CIR pour JEI", 10, "paragraph"),
            ChunkNode("c3", "BOFIP", "doc2", "BOI-B", None, "section_window", "s3", None, ["Titre B"], ["p3"], "TVA presse declaration", 10, "paragraph"),
        ]
        index = LexicalBM25Index(chunks)

        hits = index.search_documents("remboursement CIR JEI", top_k=2)

        self.assertEqual([hit.boi_reference for hit in hits], ["BOI-A", "BOI-B"])

    def test_document_lexical_index_matches_specific_reference(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-BIC-BASE-60-20120912",
                title="BIC Base d'imposition Operations de credit-bail mobilier et immobilier",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
            RawDocument(
                document_id="doc2",
                boi_reference="BOI-BIC-BASE-60-20-20120912",
                title="BIC Base d'imposition Operations de credit-bail mobilier et immobilier Obligation de constater l'amortissement en comptabilite",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
        ]
        index = DocumentLexicalIndex(documents)

        hits = index.search_documents("BOI-BIC-BASE-60-20-20120912", top_k=2)

        self.assertIn("BOI-BIC-BASE-60-20-20120912", [hit.boi_reference for hit in hits])

    def test_document_lexical_index_uses_custom_search_text(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-A",
                title="Titre A",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
            RawDocument(
                document_id="doc2",
                boi_reference="BOI-B",
                title="Titre B",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
            ),
        ]

        search_texts = {
            "BOI-A": "alpha",
            "BOI-B": "needle",
        }
        index = DocumentLexicalIndex(documents, search_text_fn=lambda document: search_texts[document.boi_reference])

        self.assertEqual(index.search_texts, ["alpha", "needle"])

    def test_document_sections_leads_mode_uses_first_section_paragraphs(self) -> None:
        document = RawDocument(
            document_id="doc1",
            boi_reference="BOI-A",
            title="Titre A",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date=None,
            source_url=None,
            language=None,
            sections=[
                RawSectionNode("s1", None, 1, 1, "I. Champ", None, ["I. Champ"]),
                RawSectionNode("s2", None, 1, 2, "II. Modalites", None, ["II. Modalites"]),
            ],
            paragraphs=[
                RawParagraph("p1", "s1", 1, "p", None, "10", "Les jeunes entreprises innovantes beneficient d'un regime favorable. Details complementaires.", [], []),
                RawParagraph("p2", "s2", 2, "p", None, "20", "Le remboursement immediat de la creance intervient sous conditions.", [], []),
            ],
        )

        text = get_document_search_text_fn("sections_leads")(document)

        self.assertIn("I. Champ", text)
        self.assertIn("II. Modalites", text)
        self.assertIn("Les jeunes entreprises innovantes beneficient d'un regime favorable", text)
        self.assertIn("Le remboursement immediat de la creance intervient sous conditions", text)

    def test_document_title_mode_keeps_title_without_subject_noise(self) -> None:
        document = RawDocument(
            document_id="doc1",
            boi_reference="BOI-A",
            title="Titre principal",
            html_title="Titre HTML",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date=None,
            source_url=None,
            language=None,
            category_path=["Categorie", "Sous-categorie"],
            subjects=["Sujet A", "Sujet B"],
        )

        text = get_document_search_text_fn("title")(document)

        self.assertIn("Titre principal", text)
        self.assertIn("Titre HTML", text)
        self.assertNotIn("Sujet A", text)

    def test_document_title_tail_mode_focuses_on_discriminative_suffix(self) -> None:
        document = RawDocument(
            document_id="doc1",
            boi_reference="BOI-IF-TU-10-20-10-20251231",
            title="IF - Taxes d'urbanisme - Taxe d'aménagement - Champ d'application - Opérations imposables",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date=None,
            source_url=None,
            language=None,
        )

        text = get_document_search_text_fn("title_tail")(document)

        self.assertIn("Champ d'application - Opérations imposables", text)
        self.assertNotIn("Taxes d'urbanisme", text)

    def test_custom_chunk_search_text_function_is_used(self) -> None:
        chunks = [
            ChunkNode(
                "c1",
                "BOFIP",
                "doc1",
                "BOI-A",
                None,
                "section_window",
                "s1",
                None,
                ["Credit d'impot recherche - Depenses de fonctionnement"],
                ["p1"],
                "Actualite liee : publication d'une mise a jour doctrinale",
                12,
                "paragraph",
            ),
            ChunkNode(
                "c2",
                "BOFIP",
                "doc1",
                "BOI-A",
                None,
                "section_window",
                "s2",
                None,
                ["I. Principes applicables"],
                ["p2"],
                "Les depenses de fonctionnement eligibles correspondent notamment a une quote-part des amortissements et depenses de personnel.",
                24,
                "paragraph_window",
            ),
        ]

        search_texts = {
            "c1": "alpha",
            "c2": "needle",
        }
        index = LexicalBM25Index(chunks, search_text_fn=lambda chunk: search_texts[chunk.chunk_id])

        self.assertEqual(index.search_texts, ["alpha", "needle"])


if __name__ == "__main__":
    unittest.main()
