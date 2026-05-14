from __future__ import annotations

import unittest

from bofip_cleanroom.family_guided_retrieval import FamilyGuidedRetriever, PriorDocumentHit
from bofip_cleanroom.models import ChunkNode, RawDocument, RawParagraph, RawSectionNode


class FamilyGuidedRetrievalTests(unittest.TestCase):
    def test_family_local_rerank_promotes_expected_sibling(self) -> None:
        documents = [
            RawDocument(
                document_id="parent",
                boi_reference="BOI-RPPM-RCM-40-50-20240730",
                title="PEA Regles generales",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s0", None, 1, 1, "Regles generales", None, ["Regles generales"])],
                paragraphs=[RawParagraph("p0", "s0", 1, "p", None, None, "Regles generales du plan d epargne en actions.", [], [])],
            ),
            RawDocument(
                document_id="child20",
                boi_reference="BOI-RPPM-RCM-40-50-20-20240730",
                title="PEA Versements",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s1", None, 1, 1, "Versements", None, ["Versements"])],
                paragraphs=[RawParagraph("p1", "s1", 1, "p", None, None, "Conditions de versement sur le plan d epargne en actions.", [], [])],
            ),
            RawDocument(
                document_id="child60",
                boi_reference="BOI-RPPM-RCM-40-50-60-20240730",
                title="PEA Dispositions diverses",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s2", None, 1, 1, "Dispositions diverses", None, ["Dispositions diverses"])],
                paragraphs=[RawParagraph("p2", "s2", 1, "p", None, None, "Dispositions diverses restant a connaitre sur le plan d epargne en actions.", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("c0", "BOFIP", "parent", "BOI-RPPM-RCM-40-50-20240730", "2024-07-30", "section_window", "s0", None, ["Regles generales"], ["p0"], "Regles generales du plan d epargne en actions.", 10, "paragraph_window"),
            ChunkNode("c1", "BOFIP", "child20", "BOI-RPPM-RCM-40-50-20-20240730", "2024-07-30", "section_window", "s1", None, ["Versements"], ["p1"], "Conditions de versement sur le plan d epargne en actions.", 11, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "child60", "BOI-RPPM-RCM-40-50-60-20240730", "2024-07-30", "section_window", "s2", None, ["Dispositions diverses"], ["p2"], "Dispositions diverses restant a connaitre sur le plan d epargne en actions.", 12, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Je veux un BOFiP qui regroupe les regles diverses restant a connaitre sur le PEA.",
            lexical_query="Je veux un BOFiP qui regroupe les regles diverses restant a connaitre sur le PEA. plan epargne actions",
            stage1_hits=[
                PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-RPPM-RCM-40-50-20240730"),
                PriorDocumentHit(rank=2, score=0.8, boi_reference="BOI-RPPM-RCM-40-50-20-20240730"),
            ],
            family_top_docs=1,
            top_docs=3,
            chunks_per_doc=1,
            max_chunks=3,
        )

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-RPPM-RCM-40-50-60-20240730")
        self.assertEqual(result.chunk_hits[0].boi_reference, "BOI-RPPM-RCM-40-50-60-20240730")

    def test_family_top_docs_two_can_recover_expected_doc_from_second_anchor_family(self) -> None:
        documents = [
            RawDocument(
                document_id="a",
                boi_reference="BOI-AAA-TEST-10-20240730",
                title="Famille A",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("sa", None, 1, 1, "A", None, ["A"])],
                paragraphs=[RawParagraph("pa", "sa", 1, "p", None, None, "Texte famille A", [], [])],
            ),
            RawDocument(
                document_id="b",
                boi_reference="BOI-BIC-CVAE-10-20240730",
                title="Famille B",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("sb", None, 1, 1, "B", None, ["B"])],
                paragraphs=[RawParagraph("pb", "sb", 1, "p", None, None, "Cotisation sur la valeur ajoutee des entreprises CVAE", [], [])],
            ),
            RawDocument(
                document_id="bchild",
                boi_reference="BOI-BIC-CVAE-10-20-20240730",
                title="Famille B Detail",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("sbc", None, 1, 1, "Detail", None, ["Detail"])],
                paragraphs=[RawParagraph("pbc", "sbc", 1, "p", None, None, "Regles detaillees de cotisation sur la valeur ajoutee des entreprises", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("ca", "BOFIP", "a", "BOI-AAA-TEST-10-20240730", "2024-07-30", "section_window", "sa", None, ["A"], ["pa"], "Texte famille A", 3, "paragraph_window"),
            ChunkNode("cb", "BOFIP", "b", "BOI-BIC-CVAE-10-20240730", "2024-07-30", "section_window", "sb", None, ["B"], ["pb"], "Cotisation sur la valeur ajoutee des entreprises CVAE", 8, "paragraph_window"),
            ChunkNode("cbc", "BOFIP", "bchild", "BOI-BIC-CVAE-10-20-20240730", "2024-07-30", "section_window", "sbc", None, ["Detail"], ["pbc"], "Regles detaillees de cotisation sur la valeur ajoutee des entreprises", 9, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Quelles sont les regles detaillees sur la CVAE ?",
            lexical_query="Quelles sont les regles detaillees sur la CVAE ? cotisation valeur ajoutee entreprises",
            stage1_hits=[
                PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-AAA-TEST-10-20240730"),
                PriorDocumentHit(rank=2, score=0.8, boi_reference="BOI-BIC-CVAE-10-20240730"),
            ],
            family_top_docs=2,
            top_docs=3,
            chunks_per_doc=1,
            max_chunks=3,
        )

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-BIC-CVAE-10-20-20240730")

    def test_preserve_stage1_top1_keeps_existing_winner_before_family_siblings(self) -> None:
        documents = [
            RawDocument(
                document_id="parent",
                boi_reference="BOI-RPPM-RCM-40-50-20240730",
                title="PEA Regles generales",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s0", None, 1, 1, "Regles generales", None, ["Regles generales"])],
                paragraphs=[RawParagraph("p0", "s0", 1, "p", None, None, "Regles generales du plan d epargne en actions.", [], [])],
            ),
            RawDocument(
                document_id="child60",
                boi_reference="BOI-RPPM-RCM-40-50-60-20240730",
                title="PEA Dispositions diverses",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s2", None, 1, 1, "Dispositions diverses", None, ["Dispositions diverses"])],
                paragraphs=[RawParagraph("p2", "s2", 1, "p", None, None, "Dispositions diverses restant a connaitre sur le plan d epargne en actions.", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("c0", "BOFIP", "parent", "BOI-RPPM-RCM-40-50-20240730", "2024-07-30", "section_window", "s0", None, ["Regles generales"], ["p0"], "Regles generales du plan d epargne en actions.", 10, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "child60", "BOI-RPPM-RCM-40-50-60-20240730", "2024-07-30", "section_window", "s2", None, ["Dispositions diverses"], ["p2"], "Dispositions diverses restant a connaitre sur le plan d epargne en actions.", 12, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Je veux un BOFiP qui regroupe les regles diverses restant a connaitre sur le PEA.",
            lexical_query="Je veux un BOFiP qui regroupe les regles diverses restant a connaitre sur le PEA. plan epargne actions",
            stage1_hits=[PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-RPPM-RCM-40-50-20240730")],
            family_top_docs=1,
            top_docs=2,
            preserve_stage1_top1=True,
        )

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-RPPM-RCM-40-50-20240730")
        self.assertEqual(result.document_hits[1].boi_reference, "BOI-RPPM-RCM-40-50-60-20240730")

    def test_overview_bonus_promotes_parent_when_multiple_descendants_support_same_branch(self) -> None:
        documents = [
            RawDocument(
                document_id="parent",
                boi_reference="BOI-IF-TU-10-20-20251231",
                title="Taxe d amenagement - Champ d application",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2025-12-31",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s0", None, 1, 1, "Champ d application", None, ["Champ d application"])],
                paragraphs=[RawParagraph("p0", "s0", 1, "p", None, None, "Document d ensemble sur le champ d application de la taxe d amenagement.", [], [])],
            ),
            RawDocument(
                document_id="child_a",
                boi_reference="BOI-IF-TU-10-20-10-20251231",
                title="Taxe d amenagement - Champ d application - Operations imposables",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2025-12-31",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s1", None, 1, 1, "Operations imposables", None, ["Operations imposables"])],
                paragraphs=[RawParagraph("p1", "s1", 1, "p", None, None, "Liste des operations imposables pour la taxe d amenagement.", [], [])],
            ),
            RawDocument(
                document_id="child_b",
                boi_reference="BOI-IF-TU-10-20-30-10-20251231",
                title="Taxe d amenagement - Champ d application - Exonerations de plein droit",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2025-12-31",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s2", None, 1, 1, "Exonerations", None, ["Exonerations"])],
                paragraphs=[RawParagraph("p2", "s2", 1, "p", None, None, "Regles d exonerations de plein droit en taxe d amenagement.", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("c0", "BOFIP", "parent", "BOI-IF-TU-10-20-20251231", "2025-12-31", "section_window", "s0", None, ["Champ d application"], ["p0"], "Document d ensemble sur le champ d application de la taxe d amenagement.", 12, "paragraph_window"),
            ChunkNode("c1", "BOFIP", "child_a", "BOI-IF-TU-10-20-10-20251231", "2025-12-31", "section_window", "s1", None, ["Operations imposables"], ["p1"], "Liste des operations imposables pour la taxe d amenagement.", 11, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "child_b", "BOI-IF-TU-10-20-30-10-20251231", "2025-12-31", "section_window", "s2", None, ["Exonerations"], ["p2"], "Regles d exonerations de plein droit en taxe d amenagement.", 11, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Je veux le document d ensemble sur la taxe d amenagement avant les sous-cas.",
            stage1_hits=[
                PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-IF-TU-10-20-10-20251231"),
                PriorDocumentHit(rank=2, score=0.8, boi_reference="BOI-IF-TU-10-20-30-10-20251231"),
            ],
            family_top_docs=2,
            top_docs=3,
            chunks_per_doc=1,
            max_chunks=3,
            overview_weight=0.5,
            overview_min_descendants=2,
            overview_top_family_ranks=3,
        )

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-IF-TU-10-20-20251231")
        self.assertGreater(result.document_hits[0].descendant_support, 0.0)

    def test_overview_bonus_does_not_apply_with_single_descendant(self) -> None:
        documents = [
            RawDocument(
                document_id="parent",
                boi_reference="BOI-IF-TU-10-20-20251231",
                title="Taxe d amenagement - Champ d application",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2025-12-31",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s0", None, 1, 1, "Champ d application", None, ["Champ d application"])],
                paragraphs=[RawParagraph("p0", "s0", 1, "p", None, None, "Document d ensemble sur le champ d application de la taxe d amenagement.", [], [])],
            ),
            RawDocument(
                document_id="child_a",
                boi_reference="BOI-IF-TU-10-20-10-20251231",
                title="Taxe d amenagement - Champ d application - Operations imposables",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2025-12-31",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s1", None, 1, 1, "Operations imposables", None, ["Operations imposables"])],
                paragraphs=[RawParagraph("p1", "s1", 1, "p", None, None, "Liste des operations imposables pour la taxe d amenagement.", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("c0", "BOFIP", "parent", "BOI-IF-TU-10-20-20251231", "2025-12-31", "section_window", "s0", None, ["Champ d application"], ["p0"], "Document d ensemble sur le champ d application de la taxe d amenagement.", 12, "paragraph_window"),
            ChunkNode("c1", "BOFIP", "child_a", "BOI-IF-TU-10-20-10-20251231", "2025-12-31", "section_window", "s1", None, ["Operations imposables"], ["p1"], "Liste des operations imposables pour la taxe d amenagement.", 11, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Je cherche les operations imposables de la taxe d amenagement.",
            stage1_hits=[PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-IF-TU-10-20-10-20251231")],
            family_top_docs=1,
            top_docs=2,
            chunks_per_doc=1,
            max_chunks=2,
            overview_weight=0.5,
            overview_min_descendants=2,
            overview_top_family_ranks=2,
        )

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-IF-TU-10-20-10-20251231")
        self.assertEqual(result.document_hits[0].descendant_count, 0)

    def test_tail_weight_uses_title_suffix_to_promote_specific_sibling(self) -> None:
        documents = [
            RawDocument(
                document_id="parent",
                boi_reference="BOI-INT-AEA-30-20231213",
                title="INT - Plateformes - Obligations des operateurs de plateforme",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2023-12-13",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s0", None, 1, 1, "Vue d'ensemble", None, ["Vue d'ensemble"])],
                paragraphs=[RawParagraph("p0", "s0", 1, "p", None, None, "Document d ensemble sur les obligations des operateurs de plateforme.", [], [])],
            ),
            RawDocument(
                document_id="child_decl",
                boi_reference="BOI-INT-AEA-30-30-20230111",
                title="INT - Plateformes - Obligations des operateurs de plateforme - Obligations declaratives",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2023-01-11",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s1", None, 1, 1, "Obligations declaratives", None, ["Obligations declaratives"])],
                paragraphs=[RawParagraph("p1", "s1", 1, "p", None, None, "Que declarer a l administration pour les operateurs de plateforme.", [], [])],
            ),
            RawDocument(
                document_id="child_sanctions",
                boi_reference="BOI-INT-AEA-30-50-20231213",
                title="INT - Plateformes - Obligations des operateurs de plateforme - Sanctions",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2023-12-13",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s2", None, 1, 1, "Sanctions", None, ["Sanctions"])],
                paragraphs=[RawParagraph("p2", "s2", 1, "p", None, None, "Sanctions applicables aux operateurs de plateforme.", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("c0", "BOFIP", "parent", "BOI-INT-AEA-30-20231213", "2023-12-13", "section_window", "s0", None, ["Vue d'ensemble"], ["p0"], "Document d ensemble sur les obligations des operateurs de plateforme.", 10, "paragraph_window"),
            ChunkNode("c1", "BOFIP", "child_decl", "BOI-INT-AEA-30-30-20230111", "2023-01-11", "section_window", "s1", None, ["Obligations declaratives"], ["p1"], "Que declarer a l administration pour les operateurs de plateforme.", 11, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "child_sanctions", "BOI-INT-AEA-30-50-20231213", "2023-12-13", "section_window", "s2", None, ["Sanctions"], ["p2"], "Sanctions applicables aux operateurs de plateforme.", 11, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Qu est-ce qu une plateforme doit declarer a l administration ?",
            stage1_hits=[
                PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-INT-AEA-30-20231213"),
                PriorDocumentHit(rank=2, score=0.8, boi_reference="BOI-INT-AEA-30-50-20231213"),
            ],
            family_top_docs=2,
            top_docs=3,
            chunks_per_doc=1,
            max_chunks=3,
            tail_weight=0.5,
        )

        self.assertEqual(result.document_hits[0].boi_reference, "BOI-INT-AEA-30-30-20230111")
        self.assertIsNotNone(result.document_hits[0].tail_rank)

    def test_ancestor_expansion_can_recover_sibling_branch_outside_initial_subfamily(self) -> None:
        documents = [
            RawDocument(
                document_id="child20a",
                boi_reference="BOI-RPPM-RCM-40-50-20-10-20240730",
                title="PEA Versements en numeraire",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s1", None, 1, 1, "Versements", None, ["Versements"])],
                paragraphs=[RawParagraph("p1", "s1", 1, "p", None, None, "Versements en numeraire sur le plan d epargne en actions.", [], [])],
            ),
            RawDocument(
                document_id="child20b",
                boi_reference="BOI-RPPM-RCM-40-50-20-20-20240730",
                title="PEA Versements en titres",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s2", None, 1, 1, "Versements en titres", None, ["Versements en titres"])],
                paragraphs=[RawParagraph("p2", "s2", 1, "p", None, None, "Versements en titres sur le plan d epargne en actions.", [], [])],
            ),
            RawDocument(
                document_id="child60",
                boi_reference="BOI-RPPM-RCM-40-50-60-20240730",
                title="PEA Dispositions diverses",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date="2024-07-30",
                source_url=None,
                language=None,
                sections=[RawSectionNode("s3", None, 1, 1, "Dispositions diverses", None, ["Dispositions diverses"])],
                paragraphs=[RawParagraph("p3", "s3", 1, "p", None, None, "Regles diverses du plan d epargne en actions.", [], [])],
            ),
        ]
        chunks = [
            ChunkNode("c1", "BOFIP", "child20a", "BOI-RPPM-RCM-40-50-20-10-20240730", "2024-07-30", "section_window", "s1", None, ["Versements"], ["p1"], "Versements en numeraire sur le plan d epargne en actions.", 11, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "child20b", "BOI-RPPM-RCM-40-50-20-20-20240730", "2024-07-30", "section_window", "s2", None, ["Versements en titres"], ["p2"], "Versements en titres sur le plan d epargne en actions.", 11, "paragraph_window"),
            ChunkNode("c3", "BOFIP", "child60", "BOI-RPPM-RCM-40-50-60-20240730", "2024-07-30", "section_window", "s3", None, ["Dispositions diverses"], ["p3"], "Regles diverses du plan d epargne en actions.", 9, "paragraph_window"),
        ]

        retriever = FamilyGuidedRetriever(documents, chunks)
        result = retriever.search(
            "Quelles sont les dispositions diverses du PEA ?",
            stage1_hits=[
                PriorDocumentHit(rank=1, score=0.9, boi_reference="BOI-RPPM-RCM-40-50-20-10-20240730"),
                PriorDocumentHit(rank=2, score=0.8, boi_reference="BOI-RPPM-RCM-40-50-20-20-20240730"),
            ],
            family_top_docs=2,
            ancestor_expansion_levels=1,
            top_docs=3,
            chunks_per_doc=1,
            max_chunks=3,
        )

        self.assertIn("BOI-RPPM-RCM-40-50-60-20240730", result.family_selection.members)
        self.assertEqual(result.document_hits[0].boi_reference, "BOI-RPPM-RCM-40-50-60-20240730")


if __name__ == "__main__":
    unittest.main()
