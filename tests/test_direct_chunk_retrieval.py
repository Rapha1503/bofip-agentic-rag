from __future__ import annotations

import unittest

from bofip_agentic.direct_chunk_retrieval import (
    DirectChunkRetriever,
    Stage1DocumentHit,
    _numeric_evidence_score,
)
from bofip_agentic.models import ChunkNode


class DirectChunkRetrievalTests(unittest.TestCase):
    def test_direct_chunk_retrieval_round_robins_top_docs(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p1"], "dépenses de recherche éligibles", 10, "paragraph_window"),
            ChunkNode("c2", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p2"], "autre paragraphe", 10, "paragraph_window"),
            ChunkNode("c3", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p3"], "redevable de la taxe sur la valeur ajoutée", 10, "paragraph_window"),
            ChunkNode("c4", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p4"], "autre contenu TVA", 10, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="body")
        result = retriever.search(
            "recherche TVA",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=0.9, boi_reference="BOI-A-10"),
                Stage1DocumentHit(rank=2, score=0.8, boi_reference="BOI-B-10"),
            ],
            top_docs=2,
            chunks_per_doc=2,
            max_candidates=4,
        )

        self.assertEqual([hit.local_rank for hit in result.chunk_hits[:2]], [1, 1])
        self.assertEqual([hit.document_rank for hit in result.chunk_hits[:2]], [1, 2])

    def test_direct_chunk_retrieval_defaults_to_full_mode(self) -> None:
        chunks = [
            ChunkNode("c1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Titre A"], ["p1"], "texte", 10, "paragraph_window"),
        ]
        retriever = DirectChunkRetriever(chunks)
        self.assertEqual(retriever.local_chunk_mode, "full")

    def test_numeric_evidence_detects_euro_symbol_amounts(self) -> None:
        chunk = ChunkNode(
            "amount",
            "BOFIP",
            "doc1",
            "BOI-A-10",
            None,
            "section_window",
            None,
            None,
            ["Conditions"],
            ["p1"],
            "Le montant applicable est de 12 300 € hors taxe.",
            10,
            "paragraph_window",
        )

        score = _numeric_evidence_score("quel montant de chiffre d'affaires", chunk)

        self.assertGreaterEqual(score, 4.0)

    def test_numeric_evidence_matches_thousands_separator_variants(self) -> None:
        matching = ChunkNode(
            "matching",
            "BOFIP",
            "doc1",
            "BOI-A-10",
            None,
            "section_window",
            None,
            None,
            ["Seuil de chiffre d'affaires"],
            ["p1"],
            "Le seuil est fixe a 5 000 euros.",
            12,
            "paragraph_window",
        )
        other = ChunkNode(
            "other",
            "BOFIP",
            "doc2",
            "BOI-B-10",
            None,
            "section_window",
            None,
            None,
            ["Seuil de chiffre d'affaires"],
            ["p1"],
            "Le seuil est fixe a 12 300 euros.",
            12,
            "paragraph_window",
        )

        query = "seuil chiffre affaires 5000"

        self.assertGreater(_numeric_evidence_score(query, matching), _numeric_evidence_score(query, other))

    def test_numeric_intent_promotes_amount_evidence_inside_document(self) -> None:
        chunks = [
            ChunkNode(
                "generic",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Champ d'application"],
                ["p1"],
                "La franchise concerne les assujettis et les operations imposables. Le seuil est defini ailleurs.",
                14,
                "paragraph_window",
            ),
            ChunkNode(
                "amount",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Chiffre d'affaires limite"],
                ["p2"],
                "La limite applicable aux ventes est fixee a 12 300 euros hors taxe.",
                12,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "a partir de quel montant de chiffre d'affaires le seuil s'applique",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "amount")

    def test_textual_amount_word_does_not_globally_evict_stronger_section(self) -> None:
        chunks = [
            ChunkNode(
                "meal-proof",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Deductibilite des frais supplementaires de repas"],
                ["p1"],
                "Les frais supplementaires de repas sont deductibles sous conditions.",
                12,
                "paragraph_window",
            ),
            ChunkNode(
                "amount-noise",
                "BOFIP",
                "doc2",
                "BOI-B-10",
                None,
                "section_window",
                None,
                None,
                ["Montant forfaitaire sans rapport"],
                ["p2"],
                "Le montant forfaitaire est de 12 300 euros.",
                10,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "frais de repas montant forfaitaire deduction",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10"),
                Stage1DocumentHit(rank=2, score=0.9, boi_reference="BOI-B-10"),
            ],
            top_docs=2,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "meal-proof")

    def test_numeric_intent_keeps_second_chunk_from_strong_document(self) -> None:
        chunks = [
            ChunkNode(
                "doc1-generic",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Champ d'application"],
                ["p1"],
                "Le seuil de franchise concerne les ventes de biens.",
                8,
                "paragraph_window",
            ),
            ChunkNode(
                "doc1-amount",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Chiffre d'affaires limite"],
                ["p2"],
                "Le seuil applicable aux ventes de biens est de 12 300 euros hors taxe.",
                12,
                "paragraph_window",
            ),
            ChunkNode("doc2-generic", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p3"], "seuil franchise operations", 4, "paragraph_window"),
            ChunkNode("doc3-generic", "BOFIP", "doc3", "BOI-C-10", None, "section_window", None, None, ["Titre C"], ["p4"], "seuil franchise operations", 4, "paragraph_window"),
            ChunkNode("doc4-generic", "BOFIP", "doc4", "BOI-D-10", None, "section_window", None, None, ["Titre D"], ["p5"], "seuil franchise operations", 4, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "a partir de quel montant de chiffre d'affaires le seuil de ventes s'applique",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10"),
                Stage1DocumentHit(rank=2, score=0.9, boi_reference="BOI-B-10"),
                Stage1DocumentHit(rank=3, score=0.8, boi_reference="BOI-C-10"),
                Stage1DocumentHit(rank=4, score=0.7, boi_reference="BOI-D-10"),
            ],
            top_docs=4,
            chunks_per_doc=2,
            max_candidates=4,
        )

        self.assertIn("doc1-amount", [hit.chunk.chunk_id for hit in result.chunk_hits])

    def test_numeric_intent_can_keep_two_amount_chunks_from_same_document(self) -> None:
        chunks = [
            ChunkNode("doc1-context", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Modification des seuils"], ["p1"], "Les seuils sont modifiés. Le nouveau plafond temporaire est de 25 000 €.", 14, "paragraph_window"),
            ChunkNode("doc1-table", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Sortie du regime"], ["p2"], "Le tableau indique un seuil de droit commun de 85 000 € et un seuil de tolerance de 93 500 €.", 18, "paragraph_window"),
            ChunkNode("doc2-generic", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Titre B"], ["p3"], "seuil franchise operations", 4, "paragraph_window"),
            ChunkNode("doc3-generic", "BOFIP", "doc3", "BOI-C-10", None, "section_window", None, None, ["Titre C"], ["p4"], "seuil franchise operations", 4, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "quel montant de chiffre d'affaires et seuil de tolerance",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10"),
                Stage1DocumentHit(rank=2, score=0.9, boi_reference="BOI-B-10"),
                Stage1DocumentHit(rank=3, score=0.8, boi_reference="BOI-C-10"),
            ],
            top_docs=3,
            chunks_per_doc=2,
            max_candidates=2,
        )

        self.assertEqual(
            {"doc1-context", "doc1-table"},
            {hit.chunk.chunk_id for hit in result.chunk_hits},
        )

    def test_numeric_intent_prefers_matching_threshold_over_unrelated_amount(self) -> None:
        chunks = [
            ChunkNode(
                "threshold",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Chiffre d'affaires limite"],
                ["p1"],
                "Le seuil de franchise pour les ventes de biens est fixe a 12 300 € hors taxe.",
                14,
                "paragraph_window",
            ),
            ChunkNode(
                "unrelated-amount",
                "BOFIP",
                "doc2",
                "BOI-B-10",
                None,
                "section_window",
                None,
                None,
                ["Montant des acomptes"],
                ["p2"],
                "L'acompte de juillet est egal a 55 % et celui de decembre a 40 % du montant du.",
                16,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "a partir de quel montant de chiffre d'affaires seuil franchise ventes de biens",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-B-10"),
                Stage1DocumentHit(rank=2, score=0.9, boi_reference="BOI-A-10"),
            ],
            top_docs=2,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "threshold")

    def test_generic_limits_word_does_not_promote_unrelated_amounts(self) -> None:
        chunks = [
            ChunkNode(
                "unrelated-amount",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Conditions de revenu"],
                ["p1"],
                "Le seuil de 93 510 euros est apprecie en totalisant les remunerations brutes.",
                14,
                "paragraph_window",
            ),
            ChunkNode(
                "semantic-target",
                "BOFIP",
                "doc2",
                "BOI-B-10",
                None,
                "section_window",
                None,
                None,
                ["Indemnites versees a l'occasion de la rupture conventionnelle"],
                ["p2"],
                "Le regime des indemnites de rupture conventionnelle precise les conditions d'exoneration.",
                16,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "indemnite rupture conventionnelle exoneration limites contrat travail",
            stage1_hits=[
                Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-B-10"),
                Stage1DocumentHit(rank=2, score=0.9, boi_reference="BOI-A-10"),
            ],
            top_docs=2,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "semantic-target")

    def test_intra_document_ranking_prefers_matching_numeric_threshold_section(self) -> None:
        chunks = [
            ChunkNode(
                "general-minimum",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Cotisation minimum", "Regles generales"],
                ["p1"],
                "La cotisation minimum est due au lieu du principal etablissement. "
                "La base minimum est fixee par deliberation de la commune.",
                32,
                "paragraph_window",
            ),
            ChunkNode(
                "matching-threshold",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Cotisation minimum", "Exoneration des contribuables a faible chiffre d'affaires"],
                ["p2"],
                "Les redevables qui realisent un chiffre d'affaires ou des recettes "
                "inferieurs ou egaux a 5 000 euros sont exoneres de cotisation minimum.",
                34,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "exoneration cotisation minimum chiffre affaires recettes inferieur egal 5000",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "matching-threshold")

    def test_user_amount_does_not_beat_specific_section_heading(self) -> None:
        chunks = [
            ChunkNode(
                "example-with-user-amount",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Cotisation minimum", "Absence de deliberation"],
                ["p1"],
                "Exemple : une commune fixe une base minimum. Pour un chiffre d'affaires de 3 200 euros, "
                "la base applicable est de 600 euros dans cet exemple local.",
                32,
                "paragraph_window",
            ),
            ChunkNode(
                "specific-exemption",
                "BOFIP",
                "doc1",
                "BOI-A-10",
                None,
                "section_window",
                None,
                None,
                ["Cotisation minimum", "Exoneration des contribuables a faible chiffre d'affaires"],
                ["p2"],
                "Les redevables qui realisent un montant de chiffre d'affaires ou de recettes "
                "inferieur ou egal au seuil legal sont exoneres de cotisation minimum.",
                34,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "faible chiffre affaires micro entrepreneur 3200 euros exonere cotisation minimum",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "specific-exemption")

    def test_query_with_exact_boi_reference_follows_reference_mentions(self) -> None:
        chunks = [
            ChunkNode("stage1", "BOFIP", "doc1", "BOI-A-10", None, "section_window", None, None, ["Sujet"], ["p1"], "Paragraphe general sans reference.", 6, "paragraph_window"),
            ChunkNode("referencing", "BOFIP", "doc2", "BOI-B-10", None, "section_window", None, None, ["Bareme"], ["p2"], "Le seuil est expose au BOI-BAREME-999999 avec un montant de 12 300 €.", 14, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "BOI-BAREME-999999 seuil franchise montant",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-A-10")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=2,
        )

        self.assertIn("referencing", [hit.chunk.chunk_id for hit in result.chunk_hits])

    def test_query_with_exact_boi_reference_retrieves_matching_child_document(self) -> None:
        chunks = [
            ChunkNode("stage1", "BOFIP", "doc1", "BOI-IR-PAS-10", None, "section_window", None, None, ["PAS"], ["p1"], "Paragraphe general.", 6, "paragraph_window"),
            ChunkNode("rsa-child", "BOFIP", "doc2", "BOI-RSA-CHAMP-20-40-10-30", None, "section_window", None, None, ["Rupture conventionnelle"], ["p2"], "Indemnites de rupture conventionnelle et limites d'exoneration.", 12, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "BOI-RSA-CHAMP-20-40-10 rupture conventionnelle exoneration",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-IR-PAS-10")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=2,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "rsa-child")

    def test_selected_candidate_internal_reference_adds_referenced_document(self) -> None:
        chunks = [
            ChunkNode("pas", "BOFIP", "doc1", "BOI-IR-PAS-10", None, "section_window", None, None, ["PAS"], ["p1"], "Le regime fiscal est precise au BOI-RSA-CHAMP-20-40-10.", 10, "paragraph_window"),
            ChunkNode("rsa", "BOFIP", "doc2", "BOI-RSA-CHAMP-20-40-10-30", None, "section_window", None, None, ["Rupture conventionnelle"], ["p2"], "Indemnites de rupture conventionnelle et limites d'exoneration.", 12, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "rupture conventionnelle exoneration",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-IR-PAS-10")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=3,
        )

        self.assertIn("rsa", [hit.chunk.chunk_id for hit in result.chunk_hits])

    def test_local_chunk_ranking_prefers_section_phrase_overlap(self) -> None:
        chunks = [
            ChunkNode("generic", "BOFIP", "doc1", "BOI-RSA-TEST", None, "section_window", None, None, ["Indemnites de licenciement"], ["p1"], "Les indemnites peuvent etre exonerees dans certaines limites.", 10, "paragraph_window"),
            ChunkNode("target", "BOFIP", "doc1", "BOI-RSA-TEST", None, "section_window", None, None, ["Indemnites versees a l'occasion de la rupture conventionnelle"], ["p2"], "Regime des indemnites de rupture conventionnelle.", 10, "paragraph_window"),
            ChunkNode("other", "BOFIP", "doc1", "BOI-RSA-TEST", None, "section_window", None, None, ["Indemnites de mise a la retraite"], ["p3"], "Indemnites versees lors du depart a la retraite.", 10, "paragraph_window"),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "rupture conventionnelle exoneration limites",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-RSA-TEST")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "target")

    def test_general_rule_query_prefers_principle_chunk_over_remark(self) -> None:
        chunks = [
            ChunkNode(
                "remark",
                "BOFIP",
                "doc1",
                "BOI-TVA-TEST",
                None,
                "section_window",
                None,
                None,
                ["Lieu des prestations de services", "Regle de territorialite", "Principes generaux"],
                ["p2"],
                "Remarque : prestations de services preneur assujetti lieu etablissement "
                "regle generale territorialite numero identification.",
                16,
                "paragraph_window",
            ),
            ChunkNode(
                "principle",
                "BOFIP",
                "doc1",
                "BOI-TVA-TEST",
                None,
                "section_window",
                None,
                None,
                ["Lieu des prestations de services", "Regle de territorialite", "Principes generaux"],
                ["p1"],
                "Le lieu des prestations de services est determine en fonction du lieu "
                "d'etablissement du preneur assujetti.",
                16,
                "paragraph_window",
            ),
        ]

        retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        result = retriever.search(
            "territorialite prestation services B2B preneur assujetti lieu etablissement regle generale",
            stage1_hits=[Stage1DocumentHit(rank=1, score=1.0, boi_reference="BOI-TVA-TEST")],
            top_docs=1,
            chunks_per_doc=1,
            max_candidates=1,
        )

        self.assertEqual(result.chunk_hits[0].chunk.chunk_id, "principle")


if __name__ == "__main__":
    unittest.main()
