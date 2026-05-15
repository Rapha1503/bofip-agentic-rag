from __future__ import annotations

import unittest

from bofip_cleanroom.models import ChunkNode
from bofip_cleanroom.passage_eval import chunk_matches_passage_gold, normalize_match_text


class PassageEvalTests(unittest.TestCase):
    def test_normalize_match_text_strips_accents_and_whitespace(self) -> None:
        self.assertEqual(normalize_match_text("  Éligibilité\tCIR  "), "eligibilite cir")
        self.assertEqual(normalize_match_text("taxe d’aménagement"), "taxe d amenagement")

    def test_chunk_must_match_expected_document(self) -> None:
        chunk = ChunkNode(
            chunk_id="c1",
            source_type="BOFIP",
            document_id="doc1",
            boi_reference="BOI-AAA-10-20240101",
            doc_version="2024-01-01",
            strategy="section_window",
            section_id="s1",
            parent_chunk_id=None,
            section_path=["I. Champ d'application"],
            paragraph_range=["p1"],
            text="Les entreprises peuvent demander un credit d'impot recherche.",
            token_count=12,
            chunk_kind="paragraph_window",
        )

        gold = {
            "expected_boi": "BOI-BBB-10-20240101",
            "text_terms_all": ["credit d'impot recherche"],
        }
        self.assertFalse(chunk_matches_passage_gold(chunk, gold))

    def test_chunk_matches_when_all_constraints_are_satisfied(self) -> None:
        chunk = ChunkNode(
            chunk_id="c2",
            source_type="BOFIP",
            document_id="doc2",
            boi_reference="BOI-BIC-CHAMP-80-20-20-20-20240703",
            doc_version="2024-07-03",
            strategy="section_window",
            section_id="s2",
            parent_chunk_id=None,
            section_path=[
                "I. Nature des avantages",
                "B. Articulation avec le bénéfice du crédit d'impôt pour dépenses de recherche",
            ],
            paragraph_range=["p10", "p11"],
            text="Les entreprises qui exposent des dépenses de recherche peuvent également solliciter le bénéfice du CIR.",
            token_count=24,
            chunk_kind="paragraph_window",
        )

        gold = {
            "expected_boi": "BOI-BIC-CHAMP-80-20-20-20-20240703",
            "section_terms_all": ["Nature des avantages"],
            "section_terms_any": ["articulation", "recherche"],
            "text_terms_all": ["dépenses de recherche", "bénéfice du CIR"],
            "text_terms_any": ["solliciter", "demander"],
            "combined_terms_all": ["entreprises"],
            "combined_terms_any": ["nature des avantages", "solliciter"],
        }

        self.assertTrue(chunk_matches_passage_gold(chunk, gold))

    def test_chunk_fails_when_required_section_term_is_missing(self) -> None:
        chunk = ChunkNode(
            chunk_id="c3",
            source_type="BOFIP",
            document_id="doc3",
            boi_reference="BOI-INT-AEA-30-40-20231213",
            doc_version="2023-12-13",
            strategy="section_window",
            section_id="s3",
            parent_chunk_id=None,
            section_path=["II. Format de la déclaration"],
            paragraph_range=["p20"],
            text="La déclaration doit être transmise sous format informatique XML.",
            token_count=15,
            chunk_kind="paragraph_window",
        )

        gold = {
            "expected_boi": "BOI-INT-AEA-30-40-20231213",
            "section_terms_all": ["Champ d'application"],
        }

        self.assertFalse(chunk_matches_passage_gold(chunk, gold))

    def test_chunk_fails_when_any_constraint_has_no_match(self) -> None:
        chunk = ChunkNode(
            chunk_id="c4",
            source_type="BOFIP",
            document_id="doc4",
            boi_reference="BOI-TVA-DECLA-10-10-20-20251022",
            doc_version="2025-10-22",
            strategy="section_window",
            section_id="s4",
            parent_chunk_id=None,
            section_path=["I. Principe général"],
            paragraph_range=["p30"],
            text="Le fournisseur ou prestataire est en principe redevable de la taxe.",
            token_count=16,
            chunk_kind="paragraph_window",
        )

        gold = {
            "expected_boi": "BOI-TVA-DECLA-10-10-20-20251022",
            "text_terms_any": ["acompte", "retenue"],
        }

        self.assertFalse(chunk_matches_passage_gold(chunk, gold))

    def test_chunk_id_constraints_are_supported(self) -> None:
        chunk = ChunkNode(
            chunk_id="c5",
            source_type="BOFIP",
            document_id="doc5",
            boi_reference="BOI-XYZ-10-20240101",
            doc_version="2024-01-01",
            strategy="section_window",
            section_id="s5",
            parent_chunk_id=None,
            section_path=["I. Regles"],
            paragraph_range=["p50"],
            text="Texte utile pour l'evaluation passage-level.",
            token_count=10,
            chunk_kind="paragraph_window",
        )

        self.assertTrue(
            chunk_matches_passage_gold(
                chunk,
                {
                    "expected_boi": "BOI-XYZ-10-20240101",
                    "chunk_ids_any": ["c4", "c5"],
                },
            )
        )
        self.assertFalse(
            chunk_matches_passage_gold(
                chunk,
                {
                    "expected_boi": "BOI-XYZ-10-20240101",
                    "chunk_ids_any": ["c6"],
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
