from __future__ import annotations

import unittest

from bofip_cleanroom.alias_expansion import build_acronym_expansion_map, expand_query_with_acronyms
from bofip_cleanroom.models import RawDocument, RawSectionNode


class AliasExpansionTests(unittest.TestCase):
    def test_build_acronym_expansion_map_recovers_pea_style_phrase(self) -> None:
        documents = [
            RawDocument(
                document_id="doc1",
                boi_reference="BOI-RPPM-RCM-40-50-20240730",
                title="RPPM - Revenus de capitaux mobiliers - Plan d'épargne en actions",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
                sections=[
                    RawSectionNode("s1", None, 1, 1, "Modalites de fonctionnement du PEA", None, ["Modalites de fonctionnement du PEA"]),
                ],
            ),
            RawDocument(
                document_id="doc2",
                boi_reference="BOI-RPPM-RCM-40-50-60-20240730",
                title="RPPM - Revenus de capitaux mobiliers - Plan d'épargne en actions - Dispositions diverses",
                document_type="Contenu",
                content_type="Commentaire",
                publication_date=None,
                source_url=None,
                language=None,
                sections=[
                    RawSectionNode("s2", None, 1, 1, "Regles diverses du PEA", None, ["Regles diverses du PEA"]),
                ],
            ),
        ]

        mapping = build_acronym_expansion_map(documents)

        self.assertIn("PEA", mapping)
        self.assertIn("plan epargne actions", mapping["PEA"])

    def test_expand_query_with_acronyms_appends_phrase_once(self) -> None:
        expansion_map = {
            "PEA": ["plan epargne actions"],
            "IFI": ["impot fortune immobiliere"],
        }

        expanded, expansions = expand_query_with_acronyms(
            "Je veux les regles du PEA et de l'IFI",
            expansion_map,
        )

        self.assertIn("plan epargne actions", expanded)
        self.assertIn("impot fortune immobiliere", expanded)
        self.assertEqual(expansions[0][0], "PEA")
        self.assertEqual(expansions[1][0], "IFI")


if __name__ == "__main__":
    unittest.main()
