from __future__ import annotations

import unittest
from pathlib import Path


class UiTextIntegrityTests(unittest.TestCase):
    def test_public_ui_copy_keeps_french_accents_and_no_replacement_markers(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        qmark = chr(63)

        broken_fragments = [
            f"R{qmark}ponse",
            f"r{qmark}ponse",
            f"cl{qmark}",
            f"Cl{qmark}",
            f"d{qmark}sactiv",
            f"Mod{qmark}le",
            f"r{qmark}el",
            f"fran{qmark}ais",
            f"T{qmark}l{qmark}chargement",
            f"v{qmark}rification",
        ]
        for fragment in broken_fragments:
            self.assertNotIn(fragment, app_source)

        expected_fragments = [
            "Réponse sourcée",
            "clé",
            "Modèle",
            "français",
            "Parcours agentique réel",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, app_source)


if __name__ == "__main__":
    unittest.main()