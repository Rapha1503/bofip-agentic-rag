from __future__ import annotations

import unittest

from bofip_agentic.prompt_utils import _extract_numbers, build_prompt


class PromptUtilsTests(unittest.TestCase):
    def test_extract_numbers_keeps_amounts_and_ignores_dates(self) -> None:
        numbers = _extract_numbers(
            "Activite creee le 15 mars 2024, chiffre d'affaires 2024 de 3 200 euros, avis CFE 2025."
        )

        self.assertEqual([item["value"] for item in numbers], [3200])
        self.assertEqual(numbers[0]["unit"], "euros")

    def test_build_prompt_does_not_label_plain_quantities_as_euros(self) -> None:
        prompt = build_prompt(
            "J'ai vendu 50 livres numeriques a 4 euros l'unite.",
            [
                {
                    "rank": 1,
                    "boi_reference": "BOI-TEST",
                    "title": "Source test",
                    "publication_date": "2026-01-01",
                    "section_path": "Section",
                    "text": "Extrait.",
                }
            ],
        )

        self.assertIn("- 50 nombre", prompt)
        self.assertIn("- 4 euros", prompt)
        self.assertNotIn("- 50 euros", prompt)


if __name__ == "__main__":
    unittest.main()
