from __future__ import annotations

import unittest

from bofip_agentic.providers import PROVIDERS, coerce_model_for_provider


class ProviderConfigTests(unittest.TestCase):
    def test_provider_change_coerces_model_to_new_provider_default(self):
        current_codex_model = PROVIDERS["Codex local"]["default_model"]

        coerced = coerce_model_for_provider("DeepSeek", current_codex_model)

        self.assertEqual(coerced, PROVIDERS["DeepSeek"]["default_model"])

    def test_provider_model_is_preserved_when_valid_for_provider(self):
        selected = "deepseek-v4-pro"

        coerced = coerce_model_for_provider("DeepSeek", selected)

        self.assertEqual(coerced, selected)


if __name__ == "__main__":
    unittest.main()
