from __future__ import annotations

import unittest

from bofip_agentic.codex_cli_client import _extract_agent_message, _messages_to_prompt
from bofip_agentic.providers import PROVIDERS


class CodexCliClientTests(unittest.TestCase):
    def test_extract_agent_message_ignores_non_json_noise(self):
        output = "\n".join(
            [
                '{"type":"thread.started","thread_id":"abc"}',
                "warning: noisy plugin log",
                '{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}',
                '{"type":"turn.completed","usage":{"input_tokens":1}}',
            ]
        )

        self.assertEqual(_extract_agent_message(output), "OK")

    def test_json_prompt_demands_json_only(self):
        prompt = _messages_to_prompt(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
            json_mode=True,
        )

        self.assertIn("objet JSON valide", prompt)
        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[USER]", prompt)

    def test_codex_provider_is_local_and_keyless(self):
        provider = PROVIDERS["Codex local"]

        self.assertEqual(provider["type"], "codex_cli")
        self.assertEqual(provider["default_model"], "gpt-5.5")
        self.assertEqual(provider["models"][0], "gpt-5.5")
        self.assertTrue(provider["local_only"])
        self.assertFalse(provider["requires_api_key"])


if __name__ == "__main__":
    unittest.main()
